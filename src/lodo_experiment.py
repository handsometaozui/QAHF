#!/usr/bin/env python3
"""
Leave-One-Dataset-Out (LODO) 跨数据集泛化实验

实验设计：
  对 6 个数据集循环 6 次，每次将 1 个数据集作为测试集（held-out），
  用剩余 5 个数据集的训练查询合并训练一个通用 MLP，
  在 held-out 数据集的测试集上评估 QAHF vs RRF。

目的：
  验证 14 维预检索查询特征是否具有跨语料库迁移能力。
  - 若 QAHF 持续优于 RRF → 特征具有领域无关的泛化性，方法的零样本实用价值得到验证；
  - 若性能大幅下降 → 方法本质是"域内自适应"，需要在论文中诚实说明使用前提。

与主实验（improved_experiment.py）的关键区别：
  主实验：train 和 test 来自同一数据集（域内有监督）
  本实验：train 来自 5 个外部数据集，test 来自 1 个未见数据集（跨域零样本）
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import BEIR_DIR, RESULTS_DIR
from baselines import BM25Retriever, DenseRetriever
from qahf_model import QAHF
from feature_extractor import QueryFeatureExtractor
from evaluator import RetrievalEvaluator
from improved_experiment import (
    load_beir_data,
    weighted_rrf_fuse,
    generate_pseudo_labels_with_features,
)

# ─────────────────────────────────────────────────────────────────────────────
# 数据集配置（与 ablation_component.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "fiqa":                    {"bm25_k1": 1.2, "bm25_b": 0.4},
    "scidocs":                 {"bm25_k1": 1.5, "bm25_b": 0.75},
    "cqadupstack/android":     {"bm25_k1": 1.2, "bm25_b": 0.4},
    "cqadupstack/english":     {"bm25_k1": 1.2, "bm25_b": 0.4},
    "cqadupstack/gaming":      {"bm25_k1": 1.2, "bm25_b": 0.4},
    "cqadupstack/physics":     {"bm25_k1": 1.2, "bm25_b": 0.4},
}

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K       = 60
RETRIEVAL_DEPTH = 500
SEED        = 42


def _set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_queries(queries: Dict, qrels: Dict, seed: int = SEED):
    """5:5 随机划分（与主实验完全一致）"""
    query_ids = list(queries.keys())
    random.seed(seed)
    random.shuffle(query_ids)
    split = int(len(query_ids) * 0.5)
    train_ids = query_ids[:split]
    test_ids  = query_ids[split:]
    train_q = {qid: queries[qid] for qid in train_ids if qid in qrels}
    test_q  = {qid: queries[qid] for qid in test_ids  if qid in qrels}
    test_qr = {qid: qrels[qid]   for qid in test_q}
    return train_q, test_q, test_qr


def run_lodo_experiment(
    limit_queries: int = 500,
    dense_model: str = DENSE_MODEL,
):
    """
    执行 Leave-One-Dataset-Out 实验，返回所有 fold 的结果字典。
    """
    _set_seeds()

    datasets = list(DATASET_CONFIGS.keys())
    all_results = {}

    # ─── 第一阶段：为所有数据集一次性加载数据 + 构建检索器 ──────────────────
    print("\n" + "=" * 80)
    print("LODO Experiment — Phase 1: Loading data and building retrievers")
    print("=" * 80)

    # 存储每个数据集的原始数据和检索器
    ds_data     = {}   # dataset -> (corpus, queries, qrels, train_q, test_q, test_qr)
    ds_bm25     = {}   # dataset -> BM25Retriever
    ds_dense    = {}   # dataset -> DenseRetriever

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n[Load] {ds}")
        corpus, queries, qrels = load_beir_data(
            BEIR_DIR / ds, limit_queries=limit_queries
        )
        train_q, test_q, test_qr = split_queries(queries, qrels)
        ds_data[ds] = (corpus, queries, qrels, train_q, test_q, test_qr)

        print(f"  Building BM25 (k1={cfg['bm25_k1']}, b={cfg['bm25_b']})...")
        bm25 = BM25Retriever(k1=cfg["bm25_k1"], b=cfg["bm25_b"])
        bm25.index(corpus)
        ds_bm25[ds] = bm25

        print(f"  Building Dense index...")
        dense = DenseRetriever(model_name=dense_model)
        dense.build_index(corpus, batch_size=128)
        ds_dense[ds] = dense

    # ─── 第二阶段：为每个数据集的训练集提取特征 + 伪标签 ───────────────────
    print("\n" + "=" * 80)
    print("LODO Experiment — Phase 2: Generating features and pseudo-labels")
    print("=" * 80)

    ds_train_features = {}   # dataset -> np.ndarray (N_train, 14)
    ds_train_labels   = {}   # dataset -> List[float]

    for ds in datasets:
        _, _, qrels, train_q, _, _ = ds_data[ds]
        bm25  = ds_bm25[ds]
        dense = ds_dense[ds]

        fe = QueryFeatureExtractor()
        fe.set_bm25(bm25)

        print(f"\n[Features] {ds} — {len(train_q)} train queries")
        features, labels = generate_pseudo_labels_with_features(
            train_q, qrels, bm25, dense, fe,
            retrieval_depth=RETRIEVAL_DEPTH, rrf_k=RRF_K,
        )
        ds_train_features[ds] = features
        ds_train_labels[ds]   = labels
        print(f"  Generated {len(labels)} pseudo-labels  "
              f"α: mean={np.mean(labels):.3f}, std={np.std(labels):.3f}")

    # ─── 第三阶段：Leave-One-Dataset-Out 循环 ────────────────────────────────
    print("\n" + "=" * 80)
    print("LODO Experiment — Phase 3: Cross-dataset evaluation (6 folds)")
    print("=" * 80)

    for held_out in datasets:
        training_sets = [ds for ds in datasets if ds != held_out]

        print(f"\n{'─'*70}")
        print(f"Fold: held-out = {held_out}")
        print(f"      train on  = {training_sets}")
        print(f"{'─'*70}")

        # ── 合并 5 个训练集的特征和标签 ──────────────────────────────────────
        combined_features = np.concatenate(
            [ds_train_features[ds] for ds in training_sets], axis=0
        )
        combined_labels = []
        for ds in training_sets:
            combined_labels.extend(ds_train_labels[ds])
        combined_labels = np.array(combined_labels, dtype=float)

        print(f"  Combined training samples: {len(combined_labels)}")
        print(f"  α distribution: mean={np.mean(combined_labels):.3f}, "
              f"std={np.std(combined_labels):.3f}")

        # ── 80/20 划分验证集（用于早停） ──────────────────────────────────────
        n_total = len(combined_labels)
        n_train = int(n_total * 0.8)
        _set_seeds()   # 重置种子确保 shuffle 可复现
        idx = np.random.permutation(n_total)
        tr_idx, vl_idx = idx[:n_train], idx[n_train:]

        tr_f = combined_features[tr_idx]
        vl_f = combined_features[vl_idx]
        tr_l = combined_labels[tr_idx].tolist()
        vl_l = combined_labels[vl_idx].tolist()

        # ── 训练通用 MLP ─────────────────────────────────────────────────────
        print(f"\n  Training universal MLP on {n_train} samples "
              f"(val: {len(vl_idx)})...")
        torch.manual_seed(SEED)
        qahf = QAHF(use_retrieval_features=False)
        # 注入 held-out 数据集的 BM25，供测试时提取 IDF 特征
        qahf.feature_extractor.set_bm25(ds_bm25[held_out])

        qahf.train(
            train_features=tr_f,
            train_labels=tr_l,
            val_features=vl_f,
            val_labels=vl_l,
            epochs=150,
            batch_size=16,
            learning_rate=0.001,
        )
        qahf.calibrate(val_features=vl_f, val_labels=vl_l)

        # ── 在 held-out 测试集上评估 ──────────────────────────────────────────
        _, _, _, _, test_q, test_qr = ds_data[held_out]
        bm25_ho  = ds_bm25[held_out]
        dense_ho = ds_dense[held_out]

        evaluator   = RetrievalEvaluator(test_qr)
        metrics_lst = ["mrr@10", "ndcg@10", "recall@100"]

        print(f"\n  Evaluating on {len(test_q)} test queries...")

        # 预缓存检索结果
        test_qids   = list(test_q.keys())
        bm25_cache  = {qid: bm25_ho.search(test_q[qid],  top_k=RETRIEVAL_DEPTH) for qid in test_qids}
        dense_cache = {qid: dense_ho.search(test_q[qid], top_k=RETRIEVAL_DEPTH) for qid in test_qids}

        def _eval(alphas: np.ndarray) -> Dict:
            fused = {}
            for i, qid in enumerate(test_qids):
                sorted_docs = weighted_rrf_fuse(
                    bm25_cache[qid], dense_cache[qid], float(alphas[i]), RRF_K
                )
                fused[qid] = sorted_docs[:100]
            return evaluator.evaluate(fused, metrics_lst)

        # RRF 基线（α=0.5）
        rrf_results  = _eval(np.full(len(test_qids), 0.5))

        # QAHF（通用 MLP，held-out BM25 已注入）
        qahf_alphas  = np.array([qahf.predict_alpha(test_q[qid]) for qid in test_qids])
        qahf_results = _eval(qahf_alphas)

        # 汇报结果
        rrf_ndcg  = rrf_results["ndcg@10"]
        qahf_ndcg = qahf_results["ndcg@10"]
        rel_gain  = (qahf_ndcg - rrf_ndcg) / rrf_ndcg * 100 if rrf_ndcg > 0 else 0.0

        print(f"\n  {'Method':<12} {'MRR@10':<10} {'NDCG@10':<10} {'Recall@100':<10}")
        print(f"  {'─'*45}")
        r = rrf_results
        print(f"  {'RRF':<12} {r['mrr@10']:<10.4f} {r['ndcg@10']:<10.4f} {r['recall@100']:<10.4f}")
        r = qahf_results
        print(f"  {'QAHF-LODO':<12} {r['mrr@10']:<10.4f} {r['ndcg@10']:<10.4f} {r['recall@100']:<10.4f}")
        print(f"  NDCG@10 relative gain vs RRF: {rel_gain:+.2f}%")
        print(f"  Predicted α: mean={np.mean(qahf_alphas):.3f}, std={np.std(qahf_alphas):.3f}")

        all_results[held_out] = {
            "rrf":         rrf_results,
            "qahf_lodo":   qahf_results,
            "ndcg_gain":   rel_gain,
            "alpha_mean":  float(np.mean(qahf_alphas)),
            "alpha_std":   float(np.std(qahf_alphas)),
            "n_train":     int(n_total),
            "n_test":      len(test_qids),
        }

    # ─── 汇总 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LODO SUMMARY — Cross-Dataset Generalization Results")
    print("=" * 80)

    print(f"\n{'Dataset':<25} {'RRF NDCG@10':<14} {'QAHF-LODO NDCG@10':<20} {'Gain':<10} {'Verdict'}")
    print("─" * 85)

    wins, losses = 0, 0
    for ds, res in all_results.items():
        rrf_n  = res["rrf"]["ndcg@10"]
        qahf_n = res["qahf_lodo"]["ndcg@10"]
        gain   = res["ndcg_gain"]
        verdict = "✓ QAHF wins" if gain > 0 else "✗ RRF wins"
        if gain > 0:
            wins += 1
        else:
            losses += 1
        ds_short = ds.replace("cqadupstack/", "CQA-")
        print(f"{ds_short:<25} {rrf_n:<14.4f} {qahf_n:<20.4f} {gain:+.2f}%   {verdict}")

    avg_rrf_ndcg  = np.mean([r["rrf"]["ndcg@10"]       for r in all_results.values()])
    avg_qahf_ndcg = np.mean([r["qahf_lodo"]["ndcg@10"] for r in all_results.values()])
    avg_gain      = np.mean([r["ndcg_gain"]             for r in all_results.values()])

    print("─" * 85)
    print(f"{'Average':<25} {avg_rrf_ndcg:<14.4f} {avg_qahf_ndcg:<20.4f} {avg_gain:+.2f}%")
    print(f"\nQAHF wins: {wins}/{len(datasets)} datasets")

    if wins >= 5:
        print("\n[结论] QAHF 在跨数据集设置下仍持续优于 RRF，预检索查询特征具有跨域迁移能力。")
        print("       可在论文中以此结果支持方法的通用性声明，有效应对 W1（评估设置）的质疑。")
    elif wins >= 3:
        print("\n[结论] QAHF 在多数数据集上优于 RRF，但泛化能力不稳定。")
        print("       建议在论文中如实呈现，承认方法在部分领域的泛化局限，")
        print("       同时保留跨域设置下的正向结果作为支撑证据。")
    else:
        print("\n[结论] QAHF 在跨数据集设置下未能稳定超越 RRF，")
        print("       说明方法的核心价值在于域内自适应（in-domain adaptation）。")
        print("       需在论文中修改贡献声明：将方法定位为'已有少量标注数据的检索系统优化工具'，")
        print("       而非'通用零样本自适应融合'，并在 Abstract 中说明使用前提。")

    # ─── 保存结果 ─────────────────────────────────────────────────────────────
    lodo_dir = RESULTS_DIR / "lodo"
    lodo_dir.mkdir(parents=True, exist_ok=True)
    out_path = lodo_dir / "lodo_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=float, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LODO cross-dataset generalization experiment")
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Max queries per dataset (default: 500, use 0 for all)"
    )
    parser.add_argument(
        "--dense_model", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    run_lodo_experiment(limit_queries=limit, dense_model=args.dense_model)
