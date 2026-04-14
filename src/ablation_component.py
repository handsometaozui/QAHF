#!/usr/bin/env python3
"""
QAHF 组件消融实验
验证各核心设计决策的必要性：
  1. 预测器架构：MLP vs 线性回归 vs 岭回归（均使用 23 维特征）
  2. 特征集：全量 23 维 vs 仅查询 14 维（MLP 预测器）
  3. 自适应预测 vs 训练集网格搜索所得的全局最优固定权重

关键约束：
  - 与 improved_experiment.py 共用相同的随机种子（42）和数据划分逻辑，
    确保 RRF / Oracle / QAHF Full 与主实验 Table 2 完全一致。
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_best_fixed_alpha(
    train_queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    bm25: BM25Retriever,
    dense: DenseRetriever,
    retrieval_depth: int = 500,
    rrf_k: int = 60,
) -> float:
    """
    在训练集上枚举 α∈[0.05, 0.95]（步长 0.05），
    选取使平均组合指标（0.35·NDCG@10 + 0.30·MRR@10 + 0.35·Recall@100）最高的 α，
    与伪标签生成时的目标函数保持一致。
    """
    alpha_grid = np.linspace(0.05, 0.95, 19)
    best_alpha, best_score = 0.5, -1.0

    for alpha in alpha_grid:
        total, count = 0.0, 0
        for qid, query in train_queries.items():
            if qid not in qrels:
                continue
            relevant_docs = set(did for did, rel in qrels[qid].items() if rel > 0)
            if not relevant_docs:
                continue

            bm25_res = bm25.search(query, top_k=retrieval_depth)
            dense_res = dense.search(query, top_k=retrieval_depth)
            sorted_docs = weighted_rrf_fuse(bm25_res, dense_res, alpha, rrf_k)

            retrieved_10 = [did for did, _ in sorted_docs[:10]]
            retrieved_100 = set(did for did, _ in sorted_docs[:100])

            ndcg = 0.0
            for rank, did in enumerate(retrieved_10, 1):
                if did in relevant_docs:
                    ndcg += 1.0 / np.log2(rank + 1)
            ideal = sum(
                1.0 / np.log2(r + 1)
                for r in range(1, min(len(relevant_docs), 10) + 1)
            )
            ndcg = ndcg / ideal if ideal > 0 else 0.0

            mrr = next(
                (1.0 / r for r, d in enumerate(retrieved_10, 1) if d in relevant_docs),
                0.0,
            )
            recall_100 = len(relevant_docs & retrieved_100) / len(relevant_docs)
            total += 0.35 * ndcg + 0.30 * mrr + 0.35 * recall_100
            count += 1

        avg = total / count if count > 0 else 0.0
        if avg > best_score:
            best_score = avg
            best_alpha = alpha

    return float(best_alpha)


def run_ablation_on_dataset(
    dataset: str,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.4,
    bm25_variant: str = "okapi",
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    limit_queries: int = 500,
) -> Dict:
    """在单个数据集上运行完整组件消融实验，返回各配置的评估结果。"""

    # ── 固定随机种子（与主实验一致）──────────────────────────────────────
    _set_seeds(42)

    print(f"\n{'='*70}")
    print(f"Ablation Experiment — {dataset.upper()}")
    print(f"{'='*70}")

    # ── 数据加载与划分（与 improved_experiment.py 完全一致）──────────────
    data_dir = BEIR_DIR / dataset
    corpus, queries, qrels = load_beir_data(data_dir, limit_queries=limit_queries)

    query_ids = list(queries.keys())
    random.seed(42)                       # 二次显式设定，确保 shuffle 结果可复现
    random.shuffle(query_ids)

    split = int(len(query_ids) * 0.5)
    train_queries = {qid: queries[qid] for qid in query_ids[:split] if qid in qrels}
    test_queries  = {qid: queries[qid] for qid in query_ids[split:]  if qid in qrels}
    test_qrels    = {qid: qrels[qid]   for qid in test_queries}

    print(f"  Train: {len(train_queries)} queries  |  Test: {len(test_queries)} queries")

    # ── 构建检索器 ─────────────────────────────────────────────────────────
    print("\nBuilding BM25 index...")
    bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b, variant=bm25_variant)
    bm25.index(corpus)

    print("Building dense index...")
    dense = DenseRetriever(model_name=dense_model)
    dense.build_index(corpus, batch_size=128)

    feature_extractor = QueryFeatureExtractor()
    feature_extractor.set_bm25(bm25)

    retrieval_depth, rrf_k = 500, 60

    # ── 生成训练集特征与伪标签（14 维） ────────────────────────────────────
    print("\nGenerating pseudo-labels and features for train set...")
    train_features_14, train_alphas = generate_pseudo_labels_with_features(
        train_queries, qrels, bm25, dense, feature_extractor,
        retrieval_depth=retrieval_depth, rrf_k=rrf_k,
    )

    n_total = len(train_alphas)
    n_train = int(n_total * 0.8)
    tr_f,  vl_f  = train_features_14[:n_train], train_features_14[n_train:]
    tr_l,  vl_l  = train_alphas[:n_train],       train_alphas[n_train:]

    # ── 预缓存测试集检索结果（所有配置共用，节省时间） ─────────────────────
    print("\nPre-caching test set retrieval results...")
    test_qids   = list(test_queries.keys())
    bm25_cache  = {}
    dense_cache = {}
    test_feat_14 = []

    for qid in test_qids:
        query = test_queries[qid]
        bm25_res  = bm25.search(query,  top_k=retrieval_depth)
        dense_res = dense.search(query, top_k=retrieval_depth)
        bm25_cache[qid]  = bm25_res
        dense_cache[qid] = dense_res

        qf = feature_extractor.extract_features(query)
        test_feat_14.append(qf)

    test_feat_14 = np.array(test_feat_14)   # (N_test, 14)

    # ── 特征标准化参数（从训练子集计算） ──────────────────────────────────
    feat_mean = tr_f.mean(axis=0)
    feat_std  = tr_f.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0

    tr_f_norm = (tr_f - feat_mean) / feat_std
    vl_f_norm = (vl_f - feat_mean) / feat_std
    test_f_norm_14 = (test_feat_14 - feat_mean) / feat_std

    evaluator   = RetrievalEvaluator(test_qrels)
    metrics_lst = ["mrr@10", "ndcg@10", "recall@100"]
    results     = {}

    # ── 公共函数：给定 α 数组，执行融合并评估 ─────────────────────────────
    def _fuse_and_eval(pred_alphas: np.ndarray) -> Dict:
        fused = {}
        for i, qid in enumerate(test_qids):
            sorted_docs = weighted_rrf_fuse(
                bm25_cache[qid], dense_cache[qid], float(pred_alphas[i]), rrf_k
            )
            fused[qid] = sorted_docs[:100]
        return evaluator.evaluate(fused, metrics_lst)

    # ════════════════════════════════════════════════════════════════════════
    # 配置 1：RRF（α=0.5）
    # ════════════════════════════════════════════════════════════════════════
    print("\n[1/6] RRF (α=0.5)...")
    results["rrf"] = _fuse_and_eval(np.full(len(test_qids), 0.5))

    # ════════════════════════════════════════════════════════════════════════
    # 配置 2：Best Fixed α（训练集网格搜索）
    # ════════════════════════════════════════════════════════════════════════
    print("[2/6] Best Fixed α (grid search on train set)...")
    best_alpha = find_best_fixed_alpha(
        train_queries, qrels, bm25, dense, retrieval_depth, rrf_k
    )
    print(f"       → best α = {best_alpha:.2f}")
    results["best_fixed"] = _fuse_and_eval(np.full(len(test_qids), best_alpha))
    results["best_fixed"]["best_alpha"] = best_alpha

    # ════════════════════════════════════════════════════════════════════════
    # 配置 3：Linear Regression（14 维）
    # ════════════════════════════════════════════════════════════════════════
    print("[3/6] Linear Regression (14-dim)...")
    lr_model_14 = LinearRegression()
    lr_model_14.fit(tr_f_norm, tr_l)
    results["linear_14"] = _fuse_and_eval(lr_model_14.predict(test_f_norm_14))

    # ════════════════════════════════════════════════════════════════════════
    # 配置 4：Ridge Regression（14 维）
    # ════════════════════════════════════════════════════════════════════════
    print("[4/6] Ridge Regression (14-dim)...")
    ridge_model_14 = Ridge(alpha=1.0)
    ridge_model_14.fit(tr_f_norm, tr_l)
    results["ridge_14"] = _fuse_and_eval(ridge_model_14.predict(test_f_norm_14))

    # ════════════════════════════════════════════════════════════════════════
    # 配置 5：QAHF（14 维 MLP，当前主模型）
    # ════════════════════════════════════════════════════════════════════════
    print("[5/6] QAHF (14-dim MLP)...")
    torch.manual_seed(42)
    qahf_14 = QAHF(use_retrieval_features=False)
    qahf_14.feature_extractor.set_bm25(bm25)
    qahf_14.train(
        train_features=tr_f,
        train_labels=tr_l,
        val_features=vl_f,
        val_labels=vl_l,
        epochs=150, batch_size=16, learning_rate=0.001,
    )
    qahf_14.calibrate(val_features=vl_f, val_labels=vl_l)
    alphas_14 = np.array(
        [qahf_14.predict_alpha(test_queries[qid]) for qid in test_qids]
    )
    results["mlp_14"] = _fuse_and_eval(alphas_14)

    # ════════════════════════════════════════════════════════════════════════
    # 配置 6：Oracle（每查询独立最优 α，理论上界）
    # ════════════════════════════════════════════════════════════════════════
    print("[6/6] Oracle...")
    alpha_grid = np.linspace(0.05, 0.95, 19)
    oracle_fused = {}
    for qid in test_qids:
        if qid not in test_qrels:
            continue
        relevant_docs = set(did for did, rel in test_qrels[qid].items() if rel > 0)
        if not relevant_docs:
            continue

        bm25_res  = bm25_cache[qid]
        dense_res = dense_cache[qid]
        best_a, best_s = 0.5, -1.0

        for alpha in alpha_grid:
            sorted_docs   = weighted_rrf_fuse(bm25_res, dense_res, alpha, rrf_k)
            retrieved_10  = [did for did, _ in sorted_docs[:10]]
            retrieved_100 = set(did for did, _ in sorted_docs[:100])

            ndcg = 0.0
            for rank, did in enumerate(retrieved_10, 1):
                if did in relevant_docs:
                    ndcg += 1.0 / np.log2(rank + 1)
            ideal = sum(
                1.0 / np.log2(r + 1)
                for r in range(1, min(len(relevant_docs), 10) + 1)
            )
            ndcg  = ndcg / ideal if ideal > 0 else 0.0
            mrr   = next(
                (1.0 / r for r, d in enumerate(retrieved_10, 1) if d in relevant_docs),
                0.0,
            )
            recall_100 = len(relevant_docs & retrieved_100) / len(relevant_docs)
            combined   = 0.35 * ndcg + 0.30 * mrr + 0.35 * recall_100
            if combined > best_s:
                best_s, best_a = combined, alpha

        oracle_fused[qid] = weighted_rrf_fuse(bm25_res, dense_res, best_a, rrf_k)[:100]

    results["oracle"] = evaluator.evaluate(oracle_fused, metrics_lst)

    # ── 打印摘要 ──────────────────────────────────────────────────────────
    rrf_ndcg = results["rrf"]["ndcg@10"]
    config_order = [
        ("rrf",          "RRF (α=0.5)"),
        ("best_fixed",   "Best Fixed α"),
        ("linear_14",    "Linear Reg. (14-dim)"),
        ("ridge_14",     "Ridge Reg. (14-dim)"),
        ("mlp_14",       "QAHF (14-dim MLP)"),
        ("oracle",       "Oracle"),
    ]

    print(f"\n{'─'*72}")
    print(f"{'Config':<28}  {'MRR@10':>8}  {'NDCG@10':>8}  {'Recall@100':>10}  Δ vs RRF")
    print(f"{'─'*72}")
    for key, label in config_order:
        r     = results[key]
        delta = (r["ndcg@10"] - rrf_ndcg) / rrf_ndcg * 100 if rrf_ndcg > 0 else 0.0
        sign  = "+" if delta >= 0 else ""
        print(
            f"{label:<28}  {r['mrr@10']:>8.4f}  {r['ndcg@10']:>8.4f}"
            f"  {r['recall@100']:>10.4f}  ({sign}{delta:.2f}%)"
        )
    print(f"{'─'*72}")

    return results


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "cqadupstack/android": {
        "bm25_k1": 1.2, "bm25_b": 0.4,
        "bm25_variant": "okapi", "limit_queries": 500,
    },
    "cqadupstack/english": {
        "bm25_k1": 1.2, "bm25_b": 0.4,
        "bm25_variant": "okapi", "limit_queries": 500,
    },
    "cqadupstack/physics": {
        "bm25_k1": 1.2, "bm25_b": 0.4,
        "bm25_variant": "okapi", "limit_queries": 500,
    },
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QAHF Component Ablation")
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to ablate (default: all three)",
    )
    args = parser.parse_args()

    all_results = {}

    for dataset in args.datasets:
        cfg = DATASET_CONFIGS.get(dataset, list(DATASET_CONFIGS.values())[0])
        res = run_ablation_on_dataset(dataset, **cfg)
        all_results[dataset] = res

        # 保存单数据集结果
        out_dir = RESULTS_DIR / dataset.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ablation_component.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: {m: float(v) for m, v in r.items() if isinstance(v, float)}
                 for k, r in res.items()},
                f, indent=2,
            )
        print(f"\nSaved → {out_path}")

    # ── 汇总打印（Paper Table 4 格式） ───────────────────────────────────
    datasets_ordered = [d for d in args.datasets]
    config_order = [
        ("rrf",          "RRF (α=0.5)"),
        ("best_fixed",   "Best Fixed α"),
        ("linear_14",    "Linear Reg. (14-dim)"),
        ("ridge_14",     "Ridge Reg. (14-dim)"),
        ("mlp_14",       "QAHF (14-dim MLP)"),
        ("oracle",       "Oracle"),
    ]

    print(f"\n{'='*90}")
    print("ABLATION SUMMARY — NDCG@10  (括号内为相对 RRF 的变化%)")
    print(f"{'='*90}")
    header = f"{'Config':<28}" + "".join(f"  {d.split('/')[-1].upper():>18}" for d in datasets_ordered)
    print(header)
    print("─" * 90)

    for key, label in config_order:
        row = f"{label:<28}"
        for dataset in datasets_ordered:
            r       = all_results[dataset][key]
            rrf_val = all_results[dataset]["rrf"]["ndcg@10"]
            ndcg    = r["ndcg@10"]
            delta   = (ndcg - rrf_val) / rrf_val * 100 if rrf_val > 0 else 0.0
            sign    = "+" if delta >= 0 else ""
            row += f"  {ndcg:.4f} ({sign}{delta:.2f}%)"
        print(row)
    print("─" * 90)
