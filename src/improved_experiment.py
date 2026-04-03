#!/usr/bin/env python3
"""
改进的 QAHF 实验 v2
核心改进：检索感知特征（retrieval-aware features）
- 在 BM25/Dense 检索完成后，提取分数分布特征（9维）
- 让模型能看到每个检索器对当前查询的置信度和一致性
- 结合查询特征（14维）共 23维，预测最优融合权重 α
"""

import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from baselines import BM25Retriever, DenseRetriever, RRFHybridRetriever
from qahf_model import QAHF
from feature_extractor import QueryFeatureExtractor
from evaluator import RetrievalEvaluator


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-Max 归一化"""
    if not scores:
        return scores
    values = list(scores.values())
    min_val, max_val = min(values), max(values)
    if max_val - min_val < 1e-9:
        return {k: 0.5 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def load_beir_data(data_dir: Path, limit_queries: int = None):
    """加载 BEIR 数据"""
    print("\n" + "=" * 60)
    print(f"Loading data from: {data_dir}")
    print("=" * 60)

    # 加载文档
    corpus = {}
    with open(data_dir / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = str(doc["_id"])
            title = doc.get("title", "")
            text = doc.get("text", "")
            corpus[doc_id] = f"{title} {text}".strip()
    print(f"  Corpus: {len(corpus)} documents")

    # 加载查询
    queries = {}
    with open(data_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            query = json.loads(line.strip())
            query_id = str(query["_id"])
            query_text = query.get("text", "")
            queries[query_id] = query_text
    print(f"  Queries: {len(queries)}")

    # 加载 qrels
    qrels = {}
    qrels_file = data_dir / "qrels" / "test.tsv"
    if not qrels_file.exists():
        qrels_file = data_dir / "qrels" / "dev.tsv"

    if qrels_file.exists():
        with open(qrels_file, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if i == 0 and (parts[0] == "query-id" or parts[0] == "query_id"):
                    continue
                if len(parts) >= 3:
                    query_id = str(parts[0])
                    doc_id = str(parts[1])
                    try:
                        relevance = int(parts[2])
                    except:
                        continue
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = relevance
    print(f"  Qrels: {len(qrels)} queries with relevance labels")

    # 限制查询数量
    if limit_queries:
        queries_with_qrels = [qid for qid in queries if qid in qrels]
        if len(queries_with_qrels) > limit_queries:
            random.seed(42)
            sampled = random.sample(queries_with_qrels, limit_queries)
            queries = {qid: queries[qid] for qid in sampled}
            qrels = {qid: qrels[qid] for qid in sampled}
        print(f"  Limited to {len(queries)} queries")

    return corpus, queries, qrels


def weighted_rrf_fuse(bm25_results, dense_results, alpha, rrf_k=60):
    """加权 RRF 融合"""
    bm25_ranks = {did: rank + 1 for rank, (did, _) in enumerate(bm25_results)}
    dense_ranks = {did: rank + 1 for rank, (did, _) in enumerate(dense_results)}

    # 未被检索到的文档排名：各自列表长度 + 1（与标准 RRF 一致）
    bm25_default = len(bm25_results) + 1
    dense_default = len(dense_results) + 1

    all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    final_scores = {}
    for doc_id in all_doc_ids:
        bm25_r = bm25_ranks.get(doc_id, bm25_default)
        dense_r = dense_ranks.get(doc_id, dense_default)
        final_scores[doc_id] = alpha / (rrf_k + bm25_r) + (1 - alpha) / (rrf_k + dense_r)

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)


def generate_pseudo_labels_with_features(queries: Dict[str, str],
                                          qrels: Dict[str, Dict[str, int]],
                                          bm25: BM25Retriever,
                                          dense: DenseRetriever,
                                          feature_extractor: QueryFeatureExtractor,
                                          retrieval_depth: int = 500,
                                          rrf_k: int = 60):
    """
    生成伪标签，同时提取查询特征 + 检索感知特征。

    Returns:
        features: np.ndarray (N, 23) 完整特征矩阵
        optimal_alphas: list[float] 最优 α 值
    """
    print("\n" + "=" * 60)
    print("Generating pseudo labels with retrieval-aware features...")
    print("=" * 60)

    all_features = []
    optimal_alphas = []

    alpha_grid = np.linspace(0.05, 0.95, 19)

    for i, (qid, query_text) in enumerate(queries.items()):
        if qid not in qrels:
            continue

        relevant_docs = set(did for did, rel in qrels[qid].items() if rel > 0)
        if not relevant_docs:
            continue

        # 检索
        bm25_results = bm25.search(query_text, top_k=retrieval_depth)
        dense_results = dense.search(query_text, top_k=retrieval_depth)

        # 提取特征（查询 14维 + 检索感知 9维 = 23维）
        query_features = feature_extractor.extract_features(query_text)
        retrieval_features = feature_extractor.extract_retrieval_features(bm25_results, dense_results)
        combined_features = np.concatenate([query_features, retrieval_features])

        # 网格搜索最优 α（加权 RRF）
        best_alpha = 0.5
        best_score = -1.0

        for alpha in alpha_grid:
            sorted_docs = weighted_rrf_fuse(bm25_results, dense_results, alpha, rrf_k)
            retrieved_10 = [did for did, _ in sorted_docs[:10]]
            retrieved_100 = set(did for did, _ in sorted_docs[:100])

            # NDCG@10
            ndcg = 0.0
            for rank, did in enumerate(retrieved_10, 1):
                if did in relevant_docs:
                    ndcg += 1.0 / np.log2(rank + 1)
            ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant_docs), 10) + 1))
            ndcg = ndcg / ideal if ideal > 0 else 0.0

            # MRR@10
            mrr = 0.0
            for rank, did in enumerate(retrieved_10, 1):
                if did in relevant_docs:
                    mrr = 1.0 / rank
                    break

            # Recall@100
            recall_100 = len(relevant_docs & retrieved_100) / len(relevant_docs)

            # 组合得分：排序 + 召回并重
            combined_score = 0.35 * ndcg + 0.30 * mrr + 0.35 * recall_100

            if combined_score > best_score:
                best_score = combined_score
                best_alpha = alpha

        all_features.append(combined_features)
        optimal_alphas.append(best_alpha)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")

    features_matrix = np.array(all_features)
    print(f"\nGenerated labels for {len(optimal_alphas)} queries")
    print(f"Alpha distribution: min={np.min(optimal_alphas):.2f}, max={np.max(optimal_alphas):.2f}, "
          f"mean={np.mean(optimal_alphas):.2f}, std={np.std(optimal_alphas):.2f}")

    return features_matrix, optimal_alphas


def run_improved_experiment(dataset: str = "fiqa", limit_queries: int = 200,
                            bm25_k1: float = 1.5, bm25_b: float = 0.75,
                            bm25_variant: str = "okapi",
                            dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """运行改进的 QAHF 实验（检索感知版本）"""
    # 固定所有随机种子，确保结果可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\n" + "=" * 80)
    print(f"QAHF Retrieval-Aware Experiment - {dataset.upper()}")
    print("=" * 80)

    data_dir = BEIR_DIR / dataset
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    corpus, queries, qrels = load_beir_data(data_dir, limit_queries=limit_queries)

    # 2. 划分
    query_ids = list(queries.keys())
    random.seed(42)
    random.shuffle(query_ids)

    split = int(len(query_ids) * 0.5)
    train_queries = {qid: queries[qid] for qid in query_ids[:split] if qid in qrels}
    test_queries = {qid: queries[qid] for qid in query_ids[split:] if qid in qrels}
    test_qrels = {qid: qrels[qid] for qid in test_queries}

    print(f"  Train queries: {len(train_queries)}")
    print(f"  Test queries: {len(test_queries)}")

    # 3. 构建检索器
    print("\n" + "=" * 60)
    print("Building retrievers...")
    print("=" * 60)

    print("\n[1/2] Building BM25 index...")
    bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b, variant=bm25_variant)
    bm25.index(corpus)

    print("\n[2/2] Building dense index...")
    dense = DenseRetriever(model_name=dense_model)
    dense.build_index(corpus, batch_size=128)

    # 4. 生成伪标签 + 提取特征
    feature_extractor = QueryFeatureExtractor()
    feature_extractor.set_bm25(bm25)

    train_features, train_alphas = generate_pseudo_labels_with_features(
        train_queries, qrels, bm25, dense, feature_extractor,
        retrieval_depth=500, rrf_k=60
    )

    # 5. 训练 QAHF
    print("\n" + "=" * 60)
    print("Training QAHF model (retrieval-aware)...")
    print("=" * 60)

    torch.manual_seed(42)  # dense encoding 会消耗 torch 随机状态，此处重置确保训练可复现
    qahf = QAHF(use_retrieval_features=True)
    qahf.feature_extractor.set_bm25(bm25)

    n = len(train_alphas)
    n_train = int(n * 0.8)

    qahf.train(
        train_features=train_features[:n_train],
        train_labels=train_alphas[:n_train],
        val_features=train_features[n_train:],
        val_labels=train_alphas[n_train:],
        epochs=150,
        batch_size=16,
        learning_rate=0.001
    )

    # 温度校准：用验证集搜索最优温度，拉大 α 方差
    print("\n" + "=" * 60)
    print("Calibrating temperature...")
    print("=" * 60)
    qahf.calibrate(
        val_features=train_features[n_train:],
        val_labels=train_alphas[n_train:]
    )

    # 6. 测试
    print("\n" + "=" * 60)
    print("Running test experiments...")
    print("=" * 60)

    evaluator = RetrievalEvaluator(test_qrels)
    results = {}
    retrieval_depth = 500
    rrf_k = 60

    # BM25
    print("\n[1/5] Running BM25...")
    bm25_all_results = {}
    for qid, query in test_queries.items():
        bm25_all_results[qid] = bm25.search(query, top_k=100)
    results["bm25"] = evaluator.evaluate(bm25_all_results, ["mrr@10", "ndcg@10", "recall@100"])

    # Dense
    print("[2/5] Running Dense...")
    dense_all_results = {}
    for qid, query in test_queries.items():
        dense_all_results[qid] = dense.search(query, top_k=100)
    results["dense"] = evaluator.evaluate(dense_all_results, ["mrr@10", "ndcg@10", "recall@100"])

    # Hybrid RRF（标准，等权重）
    print("[3/5] Running Hybrid RRF...")
    hybrid_rrf = RRFHybridRetriever(bm25, dense, k=rrf_k)
    rrf_all_results = {}
    for qid, query in test_queries.items():
        rrf_all_results[qid] = hybrid_rrf.search(query, top_k=100)
    results["hybrid_rrf"] = evaluator.evaluate(rrf_all_results, ["mrr@10", "ndcg@10", "recall@100"])

    # QAHF（检索感知加权 RRF）
    print("[4/5] Running QAHF (retrieval-aware)...")
    qahf_results = {}
    alpha_values = []
    for qid, query in test_queries.items():
        bm25_res = bm25.search(query, top_k=retrieval_depth)
        dense_res = dense.search(query, top_k=retrieval_depth)

        alpha = qahf.predict_alpha(query, bm25_res, dense_res)
        alpha_values.append(alpha)

        sorted_results = weighted_rrf_fuse(bm25_res, dense_res, alpha, rrf_k)
        qahf_results[qid] = sorted_results[:100]

    results["qahf"] = evaluator.evaluate(qahf_results, ["mrr@10", "ndcg@10", "recall@100"])

    # Oracle（每个查询用最优 α，作为上界参考）
    print("[5/5] Computing Oracle upper bound...")
    oracle_results = {}
    oracle_alphas = []
    for qid, query in test_queries.items():
        if qid not in test_qrels:
            continue
        relevant_docs = set(did for did, rel in test_qrels[qid].items() if rel > 0)
        if not relevant_docs:
            continue

        bm25_res = bm25.search(query, top_k=retrieval_depth)
        dense_res = dense.search(query, top_k=retrieval_depth)

        best_alpha = 0.5
        best_score = -1.0
        alpha_grid = np.linspace(0.05, 0.95, 19)

        for alpha in alpha_grid:
            sorted_docs = weighted_rrf_fuse(bm25_res, dense_res, alpha, rrf_k)
            retrieved_10 = [did for did, _ in sorted_docs[:10]]
            retrieved_100 = set(did for did, _ in sorted_docs[:100])

            ndcg = 0.0
            for rank, did in enumerate(retrieved_10, 1):
                if did in relevant_docs:
                    ndcg += 1.0 / np.log2(rank + 1)
            ideal_dcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant_docs), 10) + 1))
            ndcg = ndcg / ideal_dcg if ideal_dcg > 0 else 0.0

            mrr = 0.0
            for rank, did in enumerate(retrieved_10, 1):
                if did in relevant_docs:
                    mrr = 1.0 / rank
                    break

            recall_100 = len(relevant_docs & retrieved_100) / len(relevant_docs)
            combined = 0.35 * ndcg + 0.30 * mrr + 0.35 * recall_100

            if combined > best_score:
                best_score = combined
                best_alpha = alpha

        oracle_alphas.append(best_alpha)
        sorted_docs = weighted_rrf_fuse(bm25_res, dense_res, best_alpha, rrf_k)
        oracle_results[qid] = sorted_docs[:100]

    results["oracle"] = evaluator.evaluate(oracle_results, ["mrr@10", "ndcg@10", "recall@100"])

    # 7. 输出结果
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Method':<15} {'MRR@10':<12} {'NDCG@10':<12} {'Recall@100':<12}")
    print("-" * 60)
    for method in ["bm25", "dense", "hybrid_rrf", "qahf", "oracle"]:
        r = results[method]
        print(f"{method:<15} {r['mrr@10']:<12.4f} {r['ndcg@10']:<12.4f} {r['recall@100']:<12.4f}")

    # 改进分析
    print("\n" + "=" * 60)
    print("QAHF Analysis:")
    print("=" * 60)

    rrf_metrics = results["hybrid_rrf"]
    qahf_metrics = results["qahf"]
    oracle_metrics = results["oracle"]

    for metric in ["mrr@10", "ndcg@10", "recall@100"]:
        rrf_val = rrf_metrics[metric]
        qahf_val = qahf_metrics[metric]
        oracle_val = oracle_metrics[metric]
        improvement = (qahf_val - rrf_val) / rrf_val * 100 if rrf_val > 0 else 0
        oracle_gap = (oracle_val - qahf_val) / oracle_val * 100 if oracle_val > 0 else 0
        print(f"  {metric}: QAHF vs RRF = {improvement:+.2f}%, Oracle gap = {oracle_gap:.2f}%")

    print(f"\n  Alpha statistics:")
    print(f"    Mean: {np.mean(alpha_values):.3f}")
    print(f"    Std: {np.std(alpha_values):.3f}")
    print(f"    Min: {np.min(alpha_values):.3f}")
    print(f"    Max: {np.max(alpha_values):.3f}")

    if oracle_alphas:
        print(f"\n  Oracle Alpha statistics:")
        print(f"    Mean: {np.mean(oracle_alphas):.3f}")
        print(f"    Std: {np.std(oracle_alphas):.3f}")
        print(f"    Min: {np.min(oracle_alphas):.3f}")
        print(f"    Max: {np.max(oracle_alphas):.3f}")

    # 保存结果
    output = {
        "dataset": dataset,
        "test_queries": len(test_queries),
        "train_queries": len(train_queries),
        "bm25_config": {"k1": bm25_k1, "b": bm25_b, "variant": bm25_variant},
        "dense_model": dense_model,
        "retrieval_depth": retrieval_depth,
        "use_retrieval_features": True,
        "results": results,
        "temperature": qahf.temperature,
        "alpha_center": qahf.alpha_center,
        "alpha_stats": {
            "mean": float(np.mean(alpha_values)),
            "std": float(np.std(alpha_values)),
            "min": float(np.min(alpha_values)),
            "max": float(np.max(alpha_values))
        },
        "oracle_alpha_stats": {
            "mean": float(np.mean(oracle_alphas)),
            "std": float(np.std(oracle_alphas)),
            "min": float(np.min(oracle_alphas)),
            "max": float(np.max(oracle_alphas))
        } if oracle_alphas else {}
    }

    dense_model_short = dense_model.split("/")[-1]
    output_path = results_dir / f"bm25_{bm25_variant}_k{bm25_k1}_b{bm25_b}_{dense_model_short}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nResults saved to {output_path}")

    return results, alpha_values


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fiqa", type=str)
    parser.add_argument("--limit", default=200, type=int)
    parser.add_argument("--bm25_k1", default=1.5, type=float)
    parser.add_argument("--bm25_b", default=0.75, type=float)
    parser.add_argument("--bm25_variant", default="okapi", type=str, choices=["okapi", "plus"])
    parser.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2", type=str,
                        help="Dense retrieval model name")
    args = parser.parse_args()

    run_improved_experiment(args.dataset, args.limit,
                            bm25_k1=args.bm25_k1, bm25_b=args.bm25_b,
                            bm25_variant=args.bm25_variant,
                            dense_model=args.dense_model)
