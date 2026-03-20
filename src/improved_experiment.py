#!/usr/bin/env python3
"""
改进的 QAHF 实验
解决问题：alpha 分布过于极端（均值接近 0 或 1）
"""

import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

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


def generate_improved_pseudo_labels(queries: Dict[str, str],
                                     qrels: Dict[str, Dict[str, int]],
                                     bm25: BM25Retriever,
                                     dense: DenseRetriever,
                                     top_k: int = 100):
    """
    改进的伪标签生成：使用 MRR@10 和 NDCG@10 组合
    """
    print("\n" + "=" * 60)
    print("Generating improved pseudo labels...")
    print("=" * 60)
    
    query_texts = []
    optimal_alphas = []
    
    alpha_grid = np.linspace(0.1, 0.9, 9)  # [0.1, 0.2, ..., 0.9] 避免极端值
    
    for i, (qid, query_text) in enumerate(queries.items()):
        if qid not in qrels:
            continue
        
        relevant_docs = set(did for did, rel in qrels[qid].items() if rel > 0)
        if not relevant_docs:
            continue
        
        # 获取两种检索结果
        bm25_results = bm25.search(query_text, top_k=top_k * 2)
        dense_results = dense.search(query_text, top_k=top_k * 2)
        
        bm25_scores = {did: score for did, score in bm25_results}
        dense_scores = {did: score for did, score in dense_results}
        
        # 归一化
        bm25_norm = normalize_scores(bm25_scores)
        dense_norm = normalize_scores(dense_scores)
        
        # 网格搜索最优 α - 使用 MRR@10 和 NDCG@10 组合
        best_alpha = 0.5
        best_score = 0.0
        
        for alpha in alpha_grid:
            # 融合
            all_doc_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
            final_scores = {}
            
            for doc_id in all_doc_ids:
                bm25_s = bm25_norm.get(doc_id, 0)
                dense_s = dense_norm.get(doc_id, 0)
                final_scores[doc_id] = alpha * bm25_s + (1 - alpha) * dense_s
            
            # 计算 MRR@10
            sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            retrieved_10 = [did for did, _ in sorted_docs[:10]]
            
            # MRR@10
            mrr = 0.0
            for rank, did in enumerate(retrieved_10, 1):
                if did in relevant_docs:
                    mrr = 1.0 / rank
                    break
            
            # Recall@10
            recall_10 = len(set(retrieved_10) & relevant_docs) / len(relevant_docs) if relevant_docs else 0
            
            # 组合得分 (MRR 权重更高)
            combined_score = 0.7 * mrr + 0.3 * recall_10
            
            if combined_score > best_score:
                best_score = combined_score
                best_alpha = alpha
        
        query_texts.append(query_text)
        optimal_alphas.append(best_alpha)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")
    
    print(f"\nGenerated labels for {len(query_texts)} queries")
    print(f"Alpha distribution: min={np.min(optimal_alphas):.2f}, max={np.max(optimal_alphas):.2f}, mean={np.mean(optimal_alphas):.2f}")
    
    return query_texts, optimal_alphas


def run_improved_experiment(dataset: str = "fiqa", limit_queries: int = 200,
                            bm25_k1: float = 1.5, bm25_b: float = 0.75,
                            bm25_variant: str = "okapi",
                            dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):  # bge-small-en-v1.5 召回率反而下降，保留 MiniLM
    """运行改进的 QAHF 实验"""
    print("\n" + "=" * 80)
    print(f"QAHF Improved Experiment - {dataset.upper()}")
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
    
    # BM25
    print("\n[1/2] Building BM25 index...")
    bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b, variant=bm25_variant)
    bm25.index(corpus)
    print(f"  BM25 indexed {len(corpus)} documents")
    
    # Dense
    print("\n[2/2] Building dense index...")
    dense = DenseRetriever(model_name=dense_model)
    dense.build_index(corpus, batch_size=128)
    print(f"  Dense indexed {len(corpus)} documents")
    
    # 4. 训练 QAHF
    print("\n" + "=" * 60)
    print("Training QAHF model...")
    print("=" * 60)
    
    train_query_texts, train_alphas = generate_improved_pseudo_labels(
        train_queries, qrels, bm25, dense, top_k=100
    )
    
    qahf = QAHF()
    
    n = len(train_query_texts)
    n_train = int(n * 0.8)
    
    qahf.train(
        train_queries=train_query_texts[:n_train],
        train_labels=train_alphas[:n_train],
        val_queries=train_query_texts[n_train:],
        val_labels=train_alphas[n_train:],
        epochs=100,
        batch_size=16,
        learning_rate=0.001
    )
    
    # 5. 运行测试
    print("\n" + "=" * 60)
    print("Running test experiments...")
    print("=" * 60)
    
    evaluator = RetrievalEvaluator(test_qrels)
    results = {}
    
    # BM25
    print("\n[1/4] Running BM25...")
    bm25_results = {}
    for qid, query in test_queries.items():
        bm25_results[qid] = bm25.search(query, top_k=100)
    results["bm25"] = evaluator.evaluate(bm25_results, ["mrr@10", "ndcg@10", "recall@100"])
    
    # Dense
    print("[2/4] Running Dense...")
    dense_results = {}
    for qid, query in test_queries.items():
        dense_results[qid] = dense.search(query, top_k=100)
    results["dense"] = evaluator.evaluate(dense_results, ["mrr@10", "ndcg@10", "recall@100"])
    
    # Hybrid RRF
    print("[3/4] Running Hybrid RRF...")
    hybrid_rrf = RRFHybridRetriever(bm25, dense, k=60)
    rrf_results = {}
    for qid, query in test_queries.items():
        rrf_results[qid] = hybrid_rrf.search(query, top_k=100)
    results["hybrid_rrf"] = evaluator.evaluate(rrf_results, ["mrr@10", "ndcg@10", "recall@100"])
    
    # QAHF
    print("[4/4] Running QAHF...")
    qahf_results = {}
    alpha_values = []
    for qid, query in test_queries.items():
        alpha = qahf.predict_alpha(query)
        alpha_values.append(alpha)
        
        bm25_res = bm25.search(query, top_k=200)
        dense_res = dense.search(query, top_k=200)
        
        bm25_scores = {did: score for did, score in bm25_res}
        dense_scores = {did: score for did, score in dense_res}
        bm25_norm = normalize_scores(bm25_scores)
        dense_norm = normalize_scores(dense_scores)
        
        all_doc_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
        final_scores = {}
        for doc_id in all_doc_ids:
            final_scores[doc_id] = alpha * bm25_norm.get(doc_id, 0) + (1 - alpha) * dense_norm.get(doc_id, 0)
        
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        qahf_results[qid] = sorted_results
    
    results["qahf"] = evaluator.evaluate(qahf_results, ["mrr@10", "ndcg@10", "recall@100"])
    
    # 6. 输出结果
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"\n{'Method':<15} {'MRR@10':<12} {'NDCG@10':<12} {'Recall@100':<12}")
    print("-" * 60)
    for method in ["bm25", "dense", "hybrid_rrf", "qahf"]:
        r = results[method]
        print(f"{method:<15} {r['mrr@10']:<12.4f} {r['ndcg@10']:<12.4f} {r['recall@100']:<12.4f}")
    
    # 改进分析
    print("\n" + "=" * 60)
    print("QAHF Analysis:")
    print("=" * 60)
    
    best_baseline = max(
        ("bm25", results["bm25"]["mrr@10"]),
        ("dense", results["dense"]["mrr@10"]),
        ("hybrid_rrf", results["hybrid_rrf"]["mrr@10"])
    , key=lambda x: x[1])
    
    qahf_mrr = results["qahf"]["mrr@10"]
    improvement = (qahf_mrr - best_baseline[1]) / best_baseline[1] * 100
    
    print(f"  Best baseline: {best_baseline[0]} (MRR@10: {best_baseline[1]:.4f})")
    print(f"  QAHF MRR@10: {qahf_mrr:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    
    print(f"\n  Alpha statistics:")
    print(f"    Mean: {np.mean(alpha_values):.3f}")
    print(f"    Std: {np.std(alpha_values):.3f}")
    print(f"    Min: {np.min(alpha_values):.3f}")
    print(f"    Max: {np.max(alpha_values):.3f}")
    
    # 保存结果
    output = {
        "dataset": dataset,
        "test_queries": len(test_queries),
        "train_queries": len(train_queries),
        "bm25_config": {"k1": bm25_k1, "b": bm25_b, "variant": bm25_variant},
        "dense_model": dense_model,
        "results": results,
        "alpha_stats": {
            "mean": float(np.mean(alpha_values)),
            "std": float(np.std(alpha_values)),
            "min": float(np.min(alpha_values)),
            "max": float(np.max(alpha_values))
        }
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