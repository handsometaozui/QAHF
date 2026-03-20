#!/usr/bin/env python3
"""
完整 QAHF 实验流程
包括：数据加载、检索器构建、QAHF训练、对比实验
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
from baselines import BM25Retriever, DenseRetriever, HybridRetriever, RRFHybridRetriever
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


def generate_pseudo_labels(queries: Dict[str, str],
                           qrels: Dict[str, Dict[str, int]],
                           bm25: BM25Retriever,
                           dense: DenseRetriever,
                           top_k: int = 100):
    """生成伪标签：为每个查询找最优 alpha"""
    print("\n" + "=" * 60)
    print("Generating pseudo labels for QAHF training...")
    print("=" * 60)
    
    query_texts = []
    optimal_alphas = []
    
    alpha_grid = np.linspace(0, 1, 11)  # [0, 0.1, ..., 1.0]
    
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
        
        # 网格搜索最优 α
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
            
            # 计算 Recall@10
            sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            retrieved = set(did for did, _ in sorted_docs[:10])
            recall = len(relevant_docs & retrieved) / len(relevant_docs) if relevant_docs else 0
            
            if recall > best_score:
                best_score = recall
                best_alpha = alpha
        
        query_texts.append(query_text)
        optimal_alphas.append(best_alpha)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")
    
    print(f"\nGenerated labels for {len(query_texts)} queries")
    print(f"Alpha distribution: min={np.min(optimal_alphas):.2f}, max={np.max(optimal_alphas):.2f}, mean={np.mean(optimal_alphas):.2f}")
    
    return query_texts, optimal_alphas


def run_full_experiment():
    """运行完整 QAHF 实验"""
    print("\n" + "=" * 80)
    print("QAHF Full Experiment Pipeline")
    print("=" * 80)
    
    # 配置
    dataset = "scifact"
    data_dir = BEIR_DIR / dataset
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    corpus, queries, qrels = load_beir_data(data_dir)
    
    # 2. 划分训练/测试集
    print("\n" + "=" * 60)
    print("Splitting train/test sets...")
    print("=" * 60)
    
    query_ids = list(queries.keys())
    random.seed(42)
    random.shuffle(query_ids)
    
    split = int(len(query_ids) * 0.5)  # 50% 训练, 50% 测试
    train_ids = query_ids[:split]
    test_ids = query_ids[split:]
    
    train_queries = {qid: queries[qid] for qid in train_ids if qid in qrels}
    test_queries = {qid: queries[qid] for qid in test_ids if qid in qrels}
    test_qrels = {qid: qrels[qid] for qid in test_queries}
    
    print(f"  Train queries: {len(train_queries)}")
    print(f"  Test queries: {len(test_queries)}")
    
    # 3. 构建检索器
    print("\n" + "=" * 60)
    print("Building retrievers...")
    print("=" * 60)
    
    # BM25
    print("\n[1/2] Building BM25 index...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)
    print(f"  BM25 indexed {len(corpus)} documents")
    
    # Dense
    print("\n[2/2] Building dense index...")
    dense = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dense.build_index(corpus, batch_size=128)
    print(f"  Dense indexed {len(corpus)} documents")
    
    # 4. 训练 QAHF
    print("\n" + "=" * 60)
    print("Training QAHF model...")
    print("=" * 60)
    
    # 生成伪标签
    train_query_texts, train_alphas = generate_pseudo_labels(
        train_queries, qrels, bm25, dense, top_k=100
    )
    
    # 训练模型
    qahf = QAHF()
    
    # 划分训练/验证
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
    
    # 保存模型
    model_path = results_dir / "qahf_model.pt"
    qahf.save_model(model_path)
    
    # 5. 运行测试实验
    print("\n" + "=" * 60)
    print("Running test experiments...")
    print("=" * 60)
    
    evaluator = RetrievalEvaluator(test_qrels)
    
    # 基线方法
    results = {}
    
    # BM25
    print("\n[1/4] Running BM25...")
    bm25_results = {}
    for qid, query in test_queries.items():
        bm25_results[qid] = bm25.search(query, top_k=100)
    results["bm25"] = evaluator.evaluate(bm25_results, ["mrr@10", "ndcg@10", "recall@100"])
    print(f"  BM25: MRR@10={results['bm25']['mrr@10']:.4f}")
    
    # Dense
    print("\n[2/4] Running Dense...")
    dense_results = {}
    for qid, query in test_queries.items():
        dense_results[qid] = dense.search(query, top_k=100)
    results["dense"] = evaluator.evaluate(dense_results, ["mrr@10", "ndcg@10", "recall@100"])
    print(f"  Dense: MRR@10={results['dense']['mrr@10']:.4f}")
    
    # Hybrid RRF
    print("\n[3/4] Running Hybrid RRF...")
    hybrid_rrf = RRFHybridRetriever(bm25, dense, k=60)
    rrf_results = {}
    for qid, query in test_queries.items():
        rrf_results[qid] = hybrid_rrf.search(query, top_k=100)
    results["hybrid_rrf"] = evaluator.evaluate(rrf_results, ["mrr@10", "ndcg@10", "recall@100"])
    print(f"  Hybrid RRF: MRR@10={results['hybrid_rrf']['mrr@10']:.4f}")
    
    # QAHF
    print("\n[4/4] Running QAHF...")
    qahf_results = {}
    alpha_values = []
    for qid, query in test_queries.items():
        alpha = qahf.predict_alpha(query)
        alpha_values.append(alpha)
        
        # 获取两种检索结果
        bm25_res = bm25.search(query, top_k=200)
        dense_res = dense.search(query, top_k=200)
        
        # 融合
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
    print(f"  QAHF: MRR@10={results['qahf']['mrr@10']:.4f}")
    
    # 6. 输出结果
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"\n{'Method':<15} {'MRR@10':<12} {'NDCG@10':<12} {'Recall@100':<12}")
    print("-" * 60)
    for method in ["bm25", "dense", "hybrid_rrf", "qahf"]:
        r = results[method]
        print(f"{method:<15} {r['mrr@10']:<12.4f} {r['ndcg@10']:<12.4f} {r['recall@100']:<12.4f}")
    
    # 计算提升
    print("\n" + "=" * 60)
    print("QAHF Improvement over Baselines:")
    print("=" * 60)
    
    best_baseline_mrr = max(results["bm25"]["mrr@10"], results["dense"]["mrr@10"], results["hybrid_rrf"]["mrr@10"])
    qahf_mrr = results["qahf"]["mrr@10"]
    improvement = (qahf_mrr - best_baseline_mrr) / best_baseline_mrr * 100
    
    print(f"  Best baseline MRR@10: {best_baseline_mrr:.4f}")
    print(f"  QAHF MRR@10: {qahf_mrr:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    
    print(f"\n  Alpha statistics (test set):")
    print(f"    Mean: {np.mean(alpha_values):.3f}")
    print(f"    Std: {np.std(alpha_values):.3f}")
    print(f"    Min: {np.min(alpha_values):.3f}")
    print(f"    Max: {np.max(alpha_values):.3f}")
    
    # 保存结果
    output = {
        "dataset": dataset,
        "test_queries": len(test_queries),
        "train_queries": len(train_queries),
        "results": results,
        "alpha_stats": {
            "mean": float(np.mean(alpha_values)),
            "std": float(np.std(alpha_values)),
            "min": float(np.min(alpha_values)),
            "max": float(np.max(alpha_values))
        }
    }
    
    with open(results_dir / "experiment_results.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\n✓ Results saved to {results_dir / 'experiment_results.json'}")
    
    return results


if __name__ == "__main__":
    run_full_experiment()