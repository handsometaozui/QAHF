#!/usr/bin/env python3
"""
快速测试脚本 - 验证实验代码正确性
"""

import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from baselines import BM25Retriever, DenseRetriever, HybridRetriever, RRFHybridRetriever
from evaluator import RetrievalEvaluator


def run_quick_test():
    """快速测试 - 仅用 20 个查询"""
    print("=" * 60)
    print("Quick Validation Test (20 queries)")
    print("=" * 60)
    
    data_dir = Path("/home/dddddd/projects/aisearch_innovations/data/beir/scifact")
    
    # 1. 加载数据
    print("\n[1/4] Loading data...")
    
    corpus = {}
    with open(data_dir / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = str(doc["_id"])
            title = doc.get("title", "")
            text = doc.get("text", "")
            corpus[doc_id] = f"{title} {text}".strip()
    print(f"  Corpus: {len(corpus)} docs")
    
    queries = {}
    with open(data_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            query = json.loads(line.strip())
            query_id = str(query["_id"])
            query_text = query.get("text", "")
            queries[query_id] = query_text
    print(f"  Queries: {len(queries)}")
    
    qrels = {}
    with open(data_dir / "qrels" / "test.tsv", "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if i == 0:
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
    print(f"  Qrels: {len(qrels)} queries")
    
    # 选择 20 个有 qrels 的查询
    random.seed(42)
    queries_with_qrels = [qid for qid in queries if qid in qrels]
    sampled_query_ids = random.sample(queries_with_qrels, min(20, len(queries_with_qrels)))
    test_queries = {qid: queries[qid] for qid in sampled_query_ids}
    test_qrels = {qid: qrels[qid] for qid in sampled_query_ids}
    print(f"  Test queries: {len(test_queries)} (all with qrels)")
    
    # 2. 初始化检索器
    print("\n[2/4] Setting up retrievers...")
    
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)
    print("  BM25 ready")
    
    dense = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dense.build_index(corpus, batch_size=128)
    print("  Dense ready")
    
    hybrid_rrf = RRFHybridRetriever(bm25, dense, k=60)
    print("  Hybrid RRF ready")
    
    # 3. 运行检索
    print("\n[3/4] Running retrieval...")
    
    def run_search(retriever):
        results = {}
        for qid, query in test_queries.items():
            results[qid] = retriever.search(query, top_k=100)
        return results
    
    bm25_results = run_search(bm25)
    print("  BM25 done")
    
    dense_results = run_search(dense)
    print("  Dense done")
    
    hybrid_rrf_results = run_search(hybrid_rrf)
    print("  Hybrid RRF done")
    
    # 4. 评估
    print("\n[4/4] Evaluating...")
    
    evaluator = RetrievalEvaluator(test_qrels)
    
    def evaluate(results, name):
        scores = evaluator.evaluate(results, ["mrr@10", "recall@100", "ndcg@10"])
        return scores
    
    bm25_scores = evaluate(bm25_results, "BM25")
    dense_scores = evaluate(dense_results, "Dense")
    hybrid_rrf_scores = evaluate(hybrid_rrf_results, "Hybrid RRF")
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"BM25:        MRR@10={bm25_scores['mrr@10']:.4f}, NDCG@10={bm25_scores['ndcg@10']:.4f}, Recall@100={bm25_scores['recall@100']:.4f}")
    print(f"Dense:       MRR@10={dense_scores['mrr@10']:.4f}, NDCG@10={dense_scores['ndcg@10']:.4f}, Recall@100={dense_scores['recall@100']:.4f}")
    print(f"Hybrid RRF:  MRR@10={hybrid_rrf_scores['mrr@10']:.4f}, NDCG@10={hybrid_rrf_scores['ndcg@10']:.4f}, Recall@100={hybrid_rrf_scores['recall@100']:.4f}")
    
    # 检查是否有提升
    improvement = {
        "mrr": hybrid_rrf_scores["mrr@10"] - max(bm25_scores["mrr@10"], dense_scores["mrr@10"]),
        "ndcg": hybrid_rrf_scores["ndcg@10"] - max(bm25_scores["ndcg@10"], dense_scores["ndcg@10"]),
        "recall": hybrid_rrf_scores["recall@100"] - max(bm25_scores["recall@100"], dense_scores["recall@100"]),
    }
    
    print(f"\nHybrid improvement over best baseline:")
    print(f"  MRR@10: {improvement['mrr']:+.4f}")
    print(f"  NDCG@10: {improvement['ndcg']:+.4f}")
    print(f"  Recall@100: {improvement['recall']:+.4f}")
    
    return {
        "bm25": bm25_scores,
        "dense": dense_scores,
        "hybrid_rrf": hybrid_rrf_scores,
    }


if __name__ == "__main__":
    run_quick_test()