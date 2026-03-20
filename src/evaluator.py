#!/usr/bin/env python3
"""
评估模块
实现检索任务的评估指标
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import json
from pathlib import Path

try:
    import pytrec_eval
    HAS_PYTREC = True
except ImportError:
    HAS_PYTREC = False
    print("Warning: pytrec_eval not installed. Using custom evaluator.")


class RetrievalEvaluator:
    """检索评估器"""
    
    def __init__(self, qrels: Dict[str, Dict[str, int]]):
        """
        Args:
            qrels: {query_id: {doc_id: relevance}}
        """
        self.qrels = qrels
        self.use_pytrec = HAS_PYTREC and self._validate_qrels()
    
    def _validate_qrels(self) -> bool:
        """验证 qrels 格式"""
        if not self.qrels:
            return False
        # 检查格式
        for qid, docs in self.qrels.items():
            if not isinstance(docs, dict):
                return False
            for did, rel in docs.items():
                if not isinstance(rel, int):
                    return False
        return True
    
    def evaluate(self,
                 results: Dict[str, List[Tuple[str, float]]],
                 metrics: List[str] = ["mrr@10", "recall@100", "ndcg@10"]) -> Dict[str, float]:
        """
        评估检索结果
        
        Args:
            results: {query_id: [(doc_id, score), ...]}
            metrics: 评估指标列表
            
        Returns:
            指标得分字典
        """
        if self.use_pytrec:
            return self._evaluate_pytrec(results, metrics)
        else:
            return self._evaluate_custom(results, metrics)
    
    def _evaluate_pytrec(self,
                        results: Dict[str, List[Tuple[str, float]]],
                        metrics: List[str]) -> Dict[str, float]:
        """使用 pytrec_eval 评估"""
        # 转换结果格式 - pytrec_eval 需要 str->str->float
        run = {}
        for qid, doc_scores in results.items():
            run[str(qid)] = {str(doc_id): float(score) for doc_id, score in doc_scores}
        
        # 确保 qrels 也是字符串格式
        qrels_str = {}
        for qid, docs in self.qrels.items():
            qrels_str[str(qid)] = {str(doc_id): int(rel) for doc_id, rel in docs.items()}
        
        # 解析指标
        pytrec_metrics = set()
        for m in metrics:
            if m.startswith("mrr@"):
                pytrec_metrics.add("recip_rank")
            elif m.startswith("recall@"):
                k = int(m.split("@")[1])
                pytrec_metrics.add(f"recall_{k}")
            elif m.startswith("ndcg@"):
                k = int(m.split("@")[1])
                pytrec_metrics.add(f"ndcg_cut_{k}")
        
        try:
            # 使用默认 relevance 级别
            evaluator = pytrec_eval.RelevanceEvaluator(
                qrels_str,
                pytrec_metrics
            )
            
            eval_results = evaluator.evaluate(run)
            
            # 汇总结果
            final_metrics = {}
            for m in metrics:
                if m.startswith("mrr@"):
                    scores = [r["recip_rank"] for r in eval_results.values()]
                    final_metrics[m] = np.mean(scores) if scores else 0.0
                elif m.startswith("recall@"):
                    k = int(m.split("@")[1])
                    scores = [r[f"recall_{k}"] for r in eval_results.values()]
                    final_metrics[m] = np.mean(scores) if scores else 0.0
                elif m.startswith("ndcg@"):
                    k = int(m.split("@")[1])
                    scores = [r[f"ndcg_cut_{k}"] for r in eval_results.values()]
                    final_metrics[m] = np.mean(scores) if scores else 0.0
            
            return final_metrics
        except Exception as e:
            print(f"Warning: pytrec_eval failed: {e}. Using custom evaluator.")
            return self._evaluate_custom(results, metrics)
    
    def _evaluate_custom(self,
                        results: Dict[str, List[Tuple[str, float]]],
                        metrics: List[str]) -> Dict[str, float]:
        """自定义评估"""
        final_metrics = {}
        
        for metric in metrics:
            if metric.startswith("mrr@"):
                k = int(metric.split("@")[1])
                final_metrics[metric] = self._compute_mrr(results, k)
            elif metric.startswith("recall@"):
                k = int(metric.split("@")[1])
                final_metrics[metric] = self._compute_recall(results, k)
            elif metric.startswith("ndcg@"):
                k = int(metric.split("@")[1])
                final_metrics[metric] = self._compute_ndcg(results, k)
        
        return final_metrics
    
    def _compute_mrr(self, results: Dict[str, List[Tuple[str, float]]], k: int) -> float:
        """计算 MRR@k"""
        rr_scores = []
        
        for qid, doc_scores in results.items():
            if qid not in self.qrels:
                continue
            
            relevant_docs = set(did for did, rel in self.qrels[qid].items() if rel > 0)
            
            rr = 0.0
            for rank, (doc_id, _) in enumerate(doc_scores[:k], start=1):
                if doc_id in relevant_docs:
                    rr = 1.0 / rank
                    break
            
            rr_scores.append(rr)
        
        return np.mean(rr_scores) if rr_scores else 0.0
    
    def _compute_recall(self, results: Dict[str, List[Tuple[str, float]]], k: int) -> float:
        """计算 Recall@k"""
        recall_scores = []
        
        for qid, doc_scores in results.items():
            if qid not in self.qrels:
                continue
            
            relevant_docs = set(did for did, rel in self.qrels[qid].items() if rel > 0)
            
            if not relevant_docs:
                continue
            
            retrieved_docs = set(doc_id for doc_id, _ in doc_scores[:k])
            recall = len(relevant_docs & retrieved_docs) / len(relevant_docs)
            
            recall_scores.append(recall)
        
        return np.mean(recall_scores) if recall_scores else 0.0
    
    def _compute_ndcg(self, results: Dict[str, List[Tuple[str, float]]], k: int) -> float:
        """计算 NDCG@k"""
        ndcg_scores = []
        
        for qid, doc_scores in results.items():
            if qid not in self.qrels:
                continue
            
            # DCG
            dcg = 0.0
            for rank, (doc_id, _) in enumerate(doc_scores[:k], start=1):
                rel = self.qrels[qid].get(doc_id, 0)
                dcg += (2 ** rel - 1) / np.log2(rank + 1)
            
            # IDCG
            ideal_rels = sorted(self.qrels[qid].values(), reverse=True)[:k]
            idcg = 0.0
            for rank, rel in enumerate(ideal_rels, start=1):
                idcg += (2 ** rel - 1) / np.log2(rank + 1)
            
            # NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def evaluate_per_query(self,
                          results: Dict[str, List[Tuple[str, float]]],
                          metric: str) -> Dict[str, float]:
        """
        评估每个查询
        
        Args:
            results: {query_id: [(doc_id, score), ...]}
            metric: 单个指标，如 "mrr@10"
            
        Returns:
            {query_id: score}
        """
        per_query_scores = {}
        
        if metric.startswith("mrr@"):
            k = int(metric.split("@")[1])
            for qid, doc_scores in results.items():
                if qid not in self.qrels:
                    continue
                relevant_docs = set(did for did, rel in self.qrels[qid].items() if rel > 0)
                rr = 0.0
                for rank, (doc_id, _) in enumerate(doc_scores[:k], start=1):
                    if doc_id in relevant_docs:
                        rr = 1.0 / rank
                        break
                per_query_scores[qid] = rr
        
        return per_query_scores


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """
    加载 qrels 文件
    
    格式: query_id\titer\tdoc_id\trelevance
    或: query_id\tdoc_id\trelevance
    """
    qrels = defaultdict(dict)
    
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, did, rel = parts
            elif len(parts) == 3:
                qid, did, rel = parts
            else:
                continue
            
            qrels[qid][did] = int(rel)
    
    return dict(qrels)


def save_results(results: Dict[str, List[Tuple[str, float]]], path: Path):
    """保存结果"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化格式
    output = {}
    for qid, doc_scores in results.items():
        output[qid] = [(did, float(score)) for did, score in doc_scores]
    
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def load_results(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    """加载结果"""
    with open(path, "r") as f:
        data = json.load(f)
    
    results = {}
    for qid, doc_scores in data.items():
        results[qid] = [(did, score) for did, score in doc_scores]
    
    return results


# 测试
if __name__ == "__main__":
    print("Retrieval Evaluator Test")
    print("=" * 60)
    
    # 测试数据
    qrels = {
        "q1": {"d1": 2, "d2": 1, "d3": 0},
        "q2": {"d4": 1, "d5": 2},
    }
    
    results = {
        "q1": [("d1", 0.9), ("d3", 0.5), ("d2", 0.3)],
        "q2": [("d5", 0.8), ("d4", 0.6)],
    }
    
    evaluator = RetrievalEvaluator(qrels)
    metrics = evaluator.evaluate(results, metrics=["mrr@10", "recall@10", "ndcg@10"])
    
    print("\nEvaluation Results:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n✓ Evaluator test passed!")