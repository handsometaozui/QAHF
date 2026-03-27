#!/usr/bin/env python3
"""
基线检索方法实现
包括：BM25, Dense Retrieval, Fixed Hybrid, RRF
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import pickle
from collections import defaultdict

# BM25
from rank_bm25 import BM25Okapi, BM25Plus
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# 向量检索
from sentence_transformers import SentenceTransformer
import faiss

from config import EXPERIMENT_CONFIG


class BM25Retriever:
    """BM25 检索器"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, variant: str = "okapi"):
        self.k1 = k1
        self.b = b
        self.variant = variant  # "okapi" or "plus"
        self.bm25 = None
        self.doc_ids = []
        self.doc_texts = []
        self.stemmer = PorterStemmer()
        self._stopwords = set(stopwords.words("english"))

    def _tokenize(self, text: str):
        # 去标点、小写、分词、去停用词、词干提取
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text.split()
        return [self.stemmer.stem(w) for w in tokens if w not in self._stopwords and len(w) > 1]

    def index(self, documents: Dict[str, str]):
        """
        构建索引

        Args:
            documents: {doc_id: doc_text}
        """
        self.doc_ids = list(documents.keys())
        self.doc_texts = [documents[doc_id] for doc_id in self.doc_ids]

        tokenized_corpus = [self._tokenize(doc) for doc in self.doc_texts]

        # 根据 variant 选择 BM25 实现，并传入 k1/b
        if self.variant == "plus":
            self.bm25 = BM25Plus(tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        print(f"BM25({self.variant}) indexed {len(self.doc_ids)} documents")
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        检索
        
        Args:
            query: 查询字符串
            top_k: 返回文档数
            
        Returns:
            [(doc_id, score), ...]
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.doc_ids[i], scores[i]) for i in top_indices]
        
        return results
    
    def get_idf_stats(self, query: str):
        """返回查询词的平均 IDF 和最大 IDF（用于特征提取）"""
        if self.bm25 is None:
            return 0.0, 0.0
        tokens = self._tokenize(query)
        if not tokens:
            return 0.0, 0.0
        idf_values = [self.bm25.idf.get(t, 0.0) for t in tokens]
        return float(np.mean(idf_values)), float(max(idf_values))

    def save(self, path: Path):
        """保存索引"""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "doc_ids": self.doc_ids,
                "doc_texts": self.doc_texts
            }, f)
    
    def load(self, path: Path):
        """加载索引"""
        with open(path / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]


class DenseRetriever:
    """向量检索器"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        
        # 加载模型
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # FAISS 索引
        self.faiss_index = None
        self.doc_ids = []
    
    def build_index(self, documents: Dict[str, str], batch_size: int = 64):
        """
        构建向量索引
        
        Args:
            documents: {doc_id: doc_text}
            batch_size: 编码批次大小
        """
        self.doc_ids = list(documents.keys())
        doc_texts = [documents[doc_id] for doc_id in self.doc_ids]
        
        # 编码文档
        print(f"Encoding {len(doc_texts)} documents...")
        embeddings = self.model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 构建 FAISS 索引
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
        faiss.normalize_L2(embeddings)  # 归一化（用于余弦相似度）
        self.faiss_index.add(embeddings)
        
        print(f"Dense index built: {len(self.doc_ids)} documents")
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        检索
        
        Args:
            query: 查询字符串
            top_k: 返回文档数
            
        Returns:
            [(doc_id, score), ...]
        """
        # 编码查询
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = [(self.doc_ids[i], scores[0][j]) 
                   for j, i in enumerate(indices[0])]
        
        return results
    
    def save(self, path: Path):
        """保存索引"""
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存 FAISS 索引
        faiss.write_index(self.faiss_index, str(path / "faiss.index"))
        
        # 保存 doc_ids
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
    
    def load(self, path: Path):
        """加载索引"""
        self.faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "doc_ids.json", "r") as f:
            self.doc_ids = json.load(f)


class HybridRetriever:
    """混合检索基线（固定权重）"""
    
    def __init__(self, 
                 bm25_retriever: BM25Retriever,
                 dense_retriever: DenseRetriever,
                 alpha: float = 0.5):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        混合检索
        
        Args:
            query: 查询字符串
            top_k: 返回文档数
            
        Returns:
            [(doc_id, score), ...]
        """
        # BM25 检索
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        # 向量检索
        dense_results = self.dense.search(query, top_k=top_k * 2)
        dense_scores = {doc_id: score for doc_id, score in dense_results}
        
        # 归一化
        bm25_scores = self._normalize(bm25_scores)
        dense_scores = self._normalize(dense_scores)
        
        # 融合
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        final_scores = {}
        
        for doc_id in all_doc_ids:
            bm25_s = bm25_scores.get(doc_id, 0)
            dense_s = dense_scores.get(doc_id, 0)
            final_scores[doc_id] = self.alpha * bm25_s + (1 - self.alpha) * dense_s
        
        # 排序
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_results
    
    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-Max 归一化"""
        if not scores:
            return scores
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val - min_val < 1e-9:
            return {k: 0.5 for k in scores}
        
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


class RRFHybridRetriever:
    """Reciprocal Rank Fusion (RRF) 混合检索"""
    
    def __init__(self, 
                 bm25_retriever: BM25Retriever,
                 dense_retriever: DenseRetriever,
                 k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.k = k
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        RRF 融合检索
        
        Args:
            query: 查询字符串
            top_k: 返回文档数
            
        Returns:
            [(doc_id, score), ...]
        """
        # BM25 检索
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_results)}
        
        # 向量检索
        dense_results = self.dense.search(query, top_k=top_k * 2)
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_results)}
        
        # RRF 融合
        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        final_scores = {}
        
        for doc_id in all_doc_ids:
            bm25_rank = bm25_ranks.get(doc_id, len(bm25_ranks) + 1)
            dense_rank = dense_ranks.get(doc_id, len(dense_ranks) + 1)
            
            rrf_score = 1 / (self.k + bm25_rank) + 1 / (self.k + dense_rank)
            final_scores[doc_id] = rrf_score
        
        # 排序
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_results


# 测试
if __name__ == "__main__":
    print("Baseline Retrievers Test")
    print("=" * 60)
    
    # 测试文档
    docs = {
        "doc1": "Machine learning is a subset of artificial intelligence.",
        "doc2": "Python is a popular programming language for data science.",
        "doc3": "Natural language processing enables computers to understand text.",
        "doc4": "Deep learning uses neural networks with many layers.",
        "doc5": "Information retrieval is about finding relevant documents.",
    }
    
    # 测试 BM25
    print("\n[BM25]")
    bm25 = BM25Retriever()
    bm25.index(docs)
    results = bm25.search("machine learning", top_k=3)
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")
    
    print("\n✓ Baseline retrievers test passed!")