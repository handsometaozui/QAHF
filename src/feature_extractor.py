#!/usr/bin/env python3
"""
查询特征提取模块
用于 QAHF 的查询特征计算
"""

import re
import string
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter


class QueryFeatureExtractor:
    """
    提取查询的特征向量，用于查询类型预测和权重计算
    
    特征包括：
    1. 词汇特征 (Lexical Features)
    2. 语义特征 (Semantic Features)  
    3. 结构特征 (Structural Features)
    """
    
    def __init__(self):
        self._bm25_retriever = None  # 可选注入，用于 IDF 特征

        # 常见停用词
        self.stopwords = set([
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'should', 'now', 'and', 'but', 'or',
            'if', 'because', 'while', 'although', 'though', 'after', 'before',
            'until', 'unless', 'since', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
        ])
        
        # 查询特征名称
        self.feature_names = [
            'query_length', 'num_tokens', 'avg_token_length',
            'has_quotes', 'has_special_chars', 'num_entities',
            'stopword_ratio', 'unique_token_ratio', 'entity_density',
            'keyword_score', 'semantic_score', 'hybrid_score',
            'avg_idf', 'max_idf'  # IDF 特征：BM25 注入后生效
        ]

        # 检索感知特征名称（从 BM25/Dense 结果的分数分布中提取）
        self.retrieval_feature_names = [
            'bm25_top10_mean', 'bm25_top10_std',
            'dense_top10_mean', 'dense_top10_std',
            'bm25_score_gap',       # top1 - top10 分数差（衡量 BM25 置信度）
            'dense_score_gap',      # top1 - top10 分数差（衡量 Dense 置信度）
            'top10_jaccard',        # BM25/Dense top-10 Jaccard 相似度
            'top50_jaccard',        # BM25/Dense top-50 Jaccard 相似度
            'score_dominance',      # BM25 相对 Dense 的置信度优势
        ]

    def set_bm25(self, bm25_retriever):
        """注入 BM25 检索器以启用 IDF 特征"""
        self._bm25_retriever = bm25_retriever
    
    def extract_features(self, query: str) -> np.ndarray:
        """
        提取查询特征向量
        
        Args:
            query: 查询字符串
            
        Returns:
            特征向量 (12维)
        """
        features = []
        
        # 1. 查询长度（字符数）
        query_length = len(query)
        features.append(query_length)
        
        # 2. 分词
        tokens = self._tokenize(query)
        num_tokens = len(tokens)
        features.append(num_tokens)
        
        # 3. 平均词长
        avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
        features.append(avg_token_length)
        
        # 4. 是否包含引号
        has_quotes = 1.0 if ('"' in query or "'" in query) else 0.0
        features.append(has_quotes)
        
        # 5. 是否包含特殊字符
        special_chars = set(string.punctuation) - {'"', "'"}
        has_special_chars = 1.0 if any(c in special_chars for c in query) else 0.0
        features.append(has_special_chars)
        
        # 6. 命名实体数量（简化版：原始 query 中大写开头的词，不能用 lower() 后的 tokens）
        raw_tokens = re.findall(r'\b\w+\b', query)
        num_entities = len([t for t in raw_tokens if t[0].isupper() and t.isalpha()])
        features.append(num_entities)
        
        # 7. 停用词比例
        stopword_count = sum(1 for t in tokens if t.lower() in self.stopwords)
        stopword_ratio = stopword_count / num_tokens if num_tokens > 0 else 0
        features.append(stopword_ratio)
        
        # 8. 唯一词比例
        unique_token_ratio = len(set(t.lower() for t in tokens)) / num_tokens if num_tokens > 0 else 0
        features.append(unique_token_ratio)
        
        # 9. 实体密度
        entity_density = num_entities / num_tokens if num_tokens > 0 else 0
        features.append(entity_density)
        
        # 10. 关键词倾向得分
        keyword_score = self._compute_keyword_score(query, tokens)
        features.append(keyword_score)
        
        # 11. 语义倾向得分
        semantic_score = self._compute_semantic_score(query, tokens)
        features.append(semantic_score)
        
        # 12. 混合倾向得分
        hybrid_score = self._compute_hybrid_score(keyword_score, semantic_score)
        features.append(hybrid_score)

        # 13 & 14. IDF 特征（需要 BM25 注入）
        if self._bm25_retriever is not None:
            avg_idf, max_idf = self._bm25_retriever.get_idf_stats(query)
        else:
            avg_idf, max_idf = 0.0, 0.0
        features.append(avg_idf)
        features.append(max_idf)

        return np.array(features)
    
    def extract_retrieval_features(self,
                                    bm25_results: List[Tuple[str, float]],
                                    dense_results: List[Tuple[str, float]]) -> np.ndarray:
        """
        从 BM25 和 Dense 检索结果中提取分数分布特征（9维）。
        这些特征反映每个检索器对当前查询的置信度和两者的一致性，
        为权重预测提供关键信号。

        Args:
            bm25_results: [(doc_id, score), ...] 按分数降序
            dense_results: [(doc_id, score), ...] 按分数降序

        Returns:
            9维特征向量
        """
        # BM25 top-10 分数统计
        bm25_scores_10 = [s for _, s in bm25_results[:10]] if len(bm25_results) >= 10 else [s for _, s in bm25_results]
        bm25_mean_10 = float(np.mean(bm25_scores_10)) if bm25_scores_10 else 0.0
        bm25_std_10 = float(np.std(bm25_scores_10)) if bm25_scores_10 else 0.0

        # Dense top-10 分数统计
        dense_scores_10 = [s for _, s in dense_results[:10]] if len(dense_results) >= 10 else [s for _, s in dense_results]
        dense_mean_10 = float(np.mean(dense_scores_10)) if dense_scores_10 else 0.0
        dense_std_10 = float(np.std(dense_scores_10)) if dense_scores_10 else 0.0

        # 分数差距（top1 - top10）：衡量检索器置信度
        bm25_score_gap = (bm25_results[0][1] - bm25_results[min(9, len(bm25_results)-1)][1]) if bm25_results else 0.0
        dense_score_gap = (dense_results[0][1] - dense_results[min(9, len(dense_results)-1)][1]) if dense_results else 0.0

        # 跨检索器一致性：Jaccard 相似度
        bm25_top10_ids = set(did for did, _ in bm25_results[:10])
        dense_top10_ids = set(did for did, _ in dense_results[:10])
        bm25_top50_ids = set(did for did, _ in bm25_results[:50])
        dense_top50_ids = set(did for did, _ in dense_results[:50])

        union_10 = bm25_top10_ids | dense_top10_ids
        top10_jaccard = len(bm25_top10_ids & dense_top10_ids) / len(union_10) if union_10 else 0.0

        union_50 = bm25_top50_ids | dense_top50_ids
        top50_jaccard = len(bm25_top50_ids & dense_top50_ids) / len(union_50) if union_50 else 0.0

        # 置信度优势：BM25 分数差距相对于 Dense 的比值
        # > 1 表示 BM25 更有区分度，< 1 表示 Dense 更有区分度
        score_dominance = bm25_score_gap / (dense_score_gap + 1e-8)

        return np.array([
            bm25_mean_10, bm25_std_10,
            dense_mean_10, dense_std_10,
            bm25_score_gap, dense_score_gap,
            top10_jaccard, top50_jaccard,
            score_dominance
        ])

    def _tokenize(self, query: str) -> List[str]:
        """简单分词"""
        # 移除标点，转小写，分词
        query = query.lower()
        tokens = re.findall(r'\b\w+\b', query)
        return tokens
    
    def _compute_keyword_score(self, query: str, tokens: List[str]) -> float:
        """
        计算关键词倾向得分
        高得分表示查询更适合 BM25
        """
        score = 0.0
        
        # 有引号 → 精确匹配需求
        if '"' in query or "'" in query:
            score += 0.3
        
        # 有大写实体 → 专有名词
        if any(t[0].isupper() for t in tokens if t):
            score += 0.2
        
        # 短查询（< 4词）→ 关键词查询
        if len(tokens) < 4:
            score += 0.2
        
        # 高唯一词比例 → 关键词查询
        if len(tokens) > 0:
            unique_ratio = len(set(t.lower() for t in tokens)) / len(tokens)
            if unique_ratio > 0.8:
                score += 0.2
        
        # 低停用词比例 → 关键词查询
        stopword_count = sum(1 for t in tokens if t.lower() in self.stopwords)
        if len(tokens) > 0:
            stopword_ratio = stopword_count / len(tokens)
            if stopword_ratio < 0.3:
                score += 0.1
        
        return min(score, 1.0)
    
    def _compute_semantic_score(self, query: str, tokens: List[str]) -> float:
        """
        计算语义倾向得分
        高得分表示查询更适合向量检索
        """
        score = 0.0
        
        # 长查询（> 6词）→ 语义查询
        if len(tokens) > 6:
            score += 0.3
        
        # 高停用词比例 → 自然语言查询
        if len(tokens) > 0:
            stopword_count = sum(1 for t in tokens if t.lower() in self.stopwords)
            stopword_ratio = stopword_count / len(tokens)
            if stopword_ratio > 0.3:
                score += 0.3
        
        # 有疑问词 → 语义查询
        question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
        if any(t.lower() in question_words for t in tokens):
            score += 0.2
        
        # 平均词长较短 → 自然语言
        if tokens:
            avg_len = np.mean([len(t) for t in tokens])
            if avg_len < 5:
                score += 0.2
        
        return min(score, 1.0)
    
    def _compute_hybrid_score(self, keyword_score: float, semantic_score: float) -> float:
        """
        计算混合倾向得分
        高得分表示需要混合检索
        """
        # 两者都高 → 混合查询
        return min(keyword_score, semantic_score) * 2
    
    def predict_query_type(self, query: str) -> Tuple[str, float]:
        """
        预测查询类型
        
        Args:
            query: 查询字符串
            
        Returns:
            (查询类型, 置信度)
        """
        features = self.extract_features(query)
        
        keyword_score = features[9]  # keyword_score
        semantic_score = features[10]  # semantic_score
        hybrid_score = features[11]  # hybrid_score
        
        # 简单决策规则
        if keyword_score > 0.6 and keyword_score > semantic_score:
            return "keyword", keyword_score
        elif semantic_score > 0.6 and semantic_score > keyword_score:
            return "semantic", semantic_score
        else:
            return "hybrid", hybrid_score
    
    def get_feature_dict(self, query: str) -> Dict[str, float]:
        """获取特征字典（用于调试和可视化）"""
        features = self.extract_features(query)
        return dict(zip(self.feature_names, features))


# 测试
if __name__ == "__main__":
    extractor = QueryFeatureExtractor()
    
    # 测试不同类型的查询
    test_queries = [
        "machine learning",  # 关键词型
        "What is the best approach for natural language processing?",  # 语义型
        "Python tutorial",  # 关键词型
        "How does transformer architecture work in modern AI systems?",  # 混合型
        '"exact phrase match"',  # 精确匹配
    ]
    
    print("Query Feature Extraction Test")
    print("=" * 80)
    
    for query in test_queries:
        features = extractor.get_feature_dict(query)
        query_type, confidence = extractor.predict_query_type(query)
        
        print(f"\nQuery: {query}")
        print(f"  Type: {query_type} (confidence: {confidence:.3f})")
        print(f"  Features:")
        for name, value in features.items():
            print(f"    {name}: {value:.4f}")