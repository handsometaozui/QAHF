"""
QAHF: Query-Aware Adaptive Hybrid Retrieval Fusion
配置文件
"""

import os
from pathlib import Path

# 设置 HuggingFace 镜像（解决网络问题）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录
PROJECT_ROOT = Path("D:/python/pycharm/LunWen")
ACTIVE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = PROJECT_ROOT / "paper"
TOPICS_DIR = PROJECT_ROOT

# 数据集配置
MSMARCO_DIR = DATA_DIR / "msmarco"
BEIR_DIR = DATA_DIR / "beir"

# MS MARCO Passage Ranking
MSMARCO_URL = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
MSMARCO_COLLECTION = MSMARCO_DIR / "collection.tsv"
MSMARCO_QUERIES = MSMARCO_DIR / "queries.tar.gz"
MSMARCO_QRELS = MSMARCO_DIR / "qrels.train.tsv"

# BEIR benchmark
BEIR_DATASETS = ["trec-covid", "nfcorpus", "scifact", "scidocs", "fiqa"]

# 实验配置
EXPERIMENT_CONFIG = {
    # 基线方法
    "baselines": ["bm25", "dense", "hybrid_fixed", "hybrid_rrf", "qahf"],
    
    # 评估指标
    "metrics": ["mrr@10", "recall@100", "ndcg@10", "latency"],
    
    # 混合检索配置
    "hybrid": {
        "alpha_values": [0.3, 0.5, 0.7],  # 固定权重实验
        "rrf_k": 60,  # RRF 参数
    },
    
    # QAHF 配置
    "qahf": {
        "feature_dims": 12,  # 查询特征维度
        "hidden_dims": 64,
        "query_types": ["keyword", "semantic", "hybrid"],
    },
    
    # 向量检索配置
    "dense_retrieval": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "batch_size": 64,
    },
    
    # BM25 配置
    "bm25": {
        "k1": 1.5,
        "b": 0.75,
    },
}

# 日志配置
LOG_DIR = ACTIVE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 结果配置
RESULTS_DIR = ACTIVE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)