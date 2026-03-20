# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目背景

这是一篇研究生论文的实验代码，研究方向为 **QAHF（Query-Aware Adaptive Hybrid Retrieval Fusion）**。

核心思想：混合检索（BM25 + 向量检索）通常使用固定权重融合，QAHF 在**检索执行前**根据查询特征预测最优融合权重 α，实现自适应融合：

```
score_final = α × score_BM25 + (1-α) × score_Dense
```

α 越大 → 越依赖 BM25；α 越小 → 越依赖向量检索。

---

## 实验运行命令

```bash
# 主实验（唯一需要跑的脚本）
python improved_experiment.py --dataset fiqa --limit 200
python improved_experiment.py --dataset scifact --limit 200

# 快速验证环境是否正常（不需要训练）
python quick_test.py
```

结果保存在 `D:/python/pycharm/LunWen/results/{dataset}/improved_experiment_results.json`

---

## 脚本说明（按重要性排序）

| 脚本 | 用途 | 是否使用 |
|------|------|---------|
| `improved_experiment.py` | 主实验，完整流程 | ✅ 主要用这个 |
| `baselines.py` | BM25/Dense/RRF 检索器实现 | 被主实验调用 |
| `qahf_model.py` | QAHF 模型（特征提取→权重预测→融合） | 被主实验调用 |
| `feature_extractor.py` | 12维查询特征提取 | 被 qahf_model 调用 |
| `evaluator.py` | MRR/NDCG/Recall 评估 | 被主实验调用 |
| `config.py` | 路径和超参数配置 | 被所有脚本 import |
| `quick_test.py` | 快速验证，不训练 QAHF | 调试用 |
| `run_experiment.py` | 早期版本，功能不完整 | ❌ 不用 |
| `full_experiment.py` | 全量数据版本 | 暂不用 |
| `train_qahf.py` | 单独训练并保存模型 | ❌ 不用 |
| `experiment_msmarco.py` / `experiment_beir.py` | 早期实验脚本 | ❌ 不用 |
| `download_data.py` / `download_beir.py` | 数据下载 | 数据已有则不用 |

---

## improved_experiment.py 完整执行流程

### 总览

```
main()
  └─ run_improved_experiment(dataset, limit_queries)
       ├─ 1. load_beir_data()              # 加载数据
       ├─ 2. 划分 train/test (50%/50%)
       ├─ 3. 构建检索器
       │    ├─ BM25Retriever.index()
       │    └─ DenseRetriever.build_index()
       ├─ 4. generate_improved_pseudo_labels()  # 生成训练标签
       ├─ 5. QAHF.train()                  # 训练权重预测器
       ├─ 6. 测试四个方法
       │    ├─ BM25.search() × N queries
       │    ├─ Dense.search() × N queries
       │    ├─ RRFHybridRetriever.search() × N queries
       │    └─ QAHF 推理 + 融合 × N queries
       └─ 7. 输出结果 + 保存 JSON
```

### 第1步：load_beir_data()

读取三个文件：
- `data/beir/{dataset}/corpus.jsonl` → `{doc_id: "title text"}`
- `data/beir/{dataset}/queries.jsonl` → `{query_id: query_text}`
- `data/beir/{dataset}/qrels/test.tsv` → `{query_id: {doc_id: relevance}}`

只保留有 qrels 标注的查询，随机采样 `limit_queries` 条（seed=42）。

### 第2步：划分数据

```python
random.shuffle(query_ids)  # seed=42
train = 前50%   # 用于生成伪标签、训练 QAHF
test  = 后50%   # 用于最终评估
```

### 第3步：构建检索器

**BM25Retriever.index(corpus)**
- 调用路径：`index()` → `doc.lower().split()` 分词 → `BM25Okapi(tokenized_corpus)`
- 当前分词方式：仅空格分词，无词干提取（这是召回率低的原因之一）

**DenseRetriever.build_index(corpus)**
- 调用路径：`build_index()` → `SentenceTransformer.encode()` → `faiss.IndexFlatIP`
- 模型：`all-MiniLM-L6-v2`（384维，轻量但效果有限）
- 使用余弦相似度（L2归一化 + 内积）

### 第4步：generate_improved_pseudo_labels()

为每条**训练**查询生成最优 α 标签，流程：

```
for 每条训练查询:
    bm25_results  = BM25.search(query, top_k=200)
    dense_results = Dense.search(query, top_k=200)

    对两个结果分别做 Min-Max 归一化

    for α in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        融合: score = α×bm25 + (1-α)×dense
        计算: combined = 0.7×MRR@10 + 0.3×Recall@10

    选 combined 最高的 α 作为该查询的标签
```

注意：这里用的是 **Recall@10**（不是 Recall@100），对召回率优化不够充分。

### 第5步：QAHF.train()

调用路径：
```
QAHF.train(queries, labels)
  └─ feature_extractor.extract_features(query)  # 每条查询提取12维特征
       ├─ 词汇特征: query_length, num_tokens, avg_token_length
       ├─ 结构特征: has_quotes, has_special_chars, num_entities
       └─ 语义特征: stopword_ratio, unique_token_ratio, entity_density,
                    keyword_score, semantic_score, hybrid_score
  └─ WeightPredictor 训练: 12 → FC(64) → ReLU → Dropout → FC(32) → ReLU → Dropout → FC(1) → Sigmoid
  └─ 损失函数: MSELoss，优化器: Adam，早停 patience=10
```

### 第6步：测试评估

**BM25 / Dense / RRF** 直接调用各自的 `search(query, top_k=100)`

**QAHF 推理流程**（每条查询）：
```
QAHF.predict_alpha(query)
  └─ feature_extractor.extract_features(query) → 12维向量
  └─ WeightPredictor.forward(features) → α ∈ [0,1]

bm25.search(query, top_k=200)   # 注意这里取200
dense.search(query, top_k=200)

normalize_scores(bm25_scores)   # Min-Max 归一化
normalize_scores(dense_scores)

final_score[doc] = α×bm25_norm + (1-α)×dense_norm
取 top100 返回
```

### 第7步：评估

`RetrievalEvaluator.evaluate()` 优先使用 `pytrec_eval`，不可用时回退到自定义实现。
计算 MRR@10、NDCG@10、Recall@100 的宏平均（所有测试查询取均值）。

---

## 当前实验结果

### FIQA（200条查询，train/test 各100）

```
Method          MRR@10       NDCG@10     Recall@100
bm25            0.2476       0.2014       0.4276
dense           0.4458       0.3638       0.7345
hybrid_rrf      0.4208       0.3296       0.7253
qahf            0.4628       0.3961       0.7261
```

### SciFact（200条查询，train/test 各100）

```
Method          MRR@10       NDCG@10     Recall@100
bm25            0.5701       0.5897       0.8047
dense           0.6229       0.6656       0.9700
hybrid_rrf      0.6321       0.6545       0.9750
qahf            0.6229       0.6656       0.9700
```

Alpha 统计（SciFact）：mean=0.0011，std=0.0020，min≈0，max=0.0094

### NFCorpus（200条查询，train/test 各100）

```
Method          MRR@10       NDCG@10     Recall@100
bm25            0.4387       0.2498       0.2378
dense           0.5117       0.3170       0.3153
hybrid_rrf      0.4986       0.2931       0.3064
qahf            0.5166       0.3197       0.3181
```

Alpha 统计（NFCorpus）：mean=0.327，std=0.045，min=0.223，max=0.437

### 结果分析

**FIQA：**
- QAHF 的 MRR@10（0.4628）和 NDCG@10（0.3961）超过所有基线，排序质量有提升
- QAHF 的 Recall@100（0.7261）略低于 Dense（0.7345），召回层无提升
- 根本原因：QAHF 召回上限 = BM25 ∪ Dense 候选集，BM25 召回率太低（0.4276）拖累融合

**SciFact：**
- QAHF 结果与 Dense 完全相同（α 均值仅 0.001，几乎为0）
- 原因：SciFact 是科学事实验证任务，语义匹配主导，伪标签训练后模型学到"几乎全用 Dense"
- QAHF 在此数据集上退化为纯 Dense，无自适应意义
- hybrid_rrf 的 Recall@100（0.9750）略高于 Dense（0.9700），说明 BM25 仍有少量互补贡献

---

## 待解决问题与优化方向

### 问题1：BM25 召回率低（0.4276）
**原因**：当前分词仅用 `str.split()`，无词干提取，导致词形变化无法匹配（如 "running" 匹配不到 "run"）

**方案**：在 `baselines.py` 的 `BM25Retriever` 中加入 `nltk.stem.PorterStemmer`，index 和 search 时同步做词干提取

### 问题2：Dense 召回率有提升空间（0.7345）
**方案**：将 `all-MiniLM-L6-v2` 换为 `BAAI/bge-small-en-v1.5`（同样轻量，但专为检索优化）

### 问题3：QAHF 训练标签质量
**原因**：伪标签用 `0.7×MRR@10 + 0.3×Recall@10` 优化，偏向排序质量，对召回率优化不足

**方案**：在 `generate_improved_pseudo_labels()` 中调整组合权重，或改用 Recall@100 作为优化目标

### 问题4：SciFact 上 QAHF 无提升
**原因**：SciFact 是科学事实验证任务，语义匹配主导，最优 α≈0.01，QAHF 预测结果与 Dense 几乎相同

---

## 关键配置位置

| 配置项 | 文件 | 位置 |
|--------|------|------|
| BM25 参数 k1/b | `config.py:62` | `EXPERIMENT_CONFIG["bm25"]` |
| Dense 模型名 | `improved_experiment.py:221` | `DenseRetriever(model_name=...)` |
| 伪标签 α 网格 | `improved_experiment.py:114` | `alpha_grid = np.linspace(0.1, 0.9, 9)` |
| 伪标签评分权重 | `improved_experiment.py:164` | `0.7 * mrr + 0.3 * recall_10` |
| QAHF 训练轮数 | `improved_experiment.py:246` | `epochs=100` |
| 测试查询数量 | 命令行参数 | `--limit 200` |
| 数据集路径 | `config.py:21` | `BEIR_DIR = DATA_DIR / "beir"` |

---

## 数据目录结构

```
D:/python/pycharm/LunWen/
├── data/beir/
│   ├── fiqa/
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   └── qrels/test.tsv
│   └── scifact/
│       ├── corpus.jsonl
│       ├── queries.jsonl
│       └── qrels/test.tsv
├── results/
│   ├── fiqa/improved_experiment_results.json
│   └── scifact/improved_experiment_results.json
└── src/   ← 当前工作目录
```
