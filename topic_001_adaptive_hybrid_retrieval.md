# 课题 #001: 自适应混合检索融合 (Adaptive Hybrid Retrieval Fusion)

## 基本信息

- **课题编号**: #001
- **课题名称**: Query-Aware Adaptive Hybrid Retrieval Fusion (QAHF)
- **研究方向**: 混合检索优化、查询感知融合
- **开始日期**: 2026-03-11
- **当前阶段**: 第5步 (搭建实验环境)

## 创新点描述

### 核心问题

在混合检索场景中，BM25（稀疏检索）和向量检索（密集检索）各有优劣：
- **BM25**: 擅长精确关键词匹配，对实体名称、专有名词效果好
- **向量检索**: 擅长语义匹配，对同义词、概念理解效果好

现有系统通常使用**固定权重**融合两种检索结果，无法根据查询特性自适应调整。

### 提出方案

**Query-Aware Adaptive Hybrid Retrieval Fusion (QAHF)**:

1. **查询特征提取**:
   - 词汇特征: 查询长度、词数、平均词长
   - 语义特征: 实体密度、词汇重叠度预测
   - 结构特征: 是否包含引号、特殊符号

2. **查询类型预测**:
   - 关键词型 (Keyword): 适合BM25
   - 语义型 (Semantic): 适合向量检索
   - 混合型 (Hybrid): 需要平衡融合

3. **自适应权重计算**:
   ```
   score_final = α(query) × score_bm25 + (1-α(query)) × score_vector
   ```
   其中 α(query) 由轻量级模型预测

4. **工程集成**:
   - 实现 OpenSearch plugin
   - 提供 REST API
   - 支持在线学习优化

## 相关工作分析

### 已有研究

| 论文 | 核心方法 | 与本课题差异 |
|------|---------|-------------|
| arxiv 1904.05737 | 公理特征+元学习器组合 | 需要预先知道两个检索方法的结果，延迟高 |
| arxiv 2403.13468 | MoE框架组合检索 | 模型复杂，难以直接部署到搜索引擎 |
| arxiv 2601.15518 | 混合检索融合 | 固定权重，无自适应 |
| arxiv 2212.10528 | HYRR重排序 | 重排序阶段，非检索阶段 |

### 创新差异

本课题的核心创新：
1. **检索前预测**: 在执行检索前预测权重，而非检索后融合
2. **轻量级模型**: 使用简单特征+分类器，易于部署
3. **端到端集成**: 直接集成到 OpenSearch 检索流程
4. **无额外延迟**: 不增加检索延迟

## 查新验证 ✅ 完成

### 搜索关键词
- [x] "adaptive hybrid retrieval"
- [x] "query-dependent fusion retrieval" 
- [x] "pre-retrieval query prediction"
- [x] "query-aware retrieval fusion"

### 已分析论文

| 论文 | 方法 | 应用场景 | 与本方案差异 |
|------|------|---------|-------------|
| arxiv 1904.05737 | Post-retrieval 元学习器 | 检索方法组合 | 需要先检索，有额外延迟 |
| arxiv 2403.13468 | MoE框架 | 多领域检索 | 模型复杂，难以部署 |
| arxiv 2401.13351 | Pre-retrieval预测 | 个性化检索开关 | 场景不同（个性化 vs 混合检索） |
| arxiv 2212.10528 | 重排序框架 | 混合检索重排序 | 重排序阶段，非检索阶段 |
| arxiv 2601.15518 | 固定权重融合 | 混合检索 | 固定权重，无自适应 |

### 创新点确认 ✅

**本方案的核心创新：**

1. **首次将 pre-retrieval 预测应用于混合检索权重调整**
   - 现有 post-retrieval 方法（如1904.05737）需要先执行检索
   - 本方案在检索前预测，零额外延迟

2. **针对 BM25 + 向量检索的特定场景**
   - 现有 pre-retrieval 工作（如2401.13351）针对个性化检索
   - 本方案针对混合检索融合权重预测

3. **可直接在搜索引擎中部署**
   - 轻量级特征设计
   - 适配 OpenSearch/ElasticSearch 架构

## 实验计划

### 数据集
- MS MARCO Passage Ranking
- TREC Deep Learning Track
- BEIR benchmark

### 基线方法
1. BM25 only
2. Dense retrieval only (ANCE, BGE)
3. Hybrid with fixed weights (α=0.5)
4. Hybrid with Reciprocal Rank Fusion (RRF)

### 评估指标
- MRR@10
- Recall@100
- NDCG@10
- 查询延迟

### 预期提升
- MRR@10 提升 3-5%
- 无显著延迟增加

## 进度记录

### 2026-03-11 22:50
- 完成选题
- 开始查新阶段

### 2026-03-11 23:00 ✅ 查新完成
- 分析了5篇关键论文
- 确认创新点：首次将 pre-retrieval 预测应用于混合检索权重调整
- 核心差异：检索前预测 vs 检索后预测

### 2026-03-11 23:10 ✅ 可行性验证完成
**OpenSearch 技术验证：**

1. **Hybrid Query 支持** ✅
   - OpenSearch 原生支持 hybrid query
   - 最多支持 5 个查询子句组合

2. **权重组合支持** ✅
   - normalization-processor 支持设置权重
   - `combination.parameters.weights` 可设置每个查询的权重
   - 支持 arithmetic_mean, geometric_mean, harmonic_mean

3. **扩展机制** ✅
   - 支持自定义 search request processor
   - 可开发 OpenSearch 插件实现动态权重预测

**工程落地方案：**
```
┌─────────────────────────────────────────────────────────┐
│                    OpenSearch 架构                       │
├─────────────────────────────────────────────────────────┤
│  查询请求 ──► QAHF Request Processor ──► Hybrid Query   │
│                    │                        │           │
│                    ▼                        ▼           │
│              查询特征提取              BM25 + k-NN       │
│                    │                        │           │
│                    ▼                        ▼           │
│              权重预测(α)            Normalization        │
│                    │                        │           │
│                    └────────────────────────►│          │
│                                             ▼           │
│                                    加权组合 (α, 1-α)     │
└─────────────────────────────────────────────────────────┘
```

**实现路径：**
1. 阶段一：独立服务 + 脚本调用（快速验证）
2. 阶段二：OpenSearch 插件开发（正式部署）

### 2026-03-12 00:15 ✅ 实验环境搭建进行中
**代码框架已完成：**
- `src/config.py` - 实验配置
- `src/feature_extractor.py` - 查询特征提取模块（12维特征）
- `src/qahf_model.py` - QAHF 核心模型（轻量级神经网络权重预测）
- `src/baselines.py` - 基线方法实现（BM25, Dense, Hybrid, RRF）
- `src/evaluator.py` - 评估模块（MRR, Recall, NDCG）
- `src/run_experiment.py` - 主实验脚本
- `src/experiment_msmarco.py` - MS MARCO 实验类
- `src/train_qahf.py` - QAHF 模型训练脚本
- `src/download_data.py` - 数据下载脚本
- `src/download_beir.py` - BEIR 数据下载脚本

**待完成：**
- [ ] 安装 Python 依赖
- [ ] 下载 MS MARCO 数据集
- [ ] 运行快速测试验证代码

---

下一步：安装依赖并下载数据集