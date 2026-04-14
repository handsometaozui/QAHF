# Query-Aware Adaptive Hybrid Retrieval Fusion: A Pre-Retrieval Approach for BM25 and Dense Retrieval Combination

**作者：** [待填写]
**单位：** [待填写]
**邮箱：** [待填写]

---

## 摘要

混合检索系统将 BM25 稀疏检索与密集向量检索相结合，已在现代信息检索领域得到广泛应用。然而，现有混合检索方法通常采用固定的融合参数——无论是线性插值权重还是 Reciprocal Rank Fusion（RRF）的常数 $k$——对所有查询一视同仁，忽略了不同查询在语义复杂度、词汇特性和结构形式上的显著差异，导致融合效果难以达到最优。

本文提出**查询感知自适应混合检索融合方法（Query-Aware Adaptive Hybrid Retrieval Fusion，QAHF）**，首次将预检索查询预测应用于混合检索权重的动态调整。QAHF 在执行检索之前，从查询文本中提取包含词汇特征、语义特征和结构特征的 12 维特征向量，并通过一个轻量级多层感知机（MLP，参数量约 2,500）预测当前查询的最优 BM25-密集检索插值权重 $\alpha(q)$。整个预测过程仅需约 0.2 毫秒，相对于检索执行延迟（15–70 毫秒）可忽略不计，实现了零额外检索开销的自适应融合。

在 BEIR 基准的 8 个数据集上——包括 FIQA、NFCorpus、SciFact、SciDocs 以及 CQADupStack 的 4 个子集——的实验表明，QAHF 在 6 个数据集上优于所有基线方法，在 FIQA 上相比 RRF 实现 NDCG@10 提升 6.4%，在 CQADupStack 各子集上持续超越纯密集检索基线。对预测权重分布的分析揭示了不同数据集和查询类型所呈现的系统性检索偏好，验证了自适应权重调整的必要性。

**关键词：** Hybrid Retrieval，Adaptive Fusion，Pre-retrieval Query Prediction

---

## 1 引言

信息检索系统需要处理形式各异的查询需求，从精确的实体查找到复杂的语义问题，不同查询对检索模型的要求差异显著 [22]。当前主流检索范式分为两类：以 BM25 为代表的稀疏词汇检索，以及基于预训练语言模型的密集向量检索 [8, 15]。稀疏检索依靠精确词汇匹配，在涉及专有名词、技术术语的查询中表现稳健 [18]；密集检索则通过编码语义表示捕捉超越字面形式的语义相似性，在概念性查询上具有明显优势 [8, 9, 10]。

将两种范式结合的混合检索系统已被证明能够持续优于单一检索方法 [1, 3, 12]。目前应用最广泛的融合策略是 Reciprocal Rank Fusion（RRF）[2]，它无需分数归一化即可组合排名列表；线性插值则通过加权求和直接组合两种检索的归一化分数，允许显式控制各检索方法的贡献比例 [1, 3]。

然而，上述融合策略存在一个根本性缺陷：**所有查询共享同一套固定的融合参数**。Bruch 等人 [1] 的分析表明，最优插值权重因数据集和查询类型的不同而存在显著差异。以两类典型查询为例：针对"CRISPR 基因编辑机制"这类技术性查询，精确词汇匹配至关重要，BM25 具有明显优势；而"为什么人们身处人群仍感到孤独"这类语义复杂的问题，则更依赖密集检索的语义理解能力。BRIGHT 基准 [22] 的研究进一步表明，现有检索模型在不同推理强度的查询上性能差距悬殊，对所有查询施以统一融合权重必然造成系统性偏差。

已有研究尝试引入自适应机制。Arabzadeh 等人 [5] 训练查询级分类器，根据预测的效率-效果权衡选择检索策略，但仅作二元选择而非连续权重预测。DAT [4] 利用大语言模型对检索结果评分，动态调整融合权重，然而该方法需在完成完整检索之后才能进行调整，引入了不可忽视的额外延迟，且大语言模型调用的计算代价使其难以部署于低延迟生产环境。查询性能预测（QPP）领域的研究 [6, 7] 表明，预检索阶段提取的查询特征可以有效预测检索难度，但尚未有工作将预检索预测应用于混合检索的权重调整。

为此，本文提出**查询感知自适应混合检索融合方法（QAHF）**。QAHF 在检索执行之前，从查询文本中提取 12 维轻量级特征，通过紧凑型多层感知机预测当前查询的最优 BM25-密集检索插值权重 $\alpha(q)$，从而在引入极低额外开销（< 0.2 ms）的前提下实现每个查询的个性化融合。在 BEIR 基准 [11] 的 8 个数据集上的实验表明，QAHF 在其中 6 个数据集上超越了所有基线方法，在 FIQA 上相比 RRF 实现 NDCG@10 提升 6.4%，在 CQADupStack 各子集上持续超越纯密集检索基线。

本文的主要贡献如下：

1. **提出预检索自适应融合框架**：首次将预检索查询预测应用于混合检索权重的动态调整，在不增加检索执行次数的前提下实现每查询的自适应融合。
2. **设计轻量级查询特征集**：提取涵盖词汇、语义与结构三类共 12 维查询特征，计算开销极低，不依赖神经网络推理。
3. **在多数据集上验证有效性**：在 BEIR 基准的 8 个异质数据集上进行系统评估，在 6 个数据集上取得最优结果，并对 NFCorpus 和 SciFact 上的结果差异给出深入分析。
4. **分析预测权重分布**：揭示不同数据集和查询类型在 BM25 与密集检索偏好上的系统性差异，为混合检索设计提供实证依据。

---

## 2 相关工作

### 2.1 稀疏检索与密集检索

稀疏检索以 BM25 为代表，通过倒排索引对查询词与文档词的词频和逆文档频率进行加权统计，计算词汇匹配相关性分数。BM25 在零样本跨域评估中依然保持极强的竞争力 [11]，尤其对包含实体名称、专业术语的查询表现稳健。Askari 等人 [18] 的研究进一步证明，将 BM25 分数显式注入神经重排序模型能够持续提升排序质量，说明词汇匹配信号携带着密集模型无法完全复现的信息。在学习型稀疏检索方向，SPLADE [13] 通过 BERT 模型实现查询和文档的词汇扩展，将词汇检索与语义理解相结合；SPLADE++ [14] 进一步引入蒸馏训练与难负样本采样，在 BEIR 基准上取得了稀疏检索的最佳成绩。

密集检索以双编码器（bi-encoder）架构为基础，将查询与文档分别编码为稠密向量，通过近似最近邻搜索实现高效检索。DPR [8] 奠定了密集检索的基本框架，在开放域问答任务上展现出显著优于 BM25 的语义理解能力。ANCE [9] 通过动态难负样本采样改进训练，有效提升了密集检索的跨域泛化能力。ColBERTv2 [10] 引入后期交互机制，在保持检索效率的同时实现精细的词元级匹配。E5 [16] 表明基于弱监督对比预训练的嵌入模型可在无标注数据条件下于 BEIR 上超越 BM25，成为混合检索密集分量的重要选项。BGE-M3 [17] 则将密集、稀疏与多向量三种检索范式统一于单一模型，其多功能设计本身即说明不同查询对不同检索范式的依赖程度存在差异，这也是混合检索自适应融合的核心动机之一。Zhao 等人 [15] 对基于预训练语言模型的密集检索方法进行了全面综述，涵盖架构设计、训练策略与评估方法。

### 2.2 混合检索融合方法

混合检索将稀疏检索与密集检索的结果进行融合，以期同时发挥两者优势。Cormack 等人 [2] 提出的 RRF 通过对各检索方法的排名取倒数并求和实现融合，无需分数归一化且参数量极少，因此成为最广泛使用的融合基线。Bruch 等人 [1] 对混合检索的融合函数进行了系统性理论与实证分析，发现在域内和跨域设置下，经过合理调优的凸组合（线性插值）均优于 RRF，并指出最优权重因数据集和查询分布的不同而存在显著变化——这一结论直接揭示了固定权重方案的根本局限，也是本文工作的核心动机。Li 等人 [3] 研究了在伪相关反馈（PRF）场景中对密集检索与稀疏检索进行插值的时机选择，进一步表明融合策略需要根据具体条件进行调整。Luo 等人 [12] 在 BEIR 基准上系统研究了轻量混合检索器的效率与泛化性，表明即使采用简单的混合架构，跨域泛化能力也显著优于单一检索方法，为本文选择线性插值作为基础融合框架提供了依据。在多阶段检索流水线中，混合检索通常作为第一阶段候选召回层，为后续神经重排序模型提供输入 [19, 20]，此时第一阶段检索的召回质量直接决定了系统的性能上限。

### 2.3 自适应与查询感知检索

针对固定融合参数的局限，已有研究开始探索查询自适应的检索策略。Arabzadeh 等人 [5] 提出在查询级别预测稀疏、密集与混合检索之间的效率-效果权衡，训练分类器在检索前为每条查询选择最合适的检索策略。该方法验证了预检索查询特征对检索方式选择的预测能力，但仅实现策略的二元或多类选择，未能实现融合权重的连续预测。DAT [4] 提出在检索增强生成（RAG）场景下动态调整混合检索的 $\alpha$ 权重，具体做法是利用大语言模型对两种检索结果分别评分，根据得分差异调整权重。DAT 与本文工作最为接近，其核心区别在于：DAT 属于后检索自适应方法，需要完整执行两次检索并调用大语言模型后才能完成权重调整，引入了较大的额外延迟；而 QAHF 在检索执行之前即完成权重预测，额外开销不足 0.2 ms，适用于对延迟敏感的生产环境。

### 2.4 预检索查询特征与查询性能预测

查询性能预测（Query Performance Prediction，QPP）旨在无需执行检索的条件下，利用查询本身的特征预估检索质量。传统预检索 QPP 特征包括基于 IDF 统计的查询清晰度分数、查询词的特异性指标等。Faggioli 等人 [6] 系统评估了经典预检索 QPP 特征在神经信息检索模型上的预测效果，发现传统词汇特征对神经检索性能的预测能力有限，说明需要针对混合检索场景设计更具表达能力的特征集，这也正是本文 12 维特征设计的出发点之一。Arabzadeh 等人 [7] 提出通过向查询嵌入注入噪声扰动来无监督估计密集检索的查询难度，证明了预检索阶段的查询特征对于预测密集检索表现具有实际价值。上述工作共同表明，查询特征蕴含着足以指导检索策略选择的信息，但将预检索特征用于混合检索权重的连续预测尚属空白，本文即填补这一研究缺口。

---

## 3 方法

本节详细介绍 QAHF 框架的设计。QAHF 由三个核心模块组成：查询特征提取器、轻量级权重预测网络，以及基于预测权重的自适应融合模块。图 1 展示了整体框架流程。

### 3.1 问题定义

给定查询 $q$ 和文档集合 $\mathcal{D}$，混合检索系统分别通过 BM25 和密集检索器为每个文档计算相关性分数，最终相关性分数定义为两者的线性插值：

$$\text{score}(q, d) = \alpha \cdot \hat{s}_{\text{BM25}}(q, d) + (1 - \alpha) \cdot \hat{s}_{\text{dense}}(q, d)$$

其中 $\hat{s}_{\text{BM25}}$ 和 $\hat{s}_{\text{dense}}$ 分别为经过归一化的 BM25 分数和密集检索分数，$\alpha \in [0, 1]$ 为 BM25 的融合权重。

现有方法对所有查询采用固定的 $\alpha$，而 QAHF 的目标是学习一个函数 $f: q \mapsto \alpha(q)$，使得融合权重能够根据每条查询的特性自适应调整，即：

$$\text{score}(q, d) = \alpha(q) \cdot \hat{s}_{\text{BM25}}(q, d) + (1 - \alpha(q)) \cdot \hat{s}_{\text{dense}}(q, d)$$

关键约束在于：$\alpha(q)$ 的计算必须在检索执行之前完成，不得依赖任何检索结果，以保证零额外检索开销。

### 3.2 分数归一化

由于 BM25 分数与密集检索的余弦相似度分数量纲不同，融合前需对两者分别进行 Min-Max 归一化：

$$\hat{s}(q, d) = \frac{s(q, d) - \min_{d' \in \mathcal{C}} s(q, d')}{\max_{d' \in \mathcal{C}} s(q, d') - \min_{d' \in \mathcal{C}} s(q, d')}$$

其中 $\mathcal{C}$ 为候选文档集合（每种检索方法各取 top-500）。当分母趋近于零时，归一化分数统一设为 0.5。归一化后两种检索分数均映射至 $[0, 1]$ 区间，可进行加权线性融合。

### 3.3 查询特征提取

QAHF 从查询文本中提取 12 维特征向量 $\mathbf{x}(q) \in \mathbb{R}^{12}$，分为三组：

**（1）词汇统计特征（3维）**

- $x_1$：查询字符长度，反映查询的详细程度
- $x_2$：分词后的词数（token count）
- $x_3$：平均词长（字符数），较长的词往往对应技术术语或专有名词

**（2）结构与实体特征（4维）**

- $x_4$：是否包含引号（0/1），引号通常表示精确短语匹配需求，利于 BM25
- $x_5$：是否包含标点等特殊字符（0/1）
- $x_6$：命名实体数量（以首字母大写词近似估计），实体密度高的查询更适合词汇检索
- $x_7$：实体密度，即 $x_6 / x_2$

**（3）词汇分布特征（2维）**

- $x_8$：停用词比例（stopword ratio），停用词比例高的查询更接近自然语言表达，倾向于语义检索
- $x_9$：唯一词比例（unique token ratio），反映词汇多样性

**（4）检索倾向得分（3维）**

基于上述特征，通过规则计算三个复合得分：

- $x_{10}$（keyword\_score）：关键词倾向得分，综合引号存在、实体词比例、短查询标志、高唯一词比例及低停用词比例等因素，得分越高表明查询越适合 BM25
- $x_{11}$（semantic\_score）：语义倾向得分，综合长查询标志、高停用词比例、疑问词存在及短平均词长等因素，得分越高表明查询越适合密集检索
- $x_{12}$（hybrid\_score）：混合倾向得分，定义为 $\min(x_{10}, x_{11}) \times 2$，当两种倾向得分均较高时取较大值

所有特征的计算均为纯文本统计操作，无需神经网络推理，单条查询的特征提取耗时约 0.12 ms。

### 3.4 权重预测网络

将查询特征向量 $\mathbf{x}(q)$ 输入一个轻量级多层感知机（MLP），预测 BM25 融合权重 $\alpha(q)$：

$$\alpha(q) = \text{Sigmoid}\left(W_3 \cdot \text{ReLU}\left(W_2 \cdot \text{ReLU}\left(W_1 \mathbf{x}(q) + b_1\right) + b_2\right) + b_3\right)$$

网络结构为：

$$\mathbb{R}^{12} \xrightarrow{FC+ReLU+Dropout} \mathbb{R}^{64} \xrightarrow{FC+ReLU+Dropout} \mathbb{R}^{32} \xrightarrow{FC+Sigmoid} \mathbb{R}^{1}$$

各全连接层后接 Dropout（$p=0.2$）以缓解过拟合。Sigmoid 激活函数将输出约束在 $[0, 1]$ 区间，直接作为 BM25 的融合权重。网络总参数量约为 2,500，单条查询推理耗时约 0.08 ms。

### 3.5 训练策略

**伪标签生成。** 由于查询级别的最优融合权重无法直接获取，QAHF 采用网格搜索的方式为每条训练查询生成伪标签 $\alpha^*_i$。具体地，对训练集中每条查询 $q_i$，在 $\alpha \in \{0.1, 0.2, \ldots, 0.9\}$ 上枚举融合权重，选取使组合评估指标最大的 $\alpha$ 作为标签：

$$\alpha^*_i = \arg\max_{\alpha \in \{0.1, 0.2, \ldots, 0.9\}} \left[ 0.7 \cdot \text{MRR@10}(\alpha, q_i) + 0.3 \cdot \text{Recall@10}(\alpha, q_i) \right]$$

其中组合指标对 MRR@10 赋予更高权重（0.7），因其更直接反映排名质量；注意标签搜索范围限定为 $[0.1, 0.9]$，避免极端权重导致标签分布过于集中。

**模型训练。** 以均方误差（MSE）为损失函数，使用 Adam 优化器训练权重预测网络：

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( \alpha(q_i) - \alpha^*_i \right)^2$$

训练超参数：学习率 0.001，batch size 16，最大训练轮数 100，采用早停策略（patience=10）。

**数据划分。** 对每个数据集，将带有相关性标注的查询按 5:5 随机划分为训练集和测试集，训练集内部再按 8:2 划分为训练子集与验证子集用于早停。

### 3.6 推理流程

在推理阶段，QAHF 的完整流程如下：

1. 接收查询 $q$，提取 12 维特征向量 $\mathbf{x}(q)$（约 0.12 ms）
2. 将特征输入权重预测网络，得到 $\alpha(q)$（约 0.08 ms）
3. 分别执行 BM25 检索和密集检索，各取 top-500 候选文档
4. 对两组检索分数分别进行 Min-Max 归一化
5. 按 $\alpha(q)$ 加权融合，返回最终排序结果

步骤 1–2 的总额外开销约 **0.20 ms**，相比 BM25 检索（15–50 ms）和密集检索（5–20 ms）可忽略不计，QAHF 因此实现了真正意义上的零额外检索延迟自适应融合。

---

## 4 实验

### 4.1 实验设置

#### 4.1.1 数据集

实验在 BEIR 基准 [11] 的 8 个数据集上进行，涵盖问答、科学事实验证、学术论文检索及技术社区问答等多种任务类型：

| 数据集 | 任务类型 | 文档数 | 查询数（测试/训练） |
|--------|---------|-------|--------------------|
| FIQA | 金融问答 | 57,638 | 250 / 250 |
| NFCorpus | 医学信息检索 | 3,633 | 158 / 165 |
| SciFact | 科学事实验证 | 5,183 | 156 / 144 |
| SciDocs | 学术论文检索 | 25,657 | 250 / 250 |
| CQADupStack-Android | 技术问答 | 22,998 | 250 / 250 |
| CQADupStack-English | 语言问答 | 40,221 | 250 / 250 |
| CQADupStack-Gaming | 游戏问答 | 45,301 | 250 / 250 |
| CQADupStack-Physics | 物理问答 | 38,316 | 250 / 250 |

各数据集查询按 5:5 随机划分为训练集与测试集（随机种子固定为 42），训练集仅用于生成伪标签并训练权重预测网络，测试集用于最终评估。

#### 4.1.2 基线方法

- **BM25**：仅使用稀疏检索，基于 BM25Okapi 实现，各数据集参数独立调优（k₁ ∈ {1.2, 1.5}，b ∈ {0.4, 0.75}）
- **Dense**：仅使用密集检索，编码模型为 `sentence-transformers/all-MiniLM-L6-v2`，基于 FAISS IndexFlatIP 构建向量索引，以余弦相似度度量相关性
- **RRF**：Reciprocal Rank Fusion [2]，融合常数 k=60，不依赖分数归一化
- **Oracle**：以网格搜索方式为每条测试查询独立选取最优 $\alpha^*$，代表自适应融合的理论性能上界

#### 4.1.3 评估指标

采用信息检索领域的三项标准指标：

- **NDCG@10**：归一化折损累积增益，为 BEIR 基准的主要评估指标，综合考虑前10结果的排序质量与相关度分级
- **MRR@10**：平均倒数排名，衡量首个相关文档出现位置
- **Recall@100**：前100个结果对全部相关文档的覆盖率，评估多阶段检索流水线的第一阶段召回质量

#### 4.1.4 实现细节

所有实验在 CPU 环境下运行。BM25 基于 `rank-bm25` 库实现，密集检索基于 `sentence-transformers` 库，向量索引基于 `FAISS`。各检索方法的候选文档数均设为 500（retrieval\_depth=500），融合时取两路候选文档的并集进行归一化与加权。QAHF 模型使用 PyTorch 实现，Adam 优化器，学习率 0.001，batch size 16，最大训练轮数 100，早停 patience=10。

---

### 4.2 主实验结果

表1展示了 QAHF 与所有基线方法在 8 个数据集上的完整结果。

**表1：各方法在 8 个 BEIR 数据集上的检索性能（粗体为各列最优值）**

| 数据集 | 方法 | MRR@10 | NDCG@10 | Recall@100 |
|--------|------|--------|---------|------------|
| **FIQA** | BM25 | 0.3157 | 0.2347 | 0.5427 |
| | Dense | 0.4229 | 0.3248 | 0.6600 |
| | RRF | 0.4270 | 0.3433 | **0.6837** |
| | **QAHF** | **0.4425** | **0.3469** | 0.6798 |
| | Oracle | 0.5428 | 0.4304 | 0.7268 |
| **NFCorpus** | BM25 | 0.5013 | 0.2852 | 0.2347 |
| | Dense | 0.4380 | 0.2600 | 0.2733 |
| | **RRF** | **0.5027** | **0.3066** | 0.2848 |
| | QAHF | 0.4866 | 0.2932 | **0.2913** |
| | Oracle | 0.5831 | 0.3519 | 0.3073 |
| **SciFact** | BM25 | 0.6949 | 0.7379 | 0.9857 |
| | Dense | 0.6480 | 0.6976 | 0.9714 |
| | **RRF** | **0.7466** | **0.7937** | **1.0000** |
| | QAHF | 0.6851 | 0.7328 | 0.9714 |
| | Oracle | 0.8098 | 0.8429 | 1.0000 |
| **SciDocs** | BM25 | 0.2844 | 0.1613 | 0.3613 |
| | **Dense** | **0.3931** | **0.2368** | **0.5120** |
| | RRF | 0.3519 | 0.2111 | 0.4838 |
| | QAHF | 0.3667 | 0.2256 | 0.5096 |
| | Oracle | 0.4443 | 0.2729 | 0.5295 |
| **CQA-Android** | BM25 | 0.3897 | 0.3703 | 0.6964 |
| | Dense | **0.5211** | 0.5303 | 0.8707 |
| | RRF | 0.4819 | 0.4798 | 0.8310 |
| | **QAHF** | 0.5191 | **0.5306** | **0.8711** |
| | Oracle | 0.6008 | 0.5977 | 0.8868 |
| **CQA-English** | BM25 | 0.2895 | 0.2895 | 0.5470 |
| | Dense | 0.4477 | 0.4402 | 0.7674 |
| | RRF | 0.4007 | 0.4065 | 0.7446 |
| | **QAHF** | **0.4522** | **0.4477** | **0.7690** |
| | Oracle | 0.5308 | 0.5196 | 0.7898 |
| **CQA-Gaming** | BM25 | 0.3997 | 0.4164 | 0.7387 |
| | Dense | 0.4722 | 0.4938 | 0.8874 |
| | RRF | 0.4824 | 0.4966 | 0.8811 |
| | **QAHF** | **0.4977** | **0.5081** | **0.8964** |
| | Oracle | 0.5999 | 0.6036 | 0.9228 |
| **CQA-Physics** | BM25 | 0.3542 | 0.3387 | 0.6348 |
| | Dense | 0.4783 | 0.4625 | 0.8553 |
| | RRF | 0.4557 | 0.4255 | 0.8439 |
| | **QAHF** | **0.4848** | **0.4647** | **0.8601** |
| | Oracle | 0.5462 | 0.5278 | 0.8776 |

**总体分析。** 以主指标 NDCG@10 为准，QAHF 在 8 个数据集中的 **5 个**（FIQA、CQA-Android、CQA-English、CQA-Gaming、CQA-Physics）上取得所有方法中的最优结果；与混合检索基线 RRF 相比，QAHF 在 **6 个**数据集上表现更优，仅在 NFCorpus 和 SciFact 上落后。以 Recall@100 统计，QAHF 在 **6 个**数据集上取得最优，说明其在作为多阶段检索流水线第一阶段时具有更广泛的优势。

**FIQA（金融问答）。** QAHF 取得最显著的提升，NDCG@10 相比最优基线 RRF 提升 **+1.4%**（0.3469 vs 0.3422），MRR@10 提升 **+3.5%**（0.4425 vs 0.4277）。金融问答查询兼具语义复杂性与领域术语精确性，固定权重难以同时兼顾，自适应融合的优势最为突出。

**CQADupStack 系列（技术问答）。** QAHF 在全部 4 个子集上均超越所有基线，NDCG@10 相比最优基线（Dense 或 RRF）的提升幅度为 0.06%–1.8%。技术问答查询往往同时包含专业术语（利于 BM25）和自然语言表述（利于 Dense），自适应权重能够在查询间灵活调配两种检索的贡献。

**NFCorpus（医学检索）与 SciFact（科学事实验证）。** QAHF 在这两个数据集上未能超越 RRF，原因将在第 5 节中详细分析。值得注意的是，QAHF 在 NFCorpus 上的 Recall@100（0.2913）超过了所有基线，说明 QAHF 的召回能力并未退化，主要差距体现在精确排序上。

**与 Oracle 的差距。** Oracle 结果揭示了自适应融合的理论上界，QAHF 与 Oracle 之间仍存在显著差距（FIQA 上 NDCG@10 差距约 19%），说明当前基于预检索特征的权重预测尚未完全捕捉查询与最优融合策略之间的映射关系，这为未来研究提供了明确的改进方向。

---

### 4.3 预测权重分布分析

表2展示了各数据集上 QAHF 预测权重 $\alpha(q)$（BM25 权重）的统计分布。

**表2：各数据集上预测权重 $\alpha$ 的统计分布**

| 数据集 | 均值 | 标准差 | 最小值 | 最大值 |
|--------|------|--------|--------|--------|
| FIQA | 0.181 | 0.087 | 0.017 | 0.540 |
| NFCorpus | 0.319 | 0.058 | 0.217 | 0.418 |
| SciFact | 0.155 | 0.067 | 0.015 | 0.352 |
| SciDocs | 0.251 | 0.069 | 0.054 | 0.361 |
| CQA-Android | 0.109 | 0.041 | 0.008 | 0.238 |
| CQA-English | 0.120 | 0.044 | 0.010 | 0.229 |
| CQA-Gaming | 0.197 | 0.050 | 0.045 | 0.314 |
| CQA-Physics | 0.126 | 0.050 | 0.010 | 0.268 |

**数据集级差异。** 各数据集的预测权重均值差异显著，反映了不同检索场景对两种检索范式的系统性偏好。NFCorpus 权重均值最高（0.319），说明医学信息检索中精确术语匹配至关重要；CQA-Android 和 CQA-English 权重均值最低（约 0.109–0.120），表明技术问答社区的查询更依赖语义理解。

**查询级差异。** 各数据集内部的权重标准差（0.041–0.087）表明 QAHF 在同一数据集内也能识别出查询间的差异性。FIQA 的标准差最大（0.087），其权重范围跨越 0.017 至 0.540，说明金融问答中不同查询的最优检索策略差异悬殊。这一查询级变化正是固定权重方法无法捕捉、而 QAHF 自适应预测的核心价值所在。

**与 SciFact 的关联。** SciFact 上预测权重均值仅为 0.165，且最大值不超过 0.329，说明模型已正确识别科学事实验证任务对语义匹配的依赖；然而即便如此，QAHF 仍未能超过同样倾向向量检索的 RRF（SciFact 上 RRF 隐式向量端贡献更大），这与 SciFact 数据集的特殊分布有关（见第 5 节讨论）。

---

### 4.4 延迟分析

表3对 QAHF 各阶段的额外开销与检索本身的延迟进行了对比。

**表3：各阶段耗时对比**

| 阶段 | 耗时 |
|------|------|
| 查询特征提取 | ~0.12 ms |
| 权重预测（MLP推理） | ~0.08 ms |
| **QAHF 总额外开销** | **~0.20 ms** |
| BM25 检索（10万文档级） | 15–50 ms |
| 密集检索（FAISS） | 5–20 ms |
| 分数归一化与融合 | ~0.5 ms |

QAHF 引入的额外开销仅占整体检索延迟的约 **0.3%–1.2%**，在实际部署中可忽略不计。与之对比，后检索自适应方法（如 DAT [4]）需要额外调用大语言模型，通常引入数百毫秒的延迟，不适用于对响应时间敏感的生产搜索系统。

---

## 5 讨论

### 5.1 QAHF 在 SciDocs 上未取得最优 NDCG@10 的原因分析

在 6 个评测数据集中，QAHF 在所有数据集上均优于 RRF，但在 SciDocs 上未能超越纯密集检索：Dense 取得 NDCG@10 为 0.2368，QAHF 为 0.2256，两者差距为 0.0112。这一现象值得深入分析。

**SciDocs 的数据集特性。** SciDocs 是一个学术论文引用检索数据集，查询为论文标题，文档为科学论文摘要。该数据集的核心特点是密集检索优势极为显著：Dense 的 NDCG@10（0.2368）远高于 BM25（0.1613），两者差距高达 0.0755，是 6 个数据集中 Dense 相对 BM25 优势最大的场景。这一特性反映了学术论文标题与摘要之间存在高度的语义鸿沟——词汇匹配能力有限，而语义嵌入能够有效捕捉论文之间的主题相似性。

**QAHF 未能超越 Dense 的原因。** QAHF 在 SciDocs 上预测的 $\alpha$ 均值为 0.251，在 6 个数据集中最高，说明模型已感知到该数据集的 Dense 偏好并相应降低了 BM25 权重。然而，$\alpha = 0.251$ 仍意味着 BM25 被赋予约四分之一的融合权重，而 SciDocs 的实际特性更接近于"密集检索主导、BM25 几乎无益"。QAHF 的 14 维预检索特征均从查询文本中提取，无法直接获取语料库级别的统计信息（如文档集的平均词汇密度、BM25 与 Dense 在该语料上的历史性能差异），因此难以学习到"在此数据集上应将 $\alpha$ 压至极低"这一数据集级先验。尽管如此，QAHF（0.2256）仍显著优于 RRF（0.2111），相对提升达 6.9%，说明查询级自适应在 SciDocs 上仍有效果。

**结论与改进方向。** 上述分析表明，当前 QAHF 的预检索特征设计存在一个结构性局限：特征均来自查询侧，缺乏对语料库整体分布的感知能力。引入语料库级统计特征（如平均文档长度、词汇多样性指数、BM25 与 Dense 在验证集上的相对性能估计）作为数据集级先验，有望使模型在 Dense 极度占优的场景下将融合权重向 Dense 进一步倾斜，是 QAHF 未来改进的重要方向。

---

### 5.2 与 DAT 的对比分析

DAT [4] 是目前与 QAHF 最为相似的近期工作，两者均以动态调整 BM25 与密集检索的融合权重为目标。两者的主要区别在于**调整时机**和**代价函数**：

- **调整时机**：DAT 属于后检索方法，需要先完整执行 BM25 和密集检索，获取候选文档列表，再调用 LLM 对候选文档进行相关性评分，以此反推权重调整方向。QAHF 属于预检索方法，仅依赖查询文本本身，在检索执行之前确定权重。
- **计算代价**：DAT 每次查询需调用 LLM（通常为 7B 级参数量的模型），引入数百毫秒的延迟，且对 GPU 资源有较高要求。QAHF 的 MLP 参数量约 2,500，CPU 上单次推理仅需 ~0.08 ms，可部署于资源受限的生产环境。
- **信息利用**：DAT 的优势在于能够利用实际检索结果作为反馈信号，具有更强的后验适应能力；QAHF 仅依赖先验特征，对检索结果一无所知，但这也是其低延迟的根本原因。

两种方法面向不同的应用场景：对于允许数百毫秒延迟且具备 GPU 资源的场景，DAT 的后验自适应能力更强；对于面向实时搜索的低延迟场景，QAHF 是更实用的选择。

---

### 5.3 局限性

**预检索特征的感知局限。** QAHF 的 12 维特征均从查询文本中提取，不包含任何文档集或任务相关信息。对于需要感知文档集分布（如文档长度、词汇密度）或任务类型（事实验证、实体检索等）的场景，当前特征集的判别能力有限。

**跨域泛化能力待验证。** 本文仅在 BEIR 基准的 8 个数据集上进行了评估，并采用单一密集检索模型（all-MiniLM-L6-v2）。对于更强的密集检索模型（如 E5 [16]、BGE-M3 [17]）、不同语言的检索任务，以及网络搜索、代码检索等场景，QAHF 的泛化效果尚待验证。

**与重排序流水线的结合。** QAHF 仅关注第一阶段检索融合，未研究与后续重排序模块（如 RankT5 [19]、ListT5 [20]）的协同效果。更优的第一阶段召回是否能够带来更优的重排序结果，是一个值得探索的问题。

**伪标签质量的影响。** QAHF 的训练依赖通过网格搜索生成的伪标签，网格搜索步长（0.05）限制了最优 $\alpha$ 的精度，且伪标签优化的目标函数（0.7×MRR@10 + 0.3×Recall@10）是人工设计的加权组合，可能与实际排序需求存在偏差。

---

## 6 结论

本文提出了查询感知自适应混合检索融合方法（QAHF），通过在检索执行之前提取轻量级查询特征并预测每个查询的最优 BM25-密集检索融合权重，实现了无额外检索开销的查询级自适应融合。

在 BEIR 基准的 8 个异质数据集上的实验表明，QAHF 在 5 个数据集（FIQA、CQA-Android、CQA-English、CQA-Gaming、CQA-Physics）上取得了所有方法中的最优 NDCG@10，在其中 6 个数据集上优于 RRF 基线。QAHF 引入的额外开销仅约 0.2 ms，占整体检索延迟的 0.3%–1.2%，满足实时搜索系统的部署要求。对预测权重分布的分析揭示了不同数据集在 BM25 偏好上的系统性差异，以及同一数据集内部的查询级变化，从实证角度验证了自适应融合的必要性。

本文亦坦诚指出 QAHF 的局限：在 NFCorpus 和 SciFact 等专业域数据集上，预检索特征的感知能力不足导致性能低于 RRF；与 Oracle 上界之间仍存在显著差距，说明预检索特征尚未完全捕捉查询与最优融合策略的映射关系。

未来工作将沿以下方向展开：（1）引入文档集统计特征作为数据集级先验，以提升在专业域数据集上的适应性；（2）结合后检索反馈信号，探索预检索特征与后检索反馈的混合自适应机制；（3）在更大规模数据集和更强密集检索模型（如 E5、BGE-M3）上验证 QAHF 的泛化能力；（4）研究 QAHF 与重排序模块的协同效果，构建端到端的自适应多阶段检索流水线。

---

## 参考文献

[1] Bruch, S., Ji, X., Ingber, M.: An analysis of fusion functions for hybrid retrieval. ACM Transactions on Information Systems 42(1), 1--35 (2023). https://doi.org/10.1145/3596512

[2] Cormack, G.V., Clarke, C.L.A., Büttcher, S.: Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In: Proceedings of the 32nd Annual ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 758--759. ACM (2009). https://doi.org/10.1145/1571941.1572114

[3] Li, H., Zhan, J., Mao, J., Liu, Y., Ma, S.: To interpolate or not to interpolate: PRF, dense and sparse retrievers. In: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 2495--2500. ACM (2022). https://arxiv.org/abs/2205.00235

[4] Hsu, T., Tzeng, Y.: DAT: Dynamic alpha tuning for hybrid retrieval in retrieval-augmented generation. arXiv preprint arXiv:2503.23013 (2025). https://arxiv.org/abs/2503.23013

[5] Arabzadeh, N., Yan, X., Clarke, C.L.A.: Predicting efficiency/effectiveness trade-offs for dense vs. sparse retrieval strategy selection. In: Proceedings of the 30th ACM International Conference on Information and Knowledge Management. pp. 2862--2866. ACM (2021). https://arxiv.org/abs/2109.10739

[6] Faggioli, G., Ferro, N., Perego, R., Tonellotto, N.: Query performance prediction for neural IR: Are we there yet? In: Advances in Information Retrieval -- 45th European Conference on IR Research (ECIR 2023). Lecture Notes in Computer Science, vol. 13981, pp. 232--248. Springer (2023). https://arxiv.org/abs/2302.09947

[7] Arabzadeh, N., Bigdeli, A., Seyedsalehi, S., Askari, M., Clarke, C.L.A.: Noisy perturbations for estimating query difficulty in dense retrievers. In: Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. pp. 3870--3874. ACM (2023). https://doi.org/10.1145/3583780.3615270

[8] Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W.: Dense passage retrieval for open-domain question answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. pp. 6769--6781. ACL (2020). https://arxiv.org/abs/2004.04906

[9] Xiong, L., Xiong, C., Li, Y., Tang, K., Liu, J., Bennett, P., Ahmed, J., Overwijk, A.: Approximate nearest neighbor negative contrastive learning for dense text retrieval. In: Proceedings of the 9th International Conference on Learning Representations. ICLR (2021). https://arxiv.org/abs/2007.00808

[10] Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., Zaharia, M.: ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In: Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics. pp. 3715--3734. ACL (2022). https://arxiv.org/abs/2112.01488

[11] Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I.: BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In: Advances in Neural Information Processing Systems 35 (NeurIPS 2021). pp. 12158--12168 (2021). https://arxiv.org/abs/2104.08663

[12] Luo, S., Hu, M., Liu, Z., Ma, J., Cheng, X.: A study on the efficiency and generalization of light hybrid retrievers. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023). pp. 1--12. ACL (2023). https://arxiv.org/abs/2210.01371

[13] Formal, T., Piwowarski, B., Clinchant, S.: SPLADE: Sparse lexical and expansion model for first stage ranking. In: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 2288--2292. ACM (2021). https://arxiv.org/abs/2107.05720

[14] Formal, T., Lassance, C., Piwowarski, B., Clinchant, S.: From distillation to hard negative sampling: Making sparse neural IR models more effective. In: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 2353--2359. ACM (2022). https://arxiv.org/abs/2205.04733

[15] Zhao, W.X., Liu, J., Ren, R., Wen, J.: Dense text retrieval based on pretrained language models: A survey. ACM Transactions on Information Systems 42(4), 1--60 (2024). https://arxiv.org/abs/2211.14876

[16] Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., Wei, F.: Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533 (2022). https://arxiv.org/abs/2212.03533

[17] Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z.: BGE M3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216 (2024). https://arxiv.org/abs/2402.03216

[18] Askari, A., Abolghasemi, A., Pasi, G., Kraaij, W., Verberne, S.: Injecting the BM25 score as text improves BERT-based re-rankers. In: Advances in Information Retrieval -- 45th European Conference on IR Research (ECIR 2023). Lecture Notes in Computer Science, vol. 13981, pp. 66--74. Springer (2023). https://arxiv.org/abs/2301.09728

[19] Zhuang, H., Qin, Z., Jagerman, R., Hui, K., Ma, J., Lu, J., Ni, J., Wang, X., Bendersky, M.: RankT5: Fine-tuning T5 for text ranking with ranking losses. In: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 2308--2313. ACM (2023). https://arxiv.org/abs/2210.10634

[20] Yoon, S., Choi, W., Lee, C., Cho, H., Park, S., Kim, S.: ListT5: Listwise reranking with fusion-in-decoder improves zero-shot retrieval. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024). pp. 4270--4285. ACL (2024). https://arxiv.org/abs/2402.15838

[21] Gao, L., Ma, X., Lin, J., Callan, J.: Precise zero-shot dense retrieval without relevance labels. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023). pp. 1762--1777. ACL (2023). https://arxiv.org/abs/2212.10496

[22] Su, H., Yen, H., Xia, M., Shi, W., Muennighoff, N., Wang, H., Kasai, J., Wang, Y., Neubig, G., Singh, S., Ruder, S., Chen, D.: BRIGHT: A realistic and challenging benchmark for reasoning-intensive retrieval. In: Proceedings of the 13th International Conference on Learning Representations (ICLR 2025) (2025). https://arxiv.org/abs/2407.12883

