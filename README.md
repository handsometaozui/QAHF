# QAHF: Query-Aware Adaptive Hybrid Retrieval Fusion

A pre-retrieval approach for adaptive BM25 and dense retrieval combination. QAHF predicts the optimal per-query fusion weight α(q) from 14 lightweight query features before retrieval executes, adding less than 0.2 ms overhead.

---

## Requirements

```bash
pip install -r src/requirements.txt
```

**Key dependencies:** `torch`, `sentence-transformers`, `rank-bm25`, `beir`, `faiss-cpu`

---

## Data Preparation

Download the six BEIR datasets used in this paper:

```bash
python src/download_beir.py --datasets fiqa scidocs cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/physics
```

Datasets are saved to `data/beir/` by default.

---

## Reproducing Results

### 1. Main Experiment (Table 2)

Run QAHF against all baselines (BM25, Dense, RRF, Oracle) on all six datasets:

```bash
python src/improved_experiment.py --dataset fiqa       --bm25_k1 1.2 --bm25_b 0.4
python src/improved_experiment.py --dataset scidocs    --bm25_k1 1.5 --bm25_b 0.75
python src/improved_experiment.py --dataset cqadupstack/android --bm25_k1 1.2 --bm25_b 0.4
python src/improved_experiment.py --dataset cqadupstack/english --bm25_k1 1.2 --bm25_b 0.4
python src/improved_experiment.py --dataset cqadupstack/gaming  --bm25_k1 1.2 --bm25_b 0.4
python src/improved_experiment.py --dataset cqadupstack/physics --bm25_k1 1.2 --bm25_b 0.4
```

Results are saved to `results/<dataset>/`.

**Expected results (NDCG@10):**

| Dataset | BM25 | Dense | RRF | QAHF | Oracle |
|---------|------|-------|-----|------|--------|
| FIQA | 0.2347 | 0.3248 | 0.3433 | 0.3469 | 0.4304 |
| SciDocs | 0.1613 | 0.2368 | 0.2111 | 0.2256 | 0.2729 |
| CQA-Android | 0.3703 | 0.5303 | 0.4798 | 0.5306 | 0.5977 |
| CQA-English | 0.2895 | 0.4402 | 0.4065 | 0.4477 | 0.5196 |
| CQA-Gaming | 0.4164 | 0.4938 | 0.4966 | 0.5081 | 0.6051 |
| CQA-Physics | 0.3387 | 0.4625 | 0.4255 | 0.4647 | 0.5278 |

Dense model: `sentence-transformers/all-MiniLM-L6-v2`. All experiments run on CPU.

---

### 2. Ablation Study (Table 4)

Compare QAHF (MLP) against RRF, Best Fixed α, Linear Regression, and Ridge Regression on four datasets:

```bash
python src/ablation_component.py --datasets fiqa cqadupstack/android cqadupstack/english cqadupstack/physics
```

Results are saved to `results/<dataset>/ablation_component.json`.

**Expected results (NDCG@10, Δ relative to RRF):**

| Configuration | FIQA | CQA-Android | CQA-English | CQA-Physics |
|---------------|------|-------------|-------------|-------------|
| RRF (α=0.5) | 0.3433 | 0.4798 | 0.4065 | 0.4255 |
| Best Fixed α | 0.3577 (+4.17%) | 0.5394 (+12.43%) | 0.4467 (+9.89%) | 0.4692 (+10.25%) |
| Linear Regression (14-dim) | 0.3569 (+3.96%) | 0.5274 (+9.92%) | 0.4456 (+9.62%) | 0.4600 (+8.10%) |
| Ridge Regression (14-dim) | 0.3571 (+4.00%) | 0.5275 (+9.94%) | 0.4455 (+9.58%) | 0.4604 (+8.20%) |
| **QAHF (14-dim MLP)** | **0.3469 (+0.85%)** | **0.5306 (+10.59%)** | **0.4477 (+10.13%)** | **0.4647 (+9.20%)** |
| Oracle | 0.4304 | 0.5977 | 0.5196 | 0.5278 |

---

### 3. LODO Cross-Dataset Generalization (Table 5)

Leave-One-Dataset-Out experiment: train on 5 datasets, test on the held-out dataset, repeat for all 6 folds:

```bash
python src/lodo_experiment.py --limit 500
```

Results are saved to `results/lodo/lodo_results.json`.

**Expected results (NDCG@10):**

| Held-out Dataset | RRF | QAHF-LODO | Improvement |
|------------------|-----|-----------|-------------|
| FIQA | 0.3433 | 0.3523 | +2.61% |
| SciDocs | 0.2111 | 0.2307 | +9.29% |
| CQA-Android | 0.4798 | 0.5296 | +10.39% |
| CQA-English | 0.4065 | 0.4465 | +9.82% |
| CQA-Gaming | 0.4966 | 0.5090 | +2.49% |
| CQA-Physics | 0.4255 | 0.4603 | +8.17% |
| **Average** | 0.3605 | 0.3881 | **+7.13%** |

---

## Project Structure

```
src/
├── improved_experiment.py   # Main experiment (Table 2)
├── ablation_component.py    # Ablation study (Table 4)
├── lodo_experiment.py       # LODO generalization experiment (Table 5)
├── qahf_model.py            # QAHF model and weight predictor
├── feature_extractor.py     # 14-dim query feature extraction
├── baselines.py             # BM25, Dense, RRF baselines
├── evaluator.py             # NDCG@10, MRR@10, Recall@100
├── config.py                # Paths and hyperparameters
└── requirements.txt         # Dependencies
results/
├── fiqa/                    # Per-dataset results and trained models
├── scidocs/
├── cqadupstack/
└── lodo/                    # LODO experiment results
```

---

## Latency

| Stage | Time |
|-------|------|
| Query feature extraction | ~0.12 ms |
| Weight prediction (MLP inference) | ~0.08 ms |
| **QAHF total overhead** | **~0.20 ms** |
| BM25 search (100K docs) | 15–50 ms |
| Dense search (FAISS) | 5–20 ms |

QAHF overhead is 0.3–1.2% of total retrieval latency.
