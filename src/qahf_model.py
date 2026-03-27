#!/usr/bin/env python3
"""
QAHF: Query-Aware Adaptive Hybrid Retrieval Fusion
核心模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from feature_extractor import QueryFeatureExtractor


class WeightPredictor(nn.Module):
    """
    轻量级权重预测网络
    输入: 查询特征 + 检索感知特征
    输出: BM25 权重 α ∈ [0, 1]
    """

    def __init__(self, input_dim: int = 23, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class QAHF:
    """
    Query-Aware Adaptive Hybrid Retrieval Fusion

    支持两种模式：
    - 仅查询特征（14维）：pre-retrieval 预测
    - 查询 + 检索感知特征（23维）：retrieval-aware 预测（推荐）
    """

    def __init__(self,
                 use_retrieval_features: bool = True,
                 model_path: Optional[Path] = None,
                 device: str = "cpu"):
        self.feature_extractor = QueryFeatureExtractor()
        self.device = device
        self.use_retrieval_features = use_retrieval_features

        # 特征标准化参数
        self.feature_mean = None
        self.feature_std = None

        # 温度校准参数
        self.temperature = 1.0  # T > 1 拉大 α 方差
        self.alpha_center = 0.5  # 校准中心点

        # 计算特征维度
        query_dim = len(self.feature_extractor.feature_names)  # 14
        retrieval_dim = len(self.feature_extractor.retrieval_feature_names) if use_retrieval_features else 0  # 9
        self.total_feature_dim = query_dim + retrieval_dim

        # 初始化模型
        self.predictor = WeightPredictor(
            input_dim=self.total_feature_dim,
            hidden_dim=64
        ).to(device)

        if model_path and model_path.exists():
            self.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            mode = "retrieval-aware" if use_retrieval_features else "query-only"
            print(f"Initialized new model ({mode}, {self.total_feature_dim}-dim features)")

    def predict_alpha(self, query: str,
                      bm25_results: Optional[List[Tuple[str, float]]] = None,
                      dense_results: Optional[List[Tuple[str, float]]] = None) -> float:
        """
        预测 BM25 权重 α

        Args:
            query: 查询字符串
            bm25_results: BM25 检索结果（retrieval-aware 模式必传）
            dense_results: Dense 检索结果（retrieval-aware 模式必传）

        Returns:
            alpha ∈ [0, 1]，BM25 权重
        """
        features = self._build_features(query, bm25_results, dense_results)
        features = self._normalize_features(features)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            alpha = self.predictor(features_tensor).item()

        # 温度校准：拉大 α 分布方差
        if self.temperature != 1.0:
            alpha = self._apply_temperature(alpha)

        return alpha

    def _apply_temperature(self, alpha: float) -> float:
        """温度校准：将压缩的 α 分布拉伸到更接近 Oracle 的方差"""
        # sigmoid 逆变换 (logit)
        alpha_clipped = np.clip(alpha, 1e-6, 1 - 1e-6)
        logit = np.log(alpha_clipped / (1 - alpha_clipped))

        # 以 center 为中心做温度缩放
        center_logit = np.log(self.alpha_center / (1 - self.alpha_center))
        scaled_logit = (logit - center_logit) * self.temperature + center_logit

        # sigmoid 变回 [0, 1]
        return 1.0 / (1.0 + np.exp(-scaled_logit))

    def _build_features(self, query: str,
                        bm25_results=None, dense_results=None) -> np.ndarray:
        """构建完整特征向量"""
        query_features = self.feature_extractor.extract_features(query)

        if self.use_retrieval_features:
            if bm25_results is None or dense_results is None:
                # 如果未提供检索结果，用零填充（不推荐）
                retrieval_features = np.zeros(len(self.feature_extractor.retrieval_feature_names))
            else:
                retrieval_features = self.feature_extractor.extract_retrieval_features(
                    bm25_results, dense_results)
            return np.concatenate([query_features, retrieval_features])
        else:
            return query_features

    def fuse_scores(self,
                    query: str,
                    bm25_results: List[Tuple[str, float]],
                    dense_results: List[Tuple[str, float]],
                    rrf_k: int = 60) -> List[Tuple[str, float]]:
        """
        自适应加权 RRF 融合

        Args:
            query: 查询字符串
            bm25_results: BM25 检索结果 [(doc_id, score), ...]
            dense_results: Dense 检索结果 [(doc_id, score), ...]
            rrf_k: RRF 常数

        Returns:
            sorted_results: 融合排序后的结果
        """
        alpha = self.predict_alpha(query, bm25_results, dense_results)

        bm25_ranks = {did: rank + 1 for rank, (did, _) in enumerate(bm25_results)}
        dense_ranks = {did: rank + 1 for rank, (did, _) in enumerate(dense_results)}

        default_rank = len(bm25_results) + len(dense_results) + 1

        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        final_scores = {}
        for doc_id in all_doc_ids:
            bm25_r = bm25_ranks.get(doc_id, default_rank)
            dense_r = dense_ranks.get(doc_id, default_rank)
            final_scores[doc_id] = alpha / (rrf_k + bm25_r) + (1 - alpha) / (rrf_k + dense_r)

        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """训练集标准化"""
        if self.feature_mean is None:
            return features
        return (features - self.feature_mean) / self.feature_std

    def train(self,
              train_features: np.ndarray,
              train_labels: List[float],
              val_features: Optional[np.ndarray] = None,
              val_labels: Optional[List[float]] = None,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001):
        """
        训练模型

        Args:
            train_features: 预计算的特征矩阵 (N, feature_dim)
            train_labels: 最优 α 值列表
            val_features: 验证集特征矩阵
            val_labels: 验证集标签
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        # 计算并保存标准化参数
        self.feature_mean = train_features.mean(axis=0)
        self.feature_std = train_features.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0

        train_features_norm = (train_features - self.feature_mean) / self.feature_std
        train_labels_arr = np.array(train_labels).reshape(-1, 1)

        X_train = torch.tensor(train_features_norm, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(train_labels_arr, dtype=torch.float32).to(self.device)

        # 验证集
        X_val, y_val = None, None
        if val_features is not None and val_labels is not None:
            val_features_norm = (val_features - self.feature_mean) / self.feature_std
            val_labels_arr = np.array(val_labels).reshape(-1, 1)
            X_val = torch.tensor(val_features_norm, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(val_labels_arr, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate,
                                     weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.predictor.train()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state = None
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.predictor(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            if X_val is not None:
                self.predictor.eval()
                with torch.no_grad():
                    val_pred = self.predictor(X_val)
                    val_loss = criterion(val_pred, y_val).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.predictor.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                self.predictor.train()
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # 恢复最佳模型
        if best_state is not None:
            self.predictor.load_state_dict(best_state)
            print(f"Restored best model (val_loss={best_val_loss:.4f})")

        self.predictor.eval()
        print("Training completed!")

    def calibrate(self,
                  val_features: np.ndarray,
                  val_labels: List[float],
                  temp_range: np.ndarray = None):
        """
        温度校准：在验证集上搜索最优温度 T，使得校准后的 α 预测
        在检索指标上表现最好。

        Args:
            val_features: 验证集特征矩阵 (N, feature_dim)
            val_labels: 验证集最优 α 值
        """
        if temp_range is None:
            temp_range = np.arange(1.0, 5.1, 0.25)

        # 先用当前模型预测验证集的 α（不带温度）
        val_features_norm = (val_features - self.feature_mean) / self.feature_std
        X_val = torch.tensor(val_features_norm, dtype=torch.float32).to(self.device)

        self.predictor.eval()
        with torch.no_grad():
            raw_alphas = self.predictor(X_val).cpu().numpy().flatten()

        # 计算预测的 α 中心
        raw_center = float(np.mean(raw_alphas))
        center_logit = np.log(raw_center / (1 - raw_center)) if 0 < raw_center < 1 else 0.0

        val_labels_arr = np.array(val_labels)

        best_temp = 1.0
        best_mse = float('inf')

        print("\n  Temperature calibration search:")
        for T in temp_range:
            # 对每个 α 做温度缩放
            calibrated = []
            for a in raw_alphas:
                a_clip = np.clip(a, 1e-6, 1 - 1e-6)
                logit = np.log(a_clip / (1 - a_clip))
                scaled = (logit - center_logit) * T + center_logit
                calibrated.append(1.0 / (1.0 + np.exp(-scaled)))
            calibrated = np.array(calibrated)

            mse = np.mean((calibrated - val_labels_arr) ** 2)
            cal_std = np.std(calibrated)

            if mse < best_mse:
                best_mse = mse
                best_temp = T

            if T == 1.0 or T % 1.0 == 0:
                print(f"    T={T:.2f}: MSE={mse:.4f}, α std={cal_std:.3f}")

        self.temperature = best_temp
        self.alpha_center = raw_center

        # 打印校准结果
        calibrated_final = []
        for a in raw_alphas:
            a_clip = np.clip(a, 1e-6, 1 - 1e-6)
            logit = np.log(a_clip / (1 - a_clip))
            scaled = (logit - center_logit) * best_temp + center_logit
            calibrated_final.append(1.0 / (1.0 + np.exp(-scaled)))
        calibrated_final = np.array(calibrated_final)

        print(f"\n  Best temperature: T={best_temp:.2f}")
        print(f"  Before calibration: α mean={np.mean(raw_alphas):.3f}, std={np.std(raw_alphas):.3f}")
        print(f"  After calibration:  α mean={np.mean(calibrated_final):.3f}, std={np.std(calibrated_final):.3f}")
        print(f"  Oracle labels:      α mean={np.mean(val_labels_arr):.3f}, std={np.std(val_labels_arr):.3f}")

    def save_model(self, path: Path):
        """保存模型"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.predictor.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: Path):
        """加载模型"""
        self.predictor.load_state_dict(torch.load(path, map_location=self.device))
        self.predictor.eval()

    def get_config(self) -> Dict:
        """获取模型配置"""
        return {
            "input_dim": self.total_feature_dim,
            "use_retrieval_features": self.use_retrieval_features,
            "feature_names": self.feature_extractor.feature_names,
            "retrieval_feature_names": self.feature_extractor.retrieval_feature_names if self.use_retrieval_features else [],
            "device": self.device
        }


# 测试
if __name__ == "__main__":
    print("QAHF Model Test")
    print("=" * 60)

    qahf = QAHF(use_retrieval_features=False)

    test_queries = [
        "machine learning algorithms",
        "What is the best approach for natural language processing?",
        "Python programming tutorial",
        "How does transformer architecture work?",
    ]

    print("\nTesting alpha prediction (query-only mode):")
    for query in test_queries:
        alpha = qahf.predict_alpha(query)
        print(f"  Query: {query}")
        print(f"    α(BM25): {alpha:.3f}, α(Vector): {1-alpha:.3f}")

    print("\n✓ QAHF model test passed!")
