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
    输入: 查询特征向量 (12维)
    输出: BM25 权重 α ∈ [0, 1]
    """
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, input_dim)
            
        Returns:
            alpha: (batch_size, 1) BM25 权重
        """
        return self.network(features)


class QAHF:
    """
    Query-Aware Adaptive Hybrid Retrieval Fusion
    
    自适应混合检索融合框架
    """
    
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 device: str = "cpu"):
        self.feature_extractor = QueryFeatureExtractor()
        self.device = device
        
        # 初始化模型
        self.predictor = WeightPredictor(
            input_dim=len(self.feature_extractor.feature_names)
        ).to(device)
        
        # 加载预训练模型（如果存在）
        if model_path and model_path.exists():
            self.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("Initialized new model (untrained)")
    
    def predict_alpha(self, query: str) -> float:
        """
        预测 BM25 权重 α
        
        Args:
            query: 查询字符串
            
        Returns:
            alpha: BM25 的权重 ∈ [0, 1]
            (1 - alpha) 是向量检索的权重
        """
        features = self.feature_extractor.extract_features(query)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            alpha = self.predictor(features_tensor).item()
        
        return alpha
    
    def predict_batch_alpha(self, queries: List[str]) -> np.ndarray:
        """
        批量预测 BM25 权重
        
        Args:
            queries: 查询列表
            
        Returns:
            alphas: (num_queries,) 权重数组
        """
        features_list = [
            self.feature_extractor.extract_features(q) 
            for q in queries
        ]
        features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            alphas = self.predictor(features_tensor).squeeze().cpu().numpy()
        
        return alphas
    
    def fuse_scores(self,
                    query: str,
                    bm25_scores: np.ndarray,
                    vector_scores: np.ndarray,
                    doc_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        融合 BM25 和向量检索得分
        
        Args:
            query: 查询字符串
            bm25_scores: BM25 得分数组
            vector_scores: 向量检索得分数组
            doc_ids: 文档 ID 列表
            
        Returns:
            final_scores: 融合后得分
            sorted_doc_ids: 排序后的文档 ID
        """
        # 预测权重
        alpha = self.predict_alpha(query)
        
        # 归一化得分
        bm25_norm = self._normalize_scores(bm25_scores)
        vector_norm = self._normalize_scores(vector_scores)
        
        # 加权融合
        final_scores = alpha * bm25_norm + (1 - alpha) * vector_norm
        
        # 排序
        sorted_indices = np.argsort(-final_scores)
        sorted_doc_ids = [doc_ids[i] for i in sorted_indices]
        
        return final_scores, sorted_doc_ids
    
    def _normalize_scores(self, scores: np.ndarray, method: str = "minmax") -> np.ndarray:
        """归一化得分"""
        if method == "minmax":
            min_s = scores.min()
            max_s = scores.max()
            if max_s - min_s > 1e-9:
                return (scores - min_s) / (max_s - min_s)
            else:
                return np.ones_like(scores) * 0.5
        elif method == "softmax":
            exp_s = np.exp(scores - scores.max())
            return exp_s / exp_s.sum()
        else:
            return scores / scores.sum()
    
    def train(self,
              train_queries: List[str],
              train_labels: List[float],
              val_queries: Optional[List[str]] = None,
              val_labels: Optional[List[float]] = None,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001):
        """
        训练模型
        
        Args:
            train_queries: 训练查询列表
            train_labels: 训练标签（最优 α 值）
            val_queries: 验证查询列表
            val_labels: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        # 提取特征
        train_features = np.array([
            self.feature_extractor.extract_features(q) 
            for q in train_queries
        ])
        train_labels = np.array(train_labels).reshape(-1, 1)
        
        # 转换为张量
        X_train = torch.tensor(train_features, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(train_labels, dtype=torch.float32).to(self.device)
        
        # 验证集
        X_val, y_val = None, None
        if val_queries and val_labels:
            val_features = np.array([
                self.feature_extractor.extract_features(q) 
                for q in val_queries
            ])
            val_labels = np.array(val_labels).reshape(-1, 1)
            X_val = torch.tensor(val_features, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(val_labels, dtype=torch.float32).to(self.device)
        
        # 训练设置
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        self.predictor.train()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience = 10
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
            
            # 验证
            if X_val is not None:
                self.predictor.eval()
                with torch.no_grad():
                    val_pred = self.predictor(X_val)
                    val_loss = criterion(val_pred, y_val).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
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
        
        print("Training completed!")
    
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
            "input_dim": len(self.feature_extractor.feature_names),
            "feature_names": self.feature_extractor.feature_names,
            "device": self.device
        }


# 测试
if __name__ == "__main__":
    print("QAHF Model Test")
    print("=" * 60)
    
    # 初始化模型
    qahf = QAHF()
    
    # 测试查询
    test_queries = [
        "machine learning algorithms",
        "What is the best approach for natural language processing?",
        "Python programming tutorial",
        "How does transformer architecture work?",
    ]
    
    print("\nTesting alpha prediction:")
    for query in test_queries:
        alpha = qahf.predict_alpha(query)
        print(f"  Query: {query}")
        print(f"    α(BM25): {alpha:.3f}, α(Vector): {1-alpha:.3f}")
    
    print("\n✓ QAHF model test passed!")