"""任务头（多任务适配，对应赛题任务2）

架构原则：三个任务共享同一 backbone（权重共用 >80%），
每个任务只挂一个轻量 head（参数占比 <20%）：
  - SystemHead:    体制识别 8 类
  - ModulationHead: 调制识别 23 类
  - ParameterHead:  7 参数联合回归（多输出结构，参考 EMind RPE 多任务回归头）

统一从 CLS token 表征（backbone 输出第 0 个位置）读取全局特征。
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """轻量分类头：CLS -> MLP -> logits"""

    def __init__(self, d_model=128, n_classes=8, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, ctx):
        return self.net(ctx[:, 0])  # CLS token 位置


class RegressionHead(nn.Module):
    """多输出回归头：一次联合回归 7 个物理参数（EMind 式多任务回归）

    输出经 sigmoid 映射到 [0,1]，训练前对参数做 min-max 归一化，
    推理时再反归一化回原始量纲（官方要求输出原始单位）。
    """

    def __init__(self, d_model=128, n_params=7, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_params),
            nn.Sigmoid(),  # 归一化参数空间 [0,1]
        )
        self.n_params = n_params

    def forward(self, ctx):
        return self.net(ctx[:, 0])  # [B, n_params]，范围 [0,1]
