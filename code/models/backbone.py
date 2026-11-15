"""基座 Backbone（参考 EMind V 节 + IFNet/nn/Transformer.py 风格）

提供两种可选架构：
  - TransformerEncoder：标准多头自注意力 encoder（EMind 同款，纯 PyTorch 便于国产化迁移）
  - MambaEncoder：选择性状态空间模型（长序列线性复杂度，需 mamba-ssm 包，可选依赖）

两者接口一致：forward(x, mask_info) -> 上下文表征
支持按样本掩码（MAE 式预训练）：掩码 patch 不参与注意力，节省计算
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # 可选依赖，未安装则 Mamba 不可用
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class TransformerEncoder(nn.Module):
    """EMind 式 1D 时间 Transformer（eager attention，无 CUDA 专有算子）"""

    def __init__(self, d_model=128, n_layers=12, nhead=4, dim_feedforward=4 * 128,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, keep_ids=None):
        """
        Args:
            x: [B, L, D] token 序列
            mask: [B, L] bool，True=该 token 被掩码（跳过注意力）
            keep_ids: [B, K] 预筛后保留的 token 下标（MAE 加速路径，优先于 mask）
        Returns:
            ctx: [B, L, D] 与输入同位置的上下文表征（被掩码位置为零向量）
        """
        B, L, D = x.shape
        if keep_ids is not None:
            # MAE 加速路径：只对可见 token 前向
            K = keep_ids.size(1)
            gathered = torch.gather(x, 1, keep_ids.unsqueeze(-1).expand(-1, -1, D))
            encoded = self.encoder(gathered)
            ctx = torch.zeros_like(x)
            ctx.scatter_(1, keep_ids.unsqueeze(-1).expand(-1, -1, D), encoded)
            return self.norm(ctx)
        if mask is not None:
            # src_key_padding_mask: True=忽略 -> 可见 token 参与，掩码 token 跳过
            encoded = self.encoder(x, src_key_padding_mask=mask)
            return self.norm(encoded)
        return self.norm(self.encoder(x))


class MambaEncoder(nn.Module):
    """Mamba 状态空间 backbone（接口与 TransformerEncoder 对齐）

    注意：Mamba 为线性复杂度序列模型，但不支持任意位置掩码跳算，
    MAE 预训练时退化为整序列前向（keep_ids 路径仅做 gather 保证接口一致）。
    """

    def __init__(self, d_model=128, n_layers=12, d_state=16, expand=2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba-ssm 未安装。安装: pip install mamba-ssm causal-conv1d，"
                "或使用 backbone='transformer'"
            )
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, expand=expand) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, keep_ids=None):
        if keep_ids is not None:
            B, L, D = x.shape
            K = keep_ids.size(1)
            gathered = torch.gather(x, 1, keep_ids.unsqueeze(-1).expand(-1, -1, D))
            for layer in self.layers:
                gathered = layer(gathered)
            ctx = torch.zeros_like(x)
            ctx.scatter_(1, keep_ids.unsqueeze(-1).expand(-1, -1, D), gathered)
            return self.norm(ctx)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def build_backbone(name="transformer", **kwargs):
    """backbone 构建入口（configs 中 backbone 字段路由到此处）"""
    name = name.lower()
    if name == "transformer":
        return TransformerEncoder(**kwargs)
    if name == "mamba":
        return MambaEncoder(**kwargs)
    raise ValueError(f"未知 backbone: {name}，可选 'transformer' | 'mamba'")
