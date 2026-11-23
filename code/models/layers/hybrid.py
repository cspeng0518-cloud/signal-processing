"""混合架构层体系（Jamba / Qwen3-Next 式 hybrid 层栈）+ 统一层工厂

统一层契约（HybridEncoder 逐层调用，零分支）:
    forward(x, rope=None, pos_ids=None, key_padding_mask=None, time_slice=None) -> [B, L, D]

  rope / pos_ids        RoPE 旋转表与 token 原始下标（MAE gather 后的真实位置），
                        位置相关层（full_attention / mla）使用，其余忽略
  key_padding_mask      [B, L] True=忽略（512 点样本补零 token、MAE 填充槽位）
  time_slice            (s, e) 时域 token 段，仅 conformer 的卷积模块使用

层类型（layer_types 列表取值）:
  full_attention  标准全局注意力（FA），别名 fa / attn / full
  mla             DeepSeek-V2 多头潜在注意力（KV/Q 低秩压缩）
  gda             Gated DeltaNet 线性注意力（可选扩展）
  mamba           Mamba SSM（可选扩展）
  conformer       Macaron Conformer block（默认只用于微调 neck，见 conformer.py）

当前主力配置：预训练主干 full_attention : mla = 1 : 2；微调 neck 为 conformer。
"""

from abc import ABC, abstractmethod

import torch.nn as nn

from .attention import DropPath, MLAAttention, MultiHeadAttention, SwiGLU
from .conformer import ConformerLayer
from .gated_delta import GatedDeltaAttention
from .mamba import MambaBlock

LAYER_TYPES = {"full_attention", "mla", "gda", "mamba", "conformer"}
# 需要连续时间序列的层（不能用于 MAE 只编码可见 token 的 gather 路径）
CONTIGUOUS_ONLY = {"conformer"}

_LAYER_ALIASES = {"attn": "full_attention", "full": "full_attention", "fa": "full_attention"}


def normalize_layer_type(layer_type):
    """层类型归一：小写 + 别名映射，未知类型报错。"""
    t = _LAYER_ALIASES.get(str(layer_type).lower(), str(layer_type).lower())
    if t not in LAYER_TYPES:
        raise ValueError(f"未知层类型: {layer_type}，可选 {sorted(LAYER_TYPES)}")
    return t


class BaseSequenceLayer(nn.Module, ABC):
    """混合栈统一层契约（见模块文档）。"""

    @abstractmethod
    def forward(self, x, rope=None, pos_ids=None, key_padding_mask=None, time_slice=None):
        """[B, L, D] -> [B, L, D]。"""


class AttentionLayer(BaseSequenceLayer):
    """注意力层：token 混合（FA / MLA 可插拔）+ 通道混合（SwiGLU），pre-norm。"""

    def __init__(self, attn, d_model, dim_feedforward, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.attn = attn
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dp = DropPath(drop_path)

    def forward(self, x, rope=None, pos_ids=None, key_padding_mask=None, time_slice=None):
        x = x + self.dp(self.attn(self.norm1(x), rope, pos_ids, key_padding_mask))
        return x + self.dp(self.ff(self.norm2(x)))


class GatedDeltaLayer(BaseSequenceLayer):
    """Gated DeltaNet 层 + FFN（位置无关；pre-norm 对齐 Qwen3NextDecoderLayer）。"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, drop_path=0.0,
                 conv_kernel=4):
        super().__init__()
        self.gda = GatedDeltaAttention(d_model, nhead, conv_kernel=conv_kernel)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dp = DropPath(drop_path)

    def forward(self, x, rope=None, pos_ids=None, key_padding_mask=None, time_slice=None):
        x = x + self.dp(self.gda(self.norm1(x)))
        return x + self.dp(self.ff(self.norm2(x)))


class MambaLayer(BaseSequenceLayer):
    """Mamba 层（纯 torch selective scan；block 自带扩展投影，不外加 FFN）。"""

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.mamba = MambaBlock(d_model=d_model, d_state=d_state, expand=expand)

    def forward(self, x, rope=None, pos_ids=None, key_padding_mask=None, time_slice=None):
        return self.mamba(x)


def build_sequence_layer(layer_type, d_model, nhead, dim_feedforward=None, dropout=0.1,
                         drop_path=0.0, mla_kv_lora_rank=128, mla_q_lora_rank=128,
                         conv_kernel=15, d_state=16, expand=2):
    """层工厂：按层类型实例化（layer_types 列表的每一项）。"""
    if dim_feedforward is None:
        dim_feedforward = 4 * d_model
    t = normalize_layer_type(layer_type)
    if t == "full_attention":
        return AttentionLayer(MultiHeadAttention(d_model, nhead, dropout, use_rope=True),
                              d_model, dim_feedforward, dropout, drop_path)
    if t == "mla":
        return AttentionLayer(
            MLAAttention(d_model, nhead, kv_lora_rank=mla_kv_lora_rank,
                         q_lora_rank=mla_q_lora_rank, dropout=dropout),
            d_model, dim_feedforward, dropout, drop_path)
    if t == "gda":
        return GatedDeltaLayer(d_model, nhead, dim_feedforward, dropout, drop_path)
    if t == "mamba":
        return MambaLayer(d_model, d_state=d_state, expand=expand)
    if t == "conformer":
        return ConformerLayer(d_model, nhead, dim_feedforward, dropout, drop_path,
                              conv_kernel=conv_kernel)
    raise ValueError(f"未知层类型: {layer_type}")
