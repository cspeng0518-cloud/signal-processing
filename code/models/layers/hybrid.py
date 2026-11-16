"""混合架构层体系（Jamba / Qwen3-Next 式 hybrid 层栈）

现代大模型的主流做法：同一 backbone 内按层混合不同 token 混合器——
注意力（全局精确检索）+ 线性注意力 / SSM（线性复杂度长程建模），
在质量与效率之间取得折中（参考 Jamba、Qwen3-Next、Samba）。

层级类层次（与 backbone.py 的编码器级层次同构，调用方零分支）:
  BaseSequenceLayer     抽象基类：统一 forward 契约
  ├── AttentionLayer    注意力层（full / MLA 核心可插拔）+ FFN，pre-norm
  ├── GatedDeltaLayer   Gated DeltaNet 层 + FFN（对齐 Qwen3NextDecoderLayer
  │                     的 linear_attn + MLP 结构）
  └── MambaLayer        Mamba 层（block 内部自带扩展投影，与 MambaEncoder
                        用法一致，不外加 FFN）

build_sequence_layer 工厂在初始化阶段按层类型选择实现；
HybridEncoder（见 models/backbone.py）按 layer_types 逐层配置堆叠
（对齐 GLM-5.2 config 的逐层列表风格，短列表自动循环铺满；
当前主力排布为 full_attention : mla = 1 : 2，gda/mamba 为可选扩展）。
"""

from abc import ABC, abstractmethod

import torch.nn as nn

from .attention import RoPEAttention, MLAAttention, SwiGLU
from .gated_delta import GatedDeltaAttention
from .mamba import MambaBlock

# 支持的混合层类型: full_attention=标准全局注意力（精确检索），
# mla/gda/mamba=sparse attn（KV 压缩 / 线性注意力 / SSM，长程线性复杂度）
LAYER_TYPES = {"full_attention", "mla", "gda", "mamba"}

# 历史别名 -> 规范名（"attn" 为 layer_types 的旧写法）
_LAYER_ALIASES = {"attn": "full_attention", "full": "full_attention", "fa": "full_attention"}


def normalize_layer_type(layer_type):
    """层类型归一：小写 + 别名映射，未知类型报错。"""
    t = _LAYER_ALIASES.get(layer_type.lower(), layer_type.lower())
    if t not in LAYER_TYPES:
        raise ValueError(f"未知层类型: {layer_type}，可选 {sorted(LAYER_TYPES)}")
    return t


class BaseSequenceLayer(nn.Module, ABC):
    """混合栈统一层契约。

    所有层实现 [B,L,D] -> [B,L,D]；rope 为编码器持有的共享旋转表，
    pos_ids 为 token 原始下标（MAE gather 后的真实位置）——
    位置相关实现（full_attention/mla）使用，位置无关实现（gda/mamba）忽略。
    """

    @abstractmethod
    def forward(self, x, rope, pos_ids=None, key_padding_mask=None):
        """[B, L, D] -> [B, L, D]。"""


class AttentionLayer(BaseSequenceLayer):
    """注意力层：token 混合（full / MLA 核心可插拔）+ 通道混合（FFN），
    pre-norm 结构对齐 RoPEEncoderLayer / Llama。"""

    def __init__(self, attn, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = attn
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, rope, pos_ids=None, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), rope, pos_ids, key_padding_mask)
        return x + self.ff(self.norm2(x))


class GatedDeltaLayer(BaseSequenceLayer):
    """Gated DeltaNet 层 + FFN（GDA 内部已含投影/卷积/门控，
    外层补通道混合，pre-norm 对齐 Qwen3NextDecoderLayer）。"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, conv_kernel=4):
        super().__init__()
        self.gda = GatedDeltaAttention(d_model, nhead, conv_kernel=conv_kernel)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, rope, pos_ids=None, key_padding_mask=None):
        # GDA 位置无关：rope / pos_ids / key_padding_mask 均忽略
        x = x + self.gda(self.norm1(x))
        return x + self.ff(self.norm2(x))


class MambaLayer(BaseSequenceLayer):
    """Mamba 层（纯 torch 原生 selective scan；序列模型：
    rope / pos_ids / key_padding_mask 均忽略）。"""

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.mamba = MambaBlock(d_model=d_model, d_state=d_state, expand=expand)

    def forward(self, x, rope, pos_ids=None, key_padding_mask=None):
        return self.mamba(x)


# ================================================================ 工厂
def build_sequence_layer(layer_type, d_model, nhead, dim_feedforward=None,
                         dropout=0.1, mla_kv_lora_rank=32, mla_q_lora_rank=32,
                         d_state=16, expand=2):
    """层工厂：初始化时按层类型选择实现（layer_types 列表的每一项）。

    所有实现派生自 BaseSequenceLayer，forward 契约完全一致，
    HybridEncoder 逐层调用无需分支。
    """
    if dim_feedforward is None:
        dim_feedforward = 4 * d_model
    t = normalize_layer_type(layer_type)
    if t == "full_attention":
        return AttentionLayer(RoPEAttention(d_model, nhead, dropout),
                              d_model, dim_feedforward, dropout)
    if t == "mla":
        return AttentionLayer(
            MLAAttention(d_model, nhead, kv_lora_rank=mla_kv_lora_rank,
                         q_lora_rank=mla_q_lora_rank, dropout=dropout),
            d_model, dim_feedforward, dropout)
    if t == "gda":
        return GatedDeltaLayer(d_model, nhead, dim_feedforward, dropout)
    if t == "mamba":
        return MambaLayer(d_model, d_state=d_state, expand=expand)
    raise ValueError(f"未知层类型: {layer_type}，可选 {sorted(LAYER_TYPES)}")
