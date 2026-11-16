"""通用网络层子包（与赛题任务无关的可插拔组件，参考 sglang layers/ 组织）

  attention.py   注意力组件（RoPE / MLA）+ RMSNorm（纯 torch 原生实现）
  gated_delta.py Gated DeltaNet 门控增量线性注意力（纯 torch 原生递归）
  mamba.py       Mamba 选择性状态空间 block（纯 torch 原生 selective scan）
  hybrid.py      混合层体系（BaseSequenceLayer 及其子类）+ build_sequence_layer 工厂

backbone（编码器级类层次与工厂）位于 models/backbone.py 顶层；
赛题相关模块（foundation / tokenizer / heads / frequency）同样在 models/ 顶层。
"""

from .attention import RoPEAttention, MLAAttention, RoPEEncoderLayer, RMSNorm
from .gated_delta import GatedDeltaAttention
from .mamba import MambaBlock
from .hybrid import BaseSequenceLayer, build_sequence_layer, LAYER_TYPES, normalize_layer_type

__all__ = [
    "RoPEAttention", "MLAAttention", "RoPEEncoderLayer", "RMSNorm",
    "GatedDeltaAttention", "MambaBlock",
    "BaseSequenceLayer", "build_sequence_layer", "LAYER_TYPES", "normalize_layer_type",
]
