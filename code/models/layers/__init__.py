"""通用网络层子包（与赛题任务无关的可插拔组件）

  attention.py   RoPE 表 / 多头注意力(FA) / MLA / 跨域注意力 / SwiGLU / RMSNorm / DropPath
                 （注意力核心统一走 models/ops/fused.attention -> SDPA / NPU 融合算子）
  conformer.py   Conformer block（微调 neck 使用）
  gated_delta.py Gated DeltaNet 门控增量线性注意力（可选层类型）
  mamba.py       Mamba 选择性状态空间 block（可选层类型）
  hybrid.py      统一层契约 BaseSequenceLayer + build_sequence_layer 工厂

编码器（HybridEncoder，主干与 neck 共用同一个类）位于 models/backbone.py。
"""

from .attention import (CrossAttention, DropPath, MLAAttention, MultiHeadAttention,
                        RMSNorm, RoPEPositionalEncoding, SwiGLU)
from .conformer import ConformerLayer, ConvModule
from .gated_delta import GatedDeltaAttention
from .hybrid import (CONTIGUOUS_ONLY, LAYER_TYPES, AttentionLayer, BaseSequenceLayer,
                     GatedDeltaLayer, MambaLayer, build_sequence_layer,
                     normalize_layer_type)
from .mamba import MambaBlock

__all__ = [
    "CrossAttention", "DropPath", "MLAAttention", "MultiHeadAttention", "RMSNorm",
    "RoPEPositionalEncoding", "SwiGLU", "ConformerLayer", "ConvModule",
    "GatedDeltaAttention", "MambaBlock",
    "CONTIGUOUS_ONLY", "LAYER_TYPES", "AttentionLayer", "BaseSequenceLayer",
    "GatedDeltaLayer", "MambaLayer", "build_sequence_layer", "normalize_layer_type",
]
