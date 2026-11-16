"""基座 Backbone（参考 EMind V 节风格）

类层次（全部派生自 BaseEncoder，forward 接口完全一致，调用方零分支）:
  BaseEncoder                 抽象基类：统一 forward 契约与 MAE gather/scatter 公共逻辑
  ├── SinTransformerEncoder   绝对正弦位置编码 Transformer（PE 在 tokenizer 阶段叠加）
  ├── RoPETransformerEncoder  旋转位置编码 Transformer（q/k 在注意力内旋转）
  ├── MambaEncoder            选择性状态空间模型（纯 torch 原生实现，见 layers/mamba.py）
  └── HybridEncoder           混合架构（full_attention/mla 按层排布，见 layers/hybrid.py）

build_backbone 工厂在初始化时按 (backbone, pos_encoding) 配置选择实现；
注意力组件（RoPEAttention / MLAAttention / RoPEEncoderLayer）见 layers/attention.py，
混合层体系（BaseSequenceLayer 及其子类）见 layers/hybrid.py，
Mamba block（纯 torch selective scan）见 layers/mamba.py。
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .tokenizer import RoPEPositionalEncoding  # RoPE cos/sin 旋转表（tokenizer.py 中定义）
from .layers.attention import RoPEAttention, RoPEEncoderLayer  # 注意力组件
from .layers.hybrid import build_sequence_layer, normalize_layer_type  # 混合层工厂与类型归一
from .layers.mamba import MambaBlock  # 纯 torch 原生 Mamba（无 mamba_ssm 依赖）


# ==================================================================== 基类
class BaseEncoder(nn.Module, ABC):
    """Backbone 抽象基类。

    统一封装三条前向路径（调用方视角完全一致）:
      keep_ids: MAE 加速路径，gather 出可见 token 只编码 K 个，scatter 回原位
      mask:     逻辑掩码，被掩码 token 不参与注意力（不省计算）
      默认:     全序列编码

    子类只需实现 _encode 核心编码。pos_ids 为 token 原始下标（gather 后的
    紧凑序列传真实位置），依赖位置的实现（RoPE）使用，其余实现忽略。
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    @abstractmethod
    def _encode(self, x, pos_ids=None, key_padding_mask=None):
        """核心编码 [B,L,D] -> [B,L,D]，由子类实现。"""

    def forward(self, x, mask=None, keep_ids=None, pos_ids=None):
        """
        Args:
            x: [B, L, D] token 序列
            mask: [B, L] bool，True=该 token 被掩码（跳过注意力）
            keep_ids: [B, K] 可见 token 下标（MAE 加速路径，优先于 mask）
            pos_ids: [B, L] token 原始下标（保持 gather 后的真实位置）
        Returns:
            ctx: [B, L, D] 与输入同位置的上下文表征（被掩码位置为零向量）
        """
        B, L, D = x.shape
        if keep_ids is not None:
            # MAE 加速路径：只编码可见 token，再 scatter 回原位
            gathered = torch.gather(x, 1, keep_ids.unsqueeze(-1).expand(-1, -1, D))
            encoded = self._encode(gathered, pos_ids=keep_ids)  # 原始下标作位置
            ctx = torch.zeros_like(x)
            ctx.scatter_(1, keep_ids.unsqueeze(-1).expand(-1, -1, D), encoded)
            return self.norm(ctx)
        if mask is not None:
            # key_padding_mask: True=忽略 -> 可见 token 参与，掩码 token 跳过
            return self.norm(self._encode(x, pos_ids=pos_ids, key_padding_mask=mask))
        return self.norm(self._encode(x, pos_ids=pos_ids))


# ================================================================ 子类
class SinTransformerEncoder(BaseEncoder):
    """绝对正弦位置编码 Transformer（默认）。

    位置信息在 tokenizer 阶段以可加性 PE 叠加进 token，编码本身与位置无关，
    pos_ids 参数在此实现中被忽略（保持接口统一）。
    """

    def __init__(self, d_model=128, n_layers=12, nhead=4, dim_feedforward=4 * 128,
                 dropout=0.1, rope_max_len=None):
        super().__init__(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _encode(self, x, pos_ids=None, key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class RoPETransformerEncoder(BaseEncoder):
    """旋转位置编码 Transformer。

    token 不叠加位置向量，q/k 在注意力内按相对位置旋转；gather 后必须用
    原始下标（pos_ids）保持真实位置。cos/sin 表持有一次，各层共享。
    """

    def __init__(self, d_model=128, n_layers=12, nhead=4, dim_feedforward=4 * 128,
                 dropout=0.1, rope_max_len=512):
        super().__init__(d_model)
        self.rope = RoPEPositionalEncoding(d_model, nhead, max_len=rope_max_len)
        self.layers = nn.ModuleList(
            [RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
             for _ in range(n_layers)]
        )

    def _encode(self, x, pos_ids=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, self.rope, pos_ids, key_padding_mask)
        return x


class MambaEncoder(BaseEncoder):
    """Mamba 状态空间 backbone（长序列线性复杂度，纯 torch 原生实现）。

    序列模型无注意力：不支持任意位置掩码跳算（mask/pos_ids 被忽略），
    MAE keep_ids 路径由基类 gather 保证接口一致但无注意力级加速。
    """

    def __init__(self, d_model=128, n_layers=12, d_state=16, expand=2):
        super().__init__(d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model=d_model, d_state=d_state, expand=expand)
             for _ in range(n_layers)]
        )

    def _encode(self, x, pos_ids=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x)
        return x


class HybridEncoder(BaseEncoder):
    """混合架构编码器（Jamba / Qwen3-Next / GLM-5.2 式）：按 layer_types 逐层堆叠异构层。

    layer_types 显式声明每层的 token 混合器（对齐 GLM-5.2 config 的
    indexer_types / mlp_layer_types 逐层列表风格）:
      "full_attention"  标准全局注意力（精确检索）
      "mla"/"gda"/"mamba" sparse attn（KV 压缩 / 线性注意力 / SSM，长程线性复杂度）
    列表长度等于 n_layers 时逐层精确对应；短于 n_layers 时按模式串循环铺满
    （兼容旧 hybrid_layers 模式串写法）；长于 n_layers 报错。

    层级多态见 layers/hybrid.py（BaseSequenceLayer + build_sequence_layer 工厂），
    本类只持有共享 RoPE 旋转表并逐层调用，无任何层类型分支。

    注意力类层（full_attention/mla）依赖 RoPE，GDA/Mamba 层位置无关——因此
    hybrid backbone 必须搭配 pos_encoding='rope'（tokenizer 不叠加绝对位置编码）。
    """

    def __init__(self, d_model=128, n_layers=12, nhead=4, layer_types=None,
                 dim_feedforward=None, dropout=0.1, rope_max_len=512,
                 mla_kv_lora_rank=32, mla_q_lora_rank=32):
        super().__init__(d_model)
        if not layer_types:
            raise ValueError(
                "hybrid backbone 需要非空 layer_types 列表，"
                "如 ['full_attention','mla','gda','mamba']（长度不足 n_layers 时循环铺满）")
        types = [normalize_layer_type(t) for t in layer_types]
        if len(types) > n_layers:
            raise ValueError(
                f"layer_types 长度 {len(types)} 超过 n_layers={n_layers}，"
                "请提供精确逐层列表或短模式串（自动循环铺满）")
        self.rope = RoPEPositionalEncoding(d_model, nhead, max_len=rope_max_len)
        pattern = [types[i % len(types)] for i in range(n_layers)]
        self.layers = nn.ModuleList(
            [build_sequence_layer(t, d_model, nhead,
                                  dim_feedforward=dim_feedforward, dropout=dropout,
                                  mla_kv_lora_rank=mla_kv_lora_rank,
                                  mla_q_lora_rank=mla_q_lora_rank)
             for t in pattern])

    def _encode(self, x, pos_ids=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, self.rope, pos_ids, key_padding_mask)
        return x


# ================================================================ 工厂
_TRANSFORMERS = {"sin": SinTransformerEncoder, "rope": RoPETransformerEncoder}


def build_backbone(name="transformer", **kwargs):
    """backbone 工厂：初始化时按 (backbone, pos_encoding) 配置选择实现。

    所有实现派生自 BaseEncoder，forward 接口完全一致，调用方无需分支。
    """
    name = name.lower()
    pos_encoding = kwargs.pop("pos_encoding", "sin")
    # hybrid 专属参数统一弹出，仅 hybrid 分支消费（其余实现不感知）；
    # hybrid_layers 为旧配置键名，归一为 layer_types（两键并存时以新键为准）
    hybrid_kwargs = {k: kwargs.pop(k) for k in
                     ("layer_types", "hybrid_layers", "mla_kv_lora_rank", "mla_q_lora_rank")
                     if k in kwargs}
    legacy = hybrid_kwargs.pop("hybrid_layers", None)
    if "layer_types" not in hybrid_kwargs and legacy is not None:
        hybrid_kwargs["layer_types"] = legacy
    if name == "transformer":
        if pos_encoding not in _TRANSFORMERS:
            raise ValueError(
                f"未知 pos_encoding: {pos_encoding}，可选 {sorted(_TRANSFORMERS)}")
        return _TRANSFORMERS[pos_encoding](**kwargs)
    if name == "hybrid":
        if pos_encoding != "rope":
            raise ValueError("hybrid backbone 为 RoPE 原生（full_attention/mla 层需旋转位置），"
                             "请设置 pos_encoding='rope'")
        kwargs.update(hybrid_kwargs)
        return HybridEncoder(**kwargs)
    if name == "mamba":
        if pos_encoding == "rope":
            raise ValueError("Mamba 不支持 RoPE（序列模型无注意力），请用 pos_encoding='sin'")
        kwargs.pop("nhead", None)          # Mamba 无多头概念，忽略
        kwargs.pop("rope_max_len", None)   # 位置编码由 tokenizer 承担，忽略
        kwargs.pop("dim_feedforward", None)  # Mamba block 自带扩展投影，无外加 FFN
        return MambaEncoder(**kwargs)
    raise ValueError(f"未知 backbone: {name}，可选 'transformer' | 'hybrid' | 'mamba'")
