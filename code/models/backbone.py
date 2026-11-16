"""基座 Backbone（参考 EMind V 节 + IFNet/nn/Transformer.py 风格）

类层次（全部派生自 BaseEncoder，forward 接口完全一致，调用方零分支）:
  BaseEncoder                 抽象基类：统一 forward 契约与 MAE gather/scatter 公共逻辑
  ├── SinTransformerEncoder   绝对正弦位置编码 Transformer（PE 在 tokenizer 阶段叠加）
  ├── RoPETransformerEncoder  旋转位置编码 Transformer（q/k 在注意力内旋转）
  └── MambaEncoder            选择性状态空间模型（需 mamba-ssm 包，可选依赖）

build_backbone 工厂在初始化时按 (backbone, pos_encoding) 配置选择实现；
RoPEAttention / RoPEEncoderLayer 是 RoPETransformerEncoder 的内部组件。
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .tokenizer import RoPEPositionalEncoding  # RoPE cos/sin 旋转表（tokenizer.py 中定义）

try:
    from mamba_ssm import Mamba  # 可选依赖，未安装则 Mamba 不可用
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


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


# ================================================ RoPE 内部组件
class RoPEAttention(nn.Module):
    """带 RoPE 的多头自注意力（eager 实现，无 CUDA 专有算子）。

    q/k 经 rope.rotate() 按位置旋转后做标准缩放点积注意力。
    nn.TransformerEncoderLayer 的注意力不可注入旋转，故此组件必须自实现，
    但仅作为 RoPETransformerEncoder 的内部实现细节存在。
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x, rope, pos_ids=None, key_padding_mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, L, self.nhead, self.head_dim)
        q = rope.rotate(q.view(shape), pos_ids)
        k = rope.rotate(k.view(shape), pos_ids)
        v = v.view(shape)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))           # [B, H, L, Dh]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:                           # True = 忽略
            attn = attn.masked_fill(key_padding_mask[:, None, None, :],
                                    float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_drop(self.proj(out))


class RoPEEncoderLayer(nn.Module):
    """post-norm Transformer encoder layer（结构对齐 nn.TransformerEncoderLayer 默认）。"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = RoPEAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, rope, pos_ids=None, key_padding_mask=None):
        x = self.norm1(x + self.attn(x, rope, pos_ids, key_padding_mask))
        return self.norm2(x + self.ff(x))


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
    """Mamba 状态空间 backbone（长序列线性复杂度，需 mamba-ssm 包）。

    序列模型无注意力：不支持任意位置掩码跳算（mask/pos_ids 被忽略），
    MAE keep_ids 路径由基类 gather 保证接口一致但无注意力级加速。
    """

    def __init__(self, d_model=128, n_layers=12, d_state=16, expand=2):
        super().__init__(d_model)
        if not HAS_MAMBA:
            raise ImportError(
                "mamba-ssm 未安装。安装: pip install mamba-ssm causal-conv1d，"
                "或使用 backbone='transformer'"
            )
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, expand=expand) for _ in range(n_layers)]
        )

    def _encode(self, x, pos_ids=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x)
        return x


# ================================================================ 工厂
_TRANSFORMERS = {"sin": SinTransformerEncoder, "rope": RoPETransformerEncoder}


def build_backbone(name="transformer", **kwargs):
    """backbone 工厂：初始化时按 (backbone, pos_encoding) 配置选择实现。

    所有实现派生自 BaseEncoder，forward 接口完全一致，调用方无需分支。
    """
    name = name.lower()
    pos_encoding = kwargs.pop("pos_encoding", "sin")
    if name == "transformer":
        if pos_encoding not in _TRANSFORMERS:
            raise ValueError(
                f"未知 pos_encoding: {pos_encoding}，可选 {sorted(_TRANSFORMERS)}")
        return _TRANSFORMERS[pos_encoding](**kwargs)
    if name == "mamba":
        if pos_encoding == "rope":
            raise ValueError("Mamba 不支持 RoPE（序列模型无注意力），请用 pos_encoding='sin'")
        kwargs.pop("nhead", None)          # Mamba 无多头概念，忽略
        kwargs.pop("rope_max_len", None)   # 位置编码由 tokenizer 承担，忽略
        return MambaEncoder(**kwargs)
    raise ValueError(f"未知 backbone: {name}，可选 'transformer' | 'mamba'")
