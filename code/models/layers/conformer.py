"""Conformer 层（用于微调阶段叠加在预训练 FA/MLA 主干之上的 neck）

结构（Gulati et al., Interspeech 2020, Macaron 式）:
    x = x + 1/2 * FFN(LN(x))
    x = x + MHSA(LN(x))
    x = x + ConvModule(x)          # 只作用于时域 token 段
    x = x + 1/2 * FFN(LN(x))
    x = LN(x)

为什么只放在 neck、不放进预训练主干（实测结论，见 docs/REFACTOR_NOTES.md）:
  * 相同 ~100M 参数下全 Conformer 主干显存约为 FA/MLA 混合主干的 3 倍
    （每层两个 FFN + 卷积模块 + 多个 LayerNorm 的 fp32 激活），batch 与参数量都打不上去；
  * 卷积需要连续时间序列，无法使用 MAE “只编码可见 token” 的 gather 加速路径，
    预训练只能退化成全序列 SimMIM，显存再翻数倍。
  * 放在 neck：预训练仍走省显存的 FA/MLA + MAE gather；微调时用 1~4 层 Conformer
    补充局部波形建模（符号跳变、脉冲沿），代价只有 neck 层数份的全序列激活。

ConvModule 只在时域 token 段（time_slice）上做深度卷积：频谱/ACF/星座 token 不是
时间序列，跨模态卷积无物理意义；padding 位置在卷积前清零，512/1024 点样本数值一致。
"""

import torch.nn as nn
import torch.nn.functional as F

from .attention import DropPath, MultiHeadAttention, SwiGLU


class ConvModule(nn.Module):
    """Conformer 卷积模块：LN -> PW(2d)+GLU -> DW Conv(k) -> LN -> SiLU -> PW。

    用 LayerNorm 代替原论文 BatchNorm：对 padding / 小 batch 推理 / NPU 更稳。
    """

    def __init__(self, d, kernel=15, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.ln2 = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, time_slice=None, key_padding_mask=None):
        s, e = time_slice if time_slice is not None else (0, x.size(1))
        t = x[:, s:e]
        h = F.glu(self.pw1(self.ln(t)), dim=-1)
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask[:, s:e, None], 0.0)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.pw2(F.silu(self.ln2(h))))
        return F.pad(h, (0, 0, s, x.size(1) - e))        # 非时域段补零（无 in-place）


class ConformerLayer(nn.Module):
    """Macaron Conformer block，forward 契约与 hybrid.BaseSequenceLayer 一致。

    注意力不使用 RoPE：neck 输入已是主干输出（位置信息已编码），且序列尾部
    拼接的频域 token 没有时间位置；相对位置由深度卷积补充。
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, drop_path=0.0,
                 conv_kernel=15):
        super().__init__()
        self.nf1, self.ff1 = nn.LayerNorm(d_model), SwiGLU(d_model, dim_feedforward, dropout)
        self.na = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, dropout, use_rope=False)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.nf2, self.ff2 = nn.LayerNorm(d_model), SwiGLU(d_model, dim_feedforward, dropout)
        self.out = nn.LayerNorm(d_model)
        self.dp = DropPath(drop_path)

    def forward(self, x, rope=None, pos_ids=None, key_padding_mask=None, time_slice=None):
        x = x + 0.5 * self.dp(self.ff1(self.nf1(x)))
        x = x + self.dp(self.attn(self.na(x), None, None, key_padding_mask))
        x = x + self.dp(self.conv(x, time_slice, key_padding_mask))
        x = x + 0.5 * self.dp(self.ff2(self.nf2(x)))
        return self.out(x)
