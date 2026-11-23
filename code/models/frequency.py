"""频域流 + 跨域融合（时-频双流中的“频域流”，对应赛题任务 1）

频域 token（全部来自信号前端 SignalFrontEnd，物理先验、无可学习参数的部分）:
  SPEC  ×spec_tokens   全局对数谱 |X|、|FFT(x²)|、|FFT(x⁴)| 按同一频点堆叠为一个 token
                       （空间级对齐：3 个非线性谱在频点维对齐；x²/x⁴ 谱线揭示 PSK/QAM 阶数）
  ACF   ×acf_tokens    包络自相关（PRI / 脉冲数 / 占空比的强线索）
  CONST ×const_tokens  载频校正后星座密度图的 2D patch（QAM/PSK 阶数）
每类 token 有独立的可学习位置编码与模态类型编码。

FrequencyEncoder：频域 token 上的小型自注意力编码器（freq_layers 层，默认 2），
                  频域流自己的“编码”，与时域主干并行。
CrossDomainFusion：时域 token（含 CLS/STAT）作 query、频域 token 作 key/value 的
                  跨注意力 + 门控残差（注意力级动态对齐融合）。融合发生在主干之前，
                  频域 token 不进入主干序列 -> 主干序列长度不变，显存不涨。

预训练防泄漏：MAE 时前端输入的是“被掩码采样点置零”的信号，频域 token 只看到可见
部分；否则整段 FFT（幅度+相位无损）会把被掩码内容泄漏给可见 token（旧版缺陷）。
"""

import torch
import torch.nn as nn

from .layers.attention import CrossAttention, MultiHeadAttention
from .layers.hybrid import AttentionLayer

FREQ_TYPES = {"spec": 0, "acf": 1, "const": 2}


class SpectralTokenizer(nn.Module):
    def __init__(self, d_model=512, seq_len=1024, spec_tokens=32, acf_lags=512, acf_tokens=16,
                 const_bins=32, const_patch=8, use_spec=True, use_acf=True, use_const=True):
        super().__init__()
        self.use = dict(spec=use_spec, acf=use_acf, const=use_const)
        if use_spec:
            if seq_len % spec_tokens:
                raise ValueError(f"seq_len={seq_len} 必须能被 spec_tokens={spec_tokens} 整除")
            self.spec_tokens = spec_tokens
            self.spec_proj = nn.Sequential(nn.Linear(3 * seq_len // spec_tokens, d_model),
                                           nn.LayerNorm(d_model))
            self.pos_spec = nn.Parameter(torch.zeros(1, spec_tokens, d_model))
        if use_acf:
            if acf_lags % acf_tokens:
                raise ValueError(f"acf_lags={acf_lags} 必须能被 acf_tokens={acf_tokens} 整除")
            self.acf_tokens = acf_tokens
            self.acf_proj = nn.Sequential(nn.Linear(acf_lags // acf_tokens, d_model),
                                          nn.LayerNorm(d_model))
            self.pos_acf = nn.Parameter(torch.zeros(1, acf_tokens, d_model))
        if use_const:
            if const_bins % const_patch:
                raise ValueError("const_bins 必须能被 const_patch 整除")
            self.cp = const_patch
            self.n_const = (const_bins // const_patch) ** 2
            self.const_proj = nn.Sequential(nn.Linear(const_patch * const_patch, d_model),
                                            nn.LayerNorm(d_model))
            self.pos_const = nn.Parameter(torch.zeros(1, self.n_const, d_model))
        self.type_embed = nn.Embedding(len(FREQ_TYPES), d_model)
        for name in ("pos_spec", "pos_acf", "pos_const"):
            if hasattr(self, name):
                nn.init.trunc_normal_(getattr(self, name), std=0.02)

    @property
    def enabled(self):
        return any(self.use.values())

    def _typed(self, x, kind):
        return x + self.type_embed.weight[FREQ_TYPES[kind]]

    def forward(self, fe):
        """fe: SignalFrontEnd 输出 dict -> [B, Lf, d]（全部视图关闭时返回 None）。"""
        parts = []
        B = fe["spec"].size(0)
        if self.use["spec"]:
            sp = fe["spec"].reshape(B, 3, self.spec_tokens, -1).permute(0, 2, 1, 3).reshape(
                B, self.spec_tokens, -1)
            parts.append(self._typed(self.spec_proj(sp) + self.pos_spec, "spec"))
        if self.use["acf"]:
            ac = fe["acf"].reshape(B, self.acf_tokens, -1)
            parts.append(self._typed(self.acf_proj(ac) + self.pos_acf, "acf"))
        if self.use["const"]:
            G, p = fe["const"].size(-1), self.cp
            c = fe["const"].reshape(B, G // p, p, G // p, p).permute(0, 1, 3, 2, 4).reshape(
                B, self.n_const, p * p)
            parts.append(self._typed(self.const_proj(c) + self.pos_const, "const"))
        if not parts:
            return None
        return torch.cat(parts, dim=1)


class FrequencyEncoder(nn.Module):
    """频域流编码器：n_layers 层位置无关自注意力（位置由可学习编码携带）+ 末端 LN。"""

    def __init__(self, d_model, nhead, n_layers=2, dim_feedforward=None, dropout=0.1):
        super().__init__()
        ff = dim_feedforward or 4 * d_model
        self.layers = nn.ModuleList([
            AttentionLayer(MultiHeadAttention(d_model, nhead, dropout, use_rope=False),
                           d_model, ff, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class FrequencyStream(nn.Module):
    """SpectralTokenizer -> FrequencyEncoder，输出频域记忆 token [B, Lf, d]。"""

    def __init__(self, d_model, nhead, seq_len, n_layers=2, dim_feedforward=None,
                 dropout=0.1, **tok_kwargs):
        super().__init__()
        self.tokenizer = SpectralTokenizer(d_model, seq_len, **tok_kwargs)
        self.encoder = FrequencyEncoder(d_model, nhead, n_layers, dim_feedforward, dropout) \
            if n_layers > 0 else None

    @property
    def enabled(self):
        return self.tokenizer.enabled

    def forward(self, fe):
        f = self.tokenizer(fe)
        if f is None or self.encoder is None:
            return f
        return self.encoder(f)


class CrossDomainFusion(nn.Module):
    """时域 query 对频域 key/value 做跨注意力 + 门控残差融合（注意力级融合）。

    out = LN(x + σ(W[x; a]) ⊙ a),  a = CrossAttn(LN_q(x), freq)
    """

    def __init__(self, d_model, nhead=8, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.attn = CrossAttention(d_model, nhead, dropout)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, freq):
        if freq is None:
            return x
        a = self.attn(self.norm_q(x), freq)
        g = self.gate(torch.cat((x, a.to(x.dtype)), dim=-1))
        return self.norm(x + g * a)
