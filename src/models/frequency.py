"""频域流 + 跨域融合（时-频双流中的“频域流”，对应赛题任务 1）

频域 token（全部来自信号前端 SignalFrontEnd 的确定性物理视图，见 models/dsp.py）:
  SPEC    ×spec_tokens     |X|、|FFT(x²)|、|FFT(x⁴)|、共轭循环谱 |FFT(x[n]x[n-d])|，
                           同一频点的各通道拼成一个 token（频点维空间对齐）
  CAF     ×caf_tokens      非共轭循环自相关相干度 c_d(α)，α∈[0,0.5)，各滞后按 α 对齐拼接
                           （码元速率谱线、OQPSK/MSK/FSK 的循环平稳结构、雷达 1/PRI）
  DECHIRP ×dechirp_tokens  去斜谱：19 个候选调频率 × 频率（LFM/CSS 聚焦，Frank 部分聚焦）
  ACF     ×acf_tokens      包络 ACF + 复自相关 |R|、Re R、Im R（PRI / 脉冲数 / 相位编码）
  CONST   ×(G/p)²          三通道星座图（全部采样点 / 稳定点 / 符号同步点）的 2D patch
  HIST    ×6               分布直方图（平滑瞬时频率 / 延迟相位差 / 符号相位差 /
                           采样幅度 / 符号幅度环 / 符号相位），每个直方图一个 token
每类 token 有独立的可学习位置编码与模态类型编码。各视图的通道数/轴长由
SignalFrontEnd.view_shapes() 给出，tokenizer 不硬编码前端细节。

FrequencyEncoder：频域 token 上的小型自注意力编码器（freq_layers 层，默认 2）。
CrossDomainFusion：时域 token（含 CLS/STAT）作 query、频域 token 作 key/value 的
                  跨注意力 + 门控残差（注意力级动态对齐融合）。融合发生在主干之前，
                  频域 token 不进入主干序列 -> 主干序列长度不变。

预训练防泄漏：MAE 时前端输入的是“被掩码采样点置零”的信号，频域 token 只看到可见部分。
"""

import torch
import torch.nn as nn

from .layers.attention import CrossAttention, MultiHeadAttention
from .layers.hybrid import AttentionLayer

FREQ_TYPES = {"spec": 0, "acf": 1, "const": 2, "caf": 3, "dechirp": 4, "hist": 5}
MAP_VIEWS = ("spec", "caf", "dechirp", "acf")          # [B,C,M] -> 沿 M 切 token


class SpectralTokenizer(nn.Module):
    """把前端各视图切成 token 并投影到 d_model。

    Args:
        view_shapes: SignalFrontEnd.view_shapes()
        tokens:      {"spec": n, "caf": n, "dechirp": n, "acf": n}，每个 1D 视图的 token 数
        const_patch: 星座图 2D patch 边长
        use:         {视图名: bool} 开关（关闭的视图不建参数、不出 token）
    """

    def __init__(self, d_model, view_shapes, tokens, const_patch=16, use=None):
        super().__init__()
        use = dict(use or {})
        self.use = {k: bool(use.get(k, True)) and view_shapes.get(k) is not None
                    for k in FREQ_TYPES}
        self.n_tok = {}
        self.proj = nn.ModuleDict()
        self.pos = nn.ParameterDict()
        for k in MAP_VIEWS:
            if not self.use[k]:
                continue
            C, M = view_shapes[k]
            T = int(tokens[k])
            if M % T:
                raise ValueError(f"{k}: 轴长 {M} 必须能被 token 数 {T} 整除")
            self.n_tok[k] = T
            self.proj[k] = nn.Sequential(nn.Linear(C * (M // T), d_model), nn.LayerNorm(d_model))
            self.pos[k] = nn.Parameter(torch.zeros(1, T, d_model))
        if self.use["const"]:
            C, G = view_shapes["const"]
            if G % const_patch:
                raise ValueError("const_bins 必须能被 const_patch 整除")
            self.cp = const_patch
            self.n_tok["const"] = (G // const_patch) ** 2
            self.proj["const"] = nn.Sequential(nn.Linear(C * const_patch * const_patch, d_model),
                                               nn.LayerNorm(d_model))
            self.pos["const"] = nn.Parameter(torch.zeros(1, self.n_tok["const"], d_model))
        if self.use["hist"]:
            H, bins = view_shapes["hist"]
            self.n_tok["hist"] = H
            self.proj["hist"] = nn.Sequential(nn.Linear(bins, d_model), nn.LayerNorm(d_model))
            self.pos["hist"] = nn.Parameter(torch.zeros(1, H, d_model))
        self.type_embed = nn.Embedding(len(FREQ_TYPES), d_model)
        for p in self.pos.values():
            nn.init.trunc_normal_(p, std=0.02)

    @property
    def enabled(self):
        return len(self.n_tok) > 0

    @property
    def n_tokens(self):
        return sum(self.n_tok.values())

    def _typed(self, x, kind):
        return x + self.type_embed.weight[FREQ_TYPES[kind]]

    def forward(self, fe):
        """fe: SignalFrontEnd 输出 dict -> [B, Lf, d]（全部视图关闭时返回 None）。"""
        parts = []
        dt = self.type_embed.weight.dtype
        for k in ("spec", "caf", "dechirp", "acf", "const", "hist"):
            if k not in self.n_tok:
                continue
            v = fe[k]
            B = v.size(0)
            if k in MAP_VIEWS:                                   # [B,C,M] -> [B,T,C*M/T]
                C, M = v.shape[1], v.shape[2]
                T = self.n_tok[k]
                tok = v.reshape(B, C, T, M // T).permute(0, 2, 1, 3).reshape(B, T, -1)
            elif k == "const":                                   # [B,C,G,G] -> [B,(G/p)²,C*p*p]
                C, G, p = v.shape[1], v.shape[2], self.cp
                tok = v.reshape(B, C, G // p, p, G // p, p).permute(0, 2, 4, 1, 3, 5).reshape(
                    B, (G // p) ** 2, C * p * p)
            else:                                                # hist [B,H,bins]
                tok = v
            parts.append(self._typed(self.proj[k](tok.to(dt)) + self.pos[k], k))
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

    def __init__(self, d_model, nhead, view_shapes, tokens, const_patch=16, use=None,
                 n_layers=2, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.tokenizer = SpectralTokenizer(d_model, view_shapes, tokens, const_patch, use)
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
