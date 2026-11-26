"""时域流 Tokenizer（时-频双流中的“时域流”，对应赛题任务 1）

输入为信号前端（models/dsp.py, SignalFrontEnd）的输出 dict，输出 token 序列:

    [CLS] [STAT] [TIME_1 .. TIME_P]
      |      |          |
    全局读出  16 维统计量  多通道卷积 stem（I,Q,校正后 I/Q,包络,瞬时频率）
    (+任务提示)          ⊕ 门控融合的同帧 STFT 谱（通道级时-频对齐融合）

  * 位置信息：不叠加绝对位置编码，由主干 RoPE 按 token 下标旋转 q/k
    （前缀占位置 0、1，时域 token 从 2 开始）；MAE 只编码可见 token 时传原始下标。
  * 通道级融合：时域 patch token 与同一时刻的 STFT 帧（帧移 = patch，一一对齐）
    做逐 token、逐通道门控 t' = LN(t + σ(W[t; s]) ⊙ s)。门控初始偏向时域。
  * 任务提示（task prompt）：CLS 叠加任务嵌入（任务私有参数，计入 head 参数量）。
"""

import torch
import torch.nn as nn

from .dsp import N_STATS

N_PREFIX = 2                                   # [CLS] + [STAT]
TASK_IDS = {"protocol": 0, "modulation": 1, "parameter": 2}


class ConvStem(nn.Module):
    """早期卷积 stem（Xiao et al. 2021, "Early convolutions help transformers see better"）。

    [B,C,N] -> [B,N/patch,d]，总步长 = patch；比单层线性 patch embedding 保留更多
    patch 内的细粒度结构（符号跳变、脉冲前后沿——time_delay 只有几个采样点量级）。
    """

    STRIDES = {4: (2, 2), 8: (2, 4), 16: (4, 4)}

    def __init__(self, c_in, d, patch=8, width=64):
        super().__init__()
        if patch not in self.STRIDES:
            raise ValueError(f"patch_size 仅支持 {sorted(self.STRIDES)}，当前 {patch}")
        s1, s2 = self.STRIDES[patch]
        self.net = nn.Sequential(
            nn.Conv1d(c_in, width, 7, padding=3), nn.GELU(),
            nn.Conv1d(width, 2 * width, 2 * s1 + 1, stride=s1, padding=s1), nn.GELU(),
            nn.Conv1d(2 * width, d, s2, stride=s2),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.net(x).transpose(1, 2))


class GatedChannelFusion(nn.Module):
    """通道级动态对齐融合：t' = LN(t + σ(W[t; s]) ⊙ s)。"""

    def __init__(self, d):
        super().__init__()
        self.gate = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)
        self.reset_gate()

    def reset_gate(self):
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)     # 初始偏向时域，训练中逐步打开 STFT 支路

    def forward(self, t, s):
        g = torch.sigmoid(self.gate(torch.cat([t, s], dim=-1)))
        return self.norm(t + g * s)


class TimeTokenizer(nn.Module):
    def __init__(self, d_model=512, patch_size=8, stft_win=32, stem_width=64,
                 n_time_channels=6, use_stft=True, use_stats=True, task_prompt=True):
        super().__init__()
        self.patch = patch_size
        self.use_stft = use_stft
        self.use_stats = use_stats
        self.task_prompt = task_prompt
        self.stem = ConvStem(n_time_channels, d_model, patch_size, stem_width)
        if use_stft:
            self.stft_proj = nn.Sequential(nn.Linear(stft_win, d_model), nn.LayerNorm(d_model))
            self.fuse = GatedChannelFusion(d_model)
        if use_stats:
            self.stat_proj = nn.Sequential(nn.Linear(N_STATS, d_model), nn.GELU(),
                                           nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        else:
            self.stat_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.stat_token, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if task_prompt:
            self.task_embed = nn.Embedding(len(TASK_IDS), d_model)
            nn.init.normal_(self.task_embed.weight, std=0.02)

    def private_parameters(self):
        """任务私有参数（task prompt），参数核算时计入 head。"""
        return list(self.task_embed.parameters()) if self.task_prompt else []

    def forward(self, fe, task=None):
        """
        Args:
            fe:   SignalFrontEnd 输出 dict
            task: 任务名（str）或 None（预训练）
        Returns:
            prefix [B,2,d]（CLS, STAT），time [B,P,d]（P = seq_len/patch，未裁剪）
        """
        t = self.stem(fe["time_ch"].to(self.cls_token.dtype))               # [B,P,d]
        if self.use_stft:
            t = self.fuse(t, self.stft_proj(fe["stft"]))
        B = t.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        if self.task_prompt and task is not None:
            cls = cls + self.task_embed.weight[TASK_IDS[task]]
        if self.use_stats:
            stat = self.stat_proj(fe["stats"]).unsqueeze(1)
        else:
            stat = self.stat_token.expand(B, -1, -1)
        return torch.cat([cls, stat.to(cls.dtype)], dim=1), t
