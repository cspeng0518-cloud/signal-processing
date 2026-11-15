"""IQ Tokenizer（参考 EMind V-A 节）

将连续 IQ 序列离散化为 patch token 序列：
  1. [N,2] IQ -> 按 patch_size 切块（每块 patch_size 个复数点）
  2. 线性嵌入到 d_model 维 token
  3. 序列前插入两个特殊 token（仅靠位置编码区分功能）：
     - [CLS] 分类 token：输出全局表征，供任务头使用
     - [SR]  采样率 token：携带采样率元信息，预训练时永不掩码
  4. 附带幅度归一化 IQ_hat = IQ / max(|IQ|)（保物理能量分布）
"""

import math
import torch
import torch.nn as nn


class IQTokenizer(nn.Module):
    def __init__(self, d_model=128, patch_size=8, sr_bins=64):
        """
        Args:
            d_model: token 维度
            patch_size: 每个patch包含的复数采样点数（EMind: 8）
            sr_bins: 采样率分桶数（采样率跨kHz~MHz，分桶嵌入）
        """
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        # 每个 patch: patch_size 个复数点 * 2 通道(I/Q) = 2*patch_size 维实数
        self.patch_embed = nn.Linear(2 * patch_size, d_model)
        # 特殊 token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sr_embed = nn.Linear(1, d_model)  # 采样率标量 -> token
        self.sr_bins = sr_bins
        # 位置编码（正弦，非学习式，兼容变长）
        self.pos_encoding = SinPositionalEncoding(d_model, max_len=4096 // patch_size + 2)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def normalize(iq):
        """幅度归一化 IQ_hat = IQ / max(|IQ|)，返回值域约 [-1,1] 的 [B,N,2]"""
        amp = torch.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2 + 1e-8)  # [B,N]
        max_amp = amp.max(dim=-1, keepdim=True).values.unsqueeze(-1)  # [B,1,1]
        return iq / max_amp

    def to_patches(self, iq):
        """[B,N,2] -> [B, n_patch, 2*patch_size]（N 非 patch 整数倍时右侧截断）"""
        B, N, _ = iq.shape
        n_patch = N // self.patch_size
        iq = iq[:, : n_patch * self.patch_size, :]
        # [B, n_patch, patch_size, 2] -> [B, n_patch, 2*patch_size]
        return iq.reshape(B, n_patch, self.patch_size, 2).reshape(B, n_patch, -1)

    def forward(self, iq, sr):
        """
        Args:
            iq: [B, N, 2] 原始 IQ
            sr: [B] 采样率（Hz）
        Returns:
            tokens: [B, 2+n_patch, d_model]（[CLS]+[SR]+patch tokens）
            n_patch: patch 数
        """
        iq = self.normalize(iq)
        patches = self.to_patches(iq)                    # [B,P,D]
        patch_tokens = self.patch_embed(patches)         # [B,P,D]

        B = iq.size(0)
        cls = self.cls_token.expand(B, -1, -1)           # [B,1,D]
        # 采样率嵌入（log 压缩 kHz~MHz 跨度）
        sr_tok = self.sr_embed(torch.log10(sr).unsqueeze(-1).unsqueeze(-1))  # [B,1,1]->[B,1,D]

        tokens = torch.cat([cls, sr_tok, patch_tokens], dim=1)  # [B,P+2,D]
        tokens = self.pos_encoding(tokens)
        return tokens, patch_tokens.size(1)


class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[: (d_model - 1) // 2 + 1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # [B,L,D]
        return x + self.pe[:, : x.size(1)]
