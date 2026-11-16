"""FFT 幅相谱 token 与时域 token 跨域融合，无 CUDA 专有依赖。"""
import math

import torch
from torch import nn


class FrequencyTokenizer(nn.Module):
    def __init__(self, d_model, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(2 * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, iq):
        # [B,N,2] -> complex FFT, 规范化谱幅与相位，保持数值稳定性
        z = torch.complex(iq[..., 0].float(), iq[..., 1].float())
        spectrum = torch.fft.fft(z, dim=1, norm="ortho")
        spec = torch.stack((torch.log1p(torch.abs(spectrum)), torch.angle(spectrum) / 3.141592653589793), dim=-1)
        length = (spec.size(1) // self.patch_size) * self.patch_size
        spec = spec[:, :length].reshape(spec.size(0), -1, 2 * self.patch_size)
        return self.norm(self.proj(spec))


class CrossAttention(nn.Module):
    """跨域多头注意力（query 与 key/value 来自不同模态，eager 实现）。

    替代 nn.MultiheadAttention 黑盒接口，与 layers/attention.py 的
    组件同风格：手写 q/k/v 投影 + 缩放点积，无掩码（频谱 token 全局可见）。
    """

    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, Lq, D = query.shape
        Lk = key.size(1)
        q = self.q_proj(query).view(B, Lq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.nhead, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, Lq, D)
        return self.out_proj(out)


class CrossDomainFusion(nn.Module):
    """用时域 query 对频谱 key/value 做跨注意力并门控残差融合。"""
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = CrossAttention(d_model, nhead)
        self.gate = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, temporal, spectral):
        attended = self.attn(temporal, spectral, spectral)
        gate = self.gate(torch.cat((temporal, attended), dim=-1))
        return self.norm(temporal + gate * attended)
