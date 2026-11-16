"""FFT 幅相谱 token 与时域 token 跨域融合，无 CUDA 专有依赖。"""
import torch
from torch import nn
import torch.nn.functional as F


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


class CrossDomainFusion(nn.Module):
    """用时域 query 对频谱 key/value 做跨注意力并门控残差融合。"""
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, temporal, spectral):
        attended, _ = self.attn(temporal, spectral, spectral, need_weights=False)
        gate = self.gate(torch.cat((temporal, attended), dim=-1))
        return self.norm(temporal + gate * attended)
