"""编码器：HybridEncoder（预训练主干与微调 neck 共用同一个类）

按 layer_types 逐层堆叠异构层（对齐 GLM / Qwen3-Next config 的逐层列表风格）：
  * 列表长度 == n_layers 时逐层精确对应；短于 n_layers 时按模式串循环铺满；
  * 主干（backbone）默认 ["full_attention", "mla", "mla"]，RoPE 位置编码，
    支持 MAE 只编码可见 token（pos_ids 传 token 原始下标保持真实位置）；
  * neck 默认 ["conformer"]，微调/推理时叠加在主干之上（不使用 RoPE）。

旧版的 SinTransformerEncoder / RoPETransformerEncoder / MambaEncoder 已并入本类：
  纯 Transformer = layer_types ["full_attention"]；纯 Mamba = ["mamba"]。
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .layers.attention import RoPEPositionalEncoding
from .layers.hybrid import build_sequence_layer, normalize_layer_type


class HybridEncoder(nn.Module):
    """Args:
        d_model / n_layers / nhead / dim_feedforward / dropout: 常规超参
        layer_types:     逐层类型列表（短列表循环铺满）
        drop_path:       随机深度最大比例（按层线性递增）
        use_rope:        是否持有 RoPE 表（主干 True；neck False）
        rope_max_len:    RoPE 表长度（>= 最大 token 数）
        grad_checkpoint: 训练时逐层激活重计算（显存换算力，大模型/大 batch 打开）
    """

    def __init__(self, d_model=512, n_layers=12, nhead=8, layer_types=("full_attention",),
                 dim_feedforward=None, dropout=0.1, drop_path=0.0, use_rope=True,
                 rope_max_len=512, mla_kv_lora_rank=128, mla_q_lora_rank=128,
                 conv_kernel=15, grad_checkpoint=False):
        super().__init__()
        types = [normalize_layer_type(t) for t in (layer_types or [])]
        if not types:
            raise ValueError("layer_types 不能为空，如 ['full_attention','mla','mla']")
        if len(types) > n_layers:
            raise ValueError(f"layer_types 长度 {len(types)} 超过 n_layers={n_layers}")
        self.layer_types = [types[i % len(types)] for i in range(n_layers)]
        self.rope = RoPEPositionalEncoding(d_model // nhead, max_len=rope_max_len) \
            if use_rope else None
        dprs = [drop_path * i / max(1, n_layers - 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            build_sequence_layer(t, d_model, nhead, dim_feedforward=dim_feedforward,
                                 dropout=dropout, drop_path=dp,
                                 mla_kv_lora_rank=mla_kv_lora_rank,
                                 mla_q_lora_rank=mla_q_lora_rank, conv_kernel=conv_kernel)
            for t, dp in zip(self.layer_types, dprs)])
        self.norm = nn.LayerNorm(d_model)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x, key_padding_mask=None, pos_ids=None, time_slice=None):
        """x: [B,L,D]；key_padding_mask: [B,L] True=忽略；pos_ids: [B,L] token 原始下标。"""
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            if use_ckpt:
                x = checkpoint(layer, x, self.rope, pos_ids, key_padding_mask, time_slice,
                               use_reentrant=False)
            else:
                x = layer(x, self.rope, pos_ids, key_padding_mask, time_slice)
        return self.norm(x)
