"""可插拔注意力组件库（backbone / neck / 频域流 / MAE 解码器共用）

组件 forward 契约一致（调用方零分支）:
    forward(x, rope=None, pos_ids=None, key_padding_mask=None) -> [B, L, D]

  RoPEPositionalEncoding  RoPE 旋转表（编码器持有一份，各层共享）
  MultiHeadAttention      标准多头自注意力（full attention, FA）；use_rope 控制是否旋转 q/k
  MLAAttention            DeepSeek-V2 式多头潜在注意力（KV 低秩压缩 + 解耦 RoPE）
  CrossAttention          跨域注意力（query 与 key/value 来自不同模态）
  SwiGLU / RMSNorm / DropPath

所有注意力核心统一走 models/ops/fused.attention：
  CUDA -> F.scaled_dot_product_attention（FlashAttention-2 / mem-efficient 内核，
          不物化 [B,H,L,L] 注意力矩阵）
  NPU  -> torch_npu.npu_fusion_attention（首次调用数值自检，不通过自动回退 eager）
  其他 -> eager 参考实现
这是本版相对旧版（eager q@k^T + softmax）最主要的显存优化之一。

实现参考: HuggingFace transformers
  transformers/models/deepseek_v2/modeling_deepseek_v2.py (DeepseekV2Attention)
"""

import torch
import torch.nn as nn

from ..ops.fused import attention, swiglu, rmsnorm, rope_rotate, mla_preprocess


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm（MLA 潜在向量归一化用，对齐 DeepseekV2RMSNorm）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)


class DropPath(nn.Module):
    """随机深度（Stochastic Depth），深层网络正则化（逐样本丢弃残差分支）。"""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = torch.rand(x.size(0), 1, 1, device=x.device, dtype=x.dtype) >= self.p
        return x * keep / (1 - self.p)


class RoPEPositionalEncoding(nn.Module):
    """RoPE 旋转位置编码表。

    不在 token 上叠加位置向量，而是持有 cos/sin 表，由注意力对 q/k 按位置旋转。
    MAE 只编码可见 token 时，传入 pos_ids（token 原始下标）即可保持真实位置。

    Args:
        head_dim: 每头维数（须为偶数）
        max_len:  位置表长度上限（token 序列最大长度，含前缀 token）
    """

    def __init__(self, head_dim, max_len=512, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)                        # [max_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                 # [max_len, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def rotate(self, x, pos_ids=None):
        """x: [B, L, H, Dh]（H 可为 1 再广播）；pos_ids: [B, L] 或 None（0..L-1）。"""
        if pos_ids is None:
            cos = self.cos[: x.size(1)][None, :, None, :]       # [1, L, 1, Dh]
            sin = self.sin[: x.size(1)][None, :, None, :]
        else:
            cos = self.cos[pos_ids].unsqueeze(2)                # [B, L, 1, Dh]
            sin = self.sin[pos_ids].unsqueeze(2)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)             # AMP 下与 q/k 同精度
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class MultiHeadAttention(nn.Module):
    """标准多头自注意力（full attention）。

    use_rope=True 且调用方传入 rope 表时对 q/k 施加旋转；否则为位置无关注意力
    （位置信息已由上游编码器写入 token，如 Conformer neck / 频域流）。
    """

    def __init__(self, d_model, nhead, dropout=0.0, use_rope=True):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 不能被 nhead={nhead} 整除")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x, rope=None, pos_ids=None, key_padding_mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.nhead, self.head_dim).unbind(2)  # [B,L,H,Dh]
        if self.use_rope and rope is not None:
            q = rope_rotate(q, rope.cos, rope.sin, pos_ids)
            k = rope_rotate(k, rope.cos, rope.sin, pos_ids)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                         # [B,H,L,Dh]
        o = attention(q, k, v, key_padding_mask, self.dropout, self.training)
        return self.out_drop(self.proj(o.transpose(1, 2).reshape(B, L, D)))


class MLAAttention(nn.Module):
    """DeepSeek-V2 式多头潜在注意力 MLA。

    参考 transformers DeepseekV2Attention:
      - Q/KV 均先下投影到低秩潜在向量（q_lora_rank / kv_lora_rank），RMSNorm 后上投影还原
      - 位置信息走“解耦 RoPE”通道：k_rope 各头共享（MQA 式）、q_rope 每头独立，
        且不参与 KV 压缩
      - 注意力缩放取 (qk_nope + qk_rope) 总维度的 -0.5 次方（= SDPA 默认缩放）

    与参考实现的差异（适配本仓库）:
      - qk_rope_head_dim = qk_nope_head_dim = v_head_dim = d_model // nhead
      - 编码器场景无自回归 KV cache；其价值在于低秩投影的参数/激活压缩
    """

    def __init__(self, d_model, nhead, kv_lora_rank=128, q_lora_rank=128, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.dh = d_model // nhead
        self.kv_lora_rank = kv_lora_rank
        self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=False)
        self.kv_a_proj = nn.Linear(d_model, kv_lora_rank + self.dh, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, nhead * 2 * self.dh, bias=False)
        self.kv_b_proj = nn.Linear(kv_lora_rank, nhead * 2 * self.dh, bias=False)
        self.o_proj = nn.Linear(nhead * self.dh, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, rope=None, pos_ids=None, key_padding_mask=None):
        B, L, _ = x.shape
        H, dh = self.nhead, self.dh
        # Q/KV 预处理走融合入口（内部 RMSNorm 自动走 Triton / torch_npu）
        q, kv, k_pe = mla_preprocess(
            x, self.q_a_proj, self.q_a_layernorm, self.q_b_proj,
            self.kv_a_proj, self.kv_a_layernorm, self.kv_b_proj,
            H, dh, self.kv_lora_rank)
        q_nope, q_pe = q.split([dh, dh], dim=-1)
        k_nope, v = kv.split([dh, dh], dim=-1)
        k_pe = k_pe.unsqueeze(2)                                       # [B,L,1,dh]
        if rope is not None:                                           # 只旋转 rope 通道
            q_pe = rope_rotate(q_pe, rope.cos, rope.sin, pos_ids)
            k_pe = rope_rotate(k_pe, rope.cos, rope.sin, pos_ids)
        k_pe = k_pe.expand(B, L, H, dh)
        q = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)          # [B,H,L,2dh]
        k = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)          # [B,H,L,2dh]
        v = v.transpose(1, 2)                                          # [B,H,L,dh]
        o = attention(q, k, v, key_padding_mask, self.dropout, self.training)
        return self.o_proj(o.transpose(1, 2).reshape(B, L, H * dh))


class CrossAttention(nn.Module):
    """跨域多头注意力：query 来自一个模态，key/value 来自另一模态。"""

    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 不能被 nhead={nhead} 整除")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, query, memory, memory_padding_mask=None):
        B, Lq, D = query.shape
        Lk = memory.size(1)
        q = self.q_proj(query).view(B, Lq, self.nhead, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(memory).view(B, Lk, 2, self.nhead, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        o = attention(q, k, v, memory_padding_mask, self.dropout, self.training)
        return self.out_proj(o.transpose(1, 2).reshape(B, Lq, D))


class SwiGLU(nn.Module):
    """SwiGLU FFN（Llama / GLM 风格）：down(silu(gate(x)) * up(x))，投影无 bias。

    silu(gate)*up 走 ops.fused.swiglu（CUDA Triton / 昇腾 npu_swiglu 融合，自检回退）。
    """

    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(swiglu(self.gate(x), self.up(x))))
