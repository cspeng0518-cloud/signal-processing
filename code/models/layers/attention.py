"""可插拔注意力组件库（各 backbone 共用）

组件 forward 契约完全一致（调用方零分支）:
    forward(x, rope, pos_ids=None, key_padding_mask=None) -> [B, L, D]

  RoPEAttention    标准多头自注意力（q/k 按相对位置旋转）
  MLAAttention     DeepSeek-V2 式多头潜在注意力（KV 低秩压缩 + 解耦 RoPE）

RoPEEncoderLayer  注意力 + FFN 的 pre-norm 层（RoPETransformerEncoder 与
                  HybridEncoder 共用的层结构，见 backbone.py / hybrid.py）

实现参考: HuggingFace transformers
  transformers/models/deepseek_v2/modeling_deepseek_v2.py (DeepseekV2Attention)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm（MLA 潜在向量归一化用，对齐 DeepseekV2RMSNorm）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RoPEAttention(nn.Module):
    """带 RoPE 的多头自注意力（eager 实现，无 CUDA 专有算子）。

    q/k 经 rope.rotate() 按位置旋转后做标准缩放点积注意力。
    nn.TransformerEncoderLayer 的注意力不可注入旋转，故此组件必须自实现。
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


class MLAAttention(nn.Module):
    """DeepSeek-V2 式多头潜在注意力 MLA（eager 实现，无 CUDA 专有算子）。

    参考 transformers DeepseekV2Attention:
      - Q/KV 均先下投影到低秩潜在向量（q_lora_rank / kv_lora_rank），再上投影
        还原；推理时 KV cache 只需缓存压缩潜在向量 + 共享 rope key，显著降显存
      - 位置信息走"解耦 RoPE"通道：k_rope 各头共享（MQA 式）、q_rope 每头独立，
        且不参与 KV 压缩，保证相对位置建模不受低秩截断影响
      - 潜在向量经 RMSNorm 后再上投影（q_a_layernorm / kv_a_layernorm）
      - 注意力缩放取 (qk_nope + qk_rope) 总维度的 -0.5 次方

    与参考实现的差异（适配本仓库约定）:
      - qk_rope_head_dim 统一取 d_model // nhead，与共享 RoPE 旋转表
        （tokenizer.RoPEPositionalEncoding）对齐；DeepSeek-V2 中 rope 维与
        nope 维可不同
      - 仅实现训练/全序列前向，不做 KV cache（预训练-微调场景无自回归解码）
    """

    def __init__(self, d_model, nhead, kv_lora_rank=32, q_lora_rank=32, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.dh = d_model // nhead           # 每头维数（nope / v / rope 统一）
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_head_dim = 2 * self.dh       # qk_nope_head_dim + qk_rope_head_dim
        # ---- 下投影（压缩）----
        self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=False)
        self.kv_a_proj = nn.Linear(d_model, kv_lora_rank + self.dh, bias=False)
        # ---- 潜在向量归一化 ----
        self.q_a_layernorm = RMSNorm(q_lora_rank)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        # ---- 上投影（还原）----
        self.q_b_proj = nn.Linear(q_lora_rank, nhead * self.qk_head_dim, bias=False)
        self.kv_b_proj = nn.Linear(kv_lora_rank, nhead * 2 * self.dh, bias=False)
        self.o_proj = nn.Linear(nhead * self.dh, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.scaling = self.qk_head_dim ** -0.5

    def forward(self, x, rope, pos_ids=None, key_padding_mask=None):
        B, L, _ = x.shape
        H, dh = self.nhead, self.dh
        # Q: 下投影 -> RMSNorm -> 上投影，切出 nope / rope 两部分
        c_q = self.q_a_layernorm(self.q_a_proj(x))                    # [B,L,rq]
        q = self.q_b_proj(c_q).view(B, L, H, 2 * dh)
        q_nope, q_pe = q.split([dh, dh], dim=-1)
        # KV: 下投影一次切出压缩潜在向量 + 各头共享的 rope key
        c_kv = self.kv_a_proj(x)                                      # [B,L,rkv+dh]
        kv_nope, k_pe = c_kv.split([self.kv_lora_rank, dh], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        # 解耦 RoPE：只旋转 rope 通道（k_rope 广播到各头），nope 通道不携带位置
        q_pe = rope.rotate(q_pe, pos_ids)                             # [B,L,H,dh]
        k_pe = rope.rotate(k_pe.unsqueeze(2).expand(B, L, H, dh), pos_ids)
        # 上投影还原 k_nope / v
        kv = self.kv_b_proj(kv_nope).view(B, L, H, 2 * dh)
        k_nope, v = kv.split([dh, dh], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)         # [B,H,L,2dh]
        k = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)         # [B,H,L,2dh]
        v = v.transpose(1, 2)                                         # [B,H,L,dh]

        attn = (q @ k.transpose(-2, -1)) * self.scaling
        if key_padding_mask is not None:                              # True = 忽略
            attn = attn.masked_fill(key_padding_mask[:, None, None, :],
                                    float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, H * dh)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU FFN（Llama / GLM 风格）：down(silu(gate(x)) * up(x))，投影无 bias。

    相比标准两层 MLP（2*D*F 参数）多一路门控投影（3*D*F，+50%），
    门控乘法带来更强表达能力，是主流大模型的标准 FFN 形态。
    """

    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class RoPEEncoderLayer(nn.Module):
    """pre-norm Transformer encoder layer（Llama/GLM 风格）。

    残差主干为恒等通路，LayerNorm 移到子层输入侧——深栈梯度直通底层，
    训练稳定性显著优于 post-norm；栈末的 final norm 由 BaseEncoder.norm 提供。
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = RoPEAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, rope, pos_ids=None, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), rope, pos_ids, key_padding_mask)
        return x + self.ff(self.norm2(x))
