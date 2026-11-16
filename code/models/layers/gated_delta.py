"""Gated DeltaNet 门控增量规则线性注意力（eager 递归实现，无 CUDA 专有算子）

实现参考:
  - HuggingFace transformers Qwen3NextGatedDeltaNet
    (transformers/models/qwen3_next/modeling_qwen3_next.py)
  - flash-linear-attention 库 gated_deltanet（论文 "Gated Delta Networks"）

核心递推（每时间步，状态 S: [B, H, dk, dv]）:
    S_t   = exp(g_t) * S_{t-1} + k_t ⊗ [ beta_t * (v_t - k_t^T (exp(g_t) S_{t-1})) ]
    o_t   = S_t^T q_t
  即先用输入相关的衰减门 g_t 衰减旧状态，再以 delta rule 写入
  "新值 - 旧状态预测值" 的残差修正，beta_t 控制写入强度——
  线性复杂度的"可覆盖记忆"注意力。

与参考实现的差异（适配本仓库约定）:
  - 逐时间步 for 循环递归（对应参考 torch_recurrent_gated_delta_rule），
    训练序列仅百余 token，正确性与可读性优先；生产可换 fla chunk 内核加速
  - 位置无关（q/k 不加 RoPE），位置信息由门控衰减隐式建模（原论文设定），
    rope / pos_ids / key_padding_mask 参数仅为对齐层契约而保留、一律忽略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    """L2 归一化（对齐 fla / transformers 的 l2norm 实现）。"""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class RMSNormGated(nn.Module):
    """RMSNorm 后接 silu(z) 门控（对齐 Qwen3NextRMSNormGated）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dtype)
        return (x * F.silu(gate.float())).to(dtype)


class GatedDeltaAttention(nn.Module):
    """Gated DeltaNet 线性注意力层（eager 递归）。

    结构对齐 Qwen3NextGatedDeltaNet:
      in_proj_qkvz / in_proj_ba 分离投影
      -> depthwise 因果短卷积（仅 q/k/v，浅层局部上下文）
      -> q/k l2norm + 递归状态更新（fp32）
      -> RMSNormGated(z 门控) -> out_proj
    门控参数对齐参考初始化: dt_bias=1，A ~ U(0.01,16) 取 log 得 A_log，
    衰减（log 空间）g = -exp(A_log) * softplus(a + dt_bias) <= 0。
    """

    def __init__(self, d_model, nhead, conv_kernel=4):
        super().__init__()
        self.nhead = nhead
        self.dk = d_model // nhead
        self.dv = d_model // nhead
        self.key_dim = nhead * self.dk
        self.value_dim = nhead * self.dv
        # q/k/v/z 与 b/a 分离投影（对齐参考 in_proj_qkvz / in_proj_ba）
        self.in_proj_qkvz = nn.Linear(d_model, 2 * self.key_dim + 2 * self.value_dim,
                                      bias=False)
        self.in_proj_ba = nn.Linear(d_model, 2 * nhead, bias=False)
        # 因果 depthwise 短卷积（前向取前 L 步，等效只看过去 conv_kernel-1 个 token）
        self.conv_kernel = conv_kernel
        conv_dim = 2 * self.key_dim + self.value_dim
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=conv_kernel,
                                groups=conv_dim, bias=False, padding=conv_kernel - 1)
        # 门控参数（对齐参考 dt_bias / A_log）
        self.dt_bias = nn.Parameter(torch.ones(nhead))
        self.A_log = nn.Parameter(torch.log(torch.empty(nhead).uniform_(0.01, 16.0)))
        # 输出归一化 + 门控（对齐 Qwen3NextRMSNormGated）
        self.norm = RMSNormGated(self.dv)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)

    def forward(self, x, rope=None, pos_ids=None, key_padding_mask=None):
        """rope / pos_ids / key_padding_mask 保留参数以对齐层契约，均忽略
        （位置由门控衰减隐式建模；递归无法跳算被掩码 token）。"""
        B, L, _ = x.shape
        H, dk, dv = self.nhead, self.dk, self.dv

        qkvz = self.in_proj_qkvz(x)
        q, k, v, z = qkvz.split(
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1)
        b, a = self.in_proj_ba(x).split([H, H], dim=-1)

        # 因果短卷积（仅作用 q/k/v；[B,C,L] 双侧填充 k-1 后取前 L 步，
        # 等效每个位置只看过去 conv_kernel-1 个 token）
        mixed = torch.cat([q, k, v], dim=-1).transpose(1, 2)          # [B,C,L]
        mixed = F.silu(self.conv1d(mixed))[:, :, :L]
        q, k, v = (t.transpose(1, 2).reshape(B, L, H, d)
                   for t, d in zip(mixed.split(
                       [self.key_dim, self.key_dim, self.value_dim], dim=1),
                       (dk, dk, dv)))
        z = z.reshape(B, L, H, dv)

        beta = torch.sigmoid(b)                                       # [B,L,H] 写入门
        # 衰减门（log 空间，<= 0）：参考 g = -exp(A_log) * softplus(a + dt_bias)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # ---- 递归（统一 fp32 计算后回到原精度，对齐参考实现）----
        q = l2norm(q.float()) * (dk ** -0.5)
        k = l2norm(k.float())
        v = v.float()
        q, k, v, beta, g = (t.transpose(1, 2).contiguous()
                            for t in (q, k, v, beta.float(), g.float()))  # [B,H,L,*]
        S = q.new_zeros(B, H, dk, dv)                                 # 递归状态
        out = v.new_zeros(B, H, L, dv)
        for i in range(L):
            q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]        # [B,H,d]
            # 1) 门控衰减旧状态
            S = S * g[:, :, i].exp()[..., None, None]
            # 2) delta rule：写入"新值 - 旧状态预测值"的残差修正
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)              # k^T S -> [B,H,dv]
            delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)           # 外积(k, delta)
            # 3) 读出
            out[:, :, i] = (S * q_t.unsqueeze(-1)).sum(dim=-2)

        out = out.transpose(1, 2)                                     # [B,L,H,dv]
        out = self.norm(out, z)                                       # RMSNorm + z 门控
        return self.out_proj(out.reshape(B, L, self.value_dim))
