"""Mamba 选择性状态空间模块（纯 torch eager 实现，无 CUDA 专有算子）

实现参考:
  - HuggingFace transformers MambaMixer
    (transformers/models/mamba/modeling_mamba.py) 的 slow_forward 路径
  - state-spaces/mamba 官方 block 实现（mamba-minimal 风格结构）

核心递推（离散化 SSM，状态 h: [B, d_inner, d_state]）:
    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
    y_t = h_t^T C_t + D * x_t
"选择性"体现在 dt/B/C 由输入逐 token 投影得到（内容相关的记忆读写速率），
A 取负值保证指数衰减（稳定记忆）；dt 经 softplus 离散化为步长。

与参考实现的差异（适配本仓库约定）:
  - 逐时间步 for 循环递归（对应 transformers slow_forward），训练序列仅百余
    token，正确性与可读性优先；生产可换选择性 scan 融合内核加速
  - 递归统一 fp32 计算后回到输入精度（bf16 训练下保证状态累积稳定，
    对量化友好）
  - 位置无关（序列模型无注意力），rope / pos_ids / key_padding_mask
    参数仅为对齐层契约而保留、一律忽略
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 官方初始化常量：dt ∈ [1e-3, 1e-1]，bias 取 log(dt) 使 softplus 反演初值
_DT_MIN, _DT_MAX = 1e-3, 1e-1


class MambaBlock(nn.Module):
    """Mamba block（门控 SSM，与官方 Mamba 前向等价的 eager 实现）。

    in_proj(x,z) -> 因果 depthwise 短卷积 + SiLU -> x_proj(dt,B,C)
    -> dt_proj 离散化步长 -> selective scan 递归（fp32）
    -> SiLU(z) 门控 -> out_proj
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        # 门控分支 z 与 SSM 分支 x 合并投影（对齐官方 in_proj）
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # 因果 depthwise 短卷积（双侧填充 k-1 后取前 L 步 = 只看过去）
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, bias=True,
                                padding=d_conv - 1)
        # dt / B / C 联合投影；dt 低秩（对齐官方 dt_rank = ceil(d_model/16)）
        self.dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # SSM 参数：A 每通道独立（log 参数化保证负值衰减），D 为跳连
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # dt_proj 偏置对齐官方 inv_dt 初始化（softplus 反演）
        dt = torch.exp(torch.rand(self.d_inner)
                       * (math.log(_DT_MAX) - math.log(_DT_MIN))
                       + math.log(_DT_MIN)).clamp(min=_DT_MIN)
        self.dt_proj.bias.data = dt.log()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: [B, L, D] -> [B, L, D]。"""
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)                                # 各 [B,L,d_inner]

        # 因果短卷积 + SiLU（取前 L 步 = 每个位置只看过去 d_conv-1 个 token）
        x_in = F.silu(self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2))

        # 选择性参数：dt 步长 / B 写入向量 / C 读出向量（逐 token 由内容决定）
        x_dbl = self.x_proj(x_in)
        dt, Bv, Cv = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                            # [B,L,d_inner] > 0

        # ---- selective scan（统一 fp32 计算后回到原精度）----
        A = -self.A_log.float().exp()                                # [d_inner,N] < 0
        dt, Bv, Cv, x_in = (t.float() for t in (dt, Bv, Cv, x_in))
        h = x.new_zeros(B, self.d_inner, self.d_state, dtype=torch.float32)
        out = x.new_zeros(B, L, self.d_inner, dtype=torch.float32)
        for i in range(L):
            # 离散化衰减 dA = exp(dt * A)（各通道独立衰减旧状态）
            dA = (dt[:, i].unsqueeze(-1) * A).exp()                  # [B,d_inner,N]
            # 写入：dt * B * x（外积到状态维）
            delta = (dt[:, i].unsqueeze(-1) * Bv[:, i].unsqueeze(1)
                     * x_in[:, i].unsqueeze(-1))                      # [B,d_inner,N]
            h = dA * h + delta
            # 读出：h^T C + D 跳连
            out[:, i] = (h * Cv[:, i].unsqueeze(1)).sum(-1) + self.D * x_in[:, i]

        y = out.to(x.dtype) * F.silu(z)                              # SiLU 门控
        return self.out_proj(y)
