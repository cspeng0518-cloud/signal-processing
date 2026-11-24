"""Triton JIT kernels for fused operators.

Kernels 列表
-----------
- rmsnorm_fwd_kernel     : RMSNorm forward（纯 RMSNorm）
- rmsnorm_add_fwd_kernel : 先加残差再 RMSNorm（省掉 res_add 中间 tensor）
- rmsnorm_bwd_kernel     : RMSNorm backward（纯 RMSNorm 的梯度）
- rope_rotate_kernel     : RoPE 旋转（对 [B,L,H,Dh] 展平成 [rows, Dh]，cos/sin [max_len, Dh]）

所有 kernel 都是**纯设备代码**：无 torch ops、无 Python 分支、无边界外访存，
可直接 import 到 fused.py 被 autograd.Function 调用。
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                # pragma: no cover
    _HAS_TRITON = False


# ================================================================= RMSNorm kernels
if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(X, W, Y, D, eps, BLOCK_D: tl.constexpr):
        """Per-row RMSNorm。每个 program 处理一行。"""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / D
        rstd = tl.rsqrt(mean_sq + eps)
        y = x * rstd * w
        tl.store(Y + row * D + offs, y, mask=mask)

    @triton.jit
    def _rmsnorm_add_fwd_kernel(X, RES, W, Y, D, eps, BLOCK_D: tl.constexpr):
        """Fused y = rmsnorm(x + res) * weight。"""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RES + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
        x_add = x + r
        mean_sq = tl.sum(x_add * x_add, axis=0) / D
        rstd = tl.rsqrt(mean_sq + eps)
        y = x_add * rstd * w
        tl.store(Y + row * D + offs, y, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(X, W, DY, DX, D, eps, BLOCK_D: tl.constexpr):
        """RMSNorm backward：grad = dy*w*rstd - x_norm * rstd^3 * mean(dy*w*x_norm)。"""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
        dy = tl.load(DY + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / D
        rstd = tl.rsqrt(mean_sq + eps)
        x_norm = x * rstd
        grad = dy * w
        dot = tl.sum(grad * x_norm, axis=0) / D
        dx = (grad - x_norm * dot) * rstd
        tl.store(DX + row * D + offs, dx, mask=mask)


# ================================================================= RoPE kernel
# 与 attention.py RoPEPositionalEncoding.rotate 完全对齐：
#   x:        [B, L, H, Dh]           -> 展平 rows = B*L*H, 每行 Dh 元素
#   cos/sin:  [max_len, Dh]           -> 每半行 Dh 元素，pos 索引
#   旋转:     x1, x2 = x.chunk(2,-1); return x*cos + cat([-x2,x1])*sin
#   等价于原地：out[half]=x1*cos-x2*sin, out[+half]=x2*cos+x1*sin
# （attention.py 的 cos/sin 表构造时把前后两半频率 cat 成了完整 Dh，
#  所以 kernel 里只需要 load 一次 cos/sin 表，对 out 的前后两半分别应用）
if _HAS_TRITON:

    @triton.jit
    def _rope_rotate_kernel(
        X, COS, SIN, POS,
        Dh,                       # head_dim（完整，不是一半）
        BLOCK_Dh: tl.constexpr,
        HAS_POS: tl.constexpr,
    ):
        """RoPE 旋转（与 attention.py RoPEPositionalEncoding.rotate 完全对齐）。

        X:   [rows, Dh] — 原地写回
        COS/SIN: 若 HAS_POS=True 是 [max_len, Dh]；否则是 [L, Dh] 切片后
        POS: [rows] int64（HAS_POS=True）或空
        旋转: x1,x2=x.chunk(2,-1); return x*cos + cat([-x2,x1])*sin
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_Dh)
        mask = offs < Dh
        half = Dh // 2
        first = offs < half          # 前半 offs -> x1, 后半 offs -> x2

        x = tl.load(X + row * Dh + offs, mask=mask, other=0.0).to(tl.float32)
        x_other = tl.load(
            X + row * Dh + tl.where(first, offs + half, offs - half),
            mask=mask, other=0.0).to(tl.float32)

        if HAS_POS:
            pos = tl.load(POS + row).to(tl.int64)
        else:
            pos = row   # pos_ids=None 时 cos 已经切片成 [L, Dh]，pos=row 就是 l

        cos = tl.load(COS + pos * Dh + offs, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(SIN + pos * Dh + offs, mask=mask, other=0.0).to(tl.float32)

        # first=True: out = x1*cos - x2*sin  (x1=x, x2=x_other)
        # first=False: out = x2*cos + x1*sin  (x2=x, x1=x_other)
        out = tl.where(first,
                       x * cos - x_other * sin,
                       x * cos + x_other * sin)

        tl.store(X + row * Dh + offs, out.to(X.dtype.element_ty), mask=mask)


# ================================================================= Python wrappers
if _HAS_TRITON:

    class _TritonRMSNorm(torch.autograd.Function):
        """RMSNorm forward/backward，可选 fused residual add。"""

        @staticmethod
        def forward(ctx, x, weight, eps, residual=None):
            x = x.contiguous()
            D = x.size(-1)
            BLOCK_D = max(triton.next_power_of_2(D), 16)
            rows = x.numel() // D
            flat_x = x.reshape(rows, D)
            flat_y = torch.empty_like(flat_x)
            w = weight.float()
            if residual is not None:
                res = residual.contiguous().reshape(rows, D)
                _rmsnorm_add_fwd_kernel[(rows,)](
                    flat_x, res, w, flat_y, D, eps, BLOCK_D=BLOCK_D, num_warps=2)
                ctx.save_for_backward(x, weight, residual)
                ctx.has_residual = True
            else:
                _rmsnorm_fwd_kernel[(rows,)](
                    flat_x, w, flat_y, D, eps, BLOCK_D=BLOCK_D, num_warps=2)
                ctx.save_for_backward(x, weight)
                ctx.has_residual = False
            ctx.eps = eps
            return flat_y.reshape(x.shape)

        @staticmethod
        def backward(ctx, dy):
            saved = ctx.saved_tensors
            x = saved[0]; weight = saved[1]
            residual = saved[2] if ctx.has_residual else None
            D = x.size(-1)
            BLOCK_D = max(triton.next_power_of_2(D), 16)
            rows = x.numel() // D
            x_eff = (x + residual) if residual is not None else x
            flat_x = x_eff.contiguous().reshape(rows, D)
            flat_dy = dy.contiguous().reshape(rows, D)
            flat_dx = torch.empty_like(flat_x)
            w = weight.float()
            _rmsnorm_bwd_kernel[(rows,)](
                flat_x, w, flat_dy, flat_dx, D, ctx.eps, BLOCK_D=BLOCK_D, num_warps=2)
            flat_dx = flat_dx.reshape(x.shape)
            if ctx.has_residual:
                return flat_dx, None, None, flat_dx
            return flat_dx, None, None, None


    class _TritonRoPE(torch.autograd.Function):
        """RoPE rotation — forward 原地旋转，backward 用反向 cos 旋回来。"""

        @staticmethod
        def forward(ctx, x, cos, sin, pos_ids=None):
            # x: [B, L, H, Dh]  attention.py 约定
            # cos/sin: [max_len, Dh]  attention.py 构造
            orig_shape = x.shape
            dtype = x.dtype
            Dh = cos.size(-1)
            assert Dh == x.size(-1), f"cos/sin 最后一维 {Dh} != x 最后一维 {x.size(-1)}"

            x_f = x.contiguous().clone()
            cos = cos.float().contiguous()
            sin = sin.float().contiguous()

            # pos_ids 的两种形态：
            #   - None: cos = cos[:L] 切片后展平，pos_ids 不使用
            #   - [B,L]: 展平后与 rows=B*L*H 对齐，需要先 unsqueeze(2) 再 expand 到 H
            has_pos = pos_ids is not None
            if has_pos:
                # attention.py pos_ids 是 [B, L]，需要 expand 到 H
                pos = pos_ids.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2)).reshape(-1).to(torch.int64)
            else:
                cos = cos[:x.size(1)].contiguous()     # 切片成 [L, Dh]
                sin = sin[:x.size(1)].contiguous()
                pos = torch.empty(0, dtype=torch.int64, device=x_f.device)

            rows = x_f.numel() // Dh
            flat_x = x_f.reshape(rows, Dh)
            BLOCK_Dh = max(triton.next_power_of_2(Dh), 16)

            _rope_rotate_kernel[(rows,)](
                flat_x, cos, sin, pos,
                Dh,
                BLOCK_Dh=BLOCK_Dh, HAS_POS=has_pos, num_warps=1)

            ctx.save_for_backward(cos, sin)
            ctx.has_pos = has_pos
            if has_pos:
                ctx.pos_ids = pos.contiguous()
            return flat_x.to(dtype).reshape(orig_shape)

        @staticmethod
        def backward(ctx, dy):
            cos, sin = ctx.saved_tensors
            orig_shape = dy.shape
            dtype = dy.dtype
            Dh = cos.size(-1)
            dy_f = dy.contiguous().clone()
            flat_dy = dy_f.reshape(-1, Dh)
            neg_sin = -sin.contiguous()

            if ctx.has_pos:
                pos = ctx.pos_ids
            else:
                pos = torch.empty(0, dtype=torch.int64, device=dy_f.device)

            rows = flat_dy.size(0)
            BLOCK_Dh = max(triton.next_power_of_2(Dh), 16)
            _rope_rotate_kernel[(rows,)](
                flat_dy, cos.contiguous(), neg_sin, pos,
                Dh,
                BLOCK_Dh=BLOCK_Dh, HAS_POS=ctx.has_pos, num_warps=1)
            return flat_dy.to(dtype).reshape(orig_shape), None, None, None
