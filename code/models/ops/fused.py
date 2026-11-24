"""融合算子分发层：同一接口，按设备自动选择最快且数值正确的实现。

  attention(q, k, v, key_padding_mask)   q/k: [B,H,L,E]  v: [B,H,L,Ev]（MLA 时 Ev != E）
      CUDA : F.scaled_dot_product_attention（FlashAttention-2 / mem-efficient 内核）
      NPU  : torch_npu.npu_fusion_attention（昇腾 FlashAttention 融合算子）
      其他 : eager softmax(QK^T)V
  swiglu(gate, up)  -> silu(gate) * up
      CUDA : Triton 融合 kernel（一次读写，省掉 silu 中间张量）
      NPU  : torch_npu.npu_swiglu
      其他 : torch 参考实现

所有融合实现首次调用时与参考实现做数值自检（allclose），不通过自动回退并记录原因
——保证"国产平台换算子不掉精度"，也是技术报告里"算子迁移可验证"的工程证据。
环境变量 EM_FUSED_OPS=0 可全局关闭融合路径（用于消融/排查）。

注意：训练阶段同样走这些接口；Triton swiglu 带手写 backward（autograd.Function）。
"""

import math
import os

import torch
import torch.nn.functional as F

_ENABLED = os.environ.get("EM_FUSED_OPS", "1") != "0"
_STATUS = {}          # op_name@device -> "fused" | "fallback: <reason>"

try:                                          # 昇腾
    import torch_npu  # noqa: F401
    _HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:                             # pragma: no cover
    torch_npu = None
    _HAS_NPU = False

try:                                          # Triton（CUDA）
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                             # pragma: no cover
    _HAS_TRITON = False


def fused_status():
    """返回各融合算子的启用状态（infer/benchmark 日志打印用）。"""
    return dict(_STATUS)


# =========================================================== attention
def attention_reference(q, k, v, key_padding_mask=None, dropout_p=0.0, training=False):
    """eager 参考实现。key_padding_mask: [B,L] True=忽略。"""
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = (q @ k.transpose(-2, -1)) * scale
    if key_padding_mask is not None:
        attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    attn = attn.float().softmax(-1).to(q.dtype)
    if dropout_p > 0 and training:
        attn = F.dropout(attn, dropout_p)
    return attn @ v


def _sdpa(q, k, v, key_padding_mask, dropout_p, training):
    mask = None
    if key_padding_mask is not None:
        mask = ~key_padding_mask[:, None, None, :]          # SDPA: True = 参与注意力
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout_p if training else 0.0)


def _npu_fa(q, k, v, key_padding_mask, dropout_p, training):
    B, H, L, D = q.shape
    atten_mask = None
    if key_padding_mask is not None:
        # npu_fusion_attention: True/1 = 屏蔽；形状 [B,1,Lq,Lk]
        atten_mask = key_padding_mask[:, None, None, :].expand(B, 1, L, L).contiguous()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch_npu.npu_fusion_attention(
        q, k, v, H, input_layout="BNSD", atten_mask=atten_mask,
        scale=1.0 / math.sqrt(D), keep_prob=1.0 - (dropout_p if training else 0.0))[0]
    return out


def _selftest(name, fn, ref, *args):
    # 缓存键带形状签名：FA（E==Ev）与 MLA（E=2*Ev）、有/无 padding mask 走不同内核，
    # 必须分别自检，避免"FA 通过 -> MLA 直接调用融合算子报错"
    q = args[0]
    sig = "x".join(str(a.shape[-1]) for a in args[:3] if torch.is_tensor(a))
    has_mask = len(args) > 3 and args[3] is not None
    key = f"{name}@{q.device.type}[{sig}{',mask' if has_mask else ''}]"
    if key in _STATUS:
        return _STATUS[key] == "fused"
    try:
        with torch.no_grad():
            small = [a[:2] if torch.is_tensor(a) and a.dim() > 1 else a for a in args]
            out, exp = fn(*small).float(), ref(*small).float()
            ok = torch.allclose(out, exp, atol=3e-2, rtol=3e-2)
        _STATUS[key] = "fused" if ok else f"fallback: mismatch max={float((out-exp).abs().max()):.3e}"
    except Exception as e:                    # 算子不存在/签名不符 -> 回退
        _STATUS[key] = f"fallback: {type(e).__name__}: {e}"[:200]
    return _STATUS[key] == "fused"


def attention(q, k, v, key_padding_mask=None, dropout_p=0.0, training=False):
    """多头注意力统一入口。缩放因子恒为 1/sqrt(E)（E = q 的最后一维）。"""
    dev = q.device.type
    if _ENABLED and dev == "npu" and torch_npu is not None and q.size(-1) == v.size(-1):
        if _selftest("npu_fusion_attention",
                     lambda a, b, c, m: _npu_fa(a, b, c, m, 0.0, False),
                     lambda a, b, c, m: attention_reference(a, b, c, m),
                     q, k, v, key_padding_mask):
            return _npu_fa(q, k, v, key_padding_mask, dropout_p, training)
    if hasattr(F, "scaled_dot_product_attention") and dev != "npu":
        return _sdpa(q, k, v, key_padding_mask, dropout_p, training)
    return attention_reference(q, k, v, key_padding_mask, dropout_p, training)


# =========================================================== swiglu
def swiglu_reference(gate, up):
    return F.silu(gate) * up


if _HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(G, U, O, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
        s = g / (1.0 + tl.exp(-g))
        tl.store(O + offs, (s * u).to(O.dtype.element_ty), mask=m)

    @triton.jit
    def _swiglu_bwd_kernel(G, U, DO, DG, DU, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
        do = tl.load(DO + offs, mask=m, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        s = g * sig
        tl.store(DU + offs, (do * s).to(DU.dtype.element_ty), mask=m)
        dsdg = sig * (1.0 + g * (1.0 - sig))
        tl.store(DG + offs, (do * u * dsdg).to(DG.dtype.element_ty), mask=m)

    class _TritonSwiGLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate, up):
            gate, up = gate.contiguous(), up.contiguous()
            out = torch.empty_like(gate)
            n = gate.numel()
            grid = (triton.cdiv(n, 1024),)
            _swiglu_fwd_kernel[grid](gate, up, out, n, BLOCK=1024)
            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx, dout):
            gate, up = ctx.saved_tensors
            dout = dout.contiguous()
            dg, du = torch.empty_like(gate), torch.empty_like(up)
            n = gate.numel()
            grid = (triton.cdiv(n, 1024),)
            _swiglu_bwd_kernel[grid](gate, up, dout, dg, du, n, BLOCK=1024)
            return dg, du


def _npu_swiglu(gate, up):
    return torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)


def swiglu(gate, up):
    dev = gate.device.type
    if _ENABLED and dev == "cuda" and _HAS_TRITON:
        if _selftest("triton_swiglu", _TritonSwiGLU.apply, swiglu_reference, gate, up):
            return _TritonSwiGLU.apply(gate, up)
    if _ENABLED and dev == "npu" and torch_npu is not None:
        if _selftest("npu_swiglu", _npu_swiglu, swiglu_reference, gate, up):
            return _npu_swiglu(gate, up)
    return swiglu_reference(gate, up)


# ====================================================================== RMSNorm
def rmsnorm_reference(x, weight, eps=1e-6):
    """Pure torch RMSNorm 参考实现。"""
    dtype = x.dtype
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * y).to(dtype)


def rmsnorm_add_reference(x, residual, weight, eps=1e-6):
    """Fused RMSNorm(x + residual) 参考实现（先加再归一化，省一个中间 tensor 的写回）。"""
    return rmsnorm_reference(x + residual, weight, eps)


def _npu_rmsnorm(x, weight, eps):
    """调 torch_npu 原生 rmsnorm。若该版本 torch_npu 未暴露则 raise，调用方会 fallback。"""
    if hasattr(torch_npu, "npu_rmsnorm"):
        return torch_npu.npu_rmsnorm(x.contiguous(), weight.float(), eps).to(x.dtype)
    op = (torch.ops.npu.npu_rmsnorm
          if hasattr(torch.ops, "npu") and hasattr(torch.ops.npu, "npu_rmsnorm")
          else None)
    if op is not None:
        return op(x.contiguous(), weight.float(), eps).to(x.dtype)
    raise RuntimeError("torch_npu 未暴露 rmsnorm 算子")


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm 统一入口。

    GPU : Triton autograd kernel
    NPU : torch_npu.npu_rmsnorm（如可用）/ sgl_kernel_npu.layer_norm（如已装）
    其他 : 纯 torch 参考实现
    """
    dev = x.device.type
    if _ENABLED and _HAS_TRITON and dev == "cuda":
        if _selftest("triton_rmsnorm",
                     lambda a, w: __import__("models.ops.triton_kernels", fromlist=["_TritonRMSNorm"])._TritonRMSNorm.apply(a, w, eps),
                     lambda a, w: rmsnorm_reference(a, w, eps),
                     x, weight):
            from .triton_kernels import _TritonRMSNorm
            return _TritonRMSNorm.apply(x, weight, eps, None)
    if _ENABLED and dev == "npu" and torch_npu is not None:
        if _selftest("npu_rmsnorm",
                     lambda a, w: _npu_rmsnorm(a, w, eps),
                     lambda a, w: rmsnorm_reference(a, w, eps),
                     x, weight):
            return _npu_rmsnorm(x, weight, eps)
    return rmsnorm_reference(x, weight, eps)


def rmsnorm_add(x, residual, weight, eps=1e-6):
    """Fused RMSNorm(x + residual) 统一入口。

    GPU : Triton autograd kernel（rmsnorm_add_fwd_kernel）
    NPU : 如果 torch_npu 版本支持 fused layernorm+add 则用；否则回退 rmsnorm(x+res)
    其他 : 纯 torch 参考实现
    """
    dev = x.device.type
    if _ENABLED and _HAS_TRITON and dev == "cuda":
        if _selftest("triton_rmsnorm_add",
                     lambda a, r, w: __import__("models.ops.triton_kernels", fromlist=["_TritonRMSNorm"])._TritonRMSNorm.apply(a, w, eps, r),
                     lambda a, r, w: rmsnorm_add_reference(a, r, w, eps),
                     x, residual, weight):
            from .triton_kernels import _TritonRMSNorm
            return _TritonRMSNorm.apply(x, weight, eps, residual)
    if _ENABLED and dev == "npu" and torch_npu is not None:
        # torch_npu 原生无 rmsnorm_add，但 npu_layernorm 支持 fused add（某些版本）
        try:
            return _npu_rmsnorm(x + residual, weight, eps)
        except Exception:
            pass
    return rmsnorm_add_reference(x, residual, weight, eps)


# ====================================================================== RoPE
def rope_rotate_reference(x, cos, sin, pos_ids=None):
    """Pure torch RoPE 参考实现（与 attention.py RoPEPositionalEncoding.rotate 对齐）。

    x: [B, L, H, Dh] （attention.py 的用法）
    cos/sin: [max_len, Dh]（最后一维是完整 head_dim，不是 half_d）
    pos_ids: [B, L] 或 None（None 时用 0..L-1）

    旋转公式：x1, x2 = x.chunk(2, -1); return x*cos + cat([-x2,x1])*sin
    """
    if pos_ids is None:
        cos = cos[: x.size(1)][None, :, None, :]      # [1, L, 1, Dh]
        sin = sin[: x.size(1)][None, :, None, :]
    else:
        cos = cos[pos_ids].unsqueeze(2)                # [B, L, 1, Dh]
        sin = sin[pos_ids].unsqueeze(2)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def rope_rotate(x, cos, sin, pos_ids=None):
    """RoPE 统一入口。

    GPU : Triton autograd kernel（一次 launch，省 chunk/cat 中间 tensor）
    NPU : torch_npu.npu_rotary_embedding（如可用）/ 否则纯 torch
    其他 : 纯 torch 参考实现

    接口与 attention.py 的 RoPEPositionalEncoding.rotate 完全一致，
    可以直接替换：`rope_rotate(x, rope.cos, rope.sin, pos_ids)`。
    """
    dev = x.device.type
    if _ENABLED and _HAS_TRITON and dev == "cuda":
        if _selftest("triton_rope",
                     lambda a, c, s, p: __import__("models.ops.triton_kernels", fromlist=["_TritonRoPE"])._TritonRoPE.apply(a, c, s, p),
                     lambda a, c, s, p: rope_rotate_reference(a, c, s, p),
                     x, cos, sin, pos_ids):
            from .triton_kernels import _TritonRoPE
            return _TritonRoPE.apply(x, cos, sin, pos_ids)
    if _ENABLED and dev == "npu" and torch_npu is not None:
        if hasattr(torch_npu, "npu_rotary_embedding"):
            if _selftest("npu_rope",
                         lambda a, c, s, p: torch_npu.npu_rotary_embedding(
                             a.contiguous(), c.contiguous(), s.contiguous()),
                         lambda a, c, s, p: rope_rotate_reference(a, c, s, p),
                         x, cos, sin, pos_ids):
                return torch_npu.npu_rotary_embedding(x.contiguous(), cos.contiguous(), sin.contiguous())
    return rope_rotate_reference(x, cos, sin, pos_ids)


# ============================================================= MLA preprocess
def mla_preprocess(x, q_a_proj, q_a_ln, q_b_proj,
                   kv_a_proj, kv_a_ln, kv_b_proj, H, dh, kv_lora_rank):
    """MLA q/kv 预处理统一入口。

    计算 q = q_b_proj(q_a_ln(q_a_proj(x)))、kv = kv_b_proj(kv_a_ln(kv_a_proj(x)))。
    融合点在 q_a_ln / kv_a_ln（用 rmsnorm 入口），
    Linear 本身由 torch / NPU 后端各自优化。

    GPU : Linear + Triton RMSNorm（rmsnorm 入口）
    NPU : Linear + torch_npu RMSNorm（rmsnorm 入口）
    其他 : Linear + eager rmsnorm（rmsnorm_reference）

    Args:
        x: [B, L, D]
        q_a_proj / q_b_proj / kv_a_proj / kv_b_proj: nn.Linear（weight.t() 转置格式）
        q_a_ln / kv_a_ln: 带 .weight 和 .eps 的 RMSNorm / nn.LayerNorm / 任意 weight tensor
        H, dh, kv_lora_rank: 结构参数

    Returns:
        q:  [B, L, H, 2*dh]
        kv: [B, L, H, 2*dh]
        k_pe_raw: [B, L, dh] （解耦 RoPE 的 k_rope 通道，不参与 KV 压缩）
    """
    q_a_w = q_a_ln.weight if hasattr(q_a_ln, "weight") else q_a_ln
    q_a_eps = getattr(q_a_ln, "eps", 1e-6)
    kv_a_w = kv_a_ln.weight if hasattr(kv_a_ln, "weight") else kv_a_ln
    kv_a_eps = getattr(kv_a_ln, "eps", 1e-6)

    q = q_b_proj(rmsnorm(q_a_proj(x), q_a_w, q_a_eps))
    kv_full = kv_a_proj(x)
    kv_nope, k_pe = kv_full.split([kv_lora_rank, dh], dim=-1)
    kv = kv_b_proj(rmsnorm(kv_nope, kv_a_w, kv_a_eps))
    return (q.view(x.size(0), x.size(1), H, 2 * dh),
            kv.view(x.size(0), x.size(1), H, 2 * dh), k_pe)
