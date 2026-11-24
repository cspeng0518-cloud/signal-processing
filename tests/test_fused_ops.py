"""融合算子 vs 原生算子 对比测试

数据规格
--------
从 code/configs/default.json 直接读取：
  d_model=512, nhead=8, dh=64, ffn_hidden=2048,
  mla_kv_lora_rank=128, mla_q_lora_rank=128
  seq_len=1024 → patch_size=8 → ~129 token
测试时用 B=4, L=128（短序列常见场景）

覆盖
----
1. RMSNorm         ops.rmsnorm   vs rmsnorm_reference
2. RMSNorm + Add   ops.rmsnorm_add vs rmsnorm_add_reference
3. RoPE rotate     ops.rope_rotate vs rope_rotate_reference
4. SwiGLU（回归）   ops.swiglu     vs swiglu_reference
5. Attention（回归） ops.attention vs attention_reference
6. fused_status    自检融合是否生效

每个算子都对比：
  forward 数值正确性   max_abs_err + allclose
  backward 梯度正确性  max_abs_err + allclose
  延迟对比             eager vs fused（跳过 CPU）

运行
----
    pytest tests/test_fused_ops.py -v --tb=short
    python tests/test_fused_ops.py        # smoke + 延迟打印
"""

import sys
import os
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pytest

# 项目真实规格（从 default.json 读）
with open(_ROOT / "code/configs/default.json") as f:
    _CFG = json.load(f)["model"]

D_MODEL = _CFG["d_model"]            # 512
NHEAD = _CFG["nhead"]                # 8
DH = D_MODEL // NHEAD                # 64
FFN_HIDDEN = _CFG["ffn_hidden"]      # 2048
KV_LORA_RANK = _CFG["mla_kv_lora_rank"]   # 128
Q_LORA_RANK = _CFG["mla_q_lora_rank"]     # 128


# ================================================================ 数据工厂
class Spec:
    """统一规格，所有测试共享。"""
    D_MODEL = D_MODEL
    NHEAD = NHEAD
    DH = DH
    FFN_HIDDEN = FFN_HIDDEN
    KV_LORA_RANK = KV_LORA_RANK
    Q_LORA_RANK = Q_LORA_RANK

    # 测试用 batch / 序列长度（短序列场景，NPU/ONNX 友好）
    B = 4
    L = 128


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tol(dtype=torch.float32):
    if dtype == torch.float32:
        return dict(atol=1e-4, rtol=1e-4)
    if dtype == torch.bfloat16:
        return dict(atol=5e-2, rtol=5e-2)
    if dtype == torch.float16:
        return dict(atol=1e-2, rtol=1e-2)
    return dict(atol=1e-3, rtol=1e-3)


def _bench(fn, args, warmup=5, iters=20):
    dev = args[0].device.type if torch.is_tensor(args[0]) else "cpu"
    for _ in range(warmup):
        with torch.no_grad():
            fn(*args)
    if dev == "cuda":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
    return (time.perf_counter() - t0) / iters * 1e3


# ---------------------------------------------------------------- 数据申请
def rmsnorm_inputs(D, shape_prefix=(Spec.B, Spec.L), dtype=torch.float32, dev=None, requires_grad=False):
    """RMSNorm 测试数据：x + weight（带 .eps）。"""
    dev = dev or _dev()
    shape = (*shape_prefix, D)
    x = torch.randn(shape, device=dev, dtype=dtype, requires_grad=requires_grad)
    w = (torch.randn(D, device=dev, dtype=torch.float32) + 1.0).requires_grad_(requires_grad)
    return x, w


def rmsnorm_add_inputs(D, shape_prefix=(Spec.B, Spec.L), dtype=torch.float32, dev=None, requires_grad=False):
    """RMSNorm+Add 测试数据：x + residual + weight。"""
    dev = dev or _dev()
    shape = (*shape_prefix, D)
    x = torch.randn(shape, device=dev, dtype=dtype, requires_grad=requires_grad)
    r = torch.randn(shape, device=dev, dtype=dtype, requires_grad=requires_grad)
    w = (torch.randn(D, device=dev, dtype=torch.float32) + 1.0).requires_grad_(requires_grad)
    return x, r, w


def rope_inputs(B=Spec.B, L=Spec.L, H=Spec.NHEAD, Dh=Spec.DH, dtype=torch.float32, dev=None,
                pos_ids_mode="none", max_len=256):
    """RoPE 测试数据：x + cos + sin + pos_ids。

    x 形状:    [B, L, H, Dh]           （attention.py 约定）
    cos/sin:  [max_len, Dh]            （attention.py RoPEPositionalEncoding 构造方式）
    pos_ids:  None 或 [B, L] int64     （模拟 MAE gather 后的真实位置）
    """
    dev = dev or _dev()
    x = torch.randn(B, L, H, Dh, device=dev, dtype=dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, Dh, 2).float() / Dh))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)                   # [max_len, Dh/2]
    emb = torch.cat([freqs, freqs], dim=-1)             # [max_len, Dh]
    cos = emb.cos().contiguous().to(dev)
    sin = emb.sin().contiguous().to(dev)
    if pos_ids_mode == "none":
        pos_ids = None
    elif pos_ids_mode == "random":
        pos_ids = torch.randint(0, max_len, (B, L), device=dev)
    elif pos_ids_mode == "sequential":
        pos_ids = torch.arange(L, device=dev).unsqueeze(0).expand(B, L).clone()
    else:
        pos_ids = None
    return x, cos, sin, pos_ids


def swiglu_inputs(hidden=Spec.FFN_HIDDEN, shape_prefix=(Spec.B, Spec.L), dtype=torch.float32, dev=None):
    """SwiGLU 测试数据：gate + up（各自 [B, L, hidden]）。"""
    dev = dev or _dev()
    gate = torch.randn(*shape_prefix, hidden, device=dev, dtype=dtype)
    up = torch.randn(*shape_prefix, hidden, device=dev, dtype=dtype)
    return gate, up


def attention_inputs(B=Spec.B, L=Spec.L, H=Spec.NHEAD, Dh=Spec.DH, dtype=torch.float32, dev=None):
    """Attention 测试数据：q/k/v + padding_mask。"""
    dev = dev or _dev()
    q = torch.randn(B, H, L, Dh, device=dev, dtype=dtype) / math.sqrt(Dh)
    k = torch.randn(B, H, L, Dh, device=dev, dtype=dtype) / math.sqrt(Dh)
    v = torch.randn(B, H, L, Dh, device=dev, dtype=dtype)
    mask = torch.zeros(B, L, device=dev, dtype=torch.bool)
    # 每句随机 masking 末尾几个 token
    for i in range(B):
        m = torch.randint(0, L // 4, (1,)).item()
        mask[i, -m:] = True
    return q, k, v, mask


def mla_modules(d_model=Spec.D_MODEL, nhead=Spec.NHEAD,
                q_lora=Spec.Q_LORA_RANK, kv_lora=Spec.KV_LORA_RANK):
    """MLA 测试数据：返回 q_a_proj / q_a_ln / q_b_proj / kv_a_proj / kv_a_ln / kv_b_proj。"""
    dh = d_model // nhead
    class _LN(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = 1e-6
        def forward(self, x):
            dtype = x.dtype
            x = x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (self.weight * x).to(dtype)

    q_a_proj = nn.Linear(d_model, q_lora, bias=False)
    kv_a_proj = nn.Linear(d_model, kv_lora + dh, bias=False)
    q_a_ln = _LN(q_lora)
    kv_a_ln = _LN(kv_lora)
    q_b_proj = nn.Linear(q_lora, nhead * 2 * dh, bias=False)
    kv_b_proj = nn.Linear(kv_lora, nhead * 2 * dh, bias=False)
    return q_a_proj, q_a_ln, q_b_proj, kv_a_proj, kv_a_ln, kv_b_proj


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(20261124)


# ================================================================= RMSNorm
class TestRMSNorm:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_match_reference(self, dtype):
        from code.models.ops.fused import rmsnorm, rmsnorm_reference
        x, w = rmsnorm_inputs(Spec.D_MODEL, dtype=dtype)
        y_ref = rmsnorm_reference(x, w)
        y_fus = rmsnorm(x, w)
        diff = (y_ref.float() - y_fus.float()).abs().max().item()
        assert torch.allclose(y_ref.float(), y_fus.float(), **_tol(dtype)), \
            f"forward max_abs_err={diff:.3e}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_match_reference(self, dtype):
        if not torch.cuda.is_available():
            pytest.skip("backward 测试需要 CUDA Triton")
        from code.models.ops.triton_kernels import _TritonRMSNorm
        x, w = rmsnorm_inputs(Spec.D_MODEL, dtype=dtype, dev="cuda", requires_grad=True)
        DY = torch.randn_like(x)

        x_r, w_r = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
        from code.models.ops.fused import rmsnorm_reference
        rmsnorm_reference(x_r, w_r).backward(DY)

        x_t, w_t = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
        _TritonRMSNorm.apply(x_t, w_t, 1e-6, None).backward(DY)

        assert torch.allclose(x_r.grad.float(), x_t.grad.float(), **_tol(dtype))
        assert torch.allclose(w_r.grad.float(), w_t.grad.float(), **_tol(dtype))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="延迟对比需要 CUDA")
    def test_latency(self):
        from code.models.ops.fused import rmsnorm, rmsnorm_reference
        x, w = rmsnorm_inputs(Spec.D_MODEL, shape_prefix=(64, Spec.L), dev="cuda")
        t_ref = _bench(rmsnorm_reference, (x, w))
        t_fus = _bench(rmsnorm, (x, w))
        print(f"\n  rmsnorm  D={Spec.D_MODEL}, shape=[64,{Spec.L},{Spec.D_MODEL}]")
        print(f"    eager={t_ref:.3f}ms  fused={t_fus:.3f}ms  speedup={t_ref/max(t_fus,1e-9):.2f}x")


# ================================================================= RMSNorm + Add
class TestRMSNormAdd:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_match_reference(self, dtype):
        from code.models.ops.fused import rmsnorm_add, rmsnorm_add_reference
        x, r, w = rmsnorm_add_inputs(Spec.D_MODEL, dtype=dtype)
        y_ref = rmsnorm_add_reference(x, r, w)
        y_fus = rmsnorm_add(x, r, w)
        assert torch.allclose(y_ref.float(), y_fus.float(), **_tol(dtype))


# ================================================================= RoPE
class TestRoPE:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("pos_mode", ["none", "sequential", "random"])
    def test_forward_match_reference(self, dtype, pos_mode):
        from code.models.ops.fused import rope_rotate, rope_rotate_reference
        x, cos, sin, pos_ids = rope_inputs(pos_ids_mode=pos_mode, dtype=dtype)
        y_ref = rope_rotate_reference(x, cos, sin, pos_ids)
        y_fus = rope_rotate(x, cos, sin, pos_ids)
        diff = (y_ref.float() - y_fus.float()).abs().max().item()
        assert torch.allclose(y_ref.float(), y_fus.float(), atol=1e-3, rtol=1e-3), \
            f"[{pos_mode}] max_abs_err={diff:.3e}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="backward 需要 CUDA")
    def test_backward(self):
        from code.models.ops.fused import rope_rotate_reference
        from code.models.ops.triton_kernels import _TritonRoPE
        x, cos, sin, _ = rope_inputs(dtype=torch.float32, dev="cuda")
        DY = torch.randn_like(x)

        x_r = x.detach().clone().requires_grad_(True)
        rope_rotate_reference(x_r, cos, sin, None).backward(DY)

        x_t = x.detach().clone().requires_grad_(True)
        _TritonRoPE.apply(x_t, cos, sin, None).backward(DY)

        diff = (x_r.grad - x_t.grad).abs().max().item()
        assert torch.allclose(x_r.grad, x_t.grad, atol=1e-3, rtol=1e-3), \
            f"rope backward max_abs_err={diff:.3e}"


# ================================================================= SwiGLU
class TestSwiGLU:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_match_reference(self, dtype):
        from code.models.ops.fused import swiglu, swiglu_reference
        gate, up = swiglu_inputs(dtype=dtype)
        y_ref = swiglu_reference(gate, up)
        y_fus = swiglu(gate, up)
        assert torch.allclose(y_ref.float(), y_fus.float(), **_tol(dtype))


# ================================================================= Attention
class TestAttention:

    def test_forward_match_reference(self):
        from code.models.ops.fused import attention, attention_reference
        q, k, v, mask = attention_inputs()
        y_ref = attention_reference(q, k, v, mask)
        y_fus = attention(q, k, v, mask)
        assert torch.allclose(y_ref.float(), y_fus.float(), atol=5e-3, rtol=5e-3)


# ================================================================= fused_status
def test_fused_status_has_records():
    from code.models.ops.fused import fused_status, rmsnorm, swiglu, rope_rotate, rope_rotate_reference
    x, w = rmsnorm_inputs(Spec.D_MODEL)
    rmsnorm(x, w)
    gate, up = swiglu_inputs()
    swiglu(gate, up)
    x2, cos, sin, _ = rope_inputs()
    rope_rotate(x2, cos, sin, None)
    status = fused_status()
    assert any(k in status for k in ("rmsnorm", "swiglu", "rope")), status
    print("\n  fused_status keys:", list(status.keys()))


# ================================================================= smoke（直接执行时）
if __name__ == "__main__":
    from code.models.ops.fused import (
        rmsnorm, rmsnorm_add, rope_rotate, swiglu, attention,
        rmsnorm_reference, rmsnorm_add_reference, rope_rotate_reference, swiglu_reference,
        attention_reference, fused_status,
    )
    torch.manual_seed(0)
    dev = _dev()
    print(f"Smoke on device={dev}")
    print(f"spec: d_model={Spec.D_MODEL}, nhead={Spec.NHEAD}, dh={Spec.DH}, "
          f"ffn_hidden={Spec.FFN_HIDDEN}, batch={Spec.B}, seq={Spec.L}")

    def check(name, fus, ref, tol=5e-3):
        d = (fus.float() - ref.float()).abs().max().item()
        ok = torch.allclose(fus.float(), ref.float(), atol=tol, rtol=tol)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: max_err={d:.3e}")

    x, w = rmsnorm_inputs(Spec.D_MODEL, dev=dev)
    check("rmsnorm", rmsnorm(x, w), rmsnorm_reference(x, w))
    x, r, w = rmsnorm_add_inputs(Spec.D_MODEL, dev=dev)
    check("rmsnorm_add", rmsnorm_add(x, r, w), rmsnorm_add_reference(x, r, w))
    gate, up = swiglu_inputs(dev=dev)
    check("swiglu", swiglu(gate, up), swiglu_reference(gate, up))
    x, cos, sin, _ = rope_inputs(dev=dev)
    check("rope", rope_rotate(x, cos, sin, None), rope_rotate_reference(x, cos, sin, None))
    q, k, v, mask = attention_inputs(dev=dev)
    check("attention", attention(q, k, v, mask), attention_reference(q, k, v, mask), tol=5e-3)

    if dev == "cuda":
        print("\nLatency (ms, shape=[128,{0},{0}], iters=30):".format(Spec.D_MODEL))
        x2, w2 = rmsnorm_inputs(Spec.D_MODEL, shape_prefix=(128, Spec.L), dev="cuda")
        print(f"  rmsnorm:   ref={_bench(rmsnorm_reference,(x2,w2)):.3f}  fused={_bench(rmsnorm,(x2,w2)):.3f}")
        g2, u2 = swiglu_inputs(Spec.FFN_HIDDEN, shape_prefix=(128, Spec.L), dev="cuda")
        print(f"  swiglu:    ref={_bench(swiglu_reference,(g2,u2)):.3f}  fused={_bench(swiglu,(g2,u2)):.3f}")

    print("\nfused_status():")
    for k, v in fused_status().items():
        print(f"  {k}: {v}")
