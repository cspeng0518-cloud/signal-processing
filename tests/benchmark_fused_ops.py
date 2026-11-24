"""融合算子 vs 原生算子 对比 benchmark。

与训练配置对齐：从 code/configs/default.json 读 batch_size / d_model / ffn_hidden 等，
分多档 batch 测延迟曲线 + 精度 diff，最后自动给结论：每个算子建议启用 or 关闭。

用法:
    python tests/benchmark_fused_ops.py                     # 默认: fused + eager 都跑，自动给结论
    python tests/benchmark_fused_ops.py --mode eager        # 只跑 eager（基线）
    python tests/benchmark_fused_ops.py --mode fused        # 只跑 fused
    python tests/benchmark_fused_ops.py --batch 256         # 覆盖 batch 档位
    python tests/benchmark_fused_ops.py --seq 128          # 覆盖 seq_len
    python tests/benchmark_fused_ops.py --report out.md     # 把结论写到 markdown 文件

控制是否实际调用融合算子（与 benchmark 解耦）:
    EM_FUSED_OPS=0                python train.py    # 训练全局关闭融合
    EM_FORCE_EAGER=rmsnorm       python train.py    # 只关 rmsnorm，其他照旧
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# 把项目根加进 sys.path（从 tests/ 往上两级）
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from code.models.ops import (  # noqa: E402
    attention, attention_reference,
    swiglu, swiglu_reference,
    rmsnorm, rmsnorm_reference,
    rmsnorm_add, rmsnorm_add_reference,
    rope_rotate, rope_rotate_reference,
    fused_status, fused_decision, set_fused_enabled,
)

# --------------------------------------------------------------------------- 配置加载

_CONFIG_PATH = _ROOT / "code" / "configs" / "default.json"


def _load_default():
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # 兜底值（default.json 不在时也能跑）
    print(f"[warn] default.json 不存在，用兜底值: {_CONFIG_PATH}", file=sys.stderr)
    return {
        "d_model": 512, "nhead": 8, "batch_size": 512,
        "ffn_hidden": 2048, "seq_len": 1024,
        "batch_size_pretrain": 512,
    }


SPEC = _load_default()

# 结构参数
D = int(SPEC.get("d_model", 512))
NH = int(SPEC.get("nhead", 8))
DH = D // NH                       # 64
FFN_H = int(SPEC.get("ffn_hidden", D * 4))
SEQ_RAW = int(SPEC.get("seq_len", 1024))
SEQ = max(1, SEQ_RAW // 8)         # patch_size=8 → token 序列长度（约 129）
REAL_BATCH = int(SPEC.get("batch_size", 512))
PRETRAIN_BATCH = int(SPEC.get("batch_size_pretrain", REAL_BATCH))

BATCH_TIERS = (4, 32, 128, REAL_BATCH)   # 档位，最后一档是训练真实值

# --------------------------------------------------------------------------- 设备检测 + 工具

_HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
_HAS_CUDA = torch.cuda.is_available()

if _HAS_NPU:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass

# NPU 同步 API：不同版本函数名可能不一样
_NPU_SYNC = None
if _HAS_NPU:
    if hasattr(torch.npu, "synchronize"):
        _NPU_SYNC = torch.npu.synchronize
    elif hasattr(torch.cuda, "synchronize"):
        _NPU_SYNC = torch.cuda.synchronize


def _dev():
    if _HAS_NPU:
        return "npu"
    if _HAS_CUDA:
        return "cuda"
    return "cpu"


def _sync():
    if _NPU_SYNC is not None:
        _NPU_SYNC()
    elif _HAS_CUDA:
        torch.cuda.synchronize()


def _bench(fn, warmup=20, repeat=100):
    """测单次调用平均延迟（ms）。"""
    _sync()
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync()
    return (time.perf_counter() - t0) / repeat * 1000.0


def _allclose(a, b):
    """精度容忍：bf16 更宽，其他 fp32 默认 3e-2。"""
    atol = rtol = 3e-2
    if a.dtype == torch.bfloat16:
        atol = rtol = 1e-1
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- 数据工厂

def _t(shape, dev, dtype=torch.float32, scale=1.0):
    return torch.randn(shape, device=dev, dtype=dtype) * scale


def rope_inputs(dev, max_len=None):
    """RoPE cos/sin 构造与 attention.py RoPEPositionalEncoding 完全对齐。"""
    max_len = max_len or SEQ
    Dh = DH
    x = _t((REAL_BATCH, SEQ, NH, 2 * Dh), dev=dev)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, Dh, 2, device=dev).float() / Dh))
    freqs = torch.outer(torch.arange(max_len, device=dev).float(), inv_freq)  # [L, Dh/2]
    emb = torch.cat([freqs, freqs], dim=-1)                                   # [L, Dh]
    return x, emb.cos(), emb.sin(), None


# --------------------------------------------------------------------------- 基准函数封装

def _bench_rmsnorm(batch, dev):
    x = _t((batch, SEQ, D), dev=dev)
    w = torch.ones(D, device=dev)
    eager_ms = _bench(lambda: rmsnorm_reference(x, w, 1e-6))
    fused_ms = _bench(lambda: rmsnorm(x, w, 1e-6))
    return eager_ms, fused_ms


def _bench_rmsnorm_add(batch, dev):
    x = _t((batch, SEQ, D), dev=dev)
    r = _t((batch, SEQ, D), dev=dev)
    w = torch.ones(D, device=dev)
    eager_ms = _bench(lambda: rmsnorm_add_reference(x, r, w, 1e-6))
    fused_ms = _bench(lambda: rmsnorm_add(x, r, w, 1e-6))
    return eager_ms, fused_ms


def _bench_swiglu(batch, dev):
    gate = _t((batch, SEQ, FFN_H), dev=dev)
    up = _t((batch, SEQ, FFN_H), dev=dev)
    eager_ms = _bench(lambda: swiglu_reference(gate, up))
    fused_ms = _bench(lambda: swiglu(gate, up))
    return eager_ms, fused_ms


def _bench_attention(batch, dev):
    q = _t((batch, NH, SEQ, DH), dev=dev)
    k = _t((batch, NH, SEQ, DH), dev=dev)
    v = _t((batch, NH, SEQ, DH), dev=dev)
    eager_ms = _bench(lambda: attention_reference(q, k, v))
    fused_ms = _bench(lambda: attention(q, k, v))
    return eager_ms, fused_ms


def _bench_rope(batch, dev):
    x, cos, sin, _ = rope_inputs(dev)
    # 切到当前 batch 大小（rope_inputs 默认 REAL_BATCH）
    x = x[:batch]
    eager_ms = _bench(lambda: rope_rotate_reference(x, cos, sin))
    fused_ms = _bench(lambda: rope_rotate(x, cos, sin))
    return eager_ms, fused_ms


# --------------------------------------------------------------------------- 精度检查

def _check_rmsnorm(dev):
    x = _t((REAL_BATCH, SEQ, D), dev=dev)
    w = torch.ones(D, device=dev)
    ref = rmsnorm_reference(x, w, 1e-6)
    got = rmsnorm(x, w, 1e-6)
    return (got.float() - ref.float()).abs().max().item()


def _check_swiglu(dev):
    gate = _t((REAL_BATCH, SEQ, FFN_H), dev=dev)
    up = _t((REAL_BATCH, SEQ, FFN_H), dev=dev)
    ref = swiglu_reference(gate, up)
    got = swiglu(gate, up)
    return (got.float() - ref.float()).abs().max().item()


def _check_attention(dev):
    q = _t((REAL_BATCH, NH, SEQ, DH), dev=dev)
    k = _t((REAL_BATCH, NH, SEQ, DH), dev=dev)
    v = _t((REAL_BATCH, NH, SEQ, DH), dev=dev)
    ref = attention_reference(q, k, v)
    got = attention(q, k, v)
    return (got.float() - ref.float()).abs().max().item()


def _check_rope(dev):
    x, cos, sin, _ = rope_inputs(dev)
    x = x[:REAL_BATCH]
    ref = rope_rotate_reference(x, cos, sin)
    got = rope_rotate(x, cos, sin)
    return (got.float() - ref.float()).abs().max().item()


# --------------------------------------------------------------------------- 主流程

def run_benchmarks(mode="both", report_path=None):
    dev = _dev()
    print(f"\n{'='*70}")
    print(f"  设备: {dev}    batch 档位: {BATCH_TIERS}    d_model={D}  nhead={NH}")
    print(f"  seq_len={SEQ} (raw={SEQ_RAW})    ffn_hidden={FFN_H}")
    print(f"  训练真实 batch: {REAL_BATCH}    pretrain: {PRETRAIN_BATCH}")
    print(f"  当前融合决策: {fused_decision()}")
    print(f"{'='*70}\n")

    # 如果只跑 eager 或只跑 fused —— 用环境变量全局关掉对应路径
    if mode == "eager":
        set_fused_enabled({"attention": False, "swiglu": False, "rmsnorm": False,
                           "rmsnorm_add": False, "rope_rotate": False})
    elif mode == "fused":
        # 已经是默认，什么都不做
        pass

    ops = [
        ("RMSNorm",        _bench_rmsnorm,    _check_rmsnorm),
        ("RMSNormAdd",     _bench_rmsnorm_add, _check_rmsnorm_add),
        ("SwiGLU",         _bench_swiglu,     _check_swiglu),
        ("Attention",      _bench_attention,  _check_attention),
        ("RoPE",           _bench_rope,       _check_rope),
    ]

    results = {}  # op_name -> {"latency": {batch: (eager, fused, speedup)}, "precision": max_err}

    for name, bench_fn, check_fn in ops:
        print(f"--- {name} ---")
        latency_rows = []
        for b in BATCH_TIERS:
            try:
                eager_ms, fused_ms = bench_fn(b, dev)
            except Exception as e:
                print(f"  batch={b}: ERROR {type(e).__name__}: {e}")
                latency_rows.append((b, None, None, None))
                continue
            speedup = eager_ms / fused_ms if fused_ms > 0 else float("inf")
            marker = " ← 训练 batch" if b == REAL_BATCH else ""
            latency_rows.append((b, eager_ms, fused_ms, speedup))
            print(f"  batch={b:4d}{marker}:  eager={eager_ms:.4f}ms  fused={fused_ms:.4f}ms"
                  f"  speedup={speedup:.2f}x")

        try:
            max_err = check_fn(dev)
            print(f"  精度对比 (batch={REAL_BATCH}): max_abs_err = {max_err:.3e}")
        except Exception as e:
            max_err = f"ERROR: {e}"
            print(f"  精度对比: ERROR {e}")

        results[name] = {"latency": latency_rows, "precision": max_err}
        print()

    # 融合状态
    print("--- fused_status ---")
    for k, v in fused_status().items():
        print(f"  {k}: {v}")
    print()

    # ---- 结论 ----
    recommendations = []
    print("=" * 70)
    print("  结论（建议）")
    print("=" * 70)
    for name in ("Attention", "SwiGLU", "RMSNorm", "RMSNormAdd", "RoPE"):
        if name not in results:
            continue
        info = results[name]
        # 用训练真实 batch 那档的 speedup
        train_row = None
        for row in info["latency"]:
            if row[0] == REAL_BATCH:
                train_row = row
                break
        if train_row is None:
            # fallback 到最大档
            train_row = info["latency"][-1]
        b, eager, fused, speedup = train_row
        err = info["precision"]
        err_ok = isinstance(err, float) and err < 1e-2
        fused_status_dict = fused_status()
        # 找对应的 status key
        status_val = None
        for k, v in fused_status_dict.items():
            if name.lower() in k.lower():
                status_val = v
                break

        if fused is None or eager is None:
            rec = "—"
        elif speedup >= 1.2 and err_ok:
            rec = "✅ 建议启用融合"
        elif speedup >= 1.0 and err_ok:
            rec = "🟡 收益不明显，保留默认"
        elif speedup < 1.0 and err_ok:
            rec = "❌ 融合路径反而慢，建议关闭"
        elif not err_ok:
            rec = f"⚠️ 精度未达标 (err={err!s})"
        else:
            rec = "—"

        print(f"  {name:12s}  batch={b:4d}  eager={eager:.4f}ms  fused={fused:.4f}ms"
              f"  speedup={speedup:.2f}x  max_err={err!s}")
        print(f"              融合状态: {status_val or '(未触发)'}  →  {rec}")
        recommendations.append((name, speedup, rec))

    # ---- 写 markdown 报告 ----
    if report_path:
        _write_report(report_path, results, recommendations, dev, REAL_BATCH)

    return results, recommendations


def _write_report(path, results, recs, dev, real_batch):
    lines = [
        "# 融合算子对比报告",
        "",
        f"- 设备: **{dev}**",
        f"- 训练 batch: **{real_batch}**  seq_len: **{SEQ}**  d_model: **{D}**",
        f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 结论摘要",
        "",
        "| 算子 | 训练 batch 下 speedup | 建议 |",
        "|------|----------------------|------|",
    ]
    for name, speedup, rec in recs:
        lines.append(f"| {name} | {speedup:.2f}x | {rec} |")

    lines.append("")
    lines.append("## 分 batch 延迟曲线")
    lines.append("")
    for name, info in results.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| batch | eager (ms) | fused (ms) | speedup |")
        lines.append("|-------|-----------|-----------|---------|")
        for b, eager, fused, sp in info["latency"]:
            if eager is None:
                lines.append(f"| {b} | — | — | — |")
            else:
                lines.append(f"| {b} | {eager:.4f} | {fused:.4f} | {sp:.2f}x |")
        lines.append("")
        lines.append(f"- max_abs_err = {info['precision']!s}")
        lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 报告已写: {path}")


def main():
    parser = argparse.ArgumentParser(description="融合算子对比 benchmark")
    parser.add_argument("--mode", choices=("both", "fused", "eager"), default="both",
                        help="both=正常路径, fused=强制全融合, eager=强制全 eager")
    parser.add_argument("--batch", type=int, default=None,
                        help="只测指定 batch（不跑所有档位）")
    parser.add_argument("--seq", type=int, default=None,
                        help="覆盖 default.json 的 seq_len")
    parser.add_argument("--report", type=str, default=None,
                        help="把结论写到这个 markdown 文件")
    args = parser.parse_args()

    global SEQ, BATCH_TIERS, REAL_BATCH
    if args.seq is not None:
        SEQ = args.seq
    if args.batch is not None:
        BATCH_TIERS = (args.batch,)

    run_benchmarks(mode=args.mode, report_path=args.report)


if __name__ == "__main__":
    main()
