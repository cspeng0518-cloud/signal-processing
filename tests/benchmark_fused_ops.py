"""融合算子 vs 原生算子 对比 benchmark。

- 读 default.json 拿训练真实 batch / d_model / ffn_hidden 等
- 分多档 batch 测延迟 + 精度 diff
- 默认只打印建议，不写回 default.json
- --apply 时把建议值写进 default.json 的 operator 字段

用法:
    python tests/benchmark_fused_ops.py                  # 测 + 打印建议
    python tests/benchmark_fused_ops.py --apply          # 测 + 自动写回 default.json
    python tests/benchmark_fused_ops.py --batch 512      # 只测训练真实 batch
    python tests/benchmark_fused_ops.py --report out.md  # 额外写 markdown 报告

临时强制全 eager 跑：
    EM_FUSED_OPS=0 python tests/benchmark_fused_ops.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from code.models.ops import (  # noqa: E402
    attention, attention_reference,
    swiglu, swiglu_reference,
    rmsnorm, rmsnorm_reference,
    rmsnorm_add, rmsnorm_add_reference,
    rope_rotate, rope_rotate_reference,
    fused_status,
)

# --------------------------------------------------------------------------- 配置

_CONFIG_PATH = _ROOT / "code" / "configs" / "default.json"


def _load_default():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


SPEC = _load_default()
MODEL = SPEC.get("model", {})
TRAINING = SPEC.get("training", {})

D = int(MODEL.get("d_model", 512))
NH = int(MODEL.get("nhead", 8))
DH = D // NH
FFN_H = int(MODEL.get("ffn_hidden", D * 4))
SEQ_RAW = int(MODEL.get("seq_len", 1024))
SEQ = max(1, SEQ_RAW // int(MODEL.get("patch_size", 8)))
REAL_BATCH = int(TRAINING.get("batch_size", 512))

BATCH_TIERS = (4, 32, 128, REAL_BATCH)

# --------------------------------------------------------------------------- 设备

_HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
_HAS_CUDA = torch.cuda.is_available()
_NPU_SYNC = getattr(torch.npu, "synchronize", None) if _HAS_NPU else None


def _dev():
    return "npu" if _HAS_NPU else ("cuda" if _HAS_CUDA else "cpu")


def _sync():
    if _NPU_SYNC is not None:
        _NPU_SYNC()
    elif _HAS_CUDA:
        torch.cuda.synchronize()


def _bench(fn, warmup=20, repeat=100):
    _sync()
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync()
    return (time.perf_counter() - t0) / repeat * 1000.0


# --------------------------------------------------------------------------- 数据 + 基准

def _t(shape, dev, dtype=torch.float32):
    return torch.randn(shape, device=dev, dtype=dtype)


def _rope_inputs(dev):
    x = _t((REAL_BATCH, SEQ, NH, 2 * DH), dev)
    # 构造 cos/sin 最后一维 = 2*DH（完整 head_dim，对齐 fused.py rope_rotate_reference）
    half = 2 * DH // 2   # = DH
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2, device=dev).float() / half))
    freqs = torch.outer(torch.arange(SEQ, device=dev).float(), inv_freq)  # [SEQ, half/2]
    emb = torch.cat([freqs, freqs], dim=-1)                                # [SEQ, half] → 不够
    emb = emb.repeat_interleave(2, dim=-1)[:SEQ, :2 * DH]                 # 扩展到 [SEQ, 2*DH]
    return x, emb.cos(), emb.sin()


def bench_rmsnorm(batch, dev):
    x, w = _t((batch, SEQ, D), dev), torch.ones(D, device=dev)
    return (_bench(lambda: rmsnorm_reference(x, w, 1e-6)),
            _bench(lambda: rmsnorm(x, w, 1e-6)))


def bench_rmsnorm_add(batch, dev):
    x, r, w = _t((batch, SEQ, D), dev), _t((batch, SEQ, D), dev), torch.ones(D, device=dev)
    return (_bench(lambda: rmsnorm_add_reference(x, r, w, 1e-6)),
            _bench(lambda: rmsnorm_add(x, r, w, 1e-6)))


def bench_swiglu(batch, dev):
    gate, up = _t((batch, SEQ, FFN_H), dev), _t((batch, SEQ, FFN_H), dev)
    return (_bench(lambda: swiglu_reference(gate, up)),
            _bench(lambda: swiglu(gate, up)))


def bench_attention(batch, dev):
    q = _t((batch, NH, SEQ, DH), dev)
    return (_bench(lambda: attention_reference(q, q, q)),
            _bench(lambda: attention(q, q, q)))


def bench_rope(batch, dev):
    x, cos, sin = _rope_inputs(dev)
    x = x[:batch]
    return (_bench(lambda: rope_rotate_reference(x, cos, sin)),
            _bench(lambda: rope_rotate(x, cos, sin)))


# --------------------------------------------------------------------------- 精度

def _max_err(got, ref):
    return float((got.float() - ref.float()).abs().max())


def check_rmsnorm(dev):
    x, w = _t((REAL_BATCH, SEQ, D), dev), torch.ones(D, device=dev)
    return _max_err(rmsnorm(x, w, 1e-6), rmsnorm_reference(x, w, 1e-6))


def check_swiglu(dev):
    g, u = _t((REAL_BATCH, SEQ, FFN_H), dev), _t((REAL_BATCH, SEQ, FFN_H), dev)
    return _max_err(swiglu(g, u), swiglu_reference(g, u))


def check_attention(dev):
    q = _t((REAL_BATCH, NH, SEQ, DH), dev)
    return _max_err(attention(q, q, q), attention_reference(q, q, q))


def check_rope(dev):
    x, cos, sin = _rope_inputs(dev)
    return _max_err(rope_rotate(x, cos, sin), rope_rotate_reference(x, cos, sin))


# --------------------------------------------------------------------------- 主流程

def run(batch_tiers, apply=False, report_path=None):
    dev = _dev()
    cfg = _load_default()
    current_op = cfg.get("operator", {})

    print(f"\n{'='*60}")
    print(f"  设备: {dev}")
    print(f"  batch 档位: {batch_tiers}  训练 batch: {REAL_BATCH}")
    print(f"  seq={SEQ}  d_model={D}  nhead={NH}  ffn_hidden={FFN_H}")
    print(f"  current operator: {current_op}")
    print(f"{'='*60}\n")

    ops = [
        ("attention",   bench_attention,   check_attention),
        ("swiglu",      bench_swiglu,      check_swiglu),
        ("rmsnorm",     bench_rmsnorm,     check_rmsnorm),
        ("rmsnorm_add", bench_rmsnorm_add, check_rmsnorm),
        ("rope_rotate", bench_rope,        check_rope),
    ]

    results = {}   # name -> {"latency": [(batch, eager, fused, speedup)], "max_err": float}

    for name, bench_fn, check_fn in ops:
        print(f"--- {name} ---")
        rows = []
        for b in batch_tiers:
            try:
                eager_ms, fused_ms = bench_fn(b, dev)
                sp = eager_ms / fused_ms if fused_ms > 0 else float("inf")
            except Exception as e:
                print(f"  batch={b}: ERROR {type(e).__name__}: {e}")
                rows.append((b, None, None, None))
                continue
            tag = " ← 训练" if b == REAL_BATCH else ""
            rows.append((b, eager_ms, fused_ms, sp))
            print(f"  batch={b:4d}{tag}:  eager={eager_ms:.4f}  fused={fused_ms:.4f}  speedup={sp:.2f}x")
        try:
            err = check_fn(dev)
            print(f"  精度: max_abs_err = {err:.3e}")
        except Exception as e:
            err = None
            print(f"  精度: ERROR {e}")
        results[name] = {"latency": rows, "max_err": err}
        print()

    # ---- 结论 ----
    print("=" * 60)
    print("  结论（训练 batch 下）")
    print("=" * 60)
    suggestions = {}   # name -> bool  (True=保留融合, False=关融合走 eager)
    for name, info in results.items():
        # 训练 batch 那档
        row = next((r for r in info["latency"] if r[0] == REAL_BATCH), info["latency"][-1])
        b, eager, fused, sp = row
        err = info["max_err"]
        err_ok = err is not None and err < 1e-2
        status = None
        for k, v in fused_status().items():
            if name.split("_")[0] in k.lower() or name.lower() in k.lower():
                status = v
                break

        if fused is None or eager is None:
            keep = current_op.get(name, True)
            rec = f"— (测试失败, 保持 {keep})"
            print(f"  {name:12s}  (测试失败)  → {rec}")
            suggestions[name] = keep
            continue
        err_ok = err is not None and err < 1e-2
        if sp >= 1.2 and err_ok:
            keep = True
            rec = "✅ 保留融合"
        elif sp < 1.0 and err_ok:
            keep = False
            rec = "❌ 融合路径慢, 建议关闭"
        else:
            keep = current_op.get(name, True)
            rec = f"🟡 收益不明显, 保持 {keep}"
        suggestions[name] = keep

        print(f"  {name:12s}  batch={b:4d}  eager={eager:.4f}  fused={fused:.4f}  "
              f"speedup={sp:.2f}x  err={err!s}")
        print(f"              status: {status or '(未触发)'}  → {rec}")
    print()

    # ---- 写回 default.json ----
    if apply:
        cfg.setdefault("operator", {})
        changed = False
        for name, keep in suggestions.items():
            if cfg["operator"].get(name, True) != keep:
                cfg["operator"][name] = keep
                changed = True
        if changed:
            with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(f"✅ 已写回 {_CONFIG_PATH}:")
            print(f"   operator: {cfg['operator']}")
        else:
            print("(default.json 的 operator 已经是最优值, 无需改动)")

    # ---- 写 markdown ----
    if report_path:
        _write_report(report_path, results, suggestions, dev, REAL_BATCH, current_op)

    return results, suggestions


def _write_report(path, results, suggestions, dev, real_batch, current_op):
    lines = ["# 融合算子对比报告", "", f"- 设备: **{dev}**",
             f"- 训练 batch: **{real_batch}**  seq: **{SEQ}**  d_model: **{D}**",
             f"- 当前 operator 配置: `{json.dumps(current_op, ensure_ascii=False)}`", ""]
    lines.append("## 结论摘要（建议写入 default.json）")
    lines.append("")
    lines.append("| 算子 | 训练 batch speedup | max_err | 建议融合 |")
    lines.append("|------|-------------------|---------|---------|")
    for name, info in results.items():
        row = next((r for r in info["latency"] if r[0] == real_batch), info["latency"][-1])
        sp = f"{row[3]:.2f}x" if row[3] is not None else "—"
        err = f"{info['max_err']:.3e}" if info["max_err"] is not None else "—"
        lines.append(f"| {name} | {sp} | {err} | **{suggestions[name]}** |")
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 报告已写: {path}")


def main():
    parser = argparse.ArgumentParser(description="融合算子对比 benchmark")
    parser.add_argument("--apply", action="store_true",
                        help="把建议值写回 default.json 的 operator 字段")
    parser.add_argument("--batch", type=int, default=None,
                        help="只测指定 batch（不跑所有档位）")
    parser.add_argument("--report", type=str, default=None,
                        help="额外写 markdown 报告")
    args = parser.parse_args()

    tiers = (args.batch,) if args.batch else BATCH_TIERS
    run(batch_tiers=tiers, apply=args.apply, report_path=args.report)


if __name__ == "__main__":
    main()
