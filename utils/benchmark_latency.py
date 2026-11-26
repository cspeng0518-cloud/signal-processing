"""推理时延基准（复赛"计算效率提升度"按推理时延排名，占自动评分 30 分）

用法:
  python utils/benchmark_latency.py --weights_dir weights                # 测已训练权重
  python utils/benchmark_latency.py --config src/configs/default.json   # 只测结构（随机权重）

对每种 (精度 × 是否 compile × batch × 长度) 组合报告 P50/P95 单 batch 时延、
每样本时延与吞吐，结果写 logs/benchmark/latency.json，可直接贴进技术报告。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from models import build_model                      # noqa: E402
from training.common import load_config              # noqa: E402
from utils.param_codec import ParamCodec             # noqa: E402
from utils.train_utils import get_device             # noqa: E402


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "npu":
        torch.npu.synchronize()


def build(args_cfg, weights_dir, device):
    if weights_dir:
        from infer import load_model
        return load_model(weights_dir, device)
    cfg = load_config(args_cfg)
    cfg["model"]["param_head_spec"] = ParamCodec.default().head_spec()
    model = build_model(cfg["model"], with_decoder=False).to(device).eval()
    model.codec = ParamCodec.default()
    return model, cfg


@torch.no_grad()
def bench(model, device, B, L, task, precision, iters, warmup):
    iq = torch.zeros(B, 1024, 2, device=device)
    iq[:, :L] = torch.randn(B, L, 2, device=device)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    ctx = (lambda: torch.autocast(device.type, dtype=dt)) if dt and device.type != "cpu" else \
        (lambda: torch.autocast(device.type, enabled=False))

    def run():
        with ctx():
            return model.predict(iq, task)

    for _ in range(warmup):
        run()
    _sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    p50, p95 = statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]
    return {"batch": B, "length": L, "task": task, "precision": precision,
            "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
            "per_sample_us": round(1000 * p50 / B, 2), "throughput": round(B / p50 * 1000, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "src" / "configs" / "default.json"))
    ap.add_argument("--compare", default=None, help="再测一个配置做对比")
    ap.add_argument("--weights_dir", default=None)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256])
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--precisions", nargs="+", default=["fp32", "bf16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    device = get_device()
    report = []
    for cfg_path in [args.config] + ([args.compare] if args.compare else []):
        model, cfg = build(cfg_path, args.weights_dir if cfg_path == args.config else None,
                           device)
        if args.compile and device.type == "cuda":
            model.backbone = torch.compile(model.backbone, dynamic=True)
        n = model.deploy_parameter_count() / 1e6
        print(f"== {cfg_path} params={n:.2f}M device={device}")
        for prec in args.precisions:
            for B in args.batches:
                for L in args.lengths:
                    r = bench(model, device, B, L, "modulation", prec, args.iters,
                              args.warmup)
                    r.update({"config": Path(cfg_path).name, "params_M": round(n, 2),
                              "compile": bool(args.compile)})
                    print("  ", r)
                    report.append(r)
        del model
    out = ROOT / "logs" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {out / 'latency.json'}")


if __name__ == "__main__":
    main()
