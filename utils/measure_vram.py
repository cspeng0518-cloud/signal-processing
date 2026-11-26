"""显存实测工具：用合成数据真实跑"预训练 step + 微调 step（三任务）"，读取峰值显存。

为什么不再用公式估算: 旧 gen_configs.vram_estimate_gb 只在 d512/27L 一个点上校准，
且未计入反向峰值（约 +25%）与 bf16 autocast 下 LayerNorm/softmax 的 fp32 激活，
对宽配置系统性低估（详见 docs/gpu_memory_analysis.md）。
本脚本无需任何数据集，几分钟内给出真实数字。

用法:
  python utils/measure_vram.py --config src/configs/default.json            # 测当前配置
  python utils/measure_vram.py --config src/configs/default.json --find_max_bs
  python utils/measure_vram.py --grid src/configs/grid.json --limit 4        # 批量测 grid
  python utils/measure_vram.py --grid src/configs/grid.json --budget 76

输出: 每个配置 pretrain / finetune 两阶段的 max_allocated / max_reserved（GB），
      是否超过 --budget（默认 76GB，80G 卡留 4GB 余量），写入 logs/vram/report.json。
"""

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from models import build_model                                    # noqa: E402
from training.common import normalize_config                 # noqa: E402
from utils.optim import build_optimizer                      # noqa: E402
from utils.param_codec import ParamCodec                     # noqa: E402
from utils.train_utils import autocast_context               # noqa: E402

TASKS = ("protocol", "modulation", "parameter")


def _dev():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu"), torch.npu
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda
    return torch.device("cpu"), None


def synth_batch(task, B, N, device, codec=None):
    iq = torch.randn(B, N, 2, device=device)
    b = {"iq": iq}
    if task == "protocol":
        b["label"] = torch.randint(0, 8, (B,), device=device)
    elif task == "modulation":
        b["label"] = torch.randint(0, 23, (B,), device=device)
    else:
        b["pattern"] = torch.randint(0, 2, (B,), device=device)
        v = torch.zeros(B, 7, device=device)
        v[:, :3] = (b["pattern"] == 0).float()[:, None]
        v[:, 3:] = (b["pattern"] == 1).float()[:, None]
        b["valid"] = v
        lo = torch.tensor(codec.vmin, device=device).float()
        hi = torch.tensor(codec.vmax, device=device).float()
        b["raw_params"] = (lo + (hi - lo) * torch.rand(B, 7, device=device)) * v
    return b


def measure(cfg, bs, steps=2, stage="both"):
    """真实跑 steps 个预训练步 / 微调步（三任务梯度累加），返回峰值显存。"""
    device, mod = _dev()
    cfg = normalize_config(json.loads(json.dumps(cfg)))
    mcfg, tcfg = cfg["model"], dict(cfg["training"])
    N = mcfg.get("seq_len", tcfg.get("seq_len", 1024))
    codec = ParamCodec.default()
    mcfg["param_head_spec"] = codec.head_spec()
    res = {"batch_size": bs}

    def peak(fn):
        if mod is not None:
            mod.empty_cache()
            mod.reset_peak_memory_stats()
        fn()
        if mod is not None:
            mod.synchronize()
            return mod.max_memory_allocated() / 1e9, mod.max_memory_reserved() / 1e9
        return float("nan"), float("nan")

    try:
        if stage in ("both", "pretrain"):
            model = build_model(mcfg, with_decoder=True).to(device).train()
            res["params_M"] = model.deploy_parameter_count() / 1e6
            res["shared_ratio"] = model.shared_ratio()
            named = [(n, p) for n, p in model.named_parameters()
                     if not n.startswith(("task_heads.", "neck."))]
            opt = build_optimizer(named, tcfg)
            bsp = tcfg.get("batch_size_pretrain", bs) if stage == "both" else bs

            def pre():
                for _ in range(steps):
                    iq = torch.randn(bsp, N, 2, device=device)
                    with autocast_context(device, True):
                        loss, _ = model.forward_pretrain(iq)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            res["pretrain_bs"] = bsp
            res["pretrain_alloc_gb"], res["pretrain_reserved_gb"] = peak(pre)
            del opt, model
            gc.collect()
        if stage in ("both", "finetune"):
            from training.tasks import TASKS
            model = build_model(mcfg, with_decoder=False).to(device).train()
            model.codec = codec
            res.setdefault("params_M", model.deploy_parameter_count() / 1e6)
            opt = build_optimizer(list(model.named_parameters()), tcfg)

            def ft():
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    for t in TASKS:
                        b = synth_batch(t, bs, N, device, codec)
                        with autocast_context(device, True):
                            loss, _ = TASKS[t].loss(model, b, tcfg)
                        loss.backward()
                    opt.step()
            res["finetune_alloc_gb"], res["finetune_reserved_gb"] = peak(ft)
            del opt, model
        res["status"] = "ok"
    except RuntimeError as e:
        res["status"] = "OOM" if "out of memory" in str(e).lower() else f"error: {e}"[:200]
    finally:
        gc.collect()
        if mod is not None:
            mod.empty_cache()
    return res


def find_max_bs(cfg, budget, stage, start=64, stop=4096):
    """单阶段二分倍增找预算内最大 batch（pretrain 与 finetune 分别找）。"""
    best, bs = None, start
    key = f"{stage}_reserved_gb"
    while bs <= stop:
        r = measure(cfg, bs, steps=1, stage=stage)
        peak = r.get(key) or 0
        print(f"  [{stage}] bs={bs}: {r.get('status')} peak_reserved={peak:.1f}GB")
        if r["status"] != "ok" or peak > budget:
            break
        best, bs = bs, bs * 2
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=0, help="覆盖 batch_size（0=用配置值）")
    ap.add_argument("--budget", type=float, default=76.0)
    ap.add_argument("--find_max_bs", action="store_true")
    args = ap.parse_args()
    items = []
    if args.grid:
        g = json.load(open(args.grid, encoding="utf-8"))
        items = [(c.get("name", f"cfg{i}"), {"model": c["model"], "training": c["training"]})
                 for i, c in enumerate(g["configs"])]
        if args.limit:
            items = items[: args.limit]
    else:
        path = args.config or str(ROOT / "src" / "configs" / "default.json")
        items = [(Path(path).stem, json.load(open(path, encoding="utf-8")))]
    report = []
    for name, cfg in items:
        cfg = normalize_config(cfg)
        bs = args.bs or cfg["training"]["batch_size"]
        print(f"== {name} (bs={bs})")
        r = measure(cfg, bs)
        r["name"] = name
        peak = max(r.get("pretrain_reserved_gb", 0) or 0, r.get("finetune_reserved_gb", 0) or 0)
        r["fits_budget"] = r["status"] == "ok" and peak <= args.budget
        print("   " + json.dumps(r, ensure_ascii=False))
        if args.find_max_bs:
            for st in ("pretrain", "finetune"):
                r[f"max_bs_{st}"] = find_max_bs(cfg, args.budget, st)
                print(f"   [{st}] max batch size within {args.budget}GB: {r[f'max_bs_{st}']}")
        report.append(r)
    out = ROOT / "logs" / "vram"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"report -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
