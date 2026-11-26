"""网格配置生成器：按 (d_model, n_layers, ffn 倍率, neck 层数) 枚举，精确统计参数量与
共享率（在 meta 设备上实例化 EMFoundation，不占显存/内存），过滤赛题红线后写 grid.json。

显存不再用公式估算（旧公式只在单点校准、系统性低估）——生成后请用实测工具筛选:
    python utils/gen_configs.py --d_model 512 768 1024 --n_layers 24 30 --neck 2 4
    python utils/measure_vram.py --grid src/configs/grid.json --budget 76
    python utils/run_grid.py --dry

约束（违者剔除）:
  部署参数 < 1B；共享率 > 80%；d_model % nhead == 0（head_dim=64）。
lr 规则: lr_pretrain = 1.5e-4 × bs_pre/256（MAE 线性缩放），lr_finetune = 0.66 × lr_pretrain。

常用最小规模:
  d_model=512 n_layers=24 neck=2  -> ~126M   grad_checkpoint=false
  d_model=768 n_layers=24 neck=2  -> ~275M   grad_checkpoint=false
  d_model=768 n_layers=30 neck=2  -> ~357M   grad_checkpoint=false  <-- 首个 ≥300M
  d_model=1024 n_layers=24 neck=2 -> ~481M   grad_checkpoint=true
  d_model=1024 n_layers=30 neck=4 -> ~638M   grad_checkpoint=true
"""

import argparse
import copy
import itertools
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

MAX_TOTAL_PARAMS = 1_000_000_000
MIN_SHARED_RATIO = 0.80


def count_params(model_cfg):
    """meta 设备上实例化模型（失败时退回 CPU），返回 (部署参数量, 共享率)。"""
    import torch
    from models import build_model
    try:
        with torch.device("meta"):
            m = build_model(model_cfg, with_decoder=False)
    except Exception:
        m = build_model(model_cfg, with_decoder=False)
    return m.deploy_parameter_count(), m.shared_ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=os.path.join(ROOT, "src", "configs", "default.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "src", "configs", "grid.json"))
    ap.add_argument("--d_model", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--n_layers", type=int, nargs="+", default=[24, 30])
    ap.add_argument("--ffn_mult", type=float, nargs="+", default=[4.0])
    ap.add_argument("--neck", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--bs_pretrain", type=int, default=512)
    ap.add_argument("--bs_finetune", type=int, default=256)
    ap.add_argument("--min_params", type=float, default=0,
                    help="最小部署参数量（百万），低于则剔除。默认 0 不过滤。")
    ap.add_argument("--max_configs", type=int, default=20,
                    help="枚举后按参数量降序取前 N 组。默认 20。")
    ap.add_argument("--auto_grad_checkpoint", type=int, default=0,
                    help="d_model >= 此值自动开启 grad_checkpoint。0=关闭（默认）。")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    base = json.load(open(args.template, encoding="utf-8"))
    configs = []
    for d, L, fm, nk in itertools.product(args.d_model, args.n_layers, args.ffn_mult, args.neck):
        if d % 64:
            continue
        cfg = copy.deepcopy(base)
        m, t = cfg["model"], cfg["training"]
        m.update({"d_model": d, "n_layers": L, "nhead": d // 64,
                  "ffn_hidden": int(round(d * fm / 64) * 64), "neck_layers": nk})
        if args.auto_grad_checkpoint and d >= args.auto_grad_checkpoint:
            m["grad_checkpoint"] = True
        lr = round(1.5e-4 * args.bs_pretrain / 256, 6)
        t.update({"batch_size_pretrain": args.bs_pretrain, "batch_size": args.bs_finetune,
                  "lr_pretrain": lr, "lr_finetune": round(lr * 0.66, 6), "lr": round(lr * 0.66, 6)})
        total, shared = count_params(m)
        name = f"d{d}_l{L}_ffn{m['ffn_hidden']}_neck{nk}"
        if m.get("grad_checkpoint"):
            name += "_ckpt"
        if total >= MAX_TOTAL_PARAMS:
            print(f"SKIP {name}: params={total / 1e6:.1f}M ≥ 1B")
            continue
        if shared <= MIN_SHARED_RATIO:
            print(f"SKIP {name}: shared={shared:.3f} ≤ 80%")
            continue
        if args.min_params and total / 1e6 < args.min_params:
            print(f"SKIP {name}: params={total / 1e6:.1f}M < min={args.min_params}M")
            continue
        print(f"ok   {name}: params={total / 1e6:.1f}M shared={shared:.3f}")
        configs.append({"name": name, "params_M": round(total / 1e6, 1),
                        "shared_ratio": round(shared, 4), **cfg})
    # 按参数量降序取前 N
    configs.sort(key=lambda c: c["params_M"], reverse=True)
    if len(configs) > args.max_configs:
        configs = configs[:args.max_configs]
        print(f"[top-{args.max_configs}] 截取后剩余 {len(configs)} 组")
    if args.dry:
        print(f"[dry] {len(configs)} configs (top-{args.max_configs})")
        for c in configs:
            print(f"  {c['name']}: {c['params_M']}M")
        return
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"configs": configs}, f, indent=2, ensure_ascii=False)
    print(f"-> {args.out} ({len(configs)} configs)")


if __name__ == "__main__":
    main()
