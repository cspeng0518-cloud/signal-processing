"""网格配置生成器：按 (d_model, n_layers, ffn 倍率, neck 层数) 枚举，精确统计参数量与
共享率（在 meta 设备上实例化 EMFoundation，不占显存/内存），过滤赛题红线后写 grid.json。

显存不再用公式估算（旧公式只在单点校准、系统性低估）——生成后请用实测工具筛选:
    python utils/gen_configs.py --d_model 512 768 1024 --n_layers 24 30 --neck 2 4
    python utils/measure_vram.py --grid code/configs/grid.json --budget 76
    python utils/run_grid.py --dry

约束（违者剔除）: 部署参数 < 1B；共享率 > 80%；d_model % nhead == 0（head_dim=64）。
lr 规则: lr_pretrain = 1.5e-4 × bs_pre/256（MAE 线性缩放），lr_finetune = 0.66 × lr_pretrain。
"""

import argparse
import copy
import itertools
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

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
    ap.add_argument("--template", default=os.path.join(ROOT, "code", "configs", "default.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "code", "configs", "grid.json"))
    ap.add_argument("--d_model", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--n_layers", type=int, nargs="+", default=[24, 30])
    ap.add_argument("--ffn_mult", type=float, nargs="+", default=[4.0])
    ap.add_argument("--neck", type=int, nargs="+", default=[2])
    ap.add_argument("--bs_pretrain", type=int, default=512)
    ap.add_argument("--bs_finetune", type=int, default=256)
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
        lr = round(1.5e-4 * args.bs_pretrain / 256, 6)
        t.update({"batch_size_pretrain": args.bs_pretrain, "batch_size": args.bs_finetune,
                  "lr_pretrain": lr, "lr_finetune": round(lr * 0.66, 6), "lr": round(lr * 0.66, 6)})
        total, shared = count_params(m)
        name = f"d{d}_l{L}_ffn{m['ffn_hidden']}_neck{nk}"
        ok = total < MAX_TOTAL_PARAMS and shared > MIN_SHARED_RATIO
        print(f"{'ok ' if ok else 'SKIP'} {name}: params={total / 1e6:.1f}M shared={shared:.3f}")
        if ok:
            configs.append({"name": name, "params_M": round(total / 1e6, 1),
                            "shared_ratio": round(shared, 4), **cfg})
    if args.dry:
        print(f"[dry] {len(configs)} configs")
        return
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"configs": configs}, f, indent=2, ensure_ascii=False)
    print(f"-> {args.out} ({len(configs)} configs)")


if __name__ == "__main__":
    main()
