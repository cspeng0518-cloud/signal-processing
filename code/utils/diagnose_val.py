"""val 链路诊断：定位微调 val 指标异常（acc 远低于随机水平）的环节。

用法（单卡，NPU/GPU 均可，无需 torchrun）:
  python code/utils/diagnose_val.py --data_dir datasets --weights_dir weights

诊断决策树（按输出顺序对照）:
  1) train/val 标签直方图 + id 完整性  -> 切分/meta 是否损坏（应接近均匀）
  2) TRAIN 子集上的 acc               -> 评估代码正确性（应接近训练精度；
                                          若也 ~0 说明评估代码有 bug，而非数据问题）
  3) VAL 上的 acc/预测直方图/逐类召回 -> 失败模式：常数预测 / 类别系统性错位 / 全错
  4) parameter 预测 vs 真值抽样        -> 回归头行为与 precision 量纲检查
"""

import sys
import argparse
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from models import EMFoundation
from training.common import ROOT, load_config
from utils.dataset import (DEFAULT_PARAM_STATS, ID2LABEL, collate_fn,
                           load_param_stats, make_task_dataset)
from utils.train_utils import get_device

TASKS = ("protocol", "modulation", "parameter")


def parse_args():
    p = argparse.ArgumentParser("val 链路诊断")
    p.add_argument("--data_dir", default=str(ROOT / "datasets"))
    p.add_argument("--weights_dir", default=str(ROOT / "weights"))
    p.add_argument("--config", default=str(CODE_ROOT / "configs" / "default.json"))
    p.add_argument("--n_sub", type=int, default=4096,
                   help="train/val 抽样评估的样本数（直方图仍统计全量）")
    return p.parse_args()


def underlying(ds):
    """剥离 Subset 取原始数据集与索引（memmap 布局时用于直读 meta）。"""
    if isinstance(ds, Subset):
        return ds.dataset, ds.indices
    return ds, range(len(ds))


def step1_split_integrity(data_dir):
    """标签直方图 + sample_id 唯一性 + val/test 交集检查。"""
    print("=" * 72)
    print("[step 1] 切分完整性：标签直方图（train vs val 应同分布且接近均匀）")
    for task in TASKS:
        for split in ("train", "val"):
            ds = make_task_dataset(data_dir, task, split=split)
            full, idxs = underlying(ds)
            if task == "parameter":
                vals = [full.meta["patterns"][i] for i in idxs]
                names = ["comm(0)", "radar(1)"]
            else:
                vals = [full.meta["labels"][i] for i in idxs]
                names = ID2LABEL[task]
            raw = {}
            for v in vals:
                raw[v] = raw.get(v, 0) + 1
            cnt = {names[int(k)] if task == "parameter" else k: c
                   for k, c in raw.items()}
            print(f"  {task:>10} {split:>5}: n={len(idxs):>7}  {cnt}")
        full, _ = underlying(make_task_dataset(data_dir, task, split="train"))
        ids = full.meta["sample_ids"]
        dup = len(ids) - len(set(ids))
        inter = (set(full.meta.get("val_sample_ids", []))
                 & set(full.meta.get("test_sample_ids", [])))
        print(f"  {task:>10} 检查: 重复sample_id={dup}  val∩test={len(inter)}")
    print()


def collect_preds(model, ds, task, device, n_sub, seed=0):
    """抽样子集上跑前向，返回 (y_true, y_pred) numpy 数组。"""
    if n_sub and n_sub < len(ds):
        idx = np.random.default_rng(seed).choice(len(ds), n_sub, replace=False)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=256, collate_fn=collate_fn)
    ys, ps = [], []
    with torch.inference_mode():
        for batch in loader:
            feat = model.encode(batch["iq"].to(device), batch["sr"].to(device))
            logits = model.task_heads[task](feat)
            ys.append(batch["label"].numpy())
            ps.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def step23_classification(model, data_dir, device, n_sub):
    """train/val 子集分类精度 + 预测直方图 + 逐类召回。"""
    for task in ("protocol", "modulation"):
        print("=" * 72)
        for split in ("train", "val"):
            ds = make_task_dataset(data_dir, task, split=split)
            y, p = collect_preds(model, ds, task, device, n_sub)
            names = ID2LABEL[task]
            acc = float((y == p).mean())
            pred_cnt = np.bincount(p, minlength=len(names))
            recalls = [f"{(p[y == c] == c).mean():.2f}" for c in range(len(names))]
            print(f"[{task} | {split}] n={len(y)} acc={acc:.4f}")
            print(f"  预测直方图: {dict(zip(names, pred_cnt.tolist()))}")
            print(f"  逐类召回:   {recalls}")
            if split == "val":
                pairs = list(zip(p[:10].tolist(), y[:10].tolist()))
                print(f"  前10 (pred, label): {pairs}")
        print()


def step4_parameter(model, data_dir, device, param_stats, n_sub):
    """parameter 预测抽样 + 归一化/原始两个量纲下的 rmse 与 precision。"""
    print("=" * 72)
    ds = make_task_dataset(data_dir, "parameter", split="val",
                           param_stats=param_stats)
    if n_sub and n_sub < len(ds):
        idx = np.random.default_rng(0).choice(len(ds), n_sub, replace=False)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=256, collate_fn=collate_fn)
    gts, preds, valids, pats, pat_preds = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            feat = model.encode(batch["iq"].to(device), batch["sr"].to(device))
            preds.append(model.task_heads["parameter"](feat).cpu().numpy())
            pat_preds.append(
                model.task_heads["parameter_pattern"](feat).argmax(-1).cpu().numpy())
            gts.append(batch["params"].numpy())
            valids.append(batch["valid"].numpy())
            pats.append(batch["pattern"].numpy())
    p, y = np.concatenate(preds), np.concatenate(gts)
    v, pat, pp = np.concatenate(valids), np.concatenate(pats), np.concatenate(pat_preds)
    mn, mx = np.asarray(param_stats["min"]), np.asarray(param_stats["max"])
    err2 = ((y - p) ** 2 * v).sum() / max(1, v.sum())
    print(f"[parameter | val] n={len(y)}  pattern_acc={float((pat == pp).mean()):.4f}")
    print(f"  归一化空间: rmse={np.sqrt(err2):.4f}  "
          f"precision(ref=0.5)={1.0 / (1.0 + np.sqrt(err2) / 0.5):.4f}")
    err_si = ((y - p) * (mx - mn) * v) ** 2
    rmse_si = np.sqrt(err_si.sum() / max(1, v.sum()))
    print(f"  原始SI空间: rmse={rmse_si:.2f}  "
          f"precision(ref=0.5)={1.0 / (1.0 + rmse_si / 0.5):.6f}  <- 当前实现所用")
    for i in range(5):
        print(f"  样本{i}: pred_norm={np.round(p[i], 3).tolist()}")
        print(f"          gt_norm  ={np.round(y[i], 3).tolist()}  valid={v[i].astype(int).tolist()}")
        print(f"          pred_SI  ={np.round(p[i] * (mx - mn) + mn, 2).tolist()}")


def main():
    args = parse_args()
    device = get_device()
    cfg = load_config(args.config)
    model = EMFoundation(cfg["model"]).to(device)
    state_path = Path(args.weights_dir) / "model.safetensors"
    model.load_state_dict(load_file(str(state_path)))
    model.eval()
    print(f"[diagnose] device={device} weights={state_path}\n")

    stats_file = Path(args.data_dir) / "param_stats.json"
    param_stats = (load_param_stats(str(stats_file)) if stats_file.exists()
                   else {"min": DEFAULT_PARAM_STATS["min"].copy(),
                         "max": DEFAULT_PARAM_STATS["max"].copy()})

    step1_split_integrity(args.data_dir)
    step23_classification(model, args.data_dir, device, args.n_sub)
    step4_parameter(model, args.data_dir, device, param_stats, args.n_sub)


if __name__ == "__main__":
    main()
