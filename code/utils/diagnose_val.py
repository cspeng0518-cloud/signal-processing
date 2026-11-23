"""val 链路诊断：定位微调 val 指标异常的环节（单卡，NPU/GPU/CPU 均可）。

用法:
  python code/utils/diagnose_val.py --data_dir datasets --weights_dir weights [--n_sub 4096]

诊断决策树（按输出顺序对照）:
  1) train/val 标签直方图             -> 切分/meta 是否损坏（应接近均匀）
  2) TRAIN 子集 acc                  -> 评估代码正确性（应接近训练精度；也很低说明评估链路有 bug）
  3) VAL acc / 预测直方图 / 逐类召回   -> 失败模式：常数预测 / 类别系统性错位 / 全错
  4) parameter: pattern acc + 每参数 RMSE + 预测/真值抽样 -> 回归头与编解码量纲检查
（调制/体制的混淆矩阵、SNR 分桶等细粒度分析见 bad_case_analysis.py）
"""

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from torch.utils.data import DataLoader, Subset                       # noqa: E402

from training.common import ROOT                                      # noqa: E402
from training.tasks import TASKS                                      # noqa: E402
from utils.dataset import ID2LABEL, PARAM_NAMES, collate_fn, make_task_dataset  # noqa: E402
from utils.train_utils import autocast_context, get_device            # noqa: E402


def _subset(ds, n, seed=0):
    if len(ds) <= n:
        return ds
    return Subset(ds, np.random.default_rng(seed).choice(len(ds), n, replace=False).tolist())


def _loader(ds, bs=512):
    return DataLoader(ds, batch_size=bs, collate_fn=collate_fn, num_workers=2)


@torch.no_grad()
def classification(model, data_dir, device, n_sub, seq_len):
    for task in ("protocol", "modulation"):
        names = ID2LABEL[task]
        print(f"\n===== {task} =====")
        for split in ("train", "val"):
            try:
                ds = _subset(make_task_dataset(data_dir, task, split, seq_len), n_sub)
            except FileNotFoundError:
                print(f"  [{split}] missing")
                continue
            ys, ps = [], []
            with autocast_context(device, True):
                for b in _loader(ds):
                    ps.append(model.predict(b["iq"].to(device), task).cpu().numpy())
                    ys.append(b["label"].numpy())
            y, p = np.concatenate(ys), np.concatenate(ps)
            C = len(names)
            print(f"  [{split}] n={len(y)} acc={float((y == p).mean()):.4f}")
            print(f"    label hist: {np.bincount(y, minlength=C).tolist()}")
            print(f"    pred  hist: {np.bincount(p, minlength=C).tolist()}")
            if split == "val":
                for c in range(C):
                    m = y == c
                    if m.any():
                        print(f"    recall {names[c]:>28}: {float((p[m] == c).mean()):.3f}")


@torch.no_grad()
def parameter(model, data_dir, device, n_sub, seq_len):
    print("\n===== parameter (val) =====")
    try:
        ds = _subset(make_task_dataset(data_dir, "parameter", "val", seq_len), n_sub)
    except FileNotFoundError:
        print("  missing")
        return
    P, Y, V, PT, PP = [], [], [], [], []
    with autocast_context(device, True):
        for b in _loader(ds):
            pred, radar = TASKS["parameter"].predict_batch(model, b["iq"].to(device))
            P.append(pred.float().cpu().numpy())
            PP.append(radar.cpu().numpy())
            Y.append(b["raw_params"].numpy())
            V.append(b["valid"].numpy().astype(bool))
            PT.append(b["pattern"].numpy())
    p, y, v = np.concatenate(P), np.concatenate(Y), np.concatenate(V)
    print(f"  pattern acc = {float((np.concatenate(PT) == np.concatenate(PP)).mean()):.4f}")
    rng = model.codec.vmax - model.codec.vmin
    for j, n in enumerate(PARAM_NAMES):
        m = v[:, j]
        if m.any():
            r = float(np.sqrt(np.mean((p[m, j] - y[m, j]) ** 2)))
            print(f"  {n:>17} ({model.codec.modes[j]:<10}) rmse={r:.4g} nrmse={r / rng[j]:.4f}")
    print("  samples (pred | true):")
    for i in range(min(5, len(p))):
        print("   ", np.round(p[i], 7).tolist(), "|", np.round(y[i], 7).tolist())


def main():
    ap = argparse.ArgumentParser("val 链路诊断")
    ap.add_argument("--data_dir", default=str(ROOT / "datasets"))
    ap.add_argument("--weights_dir", default=str(ROOT / "weights"))
    ap.add_argument("--n_sub", type=int, default=4096)
    args = ap.parse_args()
    from infer import load_model
    device = get_device()
    model, cfg = load_model(args.weights_dir, device)
    seq_len = cfg["model"].get("seq_len", 1024)
    classification(model, args.data_dir, device, args.n_sub, seq_len)
    parameter(model, args.data_dir, device, args.n_sub, seq_len)


if __name__ == "__main__":
    main()
