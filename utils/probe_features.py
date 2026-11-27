"""前端特征可分性探针：不训练 Transformer，先量化"新特征本身能把各类分到什么程度"。

动机：bad case 分析显示 14 类数字通信块内错误近乎均匀（块内熵 0.91~0.99），说明大量样本
在当前表示里缺少可分信息。加特征之前/之后都应先回答："特征本身的上限是多少？"——
本脚本只跑无参数的 SignalFrontEnd，把输出压成扁平向量，用梯度提升树做分类，
按特征组对比准确率（组间差值 ≈ 该组特征的边际信息量）。几分钟出结果，不占 GPU 训练时间。

用法:
  python utils/probe_features.py --task modulation --n_per_class 1500
  python utils/probe_features.py --task modulation --n_per_class 3000 --device cuda --groups base16 all

输出到 logs/probe/<task>/:
  summary.md           各特征组的 acc / macro recall，及按"观测符号数"分桶的 acc
  per_class.csv        每类在各特征组下的召回
依赖：scikit-learn（仅分析用，不进推理镜像）。
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402
from torch.utils.data import DataLoader, Subset                       # noqa: E402

from models.dsp import STAT_NAMES, SignalFrontEnd                     # noqa: E402
from training.common import load_config                              # noqa: E402
from utils.dataset import ID2LABEL, collate_fn, make_task_dataset     # noqa: E402


def build_front(model_cfg):
    """按模型配置构建前端（与 EMFoundation 内的前端参数一致；前端无可学习参数）。"""
    c = model_cfg
    from models.dsp import DEFAULT_CAF_LAGS, DEFAULT_CCAF_LAGS, DEFAULT_DECHIRP_RATES
    seq_len = c.get("seq_len", 1024)
    return SignalFrontEnd(
        seq_len, c.get("patch_size", 8), c.get("stft_win", 32), c.get("acf_lags", seq_len // 2),
        const_bins=c.get("const_bins", 64), const_range=c.get("const_range", 2.5),
        dft_impl=c.get("dft_impl", "matmul"), length_quantum=c.get("length_quantum", 512),
        caf_lags=c.get("caf_lags", DEFAULT_CAF_LAGS), ccaf_lags=c.get("ccaf_lags", DEFAULT_CCAF_LAGS),
        dechirp_rates=c.get("dechirp_rates", DEFAULT_DECHIRP_RATES)
        if c.get("use_dechirp", True) else (),
        dechirp_pool=c.get("dechirp_pool", 2), hist_bins=c.get("hist_bins", 64),
        dcmp_lag=c.get("dcmp_lag", 4), if_smooth=c.get("if_smooth", 5),
        min_symbols=c.get("min_symbols", 4), cyc_whiten=c.get("cyc_whiten", 31),
        fine_cfo_bins=c.get("fine_cfo_bins", 12.0))


def _pool_last(t, n):
    """[B,...,M] -> 沿最后一维自适应均值池化到 n。"""
    shp = t.shape
    return F.adaptive_avg_pool1d(t.reshape(-1, 1, shp[-1]), n).reshape(*shp[:-1], n)


@torch.no_grad()
def extract(front, ds, device, bs=256):
    """返回 {组名: [N, D] 特征}、标签、观测符号数。"""
    loader = DataLoader(ds, batch_size=bs, collate_fn=collate_fn, num_workers=2)
    feats = {k: [] for k in ("stats", "hist", "const", "caf", "spec", "dechirp", "acf")}
    ys, nsym = [], []
    i_nsym = STAT_NAMES.index("log_nsym")
    for b in loader:
        fe = front(b["iq"].to(device))
        B = fe["stats"].size(0)
        feats["stats"].append(fe["stats"])
        feats["hist"].append(_pool_last(fe["hist"], 32).reshape(B, -1))
        G = fe["const"].size(-1)
        feats["const"].append(F.adaptive_avg_pool2d(fe["const"], max(4, G // 8)).reshape(B, -1))
        feats["caf"].append(_pool_last(fe["caf"], 64).reshape(B, -1))
        feats["spec"].append(_pool_last(fe["spec"], 64).reshape(B, -1))
        if fe["dechirp"] is not None:
            feats["dechirp"].append(fe["dechirp"].amax(-1))                    # 每个调频率的峰值
        feats["acf"].append(_pool_last(fe["acf"], 32).reshape(B, -1))
        v = fe["stats"][:, i_nsym]
        nsym.append((torch.pow(10.0, torch.sign(v) * torch.expm1(v.abs())) - 1).cpu())
        ys.append(b["label"])
    out = {k: torch.cat(v).float().cpu().numpy() for k, v in feats.items() if v}
    return out, torch.cat(ys).numpy(), torch.cat(nsym).numpy()


GROUPS = {
    "base16": lambda f: f["stats"][:, :16],                               # 旧版 16 维统计量
    "stats40": lambda f: f["stats"],                                      # 新版全部统计量
    "stats+hist": lambda f: np.hstack([f["stats"], f["hist"]]),
    "stats+hist+const": lambda f: np.hstack([f["stats"], f["hist"], f["const"]]),
    "all": lambda f: np.hstack([f[k] for k in ("stats", "hist", "const", "caf", "spec",
                                               "dechirp", "acf") if k in f]),
}


def main():
    ap = argparse.ArgumentParser("前端特征可分性探针")
    ap.add_argument("--task", default="modulation", choices=["modulation", "protocol"])
    ap.add_argument("--config", default=str(ROOT / "src" / "configs" / "default.json"))
    ap.add_argument("--data_dir", default=str(ROOT / "datasets"))
    ap.add_argument("--n_per_class", type=int, default=1500, help="每类抽样数（train 切分）")
    ap.add_argument("--n_val", type=int, default=6000, help="评估样本数（val 切分）")
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import recall_score

    cfg = load_config(args.config)
    device = torch.device(args.device)
    front = build_front(cfg["model"]).to(device).eval()
    seq_len = cfg["training"].get("seq_len", 1024)
    names = ID2LABEL[args.task]
    rng = np.random.default_rng(0)

    tr = make_task_dataset(args.data_dir, args.task, "train", seq_len=seq_len)
    idx = rng.permutation(len(tr))[: args.n_per_class * len(names)]
    va = make_task_dataset(args.data_dir, args.task, "val", seq_len=seq_len)
    vidx = rng.permutation(len(va))[: args.n_val]
    print(f"[probe] 提取特征: train {len(idx)} / val {len(vidx)}, device={device}")
    ftr, ytr, _ = extract(front, Subset(tr, idx.tolist()), device)
    fva, yva, nsym = extract(front, Subset(va, vidx.tolist()), device)

    out = Path(args.out or ROOT / "logs" / "probe" / args.task)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# 前端特征可分性探针（{args.task}，train {len(ytr)} / val {len(yva)}）", "",
             "| 特征组 | 维度 | acc | macro recall |", "|---|---|---|---|"]
    per_class, preds = {}, {}
    for g in args.groups:
        Xtr, Xva = GROUPS[g](ftr), GROUPS[g](fva)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                             early_stopping=True, random_state=0)
        clf.fit(Xtr, ytr)
        p = clf.predict(Xva)
        preds[g] = p
        acc = float((p == yva).mean())
        rec = recall_score(yva, p, labels=list(range(len(names))), average=None, zero_division=0)
        per_class[g] = rec
        lines.append(f"| {g} | {Xtr.shape[1]} | {acc:.4f} | {rec.mean():.4f} |")
        print(lines[-1])
    lines += ["", "## 准确率 vs 观测窗内符号数（最后一个特征组）", ""]
    g = args.groups[-1]
    for lo, hi in zip([0, 16, 32, 64, 128, 256, 512], [16, 32, 64, 128, 256, 512, np.inf]):
        m = (nsym >= lo) & (nsym < hi)
        if m.any():
            lines.append(f"- nsym [{lo},{hi}): n={int(m.sum())}, acc={(preds[g][m] == yva[m]).mean():.4f}")
    lines += ["", "解读：组间 acc 差值 ≈ 新增特征的边际信息量；若 'all' 的探针 acc 已接近或超过"
              "当前模型 val acc，说明主干还没把已有信息用好；若探针在某类上也只有 ~50%，"
              "且 acc 随 nsym 单调上升，则该类受观测符号数限制，属数据天花板。"]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class"] + args.groups)
        for i in range(len(names)):
            w.writerow([names[i]] + [f"{per_class[g][i]:.4f}" for g in args.groups])
    print("\n".join(lines))
    print(f"\n[probe] 输出目录: {out}")


if __name__ == "__main__":
    main()
