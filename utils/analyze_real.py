"""官方真实数据集全量分析工具（对应赛题 rf_signal_benchmark_train_v1.0）

功能:
  1. 全量扫描 3 个 Parquet 分块（跳过 iq 列，120 万行标签/参数全量统计）
  2. 生成 6 张统计图 -> figures/real_analysis/
  3. 抽样各类别样本生成 2x2 IQ 组合图 -> figures/real_samples/
  4. 输出 JSON 统计结果 -> docs/real_data_stats.json（供报告引用）

用法:
    python utils/analyze_real.py                 # 全量分析 + 画图
    python utils/analyze_real.py --row-groups 20  # 快速模式（每文件抽20行组）
"""

import os
import sys
import json
import argparse
import collections

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 复用项目根 utils/visualize.py 的绘图函数
sys.path.insert(0, os.path.dirname(__file__))
from visualize import plot_sample_grid  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRAIN_FILES = [
    os.path.join(ROOT, "datasets", "01", "rf_signal_benchmark_train_v1.0",
                 "rf_signal_benchmark_train_v1.0_part_001_of_003.parquet"),
    os.path.join(ROOT, "datasets", "02", "rf_signal_benchmark_train_v1.0",
                 "rf_signal_benchmark_train_v1.0_part_002_of_003.parquet"),
    os.path.join(ROOT, "datasets", "03", "rf_signal_benchmark_train_v1.0",
                 "rf_signal_benchmark_train_v1.0_part_003_of_003.parquet"),
]

PARAMS = ["symbol_rate", "bandwidth", "frequency_shift",
          "pulse_width", "pri", "time_delay", "number_of_pulses"]
COMM_PARAMS = ["symbol_rate", "bandwidth", "frequency_shift"]      # 通信类有效
RADAR_PARAMS = ["pulse_width", "pri", "time_delay", "number_of_pulses"]  # 雷达类有效

FIG_STATS_DIR = os.path.join(ROOT, "figures", "real_analysis")
FIG_SAMPLE_DIR = os.path.join(ROOT, "figures", "real_samples")
STATS_JSON = os.path.join(ROOT, "files", "real_data_stats.json")

# 标签列不做字符串处理，直接按行组累计
LABEL_COLS = ["task", "modulation_label", "protocol_label"]
VALID_COLS = [p + "_valid" for p in PARAMS] + ["modulation_valid", "protocol_valid"]


# ============================================================
# 1. 全量扫描
# ============================================================

def scan_all(row_groups=None):
    """逐行组扫描 3 个分块，返回统计字典。

    row_groups: 每文件读取的行组数（None=全量；快速模式可设 20）
    """
    stats = {
        "files": {},
        "task_counter": collections.Counter(),
        "mod_counter": collections.Counter(),
        "proto_counter": collections.Counter(),
        "iq_len_counter": collections.Counter(),
        "param_values": {p: [] for p in PARAMS},
        "param_valid": {p: [0, 0] for p in PARAMS},  # [True 数, False 数]
        "valid_combo_counter": collections.Counter(),  # 7 个 valid 的组合模式
        "n_rows": 0,
    }

    for path in TRAIN_FILES:
        f = pq.ParquetFile(path)
        n_rg = f.num_row_groups if row_groups is None else min(row_groups, f.num_row_groups)
        step = 1 if row_groups is None else max(1, f.num_row_groups // n_rg)
        f_rows = 0
        for i in range(0, f.num_row_groups, step):
            if row_groups is not None and f_rows >= row_groups * 1024:
                break
            tbl = f.read_row_group(i, columns=None).to_pandas()
            f_rows += len(tbl)
            _accumulate(stats, tbl)
        stats["files"][os.path.relpath(path, ROOT)] = {
            "row_groups_total": f.num_row_groups,
            "row_groups_read": len(range(0, f.num_row_groups, step)),
            "rows_read": f_rows,
        }
        stats["n_rows"] += f_rows
        print(f"[scan] {os.path.relpath(path, ROOT)}: 读取 {f_rows} 行")
    return stats


def _accumulate(stats, df):
    """把一个行组的统计累加进 stats"""
    stats["task_counter"].update(df["task"])
    m = df["task"] == "modulation"
    stats["mod_counter"].update(df["modulation_label"][m])
    p = df["task"] == "protocol"
    stats["proto_counter"].update(df["protocol_label"][p])
    stats["iq_len_counter"].update(df["iq"].apply(len))

    par = df["task"] == "parameter"
    for pname in PARAMS:
        v = df[pname][par].tolist()
        valid = df[pname + "_valid"][par].tolist()
        stats["param_values"][pname].extend(x for x, ok in zip(v, valid) if ok)
        stats["param_valid"][pname][0] += sum(valid)
        stats["param_valid"][pname][1] += len(valid) - sum(valid)

    # parameter 行的 7 valid 组合模式（识别通信/雷达互斥）
    if par.any():
        vmat = df.loc[par, [p + "_valid" for p in PARAMS]].astype(int)
        for tup in vmat.itertuples(index=False):
            stats["valid_combo_counter"]["".join(map(str, tup))] += 1


# ============================================================
# 2. 统计图（6 张）
# ============================================================

def _save(fig, name):
    os.makedirs(FIG_STATS_DIR, exist_ok=True)
    p = os.path.join(FIG_STATS_DIR, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {p}")
    return p


def plot_task_distribution(stats):
    c = stats["task_counter"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = sorted(c.keys())
    vals = [c[k] for k in keys]
    bars = ax.bar(keys, vals, color=["#4C72B0", "#DD8452", "#55A868"][:len(keys)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=10)
    ax.set_title(f"任务分布（共 {sum(vals):,} 行）")
    ax.set_ylabel("样本数")
    fig.tight_layout()
    return _save(fig, "01_task_distribution.png")


def plot_modulation_distribution(stats):
    c = stats["mod_counter"]
    fig, ax = plt.subplots(figsize=(10, 6))
    keys = [k for k, _ in c.most_common()]
    vals = [c[k] for k in keys]
    colors = ["#DD8452" if k.startswith("Radar") else "#4C72B0" for k in keys]
    ax.barh(keys[::-1], vals[::-1], color=colors[::-1])
    ax.set_title(f"调制类别分布（23 类，橙色=雷达波形，蓝色=通信波形）")
    ax.set_xlabel("样本数")
    fig.tight_layout()
    return _save(fig, "02_modulation_distribution.png")


def plot_protocol_distribution(stats):
    c = stats["proto_counter"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = [k for k, _ in c.most_common()]
    vals = [c[k] for k in keys]
    ax.bar(keys, vals, color="#55A868")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"协议类别分布（8 类）")
    ax.set_ylabel("样本数")
    plt.xticks(rotation=20)
    fig.tight_layout()
    return _save(fig, "03_protocol_distribution.png")


def plot_param_histograms(stats):
    """7 参数直方图网格（仅 valid=True），通信 3 参 + 雷达 4 参"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i, p in enumerate(PARAMS):
        ax = axes[i]
        a = np.asarray(stats["param_values"][p], dtype=np.float64)
        if len(a) == 0:
            ax.set_title(p + " (无数据)")
            continue
        if p in RADAR_PARAMS and p != "number_of_pulses":
            a = a * 1e6  # 秒 -> 微秒，更易读
            unit = "µs"
        elif p == "number_of_pulses":
            unit = "个"
        else:
            unit = "Hz"
        ax.hist(a, bins=40, color="#4C72B0", alpha=0.8)
        ax.set_title(f"{p} ({unit})")
        ax.set_ylabel("样本数")
        ax.grid(alpha=0.3)
    # 第 8 格画 valid 组合模式
    ax = axes[7]
    cc = stats["valid_combo_counter"]
    keys = list(cc.keys())
    vals = [cc[k] for k in keys]
    ax.bar(range(len(keys)), vals, color="#55A868")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("1", "V").replace("0", "-") for k in keys], fontsize=7)
    ax.set_title("7 参数有效位组合模式\n(V=有效；可见通信3位与雷达4位互斥)")
    ax.grid(alpha=0.3)
    fig.suptitle("parameter 任务参数分布（仅统计 valid=True 样本）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "04_param_histograms.png")


def plot_amplitude_distribution(stats, sample_rows=2000):
    """峰值幅度分布（log-x 直方图，按 task 分色）"""
    # 抽样读 iq 计算幅度（全量读 iq 太慢，均匀抽行组）
    amps = {"modulation": [], "protocol": [], "parameter": []}
    for path in TRAIN_FILES:
        f = pq.ParquetFile(path)
        step = max(1, f.num_row_groups // 30)
        got = 0
        for i in range(0, f.num_row_groups, step):
            if got >= sample_rows // 3:
                break
            tbl = f.read_row_group(i, columns=["task", "iq"]).to_pandas()
            for t, iq in zip(tbl["task"], tbl["iq"]):
                if len(amps[t]) < sample_rows // 3:
                    a = np.asarray(iq.tolist() if hasattr(iq, "tolist") else iq, dtype=np.float32)
                    amps[t].append(float(np.sqrt((a.astype(np.float64) ** 2).sum(1)).max()))
            got += 1
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(-4, 2, 40)
    for t, color in zip(["modulation", "protocol", "parameter"],
                        ["#4C72B0", "#55A868", "#DD8452"]):
        if amps[t]:
            ax.hist(amps[t], bins=bins, alpha=0.55, label=f"{t} (n={len(amps[t])})", color=color)
    ax.set_xscale("log")
    ax.set_xlabel("峰值幅度（每样本 |IQ| 最大值，log 刻度）")
    ax.set_ylabel("样本数")
    ax.set_title("信号峰值幅度分布：跨 5 个数量级（Idle background 极弱 ~ 强协议信号）")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return _save(fig, "05_amplitude_distribution.png")


def plot_valid_rates(stats):
    """各参数 valid=True 比例"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = PARAMS
    rates = [100 * stats["param_valid"][p][0] /
             max(stats["param_valid"][p][0] + stats["param_valid"][p][1], 1)
             for p in names]
    colors = ["#4C72B0" if p in COMM_PARAMS else "#DD8452" for p in names]
    bars = ax.bar(names, rates, color=colors)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.0f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("valid=True 比例 (%)")
    ax.set_title("parameter 任务各参数有效率（蓝=通信参数，橙=雷达参数）")
    plt.xticks(rotation=15)
    fig.tight_layout()
    return _save(fig, "06_valid_rates.png")


# ============================================================
# 3. 样本图（各类别抽样 2x2 组合图）
# ============================================================

def _find_rows(task, label_col, label, per_file=30):
    """在 3 个分块中找指定 task+label 的行（返回 DataFrame 列表）"""
    out = []
    for path in TRAIN_FILES:
        f = pq.ParquetFile(path)
        step = max(1, f.num_row_groups // 60)
        for i in range(0, f.num_row_groups, step):
            tbl = f.read_row_group(i).to_pandas()
            m = (tbl["task"] == task) & (tbl[label_col] == label)
            if m.any():
                out.append(tbl[m].head(per_file))
            if sum(len(x) for x in out) >= per_file:
                return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def generate_sample_figures(max_per_class=1):
    """各类别抽样生成 2x2 IQ 组合图"""
    os.makedirs(FIG_SAMPLE_DIR, exist_ok=True)
    plan = (
        [("modulation", "modulation_label", k) for k in
         ["Radar/LFM", "Radar/Barker Code", "Radar/Frank Code",
          "Radar/Coherent Pulse Train", "Radar/Polyphase Barker Code",
          "64QAM", "QPSK", "MSK", "GFSK", "FM", "CSS"]]
        + [("protocol", "protocol_label", k) for k in
           ["WiFi", "Bluetooth", "LoRa", "5G", "ZigBee", "FM broadcast",
            "Radar", "Idle background"]]
        + [("parameter", None, None)]
    )
    saved = []
    for task, col, label in plan:
        sub_dir = os.path.join(FIG_SAMPLE_DIR, task)
        os.makedirs(sub_dir, exist_ok=True)
        if task == "parameter":
            # 通信参数样本 + 雷达参数样本各 1 张
            df = _find_rows(task, "bandwidth_valid", True).head(1)
            df2 = _find_rows(task, "pulse_width_valid", True).head(1)
            rows = pd.concat([df, df2], ignore_index=True)
        else:
            rows = _find_rows(task, col, label).head(max_per_class)
        for _, r in rows.iterrows():
            raw = r["iq"]
            iq = np.asarray(raw.tolist() if hasattr(raw, "tolist") else raw, dtype=np.float32)
            sample = {
                "sample_id": r["sample_id"], "task": r["task"], "iq": iq,
                "label": r.get("modulation_label") or r.get("protocol_label") or "",
                "params": {p: r[p] for p in PARAMS if r[p + "_valid"]},
            }
            name = f"{(label or task).replace('/', '_').replace(' ', '_')}_{r['sample_id'][:8]}.png"
            p = os.path.join(sub_dir, name)
            try:
                plot_sample_grid(sample, p)
                saved.append(p)
                print(f"[sample-fig] {task}/{label}: {p}")
            except Exception as e:
                print(f"[sample-fig] 失败 {task}/{label}: {e}")
    return saved


# ============================================================
# 4. 统计汇总 JSON
# ============================================================

def dump_stats(stats, path=STATS_JSON):
    def percentile(a, q):
        a = np.asarray(a, dtype=np.float64)
        return float(np.percentile(a, q)) if len(a) else None

    out = {
        "files": stats["files"],
        "n_rows_scanned": stats["n_rows"],
        "task_distribution": dict(stats["task_counter"]),
        "iq_length_distribution": {str(k): v for k, v in stats["iq_len_counter"].items()},
        "modulation_distribution": dict(stats["mod_counter"]),
        "protocol_distribution": dict(stats["proto_counter"]),
        "valid_combo_patterns": dict(stats["valid_combo_counter"]),
        "param_stats": {},
    }
    for p in PARAMS:
        a = stats["param_values"][p]
        t, f = stats["param_valid"][p]
        out["param_stats"][p] = {
            "n_valid": len(a),
            "valid_rate_pct": round(100 * t / max(t + f, 1), 1),
            "min": float(min(a)) if a else None,
            "p25": percentile(a, 25), "median": percentile(a, 50),
            "p75": percentile(a, 75), "max": float(max(a)) if a else None,
        }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"[stats] saved {path}")
    return out


# ============================================================
# CLI
# ============================================================

def load_stats_from_json(path=STATS_JSON):
    """从已落盘的 JSON 重建 stats（避免全量重扫）"""
    with open(path, encoding="utf-8") as fp:
        d = json.load(fp)
    stats = {
        "files": d["files"],
        "task_counter": collections.Counter(d["task_distribution"]),
        "mod_counter": collections.Counter(d["modulation_distribution"]),
        "proto_counter": collections.Counter(d["protocol_distribution"]),
        "iq_len_counter": collections.Counter(
            {int(k): v for k, v in d["iq_length_distribution"].items()}),
        "param_values": {},
        "param_valid": {p: [d["param_stats"][p]["n_valid"],
                            int(d["param_stats"][p]["n_valid"] /
                                max(d["param_stats"][p]["valid_rate_pct"] / 100, 1e-9)
                                - d["param_stats"][p]["n_valid"])]
                       for p in PARAMS},
        "valid_combo_counter": collections.Counter(d["valid_combo_patterns"]),
        "n_rows": d["n_rows_scanned"],
    }
    # 直方图/有效率图所需的数值：从 param_stats 恢复近似分布不可行，
    # 幅度/直方图这类需要原始值的图在全量 JSON 模式下重抽行组即可
    return stats


def main():
    ap = argparse.ArgumentParser(description="官方真实数据集分析")
    ap.add_argument("--row-groups", type=int, default=None,
                    help="每文件读取行组数（默认全量；快速模式如 20）")
    ap.add_argument("--skip-figures", action="store_true", help="只统计不画图")
    ap.add_argument("--figures-from-json", action="store_true",
                    help="用已落盘的全量统计 JSON 直接画分布图（参数直方图除外）")
    args = ap.parse_args()

    if args.figures_from_json:
        stats = load_stats_from_json()
        plot_task_distribution(stats)
        plot_modulation_distribution(stats)
        plot_protocol_distribution(stats)
        plot_valid_rates(stats)
        plot_valid_combos = collections.Counter(stats["valid_combo_counter"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(range(len(plot_valid_combos)), list(plot_valid_combos.values()), color="#55A868")
        ax.set_xticks(range(len(plot_valid_combos)))
        ax.set_xticklabels(list(plot_valid_combos.keys()), fontsize=9)
        for i, v in enumerate(plot_valid_combos.values()):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
        ax.set_title("参数有效位组合模式：通信(1110000) 与 雷达(0001111) 严格二分")
        ax.set_ylabel("样本数")
        fig.tight_layout()
        _save(fig, "07_valid_combo_patterns.png")
        return

    stats = scan_all(row_groups=args.row_groups)
    dump_stats(stats)

    if not args.skip_figures:
        plot_task_distribution(stats)
        plot_modulation_distribution(stats)
        plot_protocol_distribution(stats)
        plot_param_histograms(stats)
        plot_amplitude_distribution(stats)
        plot_valid_rates(stats)
        generate_sample_figures()


if __name__ == "__main__":
    main()
