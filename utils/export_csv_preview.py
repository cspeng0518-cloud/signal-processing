"""官方数据集 CSV 预览导出（供人工查看数据长什么样）

用法:
    python utils/export_csv_preview.py                    # 每任务前 20 行
    python utils/export_csv_preview.py --limit 50         # 每任务前 50 行

产出 (out/preview/, utf-8-sig 编码可直接用 Excel 打开):
    samples_<task>.csv     每任务前 N 行字段总览:
                           标签 / 7 参数 / 7 有效位 / iq 长度与峰值 / 前 8 对 I/Q 预览
    iq_series_<task>.csv   每任务第一条样本的完整 I/Q 时序 (t, i, q)，
                           可直接在 Excel 里画折线图看波形
    summary.txt            三任务行数与字段说明

只读原始 parquet 的前几个行组，内存占用极小，分钟级完成。
"""

import os
import sys
import csv
import glob
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
# 复用 code/utils/dataset.py 的 iq 解析（按路径加载，避免同名包冲突）
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_preview_ds", os.path.join(ROOT, "code", "utils", "dataset.py"))
_ds = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ds)
to_iq_array = _ds.to_iq_array
PARAM_NAMES = _ds.PARAM_NAMES

CHUNK_GLOB = os.path.join(ROOT, "datasets", "**",
                          "rf_signal_benchmark_train_v1.0", "*part_*of_003.parquet")
TASKS = ("modulation", "protocol", "parameter")


def find_chunks():
    files = sorted(glob.glob(CHUNK_GLOB, recursive=True))
    if len(files) != 3:
        raise FileNotFoundError(
            f"期望 3 个官方分块，实际 {len(files)} 个: {files}")
    return files


def collect_rows(limit):
    """流式扫描分块行组，每任务收集前 limit 行（原始 DataFrame 行）。"""
    rows = {t: [] for t in TASKS}
    for chunk in find_chunks():
        pf = pq.ParquetFile(chunk)
        for rg in range(pf.num_row_groups):
            df = pf.read_row_group(rg).to_pandas()
            for t in TASKS:
                if len(rows[t]) >= limit:
                    continue
                need = limit - len(rows[t])
                rows[t].extend(
                    df[df["task"] == t].head(need).to_dict("records"))
        if all(len(v) >= limit for v in rows.values()):
            break
    return rows


def iq_summary(iq_value):
    """单行 iq -> (长度, 峰值, 前 8 对预览串)。"""
    arr = to_iq_array(iq_value)
    peak = float(np.abs(arr).max())
    preview = ";".join(f"{arr[k, 0]:.4f},{arr[k, 1]:.4f}" for k in range(8))
    return arr, peak, preview


def write_samples_csv(task, records, out_dir):
    """字段总览表：一行 = 一条样本。"""
    path = os.path.join(out_dir, f"samples_{task}.csv")
    cols = (["sample_id", "task", "iq_len", "iq_peak_abs", "iq_preview_8pairs",
             "protocol_label", "protocol_valid",
             "modulation_label", "modulation_valid"]
            + [f"{n}({u})" for n, u in zip(PARAM_NAMES,
                ["Hz", "Hz", "Hz", "s", "s", "s", "count"])]
            + [n + "_valid" for n in PARAM_NAMES])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            arr, peak, preview = iq_summary(r["iq"])
            row = [r["sample_id"], r["task"], len(arr), f"{peak:.4f}", preview,
                   r.get("protocol_label"), r.get("protocol_valid"),
                   r.get("modulation_label"), r.get("modulation_valid")]
            row += [f"{float(r[n]):.6g}" if r.get(n) is not None else ""
                    for n in PARAM_NAMES]
            row += [r.get(n + "_valid") for n in PARAM_NAMES]
            w.writerow(row)
    return path


def write_series_csv(task, record, out_dir):
    """完整 I/Q 时序表：一行 = 一个采样点（第一条样本），可画波形图。"""
    path = os.path.join(out_dir, f"iq_series_{task}.csv")
    arr = to_iq_array(record["iq"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        label = record.get("modulation_label") or record.get("protocol_label") or ""
        w.writerow(["sample_id", "label", "t", "i", "q"])
        for t, (i, q) in enumerate(arr):
            w.writerow([record["sample_id"], label, t,
                        f"{i:.6f}", f"{q:.6f}"])
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="每任务导出前 N 行")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "out", "preview"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = collect_rows(args.limit)
    summary = []
    for t in TASKS:
        if not rows[t]:
            print(f"[preview] warning: no rows for task={t}")
            continue
        p1 = write_samples_csv(t, rows[t], args.out_dir)
        p2 = write_series_csv(t, rows[t][0], args.out_dir)
        summary.append(f"{t}: {len(rows[t])} rows -> {p1}\n         "
                       f"first-sample waveform -> {p2}")
        print(f"[preview] {t}: {len(rows[t])} rows done")

    with open(os.path.join(args.out_dir, "summary.txt"), "w",
              encoding="utf-8-sig") as f:
        f.write("官方数据集 CSV 预览\n" + "=" * 40 + "\n")
        f.write("\n".join(summary) + "\n\n")
        f.write("字段说明:\n"
                "  iq_len / iq_peak_abs: 样本点数(512/1024)与峰值幅度\n"
                "  iq_preview_8pairs:    前 8 对 I/Q 值 (i,q;i,q;...)\n"
                "  *_label / *_valid:    分类真值与适用位\n"
                "  7 参数 + *_valid:     SI 单位回归真值与逐参数有效位\n"
                "  (无效参数原始真值为 0.0，以 valid 位为准)\n")
    print(f"[preview] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
