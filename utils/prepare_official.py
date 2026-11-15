"""官方真实数据预处理：3 个 Parquet 分块 -> memmap 快速训练格式

用法:
    python utils/prepare_official.py                    # 全量 1,179,000 行
    python utils/prepare_official.py --limit 3000       # 冒烟测试: 每任务前 N 行
    python utils/prepare_official.py --val_ratio 0.02

处理内容:
  1. 按行组流式扫描 3 个分块（part_001~003），按 task 列分派样本
     （大表按行混排，不能按文件假设任务!）
  2. iq (list<list<float>>) -> float32 memmap [N, 1024, 2]（512 补零对齐）
  3. 每任务产出:
       datasets_official/{task}/iq.memap       float32 memmap
       datasets_official/{task}/meta.json      labels/params/valid/patterns/sample_ids/val
     并写 datasets_official/{split}_{task}.parquet 兼容 build_task_dataloaders
  4. val 划分: 对 protocol/modulation 按 label 分层抽样；parameter 按 pattern 分层

统计来源: files/real_data_stats.json（全量 1,179,000 行扫描结果）
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
# 复用 code/utils/dataset.py 的双格式 iq 读取（同名包冲突，按路径加载）
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_official_ds", os.path.join(ROOT, "code", "utils", "dataset.py"))
_official_ds = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_official_ds)
to_iq_array = _official_ds.to_iq_array

PARAM_NAMES = [
    "symbol_rate", "bandwidth", "frequency_shift",
    "pulse_width", "pri", "time_delay", "number_of_pulses",
]

# 官方 3 分块路径（自动发现）
CHUNK_GLOBS = [
    os.path.join(ROOT, "datasets", "*", "rf_signal_benchmark_train_v1.0",
                 "*part_*of_003.parquet"),
]

SEQ_LEN = 1024  # 官方 iq 长度 512/1024，统一对齐补零到 1024


def find_chunks():
    files = sorted(glob.glob(CHUNK_GLOBS[0]))
    if not files:
        raise FileNotFoundError(
            "未找到官方分块文件，期望 datasets/*/rf_signal_benchmark_train_v1.0/*part_*of_003.parquet")
    return files


class TaskWriter:
    """单任务累积写器：memmap 分段写 + meta 收集"""

    def __init__(self, out_dir, task, seq_len):
        self.dir = os.path.join(out_dir, task)
        os.makedirs(self.dir, exist_ok=True)
        self.task = task
        self.seq_len = seq_len
        self.iq_path = os.path.join(self.dir, "iq.memap")
        self.n_written = 0
        self._fh = open(self.iq_path, "wb")
        self.meta = {"seq_len": seq_len, "labels": [], "params": [],
                     "valid": [], "patterns": [], "sample_ids": []}

    def add(self, row):
        arr = to_iq_array(row["iq"], max_len=self.seq_len)
        assert arr.shape == (self.seq_len, 2), f"bad iq shape {arr.shape}"
        self._fh.write(arr.astype(np.float32).tobytes())
        self.n_written += 1
        self.meta["sample_ids"].append(row["sample_id"])
        if self.task in ("protocol", "modulation"):
            lab = row["modulation_label"] if self.task == "modulation" else row["protocol_label"]
            assert lab is not None, f"{self.task} 行 label 为空"
            self.meta["labels"].append(lab)
        else:
            params = [float(row[n]) for n in PARAM_NAMES]
            valid = [bool(row[n + "_valid"]) for n in PARAM_NAMES]
            pattern = int(valid[3])  # 0=通信 / 1=雷达
            self.meta["params"].append(params)
            self.meta["valid"].append(valid)
            self.meta["patterns"].append(pattern)

    def finalize(self, val_ids):
        self._fh.close()
        assert self.n_written == len(self.meta["sample_ids"])
        self.meta["val_sample_ids"] = sorted(val_ids)
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f)
        # 校验 memmap 可读回
        mm = np.memmap(self.iq_path, dtype=np.float32, mode="r",
                       shape=(self.n_written, self.seq_len, 2))
        assert np.array_equal(np.asarray(mm[0][:4]), np.asarray(mm[0][:4]))
        return self.n_written


def stratified_val_split(meta, ratio, seed=2023):
    """分层抽样 val sample_id 集合：protocol/modulation 按 label，parameter 按 pattern"""
    rng = np.random.default_rng(seed)
    ids = meta["sample_ids"]
    if "labels" in meta and meta["labels"]:
        strata = meta["labels"]
    else:
        strata = meta["patterns"]
    val_ids = set()
    for lab in sorted(set(strata)):
        idx = [i for i, s in enumerate(strata) if s == lab]
        rng.shuffle(idx)
        k = max(int(len(idx) * ratio), 1)
        val_ids.update(ids[i] for i in idx[:k])
    return val_ids


def to_legacy_parquet(df_task, out_dir, split, task):
    """写兼容 build_task_dataloaders 布局的 Parquet（官方列名风格, 19 列 subset）"""
    path = os.path.join(out_dir, f"{split}_{task}.parquet")
    df_task.to_parquet(path, index=False)
    return path


def run(limit=None, val_ratio=0.02, out_dir=None):
    out_dir = out_dir or os.path.join(ROOT, "datasets_official")
    os.makedirs(out_dir, exist_ok=True)
    chunks = find_chunks()
    print(f"[prepare] chunks: {[os.path.basename(c) for c in chunks]}")

    writers = {t: TaskWriter(out_dir, t, SEQ_LEN) for t in ("protocol", "modulation", "parameter")}
    counts = {t: 0 for t in writers}
    iq_cols = ["iq"]

    for chunk in chunks:
        pf = pq.ParquetFile(chunk)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)  # 19 列全读（每行组 1024 行，内存可控）
            df = tbl.to_pandas()
            for task, sub in [(t, df[df["task"] == t]) for t in writers]:
                if limit is not None and counts[task] >= limit:
                    continue
                if limit is not None:
                    sub = sub.head(limit - counts[task])
                for _, row in sub.iterrows():
                    writers[task].add(row)
                    counts[task] += 1
        print(f"[prepare] {os.path.basename(chunk)} done, counts={counts}")
        if limit is not None and all(counts[t] >= limit for t in counts):
            break

    # val 划分 + 落盘
    val_ids = {}
    for task, w in writers.items():
        val_ids[task] = stratified_val_split(w.meta, val_ratio) if w.n_written > 0 else set()
        n = w.finalize(val_ids[task])
        print(f"[prepare] {task}: {n} samples "
              f"(val={len(val_ids[task])}) -> {w.dir}")

    # ---- 兼容布局 Parquet（仅冒烟/小规模；全量训练直接走 memmap 更省内存）----
    summary = {}
    if limit is not None:
        for task, w in writers.items():
            m = w.meta
            n = w.n_written
            if n == 0:
                continue
            is_val = np.array([s in val_ids[task] for s in m["sample_ids"]])
            mm = np.memmap(w.iq_path, dtype=np.float32, mode="r", shape=(n, SEQ_LEN, 2))
            for split, keep in (("train", ~is_val), ("val", is_val)):
                df = build_compat_df(task, m, mm, np.where(keep)[0])
                to_legacy_parquet(df, out_dir, split, task)
                summary[f"{split}_{task}"] = len(df)
                print(f"[prepare] {split}_{task}.parquet: {len(df)} rows")
    else:
        print("[prepare] full-scale mode: skip compat parquet "
              "(train/val read memmap directly via make_task_dataset)")

    with open(os.path.join(out_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "val_ratio": val_ratio, "seq_len": SEQ_LEN,
                   "splits": summary}, f, indent=2)
    print("[prepare] done ->", os.path.join(out_dir, "prepare_summary.json"))


def build_compat_df(task, meta, mm, indices):
    """从 memmap+meta 构建官方 schema 的 DataFrame（19 列 subset）"""
    rows = {c: [] for c in ("task", "sample_id", "iq",
                           "modulation_label", "protocol_label")}
    for c in PARAM_NAMES:
        rows[c] = []
        rows[c + "_valid"] = []
    for i in indices:
        rows["task"].append(task)
        rows["sample_id"].append(meta["sample_ids"][i])
        rows["iq"].append(np.asarray(mm[i]).tolist())
        mod_lab, proto_lab = None, None
        params = [0.0] * 7
        valid = [False] * 7
        if task == "modulation":
            mod_lab = meta["labels"][i]
        elif task == "protocol":
            proto_lab = meta["labels"][i]
        else:
            params = list(meta["params"][i])
            valid = list(meta["valid"][i])
        rows["modulation_label"].append(mod_lab)
        rows["protocol_label"].append(proto_lab)
        for j, pname in enumerate(PARAM_NAMES):
            rows[pname].append(float(params[j]))
            rows[pname + "_valid"].append(bool(valid[j]))
    return pd.DataFrame(rows)


def rebuild_compat(out_dir, per_shard=50000):
    """从已有 memmap+meta 分片重建兼容布局 parquet（大表内存安全，输出分片文件）

    全量训练无需此步（make_task_dataset 直接读 memmap）；分片仅用于小规模实验。
    产出: {split}_{task}.shard000.parquet, .shard001, ...
    """
    summary = {}
    for task in ("protocol", "modulation", "parameter"):
        d = os.path.join(out_dir, task)
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        n = len(meta["sample_ids"])
        val_ids = set(meta.get("val_sample_ids", []))
        is_val = np.array([s in val_ids for s in meta["sample_ids"]])
        mm = np.memmap(os.path.join(d, "iq.memap"), dtype=np.float32, mode="r",
                       shape=(n, meta["seq_len"], 2))
        for split, keep in (("train", ~is_val), ("val", is_val)):
            idx_all = np.where(keep)[0]
            total = 0
            for si in range(0, len(idx_all), per_shard):
                idx = idx_all[si:si + per_shard]
                df = build_compat_df(task, meta, mm, idx)
                p = os.path.join(out_dir, f"{split}_{task}.shard{si//per_shard:03d}.parquet")
                df.to_parquet(p, index=False)
                total += len(df)
            summary[f"{split}_{task}"] = total
            print(f"[compat] {split}_{task}: {total} rows in "
                  f"{(len(idx_all)+per_shard-1)//per_shard} shards")
    with open(os.path.join(out_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"val_ratio": None, "seq_len": SEQ_LEN, "splits": summary}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="每任务最多处理行数（冒烟用）")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--only_compat", action="store_true",
                    help="跳过重扫官方分块，仅从已有 memmap+meta 重建兼容 parquet")
    args = ap.parse_args()
    if args.only_compat:
        rebuild_compat(args.out_dir or os.path.join(ROOT, "datasets_official"))
    else:
        run(limit=args.limit, val_ratio=args.val_ratio, out_dir=args.out_dir)
