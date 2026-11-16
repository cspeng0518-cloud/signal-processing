"""数据集读取（完全对齐官方真实数据格式）

官方训练数据 schema (rf_signal_benchmark_train_v1.0, 19 列):
  task: protocol / modulation / parameter        （注意: 大表按行混排后分块，
  sample_id: 唯一标识                              必须按 task 列过滤，不能按文件假设）
  iq: list<list<float>>  [N,2] I/Q 时序
  modulation_label / modulation_valid             （task=protocol 行为 None）
  protocol_label  / protocol_valid                （task=modulation 行为 None）
  symbol_rate ... number_of_pulses                （7 参数列, 无前缀, SI 单位）
  symbol_rate_valid ... number_of_pulses_valid     （不适用参数真值 = 0.0）

  官方测试输入 (sample_io) 仅 3 列: task / sample_id / iq（混合三任务）

兼容性:
  - iq 双格式: 官方原生 list<list<float>> / 本框架模拟数据的扁平 bytes
  - 列名双风格: 官方 (modulation_label 等) / 旧模拟 (label / param_* / valid_*)
  - 任务名: protocol（官方）; 旧 "system" 已废弃

归一化策略:
  - IQ: 模型内部按样本峰值归一化（tokenizer.normalize），此处不动
  - 回归参数: min-max 归一化到 [0,1]，无效维目标置 0，推理时反归一化
"""

import os
import io
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PARAM_NAMES = [
    "symbol_rate", "bandwidth", "frequency_shift",
    "pulse_width", "pri", "time_delay", "number_of_pulses",
]

# ---- 官方类别表（task_schema.json 顺序，唯一可信来源）----
MODULATION_CLASSES = [
    "OOK", "BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM", "256QAM",
    "OQPSK", "2FSK", "4FSK", "MSK", "128QAM", "AQPSK", "CSS", "GFSK",
    "pi_4DQPSK", "FM",
    "Radar/Coherent Pulse Train", "Radar/Barker Code",
    "Radar/Polyphase Barker Code", "Radar/Frank Code", "Radar/LFM",
]
PROTOCOL_CLASSES = [
    "WiFi", "Bluetooth", "LoRa", "5G", "ZigBee",
    "FM broadcast", "Radar", "Idle background",
]
# 向后兼容别名（旧代码引用）
SYSTEM_CLASSES = PROTOCOL_CLASSES

LABEL2ID = {
    "protocol": {c: i for i, c in enumerate(PROTOCOL_CLASSES)},
    "modulation": {c: i for i, c in enumerate(MODULATION_CLASSES)},
}
ID2LABEL = {t: {i: c for c, i in m.items()} for t, m in LABEL2ID.items()}

# ---- 官方全量统计 (1,179,000 行, docs/real_data_stats.json, SI 单位) ----
DEFAULT_PARAM_STATS = {
    "min": np.array([62.5, 65.0, -50.0,
                     1.0000078873417806e-05, 1.7000029401970096e-05,
                     1.000015004137822e-06, 2.0], dtype=np.float32),
    "max": np.array([2000.0, 3500.0, 1000.0,
                     1.5999929019017145e-05, 2.2999996872385964e-05,
                     9.999991561926436e-06, 6.0], dtype=np.float32),
}

FS = 1_000_000.0  # 默认采样率（官方数据无采样率字段）


def to_iq_array(v, max_len=None):
    """双格式 iq 读取 -> [N,2] float32

    官方原生列 (list<list<float>> / ndarray of ndarray) 或本框架扁平 bytes 均支持。
    max_len 不为 None 时截断/右侧补零。
    """
    if isinstance(v, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(v, dtype=np.float32).reshape(-1, 2).copy()
    else:
        arr = np.asarray(v)
        if arr.dtype == object:  # pandas 读原生嵌套列的 object 行
            arr = np.asarray(arr.tolist(), dtype=np.float32)
        else:
            arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"unexpected iq shape {arr.shape}")
    if max_len is not None:
        if arr.shape[0] > max_len:
            arr = arr[:max_len]
        elif arr.shape[0] < max_len:
            arr = np.pad(arr, ((0, max_len - arr.shape[0]), (0, 0)))
    return arr


# 向后兼容别名
def bytes_to_iq(b, max_len=None):
    return to_iq_array(b, max_len)


def _detect_style(df):
    """检测列名风格: 'official' 或 'legacy'(模拟数据)"""
    official_marks = ("modulation_label", "protocol_label", "symbol_rate_valid")
    return "official" if any(c in df.columns for c in official_marks) else "legacy"


class EMSignalDataset(Dataset):
    """单任务数据集（一个 Parquet 文件 -> 一个 Dataset）

    自动适配官方 / 模拟两种列名风格；parameter 任务额外返回 pattern 标签
    （0=通信模式: symbol_rate/bandwidth/frequency_shift 有效
      1=雷达模式: pulse_width/pri/time_delay/number_of_pulses 有效）
    """

    def __init__(self, parquet_path, task, seq_len=1024, param_stats=None):
        self.df = pd.read_parquet(parquet_path)
        self.task = "protocol" if task == "system" else task
        if "task" in self.df.columns:
            self.df = self.df[self.df["task"].replace({"system":"protocol"}) == self.task].reset_index(drop=True)
        self.seq_len = seq_len
        self.param_stats = param_stats
        self.style = _detect_style(self.df)
        if self.style == "official":
            self._label_col = "modulation_label" if task == "modulation" else "protocol_label"
            self._param_cols = PARAM_NAMES
            self._valid_cols = [n + "_valid" for n in PARAM_NAMES]
        else:
            self._label_col = "label"
            self._param_cols = [f"param_{n}" for n in PARAM_NAMES]
            self._valid_cols = [f"valid_{n}" for n in PARAM_NAMES]
        self.has_label = self._label_col in self.df.columns
        self.has_params = self._param_cols[0] in self.df.columns

    def __len__(self):
        return len(self.df)

    def _label_of(self, row):
        v = row[self._label_col]
        return None if v is None or pd.isna(v) else v

    def __getitem__(self, i):
        row = self.df.iloc[i]
        iq = to_iq_array(row["iq"], self.seq_len)  # [N,2]
        item = {
            "iq": torch.from_numpy(iq),
            "sr": torch.tensor(FS, dtype=torch.float32),
            "task": self.task,
            "sample_id": row["sample_id"],
        }
        if self.task in ("protocol", "modulation") and self.has_label:
            lab = self._label_of(row)
            item["label"] = torch.tensor(LABEL2ID[self.task][lab], dtype=torch.long)
        if self.task == "parameter" and self.has_params:
            params = np.array([row[c] for c in self._param_cols], dtype=np.float32)
            valid = np.array([bool(row[c]) for c in self._valid_cols], dtype=np.float32)
            # pattern: 通信(前3参数有效)=0 / 雷达(后4参数有效)=1
            pattern = float(valid[3])  # pulse_width_valid
            if self.param_stats is not None:
                mn, mx = self.param_stats["min"], self.param_stats["max"]
                params = (params - mn) / (mx - mn + 1e-12)
                params = np.clip(params, 0.0, 1.0)
            # 无效维目标置 0（原始真值不适用项 = 0.0；pattern 头推理时硬掩码）
            params = params * valid
            item["params"] = torch.from_numpy(params.astype(np.float32))
            item["valid"] = torch.from_numpy(valid)
            item["pattern"] = torch.tensor(int(pattern), dtype=torch.long)
        return item


class PreparedOfficialDataset(Dataset):
    """官方数据预处理后的快速数据集（memmap 版，配合 utils/prepare_official.py）

    目录结构:
      data_dir/{task}/iq.memap   —— float32 [total, seq_len, 2]（每个样本对齐到 seq_len）
      data_dir/{task}/meta.json  —— {"seq_len": N, "labels": [...], "params": [[...]],
                                     "valid": [[...]], "patterns": [...], "sample_ids": [...]}
    """

    def __init__(self, data_dir, task, seq_len=1024, param_stats=None):
        d = os.path.join(data_dir, task)
        with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta["seq_len"] != seq_len:
            raise ValueError(f"{task} meta seq_len={self.meta['seq_len']} != {seq_len}")
        self.task = task
        self.seq_len = seq_len
        self.param_stats = param_stats
        self.n = len(self.meta["sample_ids"])
        self.iq = np.memmap(os.path.join(d, "iq.memap"), dtype=np.float32, mode="r",
                            shape=(self.n, seq_len, 2))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        iq = np.array(self.iq[i])  # memmap 切片拷贝
        item = {
            "iq": torch.from_numpy(iq),
            "sr": torch.tensor(FS, dtype=torch.float32),
            "task": self.task,
            "sample_id": self.meta["sample_ids"][i],
        }
        if self.task in ("protocol", "modulation"):
            item["label"] = torch.tensor(LABEL2ID[self.task][self.meta["labels"][i]],
                                         dtype=torch.long)
        else:
            params = np.array(self.meta["params"][i], dtype=np.float32)
            valid = np.array(self.meta["valid"][i], dtype=np.float32)
            pattern = int(self.meta["patterns"][i])
            if self.param_stats is not None:
                mn, mx = self.param_stats["min"], self.param_stats["max"]
                params = np.clip((params - mn) / (mx - mn + 1e-12), 0.0, 1.0)
            params = params * valid
            item["params"] = torch.from_numpy(params.astype(np.float32))
            item["valid"] = torch.from_numpy(valid)
            item["pattern"] = torch.tensor(int(pattern), dtype=torch.long)
        return item


def make_task_dataset(data_dir, task, split="train", seq_len=1024, param_stats=None):
    """统一工厂：优先用预处理 memmap（官方数据），否则退回 Parquet。

    data_dir 布局二选一:
      官方预处理: {task}/iq.memap + {task}/meta.json（val 划分记录在 meta 的
                  val_sample_ids，train/val 均从同一 memmap 过滤）
      模拟数据:   {split}_{task}.parquet（data_gen.py 产出）
    """
    mm_dir = os.path.join(data_dir, task)
    has_mm = os.path.exists(os.path.join(mm_dir, "meta.json"))
    if has_mm:
        full = PreparedOfficialDataset(data_dir, task, seq_len, param_stats)
        val_ids = set(full.meta.get("val_sample_ids", []))
        if split == "val" and not val_ids:
            raise FileNotFoundError(f"{task} has no val split in meta")
        keep = [i for i, s in enumerate(full.meta["sample_ids"])
                if (s in val_ids) == (split == "val")]
        return torch.utils.data.Subset(full, keep)
    pq_path = os.path.join(data_dir, f"{split}_{task}.parquet")
    if os.path.exists(pq_path):
        return EMSignalDataset(pq_path, task, seq_len, param_stats)
    # 兼容旧模拟数据命名：protocol 任务旧名为 system（EMSignalDataset 内部已做映射）
    if task == "protocol":
        legacy_path = os.path.join(data_dir, f"{split}_system.parquet")
        if os.path.exists(legacy_path):
            return EMSignalDataset(legacy_path, task, seq_len, param_stats)
    raise FileNotFoundError(f"no dataset for task={task}, split={split} under {data_dir}")


def collate_fn(batch):
    """自定义 collate：iq 堆叠为 [B,N,2]，其余字段对齐"""
    out = {
        "iq": torch.stack([b["iq"] for b in batch]),
        "sr": torch.stack([b["sr"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "task": batch[0]["task"],
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    if "params" in batch[0]:
        out["params"] = torch.stack([b["params"] for b in batch])
        out["valid"] = torch.stack([b["valid"] for b in batch])
        out["pattern"] = torch.stack([b["pattern"] for b in batch])
    return out


def compute_param_stats(parquet_path):
    """参数归一化统计。官方风格直接返回全量统计常量（避免重扫 GB 级文件）"""
    df_cols = pd.read_parquet(parquet_path).columns
    if "symbol_rate_valid" in df_cols:
        return {"min": DEFAULT_PARAM_STATS["min"].copy(),
                "max": DEFAULT_PARAM_STATS["max"].copy()}
    df = pd.read_parquet(parquet_path)
    mins, maxs = [], []
    for n in PARAM_NAMES:
        col = n if n in df.columns else f"param_{n}"
        flag = n+"_valid" if n+"_valid" in df.columns else f"valid_{n}"
        v = df.loc[df[flag].astype(bool), col].to_numpy(dtype=np.float64)
        if not len(v):
            raise ValueError(f"No valid observations for {n}")
        mins.append(float(v.min())); maxs.append(float(v.max()))
    return {"min": np.array(mins, dtype=np.float32), "max": np.array(maxs, dtype=np.float32)}


def build_task_dataloaders(data_dir, batch_size=32, seq_len=1024, splits=("train", "val"),
                           tasks=("protocol", "modulation", "parameter"), num_workers=0):
    """构建多任务 DataLoader 字典

    优先使用预处理 memmap 布局（{task}/iq.memap + meta.json，val 划分在 meta 内），
    否则退回 Parquet 布局（{split}_{task}.parquet，模拟数据 data_gen.py 产出）。

    Returns:
        loaders = {
          "train": {"protocol": DataLoader, "modulation": ..., "parameter": ...},
          "val": {...},
        }, param_stats
    """
    stats_file = os.path.join(data_dir, "param_stats.json")
    if os.path.exists(stats_file):
        param_stats = load_param_stats(stats_file)
    else:
        param_stats = {"min": DEFAULT_PARAM_STATS["min"].copy(),
                       "max": DEFAULT_PARAM_STATS["max"].copy()}
    loaders = {}
    for split in splits:
        loaders[split] = {}
        for task in tasks:
            try:
                ds = make_task_dataset(data_dir, task, split=split, seq_len=seq_len,
                                       param_stats=param_stats if task == "parameter" else None)
            except FileNotFoundError:
                continue
            loaders[split][task] = DataLoader(
                ds, batch_size=batch_size, shuffle=(split == "train"),
                collate_fn=collate_fn, num_workers=num_workers,
            )
    return loaders, param_stats


def save_param_stats(stats, path):
    """保存参数归一化统计（推理时需要反归一化）"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: np.asarray(v).tolist() for k, v in stats.items()}, f)


def load_param_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {"min": np.array(d["min"], dtype=np.float32),
            "max": np.array(d["max"], dtype=np.float32)}
