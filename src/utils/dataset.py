"""数据集读取（对齐官方真实数据格式）+ 长度分桶 DataLoader

官方训练数据 schema (rf_signal_benchmark_train_v1.0, 19 列):
  task: protocol / modulation / parameter        （大表按行混排后分块，必须按 task 列过滤）
  sample_id: 唯一标识
  iq: list<list<float>>  [N,2] I/Q 时序（N = 512 或 1024）
  modulation_label / protocol_label               分类标签字符串
  symbol_rate ... number_of_pulses                7 参数列（SI 单位，不适用参数 = 0.0）
  symbol_rate_valid ... number_of_pulses_valid    有效性标识
官方测试输入 (sample_io) 仅 3 列: task / sample_id / iq（混合三任务）

两种数据布局（make_task_dataset 自动选择）:
  官方预处理 memmap: {data_dir}/{task}/iq.memap + meta.json（utils/prepare_official.py 产出，
                     train/val/test 划分记录在 meta）
  Parquet:           {data_dir}/{split}_{task}.parquet（src/utils/data_gen.py 模拟数据）

返回样本字段:
  iq [N,2] float32（右侧补零到 seq_len；单位功率归一化在模型前端做）
  label（分类任务）/ raw_params [7] SI 原值（无效位为 0）、valid [7]、pattern（参数任务，
  0=通信 1=雷达）。参数编码（标准化 / 对数 / 离散分类）由 utils/param_codec.py 在 GPU 上完成。

长度分桶（LengthGroupedBatchSampler）: 512 点与 1024 点样本不混批 -> 模型 trim_padding
后 512 点 batch 只跑 64 个时域 token，且批内无 padding mask（SDPA 可走 Flash 内核）。
多卡时各 rank 用同一随机种子生成同一 batch 列表，再按 rank 交错切分，分桶对多卡同样生效。
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

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
SYSTEM_CLASSES = PROTOCOL_CLASSES          # 向后兼容别名

LABEL2ID = {
    "protocol": {c: i for i, c in enumerate(PROTOCOL_CLASSES)},
    "modulation": {c: i for i, c in enumerate(MODULATION_CLASSES)},
}
ID2LABEL = {t: {i: c for c, i in m.items()} for t, m in LABEL2ID.items()}

# ---- 官方全量统计 (1,179,000 行, docs/real_data_stats.json, SI 单位；仅作参考/默认编解码) ----
DEFAULT_PARAM_STATS = {
    "min": np.array([62.5, 65.0, -50.0, 1.0000078873417806e-05, 1.7000029401970096e-05,
                     1.000015004137822e-06, 2.0], dtype=np.float32),
    "max": np.array([2000.0, 3500.0, 1000.0, 1.5999929019017145e-05, 2.2999996872385964e-05,
                     9.999991561926436e-06, 6.0], dtype=np.float32),
}

FS = 1_000_000.0  # 默认采样率（官方数据无采样率字段；仅可视化工具使用）
TASKS = ("protocol", "modulation", "parameter")


def to_iq_array(v, max_len=None):
    """双格式 iq 读取 -> [N,2] float32（官方 list<list<float>> / 模拟数据扁平 bytes）。

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


def bytes_to_iq(b, max_len=None):
    """向后兼容别名（utils/visualize.py 使用）。"""
    return to_iq_array(b, max_len)


def _param_item(params, valid):
    """参数任务公共字段（raw_params 无效位置 0）。"""
    params = np.asarray(params, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    return {
        "raw_params": torch.from_numpy((params * valid).astype(np.float32)),
        "valid": torch.from_numpy(valid),
        "pattern": torch.tensor(int(valid[3] > 0), dtype=torch.long),   # pulse_width_valid
    }


class EMSignalDataset(Dataset):
    """单任务 Parquet 数据集（官方列名 / 旧模拟数据列名自动适配）。"""

    def __init__(self, parquet_path, task, seq_len=1024):
        self.df = pd.read_parquet(parquet_path)
        self.task = "protocol" if task == "system" else task
        if "task" in self.df.columns:
            t = self.df["task"].replace({"system": "protocol"})
            self.df = self.df[t == self.task].reset_index(drop=True)
        self.seq_len = seq_len
        official = any(c in self.df.columns
                       for c in ("modulation_label", "protocol_label", "symbol_rate_valid"))
        if official:
            self._label_col = "modulation_label" if self.task == "modulation" else "protocol_label"
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

    def lengths(self):
        return np.array([len(to_iq_array(v)) for v in self.df["iq"]], dtype=np.int32)

    def param_arrays(self):
        p = self.df[self._param_cols].to_numpy(dtype=np.float64)
        v = self.df[self._valid_cols].to_numpy().astype(bool)
        return p, v

    def __getitem__(self, i):
        row = self.df.iloc[i]
        item = {"iq": torch.from_numpy(to_iq_array(row["iq"], self.seq_len)),
                "task": self.task, "sample_id": row["sample_id"]}
        if self.task in ("protocol", "modulation") and self.has_label:
            item["label"] = torch.tensor(LABEL2ID[self.task][row[self._label_col]],
                                         dtype=torch.long)
        if self.task == "parameter" and self.has_params:
            item.update(_param_item([row[c] for c in self._param_cols],
                                    [bool(row[c]) for c in self._valid_cols]))
        return item


class PreparedOfficialDataset(Dataset):
    """官方数据预处理后的 memmap 数据集（配合 utils/prepare_official.py）。

      data_dir/{task}/iq.memap   float32 [total, seq_len, 2]（512 点样本右侧补零）
      data_dir/{task}/meta.json  {"seq_len", "labels" | "params"/"valid"/"patterns",
                                  "sample_ids", "val_sample_ids", "test_sample_ids"}
    """

    def __init__(self, data_dir, task, seq_len=1024):
        d = os.path.join(data_dir, task)
        with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta["seq_len"] != seq_len:
            raise ValueError(f"{task} meta seq_len={self.meta['seq_len']} != {seq_len}")
        self.task = task
        self.seq_len = seq_len
        self.n = len(self.meta["sample_ids"])
        self.iq = np.memmap(os.path.join(d, "iq.memap"), dtype=np.float32, mode="r",
                            shape=(self.n, seq_len, 2))

    def __len__(self):
        return self.n

    def lengths(self, chunk=8192):
        """每个样本的原始长度（seq_len 或 seq_len/2，按后半段是否全零判断；与模型前端
        length_quantum=512 的规则一致），首次计算后缓存为 lengths.npy。"""
        cache = os.path.join(os.path.dirname(self.iq.filename), "lengths.npy")
        if os.path.exists(cache):
            arr = np.load(cache)
            if len(arr) == self.n:
                return arr
        half = self.seq_len // 2
        out = np.empty(self.n, dtype=np.int32)
        for s in range(0, self.n, chunk):
            blk = np.asarray(self.iq[s:s + chunk, half:, :])
            out[s:s + chunk] = np.where(np.any(blk != 0, axis=(1, 2)), self.seq_len, half)
        try:
            np.save(cache, out)
        except OSError:
            pass
        return out

    def param_arrays(self):
        return (np.asarray(self.meta["params"], dtype=np.float64),
                np.asarray(self.meta["valid"], dtype=bool))

    def __getitem__(self, i):
        item = {"iq": torch.from_numpy(np.array(self.iq[i])), "task": self.task,
                "sample_id": self.meta["sample_ids"][i]}
        if self.task in ("protocol", "modulation"):
            item["label"] = torch.tensor(LABEL2ID[self.task][self.meta["labels"][i]],
                                         dtype=torch.long)
        else:
            item.update(_param_item(self.meta["params"][i], self.meta["valid"][i]))
        return item


def make_task_dataset(data_dir, task, split="train", seq_len=1024):
    """统一工厂：优先用预处理 memmap（官方数据），否则退回 Parquet。"""
    if os.path.exists(os.path.join(data_dir, task, "meta.json")):
        full = PreparedOfficialDataset(data_dir, task, seq_len)
        val_ids = set(full.meta.get("val_sample_ids", []))
        test_ids = set(full.meta.get("test_sample_ids", []))
        if split == "val" and not val_ids:
            raise FileNotFoundError(f"{task} has no val split in meta")
        if split == "test" and not test_ids:
            raise FileNotFoundError(f"{task} has no test split in meta; "
                                    "re-run utils/prepare_official.py")
        sids = full.meta["sample_ids"]
        if split == "test":
            keep = [i for i, s in enumerate(sids) if s in test_ids]
        elif split == "val":
            keep = [i for i, s in enumerate(sids) if s in val_ids]
        else:                       # train = 全量 - val - test
            keep = [i for i, s in enumerate(sids) if s not in val_ids and s not in test_ids]
        return torch.utils.data.Subset(full, keep)
    for name in (f"{split}_{task}.parquet",
                 f"{split}_system.parquet" if task == "protocol" else None):
        if name and os.path.exists(os.path.join(data_dir, name)):
            return EMSignalDataset(os.path.join(data_dir, name), task, seq_len)
    raise FileNotFoundError(f"no dataset for task={task}, split={split} under {data_dir}")


def collate_fn(batch):
    """iq 堆叠为 [B,N,2]，其余张量字段堆叠，字符串字段保留为 list。"""
    out = {"iq": torch.stack([b["iq"] for b in batch]),
           "sample_id": [b["sample_id"] for b in batch],
           "task": batch[0]["task"]}
    for k in ("label", "raw_params", "valid", "pattern"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def dataset_lengths(ds):
    """数据集（含 Subset / ConcatDataset）每个样本的原始长度；无法获得时返回 None。"""
    if isinstance(ds, torch.utils.data.Subset):
        base = dataset_lengths(ds.dataset)
        return None if base is None else base[np.asarray(ds.indices)]
    if isinstance(ds, torch.utils.data.ConcatDataset):
        parts = [dataset_lengths(d) for d in ds.datasets]
        return None if any(p is None for p in parts) else np.concatenate(parts)
    if hasattr(ds, "lengths"):
        return ds.lengths()
    return None


class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """按长度分桶组 batch（512 点与 1024 点样本不混批），支持多卡。

    每次迭代：桶内打乱 -> 切 batch -> batch 顺序全局打乱 -> 按 rank 交错取
    （batch 数截断为 world_size 的整数倍，各 rank 步数一致）。各 rank 种子相同，
    且每次 __iter__ 自动推进 epoch，多卡下 batch 划分天然一致、不重叠。
    """

    def __init__(self, lengths, batch_size, shuffle=True, drop_last=True, seed=42,
                 rank=0, world_size=1):
        self.lengths = np.asarray(lengths)
        self.bs = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rank, self.world = rank, max(1, world_size)
        self.epoch = 0
        self.groups = [np.where(self.lengths == v)[0] for v in np.unique(self.lengths)]

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for g in self.groups:
            idx = rng.permutation(g) if self.shuffle else g
            for s in range(0, len(idx), self.bs):
                b = idx[s:s + self.bs]
                if len(b) < self.bs and self.drop_last:
                    continue
                batches.append(b.tolist())
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        if self.world > 1:
            if self.drop_last:
                batches = batches[: len(batches) // self.world * self.world]
            batches = batches[self.rank::self.world]
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self):
        n = sum(len(g) // self.bs if self.drop_last else -(-len(g) // self.bs)
                for g in self.groups)
        if self.world > 1:
            n = n // self.world if self.drop_last else -(-(n - self.rank) // self.world)
        return n


def make_loader(ds, batch_size, train, num_workers=0, bucket=True, collate=collate_fn,
                seed=42, rank=0, world_size=1):
    """统一 DataLoader：分桶（默认）或普通随机/顺序采样；多卡按 rank 切分。"""
    common = dict(collate_fn=collate, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    if num_workers > 0:
        common["prefetch_factor"] = 4
    lengths = dataset_lengths(ds) if bucket else None
    if lengths is not None:
        bs = LengthGroupedBatchSampler(lengths, batch_size, shuffle=train, drop_last=train,
                                       seed=seed, rank=rank, world_size=world_size)
        return DataLoader(ds, batch_sampler=bs, **common)
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=train)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=train, **common)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, **common)


def build_task_dataloaders(data_dir, batch_size=256, seq_len=1024, splits=("train", "val"),
                           tasks=TASKS, num_workers=4, bucket=True, eval_batch_size=None,
                           rank=0, world_size=1, seed=42):
    """多任务 loader 字典 {split: {task: DataLoader}}（缺失的 task/split 自动跳过）。"""
    loaders = {}
    for split in splits:
        loaders[split] = {}
        for task in tasks:
            try:
                ds = make_task_dataset(data_dir, task, split=split, seq_len=seq_len)
            except FileNotFoundError:
                continue
            bs = batch_size if split == "train" else (eval_batch_size or batch_size)
            loaders[split][task] = make_loader(ds, bs, split == "train", num_workers,
                                               bucket=bucket, seed=seed, rank=rank,
                                               world_size=world_size)
    return loaders


def load_param_arrays(data_dir, split="train", seq_len=1024):
    """读取 parameter 任务原始 SI 参数与有效位（拟合 ParamCodec 用，默认只用 train 划分）。"""
    ds = make_task_dataset(data_dir, "parameter", split, seq_len)
    if isinstance(ds, torch.utils.data.Subset):
        p, v = ds.dataset.param_arrays()
        idx = np.asarray(ds.indices)
        return p[idx], v[idx]
    return ds.param_arrays()
