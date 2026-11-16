"""统一推理入口（官方评测程序直接调用）

官方要求:
  - 位于 $HOME/code/infer.py
  - 输入: test Parquet（官方 sample_io 仅 task/sample_id/iq 3 列，混合三任务）
  - 输出: JSONL，每行 {task, sample_id, prediction}，顺序与输入行序一致
    分类: prediction 为字符串（官方精确类别名）
    参数: prediction 为 {参数名: 数值}（SI 原始单位，不适用参数填 0.0）

用法:
    python code/infer.py --input datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet --output out/pred.jsonl
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))  # code/
sys.path.insert(0, ROOT)

from models import EMFoundation
from training.common import load_config  # 归一化新旧配置格式 -> {"model","training"}
from utils.dataset import to_iq_array, PARAM_NAMES, ID2LABEL, FS
from utils.logger import get_logger


def load_model(weights_dir, device):
    """加载 weights/ 下的 model.safetensors + config.json + param_stats.json"""
    cfg = load_config(os.path.join(weights_dir, "config.json"))
    model = EMFoundation(cfg["model"]).to(device)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(weights_dir, "model.safetensors"))
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    stats_path = os.path.join(weights_dir, "param_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        param_stats = {"min": np.array(s["min"], dtype=np.float32),
                       "max": np.array(s["max"], dtype=np.float32)}
    else:
        param_stats = None
    return model, cfg, param_stats


@torch.no_grad()
def infer_file(model, parquet_path, out_path, seq_len, param_stats, device, batch_size=32):
    """官方输入（混合三任务，行序任意混排）-> JSONL。

    批内按 task 分组前向；输出严格保持输入行序（与官方 expected_output 对齐）。
    parameter 样本由 parameter_pattern 头决定模式：
      通信(前3参有效) -> 后4参填 0.0；雷达(后4参有效) -> 前3参填 0.0
    """
    df = pd.read_parquet(parquet_path, columns=["task", "sample_id", "iq"])
    df["task"] = df["task"].replace({"system": "protocol"})
    if not set(df["task"]).issubset({"protocol","modulation","parameter"}):
        raise ValueError(f"unknown tasks: {set(df['task'])}")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    # 结果按全局行号存放，最后顺序写出
    results = [None] * len(df)
    for s in range(0, len(df), batch_size):
        chunk = df.iloc[s : s + batch_size]
        # 同批内按 task 分组前向（批内行序不再要求一致）
        for task, idx in chunk.groupby("task").indices.items():
            rows = chunk.iloc[idx]
            iqs = [to_iq_array(v, seq_len) for v in rows["iq"]]
            iq = torch.from_numpy(np.stack(iqs)).to(device)
            sr = torch.full((len(rows),), FS, dtype=torch.float32, device=device)

            out = model(iq, sr, task)
            if task == "parameter":
                # pattern 头 -> 模式硬掩码（官方 expected 输出不适用项 = 0.0）
                pat = model(iq, sr, "parameter_pattern").argmax(-1)  # [B] 0=通信 1=雷达
                mask = torch.zeros_like(out)
                mask[:, :3] = (pat == 0).float().unsqueeze(-1)
                mask[:, 3:] = (pat == 1).float().unsqueeze(-1)
                pred = out.cpu().numpy()
                if param_stats is not None:
                    pred = pred * (param_stats["max"] - param_stats["min"]) + param_stats["min"]
                pred = pred * mask.cpu().numpy()  # 必须先反归一化后清零
                for j, (_, row) in enumerate(rows.iterrows()):
                    results[s + int(idx[j])] = {
                        "task": task,
                        "sample_id": row["sample_id"],
                        "prediction": {n: float(pred[j, k])
                                       for k, n in enumerate(PARAM_NAMES)},
                    }
            else:
                ids = out.argmax(-1).cpu().numpy()
                for j, (_, row) in enumerate(rows.iterrows()):
                    results[s + int(idx[j])] = {
                        "task": task,
                        "sample_id": row["sample_id"],
                        "prediction": ID2LABEL[task][int(ids[j])],
                    }

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[infer] {parquet_path} -> {out_path} ({len(df)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="测试 Parquet 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--weights_dir", default=os.path.join(ROOT, "..", "weights"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    from utils.train_utils import get_device
    device = get_device()
    model, cfg, param_stats = load_model(args.weights_dir, device)
    import time
    logger = get_logger("inference", os.path.join(ROOT,"..","logs"), "inference.log")
    start = time.perf_counter()
    infer_file(model, args.input, args.output,
               seq_len=cfg["training"].get("seq_len", 1024), param_stats=param_stats,
               device=device, batch_size=args.batch_size)
    logger.info("device=%s input=%s output=%s elapsed_s=%.3f",device,args.input,args.output,time.perf_counter()-start)


if __name__ == "__main__":
    main()
