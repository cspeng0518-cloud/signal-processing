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
from utils.dataset import to_iq_array, PARAM_NAMES, ID2LABEL, FS


def load_model(weights_dir, device):
    """加载 weights/ 下的 model.safetensors + config.json + param_stats.json"""
    with open(os.path.join(weights_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = EMFoundation(cfg).to(device)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(weights_dir, "model.safetensors"))
    sd = {k[len("model."):]: v for k, v in sd.items()}
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
    """官方输入（混合三任务）-> JSONL。

    按行分批处理；批内行序即输出顺序，与官方 expected_output 对齐。
    parameter 样本由 parameter_pattern 头决定模式：
      通信(前3参有效) -> 后4参填 0.0；雷达(后4参有效) -> 前3参填 0.0
    """
    df = pd.read_parquet(parquet_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for s in range(0, len(df), batch_size):
            chunk = df.iloc[s : s + batch_size]
            iqs = [to_iq_array(v, seq_len) for v in chunk["iq"]]
            iq = torch.from_numpy(np.stack(iqs)).to(device)
            sr = torch.full((len(chunk),), FS, dtype=torch.float32, device=device)
            task = chunk["task"].iloc[0]

            out = model(iq, sr, task)
            if task == "parameter":
                # pattern 头 -> 模式硬掩码（官方 expected 输出不适用项 = 0.0）
                pat = model(iq, sr, "parameter_pattern").argmax(-1)  # [B] 0=通信 1=雷达
                mask = torch.zeros_like(out)
                mask[:, :3][pat == 0] = 1.0
                mask[:, 3:][pat == 1] = 1.0
                pred = (out * mask).cpu().numpy()
                if param_stats is not None:  # 反归一化 -> 原始 SI 单位
                    pred = pred * (param_stats["max"] - param_stats["min"]) + param_stats["min"]
                for j in range(len(chunk)):
                    p = {n: float(pred[j, k]) for k, n in enumerate(PARAM_NAMES)}
                    fout.write(json.dumps({
                        "task": task,
                        "sample_id": chunk["sample_id"].iloc[j],
                        "prediction": p,
                    }, ensure_ascii=False) + "\n")
            else:
                ids = out.argmax(-1).cpu().numpy()
                for j in range(len(chunk)):
                    fout.write(json.dumps({
                        "task": task,
                        "sample_id": chunk["sample_id"].iloc[j],
                        "prediction": ID2LABEL[task][int(ids[j])],
                    }, ensure_ascii=False) + "\n")
    print(f"[infer] {parquet_path} -> {out_path} ({len(df)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="测试 Parquet 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--weights_dir", default=os.path.join(ROOT, "..", "weights"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, param_stats = load_model(args.weights_dir, device)
    infer_file(model, args.input, args.output,
               seq_len=cfg.get("seq_len", 1024), param_stats=param_stats,
               device=device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
