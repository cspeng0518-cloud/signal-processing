"""统一推理入口（官方评测程序直接调用）

官方要求:
  - 位于 $HOME/src/infer/infer.py
  - 输入: test Parquet（task / sample_id / iq 3 列，混合三任务，512/1024 两种长度）
  - 输出: JSONL，每行 {task, sample_id, prediction}，顺序与输入行序一致
    分类: prediction 为官方类别名字符串
    参数: prediction 为 {参数名: 数值}（SI 原始单位，不适用参数填 0.0）

推理优化（复赛“计算效率提升度”按推理时延排名）:
  1. 按 (task, 原始长度) 分组成批：512 点样本只跑 64 个时域 token，批内无 padding mask
     -> SDPA 可走 FlashAttention 内核
  2. 默认 bf16 autocast（信号前端内部强制 fp32，精度不受影响）
  3. 可选 --compile（torch.compile 主干，CUDA）
  4. 预热 + 计时：日志输出解析/模型耗时、每样本耗时、吞吐，以及融合算子启用状态

用法:
  python src/infer/infer.py --input test.parquet --output out/pred.jsonl
  python src/infer/infer.py --input test.parquet --output out/pred.jsonl --precision fp32 --batch_size 256
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
PKG = os.path.dirname(SRC)                                         # 仓库根目录
sys.path.insert(0, SRC)

from models import build_model                                  # noqa: E402
from training.common import load_config                         # noqa: E402
from utils.dataset import ID2LABEL, PARAM_NAMES, to_iq_array   # noqa: E402
from utils.logger import get_logger                             # noqa: E402
from utils.param_codec import ParamCodec                        # noqa: E402


def load_model(weights_dir, device):
    """加载 weights/ 下 config.json + model.safetensors（+ param_codec），返回 (model, cfg)。"""
    from safetensors.torch import load_file
    cfg = load_config(os.path.join(weights_dir, "config.json"))
    model = build_model(cfg["model"], with_decoder=False).to(device)
    sd = load_file(os.path.join(weights_dir, "model.safetensors"))
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"权重与配置不匹配: missing={missing[:5]} unexpected={unexpected[:5]}")
    spec = cfg["model"].get("param_codec")
    model.codec = ParamCodec(spec) if spec else \
        ParamCodec.load(os.path.join(weights_dir, "param_codec.json"))
    model.eval()
    return model, cfg


def _autocast(device, precision):
    if precision == "fp32" or device.type not in ("cuda", "npu"):
        return torch.autocast(device_type=device.type, enabled=False)
    dt = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dt)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def _predict(model, iq, task):
    out = model.predict(iq, task)
    return out.float().cpu().numpy() if task == "parameter" else out.cpu().numpy()


@torch.no_grad()
def infer_file(model, parquet_path, out_path, seq_len, device, batch_size=256,
               precision="bf16", logger=None, warmup=True):
    """官方输入（混合三任务，行序任意）-> JSONL（严格保持输入行序）。"""
    df = pd.read_parquet(parquet_path, columns=["task", "sample_id", "iq"])
    task_out = df["task"].astype(str).tolist()          # 原样回显到输出
    df["task"] = df["task"].replace({"system": "protocol"})
    unknown = set(df["task"]) - {"protocol", "modulation", "parameter"}
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    t0 = time.perf_counter()
    raw = [to_iq_array(v) for v in df["iq"]]
    lengths = [min(len(a), seq_len) for a in raw]
    iqs = [to_iq_array(a, seq_len) for a in raw]
    del raw
    t_parse = time.perf_counter() - t0

    groups = {}
    for i, (t, ln) in enumerate(zip(df["task"].to_numpy(), lengths)):
        groups.setdefault((t, int(ln)), []).append(i)

    if warmup and len(df):
        k = next(iter(groups))
        x = torch.from_numpy(np.stack([iqs[i] for i in groups[k][:4]])).to(device)
        with _autocast(device, precision):
            _predict(model, x, k[0])
        _sync(device)

    results = [None] * len(df)
    sids = df["sample_id"].to_numpy()
    t1 = time.perf_counter()
    for (task, _), idx_all in groups.items():
        for s in range(0, len(idx_all), batch_size):
            idx = idx_all[s:s + batch_size]
            x = torch.from_numpy(np.stack([iqs[i] for i in idx])).to(device, non_blocking=True)
            with _autocast(device, precision):
                pred = _predict(model, x, task)
            for j, i in enumerate(idx):
                if task == "parameter":
                    p = {n: float(pred[j, k]) for k, n in enumerate(PARAM_NAMES)}
                else:
                    p = ID2LABEL[task][int(pred[j])]
                results[i] = {"task": task_out[i], "sample_id": sids[i], "prediction": p}
    _sync(device)
    t_model = time.perf_counter() - t1

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    msg = (f"[infer] {parquet_path} -> {out_path} n={len(df)} parse_s={t_parse:.3f} "
           f"model_s={t_model:.3f} per_sample_ms={1000 * t_model / max(1, len(df)):.4f} "
           f"throughput={len(df) / max(t_model, 1e-9):.1f}/s precision={precision}")
    print(msg)
    if logger:
        logger.info(msg)
    return results


def load_env_config():
    """读取 config/infer.json（运行环境配置：设备/精度/批大小/权重目录）。"""
    path = os.path.join(PKG, "config", "infer.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    env = load_env_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="测试 Parquet 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--weights_dir",
                        default=os.path.join(PKG, env.get("weights_dir", "weights")))
    parser.add_argument("--batch_size", type=int, default=int(env.get("batch_size", 256)))
    parser.add_argument("--precision", default=env.get("precision", "bf16"),
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true", help="torch.compile 主干（CUDA）")
    args = parser.parse_args()

    from utils.train_utils import get_device
    dev_pref = str(env.get("device", "auto")).lower()
    device = get_device() if dev_pref == "auto" else torch.device(dev_pref)
    logger = get_logger("inference", os.path.join(PKG, "logs"), "inference.log")
    model, cfg = load_model(args.weights_dir, device)
    if args.compile and device.type == "cuda":
        model.backbone = torch.compile(model.backbone, dynamic=True)
    start = time.perf_counter()
    infer_file(model, args.input, args.output, seq_len=cfg["model"].get(
        "seq_len", cfg["training"].get("seq_len", 1024)), device=device,
        batch_size=args.batch_size, precision=args.precision, logger=logger)
    try:
        from models.ops import fused_status
        logger.info("fused ops: %s", fused_status())
    except Exception:
        pass
    logger.info("device=%s params=%.2fM elapsed_total_s=%.3f", device,
                model.deploy_parameter_count() / 1e6, time.perf_counter() - start)


if __name__ == "__main__":
    main()
