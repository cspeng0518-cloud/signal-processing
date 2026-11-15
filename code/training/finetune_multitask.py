"""多任务联合微调（对应赛题任务2：单一基座 + 三个任务头）

用法:
    python code/training/finetune_multitask.py --config code/configs/default.json

流程:
  1. 加载（可选）MAE 预训练的 tokenizer+backbone 权重
  2. 每个 epoch 在三任务间轮转：共享 backbone 接收三任务梯度
  3. 保存完整模型 -> weights/model.safetensors + weights/config.json
     同时保存参数归一化统计 -> weights/param_stats.json（推理反归一化用）
"""

import os
import sys
import json
import argparse
import logging

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_ROOT)  # models/utils 包根（code/）

from models import EMFoundation
from utils.dataset import (
    build_task_dataloaders, save_param_stats, LABEL2ID, ID2LABEL,
)
from utils.metrics import accuracy, macro_recall, precision_score

TASKS = ("protocol", "modulation", "parameter")
PATTERN_LOSS_W = 0.5  # parameter_pattern 辅助头权重


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(
            os.path.join(log_dir, "train.log"), encoding="utf-8")],
    )
    return logging.getLogger("finetune")


def evaluate(model, loader, task, device, param_stats, ref_scale):
    """验证集评估：返回各任务指标"""
    import numpy as np
    model.eval()
    ys, ps, vs = [], [], []
    with torch.no_grad():
        for batch in loader:
            iq, sr = batch["iq"].to(device), batch["sr"].to(device)
            out = model(iq, sr, task)
            if task == "parameter":
                # pattern 头硬掩码：预测模式决定不适用参数置 0（归一化空间）
                pattern_logits = model(iq, sr, "parameter_pattern")
                pat = pattern_logits.argmax(-1)  # [B] 0=通信 1=雷达
                mask = torch.zeros_like(out)
                mask[:, :3][pat == 0] = 1.0   # 通信: 前 3 参数有效
                mask[:, 3:][pat == 1] = 1.0   # 雷达: 后 4 参数有效
                out = out * mask
                pred = out.cpu().numpy()
                # 反归一化到原始量纲（无效维保持 0.0）
                mn, mx = param_stats["min"], param_stats["max"]
                pred = pred * (mx - mn) + mn
                ps.append(pred)
                vs.append(batch["valid"].numpy())
                ys.append(batch["params"].numpy() * (mx - mn) + mn)
            else:
                ps.append(out.argmax(-1).cpu().numpy())
                ys.append(batch["label"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    if task == "parameter":
        v = np.concatenate(vs)
        prec = precision_score(y, p, ref_scale, valid=v)
        return {"precision": prec, "rmse": None}
    n_cls = 8 if task == "protocol" else 23
    return {"acc": accuracy(y, p), "macro_recall": macro_recall(y, p, n_cls)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "code", "configs", "default.json"))
    parser.add_argument("--data_dir", default=os.path.join(ROOT, "datasets_official"),
                        help="官方预处理目录（datasets_official/）或模拟数据目录（datasets/）")
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "weights"))
    parser.add_argument("--from_mae", default=None,
                        help="MAE 预训练权重路径（缺省则用 out_dir/mae_pretrained.pt，不存在则从零开始）")
    parser.add_argument("--smoke_steps", type=int, default=None,
                        help="冒烟测试：每 epoch 限制优化步数（跳过验证集评估）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    logger = setup_logger(os.path.join(ROOT, "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}, backbone: {cfg['backbone']}")

    # ---- 数据 ----
    loaders, param_stats = build_task_dataloaders(
        args.data_dir, batch_size=cfg["batch_size"], seq_len=cfg["seq_len"])
    logger.info("dataloaders ready: "
                + ", ".join(f"{t}(train={len(loaders['train'][t].dataset)})" for t in TASKS))

    # ---- 模型（可选加载 MAE 预训练权重）----
    model = EMFoundation(cfg).to(device)
    mae_path = args.from_mae or os.path.join(args.out_dir, "mae_pretrained.pt")
    if os.path.exists(mae_path):
        state = torch.load(mae_path, map_location=device)
        model.tokenizer.load_state_dict(state["tokenizer"])
        model.backbone.load_state_dict(state["backbone"])
        logger.info(f"loaded MAE pretrained weights from {mae_path}")
    else:
        logger.info("no MAE checkpoint found, training from scratch")

    logger.info(f"shared_ratio = {model.shared_ratio():.2%} (官方红线 >80%)")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                              weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    ref_scale = cfg.get("ref_scale", 0.5)
    epochs = cfg["epochs_finetune"]

    best_score = -1.0
    for ep in range(epochs):
        model.train()
        losses = {t: 0.0 for t in TASKS}
        # ---- 三任务轮转训练（同一 optimizer，共享 backbone）----
        iters = {t: iter(loaders["train"][t]) for t in TASKS}
        n_steps = max(len(loaders["train"][t]) for t in TASKS)
        if args.smoke_steps:
            n_steps = min(n_steps, args.smoke_steps)
        for step in range(n_steps):
            for t in TASKS:
                try:
                    batch = next(iters[t])
                except StopIteration:
                    iters[t] = iter(loaders["train"][t])
                    batch = next(iters[t])
                iq, sr = batch["iq"].to(device), batch["sr"].to(device)
                out = model(iq, sr, t)
                if t == "parameter":
                    # 回归损失：仅对有效参数计算（valid 掩码参与）
                    target = batch["params"].to(device)
                    valid = batch["valid"].to(device)          # [B,7]
                    reg_loss = (mse(out, target) * valid).sum() / (valid.sum() + 1e-8)
                    # pattern 辅助损失：通信/雷达模式二分类
                    pat_logits = model(iq, sr, "parameter_pattern")
                    pat_loss = ce(pat_logits, batch["pattern"].to(device))
                    loss = reg_loss + PATTERN_LOSS_W * pat_loss
                else:
                    loss = ce(out, batch["label"].to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses[t] += loss.item()

        msg = " | ".join(f"{t}: {losses[t]/max(n_steps,1):.4f}" for t in TASKS)
        logger.info(f"epoch {ep+1}/{epochs}  loss[{msg}]")

        # ---- 验证 ----
        if args.smoke_steps:
            logger.info("smoke mode: skip evaluation")
            break
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            evals = {t: evaluate(model, loaders["val"][t], t, device, param_stats, ref_scale)
                     for t in TASKS}
            logger.info(
                f"  val protocol: acc={evals['protocol']['acc']:.3f} "
                f"recall={evals['protocol']['macro_recall']:.3f} | "
                f"modulation: acc={evals['modulation']['acc']:.3f} "
                f"recall={evals['modulation']['macro_recall']:.3f} | "
                f"parameter: precision={evals['parameter']['precision']:.3f}"
            )
            cur = (evals["protocol"]["acc"] + evals["modulation"]["acc"]
                   + evals["parameter"]["precision"]) / 3
            if cur > best_score:
                best_score = cur
                save_model(model, cfg, param_stats, args.out_dir)
                logger.info(f"  saved best (mean={best_score:.4f})")

    save_model(model, cfg, param_stats, args.out_dir, suffix="final")
    logger.info("finetune done.")


def save_model(model, cfg, param_stats, out_dir, suffix="best"):
    """保存官方交付格式：
      weights/model.safetensors   —— 完整模型权重
      weights/config.json         —— 模型结构配置
      weights/param_stats.json    —— 参数归一化统计
    """
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {f"model.{k}": v for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    save_param_stats(param_stats, os.path.join(out_dir, "param_stats.json"))
    if suffix != "best":
        # final 版本另存一份数据（best 已保存）
        save_file(sd, os.path.join(out_dir, f"model_{suffix}.safetensors"))


if __name__ == "__main__":
    main()
