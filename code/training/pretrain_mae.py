"""阶段一：MAE 自监督预训练（任务无关，仅需 IQ，无需标签）。

用法:
    python code/training/pretrain_mae.py [--config ...] [--data_dir ...] [--smoke_steps N]

流程:
  1. 三任务训练集混合，只取 IQ 与采样率
  2. 随机掩码 patch，backbone 只编码可见 token，轻量 decoder 重建被掩码 patch
  3. 导出 tokenizer + backbone（+ 频域分支）权重，供微调初始化
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from models import EMFoundation
from training.common import ROOT, base_arg_parser, load_config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.dataset import make_task_dataset
from utils.logger import get_logger, record_metrics
from utils.train_utils import (autocast_context, build_scheduler, get_device,
                               optimizer_step, seed_all)

PRETRAIN_TASKS = ("protocol", "modulation", "parameter")


class UnlabeledIQ(Dataset):
    """包装单任务数据集，丢弃标签，仅返回 (iq, sr)。"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        return item["iq"], item["sr"]


def build_pretrain_loader(data_dir, cfg, device, logger):
    """三任务训练集混合为一个无标签 DataLoader。"""
    parts = []
    for task in PRETRAIN_TASKS:
        try:
            parts.append(UnlabeledIQ(make_task_dataset(data_dir, task, "train", cfg["seq_len"])))
        except FileNotFoundError:
            logger.warning("missing training data for task=%s, skipped", task)
    if not parts:
        raise FileNotFoundError(f"no train data found under {data_dir}")
    return DataLoader(ConcatDataset(parts), batch_size=cfg["batch_size"], shuffle=True,
                      num_workers=cfg.get("num_workers", 0),
                      pin_memory=(device.type == "cuda"))


def pretrain_parameters(model):
    """预训练优化参数：共享主干 + MAE 专用模块，任务头不参与（无梯度，省优化器状态）。"""
    return [p for name, p in model.named_parameters()
            if not name.startswith("task_heads")]


def pretrain_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, smoke_steps):
    """跑一个 epoch，返回整池平均掩码重建 MSE。"""
    model.train()
    total, count = 0.0, 0
    for step, (iq, sr) in enumerate(loader):
        if smoke_steps and step >= smoke_steps:
            break
        iq, sr = iq.to(device), sr.to(device)
        with autocast_context(device, cfg.get("amp", True)):
            recon, target = model.forward_mae(iq, sr)
            loss = (recon.float() - target.float()).square().mean()
        optimizer_step(loss, optimizer, model, scaler, cfg.get("grad_clip", 1.0))
        scheduler.step()
        total += loss.item() * iq.size(0)
        count += iq.size(0)
    return total / max(1, count)


def export_pretrained(model, out_dir):
    """导出微调初始化所需的共享权重（任务头与 MAE decoder 不导出）。"""
    state = {"tokenizer": model.tokenizer.state_dict(),
             "backbone": model.backbone.state_dict()}
    if model.use_frequency:
        state["frequency_tokenizer"] = model.frequency_tokenizer.state_dict()
        state["cross_domain_fusion"] = model.cross_domain_fusion.state_dict()
    torch.save(state, Path(out_dir) / "mae_pretrained.pt")


def main():
    args = base_arg_parser("MAE self-supervised pretraining").parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.get("seed", 42))
    device = get_device()
    logger = get_logger("pretrain", ROOT / "logs")

    loader = build_pretrain_loader(args.data_dir, cfg, device, logger)
    model = EMFoundation(cfg).to(device)
    optimizer = torch.optim.AdamW(pretrain_parameters(model),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = min(len(loader), args.smoke_steps) if args.smoke_steps else len(loader)
    scheduler = build_scheduler(optimizer, max(1, steps_per_epoch * cfg["epochs_pretrain"]),
                                cfg.get("warmup_ratio", 0.1))
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(device.type == "cuda" and cfg.get("amp", True)))

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model=model, optimizer=optimizer,
                                         scheduler=scheduler, scaler=scaler)
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logger.info("device=%s samples=%d shared_ratio=%.3f",
                device, len(loader.dataset), model.shared_ratio())

    for epoch in range(start_epoch, cfg["epochs_pretrain"]):
        mse = pretrain_one_epoch(model, loader, optimizer, scheduler, scaler,
                                 device, cfg, args.smoke_steps)
        record_metrics(ROOT / "logs", "pretrain", epoch + 1,
                       {"recon_mse": mse, "lr": scheduler.get_last_lr()[0]})
        logger.info("epoch=%d masked_mse=%.6f", epoch + 1, mse)
        save_checkpoint(Path(args.out_dir) / "mae_resume.pt", epoch=epoch, model=model,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler)

    export_pretrained(model, args.out_dir)
    logger.info("saved MAE pretrained weights -> %s",
                Path(args.out_dir) / "mae_pretrained.pt")


if __name__ == "__main__":
    main()
