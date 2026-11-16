"""阶段二：单基座三任务联合微调（体制/调制/参数），校验后保存最佳 checkpoint。

用法:
    python code/training/finetune_multitask.py [--config ...] [--data_dir ...]
        [--from_mae ...] [--smoke_steps N]

流程:
  1. （可选）加载 MAE 预训练的 tokenizer + backbone 权重
  2. 三任务轮转训练：每个 step 依次取各任务 batch，共享 backbone 接收三任务梯度
  3. 按本地代理得分（30% 体制 + 30% 调制 + 40% 参数精度）保存最佳模型
     -> weights/model.safetensors + config.json + param_stats.json
"""

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from models import EMFoundation
from training.common import ROOT, base_arg_parser, load_config, load_mae_pretrained
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.dataset import build_task_dataloaders, save_param_stats
from utils.logger import get_logger, record_metrics
from utils.metrics import accuracy, macro_recall, precision_score
from utils.train_utils import (autocast_context, build_scheduler, get_device,
                               optimizer_step, seed_all)

TASKS = ("protocol", "modulation", "parameter")

# 官方类别数（task_schema.json）
N_CLASSES = {"protocol": 8, "modulation": 23}


# ---------------------------------------------------------------- 损失
def parameter_loss(pred, target, valid):
    """逐元素 masked smooth L1，按有效参数个数归一化。"""
    loss = F.smooth_l1_loss(pred, target, reduction="none") * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def compute_task_loss(model, feat, batch, task, cfg):
    """单任务损失：分类 = CE；参数 = 有效维 smooth L1 + pattern 辅助 CE。"""
    if task == "parameter":
        pred = model.task_heads["parameter"](feat)
        reg = parameter_loss(pred, batch["params"], batch["valid"])
        pattern_logits = model.task_heads["parameter_pattern"](feat)
        aux = F.cross_entropy(pattern_logits, batch["pattern"])
        return reg + cfg.get("pattern_loss_weight", 0.3) * aux
    pred = model.task_heads[task](feat)
    return F.cross_entropy(pred, batch["label"])


# ---------------------------------------------------------------- 训练
def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def next_batch(iters, task_loaders, task):
    """取下一个 batch，耗尽时自动重开 epoch（任务间 loader 长度不齐）。"""
    try:
        return next(iters[task])
    except StopIteration:
        iters[task] = iter(task_loaders[task])
        return next(iters[task])


def train_one_epoch(model, loaders, optimizer, scheduler, scaler, device, cfg, smoke_steps):
    """三任务轮转一个 epoch。

    Returns:
        (avg_losses, n_steps): 各任务加权后损失均值（按 step 数平均），实际步数
    """
    model.train()
    n_steps = max(len(loaders["train"][t]) for t in TASKS)
    if smoke_steps:
        n_steps = min(n_steps, smoke_steps)
    iters = {t: iter(loaders["train"][t]) for t in TASKS}
    weights = cfg.get("task_weights", {"protocol": 0.3, "modulation": 0.3, "parameter": 0.4})
    losses = {t: 0.0 for t in TASKS}

    for _ in range(n_steps):
        for task in TASKS:
            batch = to_device(next_batch(iters, loaders["train"], task), device)
            with autocast_context(device, cfg.get("amp", True)):
                feat = model.encode(batch["iq"], batch["sr"])
                loss = compute_task_loss(model, feat, batch, task, cfg)
                loss = loss * float(weights[task])
            optimizer_step(loss, optimizer, model, scaler, cfg.get("grad_clip", 1.0))
            scheduler.step()
            losses[task] += loss.item()
    return {t: losses[t] / max(1, n_steps) for t in TASKS}, n_steps


# ---------------------------------------------------------------- 评估
def evaluate(model, loader, task, device, param_stats, ref_scale):
    """验证集指标。

    分类任务: acc / macro_recall；参数任务: precision（本地参考尺度）与
    rmse（仅 ground-truth valid 维参与；官方测试无 valid，推理时用 pattern 头硬掩码）。
    """
    model.eval()
    gt, pred, valid = [], [], []
    with torch.inference_mode():
        for batch in loader:
            feat = model.encode(batch["iq"].to(device), batch["sr"].to(device))
            out = model.task_heads[task](feat)
            if task == "parameter":
                mn, mx = param_stats["min"], param_stats["max"]
                pred.append(out.float().cpu().numpy() * (mx - mn) + mn)
                gt.append(batch["params"].numpy() * (mx - mn) + mn)
                valid.append(batch["valid"].numpy())
            else:
                gt.append(batch["label"].numpy())
                pred.append(out.argmax(-1).cpu().numpy())

    y, p = np.concatenate(gt), np.concatenate(pred)
    if task == "parameter":
        v = np.concatenate(valid)
        return {
            "precision": float(precision_score(y, p, ref_scale, valid=v)),
            "rmse": float(np.sqrt(np.sum((y - p) ** 2 * v) / max(1, v.sum()))),
        }
    return {"acc": float(accuracy(y, p)),
            "macro_recall": float(macro_recall(y, p, N_CLASSES[task]))}


# ---------------------------------------------------------------- 导出
def export_model(model, cfg, param_stats, out_dir):
    """保存官方交付格式：model.safetensors + config.json + param_stats.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {f"model.{k}": v.detach().cpu().contiguous()
               for k, v in model.state_dict().items()}
    save_file(tensors, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    save_param_stats(param_stats, out_dir / "param_stats.json")


def assert_competition_limits(model):
    """赛题红线：权重共用率 >80%，总参数量 <1B。"""
    shared = model.shared_ratio()
    total = sum(p.numel() for p in model.parameters())
    if shared <= 0.8:
        raise RuntimeError(f"shared parameter ratio {shared:.2%} must exceed 80%")
    if total >= 1_000_000_000:
        raise RuntimeError("model exceeds 1B parameters")
    return shared, total


# ---------------------------------------------------------------- 主流程
def main():
    parser = base_arg_parser("Single-backbone multi-task fine-tuning")
    parser.add_argument("--from_mae", default=None,
                        help="MAE 预训练权重（默认 out_dir/mae_pretrained.pt，缺省从零训练）")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.get("seed", 42))
    device = get_device()
    logger = get_logger("finetune", ROOT / "logs")

    loaders, param_stats = build_task_dataloaders(
        args.data_dir, batch_size=cfg["batch_size"], seq_len=cfg["seq_len"],
        num_workers=cfg.get("num_workers", 0))
    missing = [t for t in TASKS if t not in loaders["train"]]
    if missing:
        raise RuntimeError(f"missing required train tasks: {missing}")

    model = EMFoundation(cfg).to(device)
    shared, total = assert_competition_limits(model)
    logger.info("device=%s shared_ratio=%.2f%% params=%d", device, shared * 100, total)

    if not args.resume:
        mae_path = args.from_mae or str(Path(args.out_dir) / "mae_pretrained.pt")
        load_mae_pretrained(model, mae_path, logger)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(len(loaders["train"][t]) for t in TASKS)
    if args.smoke_steps:
        steps_per_epoch = min(steps_per_epoch, args.smoke_steps)
    scheduler = build_scheduler(
        optimizer, max(1, steps_per_epoch * len(TASKS) * cfg["epochs_finetune"]),
        cfg.get("warmup_ratio", 0.1))
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(device.type == "cuda" and cfg.get("amp", True)))

    start_epoch, best, bad = 0, -float("inf"), 0
    if args.resume:
        start_epoch, extra = load_checkpoint(args.resume, model=model, optimizer=optimizer,
                                             scheduler=scheduler, scaler=scaler)
        best = extra.get("best", -float("inf"))
        bad = extra.get("bad", 0)
        logger.info("resumed from %s at epoch %d (best=%.5f)", args.resume, start_epoch, best)

    for epoch in range(start_epoch, cfg["epochs_finetune"]):
        avg_losses, _ = train_one_epoch(model, loaders, optimizer, scheduler, scaler,
                                        device, cfg, args.smoke_steps)
        values = {f"train_{t}_weighted_loss": avg_losses[t] for t in TASKS}
        values["lr"] = scheduler.get_last_lr()[0]

        if not args.smoke_steps and (epoch + 1) % cfg.get("eval_every", 1) == 0:
            if any(t not in loaders.get("val", {}) for t in TASKS):
                raise RuntimeError("missing validation split for one or more tasks")
            results = {t: evaluate(model, loaders["val"][t], t, device, param_stats,
                                   cfg.get("ref_scale", 0.5)) for t in TASKS}
            for t, metrics in results.items():
                for key, val in metrics.items():
                    values[f"val_{t}_{key}"] = val
            # 与初赛 30/30/40 权重一致的本地代理得分（参数精度为非官方参考尺度）
            score = (0.3 * results["protocol"]["acc"]
                     + 0.3 * results["modulation"]["acc"]
                     + 0.4 * results["parameter"]["precision"])
            values["val_proxy_score"] = score
            if score > best:
                best, bad = score, 0
                export_model(model, cfg, param_stats, args.out_dir)
                logger.info("new best proxy score=%.5f", best)
            else:
                bad += 1

        record_metrics(ROOT / "logs", "finetune", epoch + 1, values)
        logger.info("epoch=%d %s", epoch + 1,
                    " ".join(f"{k}={v:.5f}" for k, v in values.items()))
        save_checkpoint(Path(args.out_dir) / "finetune_resume.pt", epoch=epoch, model=model,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        best=best, bad=bad)

        if args.smoke_steps:
            break
        if bad >= cfg.get("patience", 7):
            logger.info("early stopping at epoch %d", epoch + 1)
            break

    if args.smoke_steps:
        export_model(model, cfg, param_stats, args.out_dir)
    logger.info("finished; official inference weights: %s",
                Path(args.out_dir) / "model.safetensors")


if __name__ == "__main__":
    main()
