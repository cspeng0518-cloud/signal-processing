"""阶段一：MAE 自监督预训练（三任务训练数据混用，任务无关）。

用法:
  python src/training/pretrain_mae.py --data_dir datasets --out_dir weights          # 单卡
  torchrun --nproc_per_node=N src/training/pretrain_mae.py --data_dir datasets        # 多卡
  python src/training/pretrain_mae.py --smoke_steps 2                                # 冒烟

产出: weights/mae_pretrained.pt（val 重建损失最佳时刷新）、weights/mae_resume.pt（断点）
主干: FA + MLA 混合（只编码约 25% 可见 token，省显存）；Conformer neck 不参与预训练。
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import torch                                                     # noqa: E402
from torch.utils.data import ConcatDataset                       # noqa: E402

from models import build_model                                   # noqa: E402
from training.common import ROOT, base_arg_parser, load_config   # noqa: E402
from training.trainer import MAEPretrainer                       # noqa: E402
from utils.dataset import TASKS, make_loader, make_task_dataset  # noqa: E402
from utils.logger import get_logger                              # noqa: E402
from utils.train_utils import (get_device, init_distributed, seed_all,  # noqa: E402
                               teardown_distributed)


def collate_iq(batch):
    """预训练只需要 iq（三任务样本字段不同，不能用任务 collate）。"""
    return {"iq": torch.stack([b["iq"] for b in batch])}


def build_pretrain_loaders(data_dir, tc, logger, rank=0, world_size=1):
    """train: 三任务 train 划分合并 + 长度分桶；val: 三任务 val 划分合并（早停用）。"""
    out = {}
    for split in ("train", "val"):
        dss = []
        for t in TASKS:
            try:
                ds = make_task_dataset(data_dir, t, split, seq_len=tc.get("seq_len", 1024))
            except FileNotFoundError:
                continue
            dss.append(ds)
            logger.info("pretrain %s: %s -> %d samples", split, t, len(ds))
        if not dss:
            if split == "train":
                raise FileNotFoundError(f"no training data under {data_dir}; "
                                        "run `python utils/prepare_official.py` first")
            out[split] = None
            continue
        train = split == "train"
        bs = tc["batch_size"] if train else tc.get("eval_batch_size", tc["batch_size"])
        out[split] = make_loader(ConcatDataset(dss), bs, train, tc.get("num_workers", 4),
                                 bucket=tc.get("length_bucketing", True), collate=collate_iq,
                                 seed=tc.get("seed", 42), rank=rank, world_size=world_size)
    return out["train"], out["val"]


def main():
    args = base_arg_parser("MAE self-supervised pretraining").parse_args()
    cfg = load_config(args.config)
    tc = cfg["training"]
    tc["lr"] = tc.get("lr_pretrain", tc["lr"])
    tc["batch_size"] = tc.get("batch_size_pretrain", tc["batch_size"])

    seed_all(tc.get("seed", 42))           # 各卡同 seed：初始化一致 + 梯度平均 => 权重同步
    device, rank, world_size, is_main = init_distributed()
    if device is None:
        device = get_device()
    logger = get_logger("pretrain", ROOT / "logs", quiet=not is_main)

    model = build_model(cfg["model"], with_decoder=True).to(device)
    loader, val_loader = build_pretrain_loaders(args.data_dir, tc, logger, rank, world_size)
    dec = sum(p.numel() for p in model.mae_decoder.parameters())
    logger.info("device=%s world_size=%d deploy_params=%.2fM mae_decoder=%.2fM "
                "shared_ratio=%.4f backbone=%s", device, world_size,
                model.deploy_parameter_count() / 1e6, dec / 1e6, model.shared_ratio(),
                model.backbone.layer_types)
    MAEPretrainer(cfg, model, device, logger, out_dir=args.out_dir, loader=loader,
                  val_loader=val_loader, resume=args.resume, smoke_steps=args.smoke_steps,
                  rank=rank, world_size=world_size).fit()
    teardown_distributed()


if __name__ == "__main__":
    main()
