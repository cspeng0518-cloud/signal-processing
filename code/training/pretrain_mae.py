"""阶段一：MAE 自监督预训练（任务无关，全部任务训练数据混用）。

用法:
  python training/pretrain_mae.py --data_dir <dir> [--config ...] [--resume ...]     # 单卡
  torchrun --nproc_per_node=N training/pretrain_mae.py --data_dir <dir> [...]        # 多卡

本文件只负责：参数/配置/数据组装 -> MAEPretrainer.fit()。
训练生命周期（调度/断点/日志/导出）见 training/trainer.py，
任务损失/评估的多态实现见 training/tasks.py。
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from models import EMFoundation
from training.common import ROOT, base_arg_parser, load_config
from training.trainer import MAEPretrainer
from utils.dataset import make_task_dataset
from utils.logger import get_logger
from utils.train_utils import get_device, init_distributed, seed_all, teardown_distributed

# 预训练混用全部任务的训练集（任务无关）
PRETRAIN_TASKS = ("protocol", "modulation", "parameter")


class UnlabeledIQ(Dataset):
    """无标签 IQ 包装（MAE 只需 iq/sr）。"""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]["iq"], self.ds[i]["sr"]


def build_pretrain_loader(data_dir, cfg, device, logger, rank=0, world_size=1):
    """合并各任务训练集 -> 无标签 loader（多卡时 DistributedSampler 分片）。"""
    datasets, n_total = [], 0
    for t in PRETRAIN_TASKS:
        try:
            ds = make_task_dataset(data_dir, t, "train")
        except FileNotFoundError:
            logger.info("skip %s (no train split)", t)
            continue
        datasets.append(ds)
        n_total += len(ds)
        logger.info("pretrain: %s -> %d samples", t, len(ds))
    if not datasets:
        raise FileNotFoundError(
            f"no training data under {data_dir}; run `python utils/prepare_official.py` "
            f"first (produces datasets/<task>/iq.memap + meta.json), or check --data_dir")
    merged = UnlabeledIQ(ConcatDataset(datasets))
    sampler = (DistributedSampler(merged, num_replicas=world_size, rank=rank,
                                  shuffle=True)
               if world_size > 1 else None)
    return DataLoader(
        merged,
        batch_size=cfg["batch_size"], shuffle=(sampler is None), sampler=sampler,
        drop_last=True, num_workers=cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"), persistent_workers=False)


def main():
    args = base_arg_parser("MAE self-supervised pretraining").parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["training"].get("seed", 42))  # 各卡同 seed：初始化一致 + 梯度平均 => 权重同步
    device, rank, world_size, is_main = init_distributed()
    if device is None:
        device = get_device()
    logger = get_logger("pretrain", ROOT / "logs", quiet=not is_main)

    loader = build_pretrain_loader(args.data_dir, cfg["training"], device, logger,
                                   rank, world_size)
    model = EMFoundation(cfg["model"]).to(device)
    MAEPretrainer(cfg, model, device, logger, out_dir=args.out_dir, loader=loader,
                  resume=args.resume, smoke_steps=args.smoke_steps,
                  rank=rank, world_size=world_size).fit()
    teardown_distributed()


if __name__ == "__main__":
    main()
