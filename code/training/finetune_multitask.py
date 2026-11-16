"""阶段二：单 backbone 多任务微调（protocol / modulation / parameter）。

用法:
  python training/finetune_multitask.py --data_dir <dir> [--from_mae ...]
                                       [--config ...] [--resume ...]

本文件只负责：参数/配置/数据组装 -> 多任务微调 -> fit()。
训练生命周期（调度/断点/早停/评估/导出）见 training/trainer.py，
任务损失/评估的多态实现见 training/tasks.py。
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from models import EMFoundation
from training.common import ROOT, base_arg_parser, load_config, load_mae_pretrained
from training.tasks import TASK_ORDER
from training.trainer import MultitaskFinetuner
from utils.dataset import build_task_dataloaders
from utils.logger import get_logger
from utils.train_utils import get_device, seed_all

MAX_TOTAL_PARAMS = 1_000_000_000   # 赛题红线：总参数量 < 1B
MIN_SHARED_RATIO = 0.80            # 赛题红线：权重共用率 > 80%


def assert_competition_limits(model):
    """赛题红线校验，返回 (shared_ratio, total_params)。"""
    shared_ratio, total = model.shared_ratio(), \
        sum(p.numel() for p in model.parameters())
    if total >= MAX_TOTAL_PARAMS:
        raise ValueError(f"总参数量 {total} 超过 1B 红线")
    if shared_ratio <= MIN_SHARED_RATIO:
        raise ValueError(f"权重共用率 {shared_ratio:.2%} 未达 80% 红线")
    return shared_ratio, total


def main():
    parser = base_arg_parser("Single-backbone multi-task fine-tuning")
    parser.add_argument("--from_mae", default=None,
                        help="MAE 预训练权重路径（默认: out_dir/mae_pretrained.pt）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    seed_all(train_cfg.get("seed", 42))
    device = get_device()
    logger = get_logger("finetune", ROOT / "logs")

    # ---- 数据 ----
    loaders, param_stats = build_task_dataloaders(
        args.data_dir, batch_size=train_cfg["batch_size"], seq_len=train_cfg["seq_len"],
        num_workers=train_cfg.get("num_workers", 0))
    missing = [t for t in TASK_ORDER if t not in loaders["train"]]
    if missing:
        raise RuntimeError(f"missing required train tasks: {missing}")

    # ---- 模型 ----
    model = EMFoundation(cfg["model"]).to(device)
    shared, total = assert_competition_limits(model)
    logger.info("device=%s shared_ratio=%.2f%% params=%d", device, shared * 100, total)

    if not args.resume:  # resume 时继续原权重，不重复加载 MAE 初始化
        mae_path = args.from_mae or str(Path(args.out_dir) / "mae_pretrained.pt")
        load_mae_pretrained(model, mae_path, logger)

    MultitaskFinetuner(cfg, model, device, logger, out_dir=args.out_dir,
                       loaders=loaders, param_stats=param_stats,
                       resume=args.resume, smoke_steps=args.smoke_steps).fit()


if __name__ == "__main__":
    main()
