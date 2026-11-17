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
from training.tasks import TASK_ORDER, TASKS
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


def evaluate_on_test(model, data_dir, train_cfg, param_stats, device, logger, out_dir):
    """在 test 划分上做无偏评估（加载导出的 best 权重）。

    test 从未参与训练/早停/导出决策，其指标是官方隐藏测试集出分前
    最可信的本地水位参考；频繁查看 test 会引入选择偏差，只在此处用一次。
    """
    from safetensors.torch import load_file
    best_path = Path(out_dir) / "model.safetensors"
    model.load_state_dict(load_file(str(best_path)))
    logger.info("[test] loaded best weights: %s", best_path)
    loaders, _ = build_task_dataloaders(
        data_dir, batch_size=train_cfg["batch_size"], seq_len=train_cfg["seq_len"],
        splits=("test",), num_workers=train_cfg.get("num_workers", 0))
    results = {t: TASKS[t].evaluate(model, loaders["test"][t], device, param_stats,
                                    train_cfg.get("ref_scale", 0.5))
               for t in TASK_ORDER if t in loaders["test"]}
    for t, m in results.items():
        logger.info("[test] %s: %s", t, {k: round(v, 5) for k, v in m.items()})
    score = (0.3 * results["protocol"]["acc"]
             + 0.3 * results["modulation"]["acc"]
             + 0.4 * results["parameter"]["precision"])
    logger.info("[test] proxy score=%.5f (unbiased; val best was the model-selection "
                "criterion, test was never seen during training)", score)
    return score


def main():
    parser = base_arg_parser("Single-backbone multi-task fine-tuning")
    parser.add_argument("--from_mae", default=None,
                        help="MAE 预训练权重路径（默认: out_dir/mae_pretrained.pt）")
    parser.add_argument("--eval_test", action="store_true",
                        help="训练结束后在 test 划分上无偏评估（加载导出的 best 权重）")
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

    if args.eval_test:
        evaluate_on_test(model, args.data_dir, train_cfg, param_stats,
                         device, logger, args.out_dir)


if __name__ == "__main__":
    main()
