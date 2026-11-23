"""阶段二：单一共享基座多任务微调（protocol / modulation / parameter 联合训练）。

用法:
  python code/training/finetune_multitask.py --data_dir datasets --out_dir weights [--eval_test]
  torchrun --nproc_per_node=N code/training/finetune_multitask.py --data_dir datasets
  python code/training/finetune_multitask.py --smoke_steps 2                         # 冒烟

流程: 拟合/加载 ParamCodec -> 构建模型（无 MAE 解码器）-> 加载 mae_pretrained.pt 共享权重
      （主干 FA/MLA + 双流 tokenizer + 融合；Conformer neck 与任务头随机初始化）
      -> 赛题红线校验 -> MultitaskFinetuner.fit() -> 可选 test 无偏评估
产出: weights/config.json + model.safetensors + param_codec.json（官方推理格式）
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import torch.distributed as dist                                              # noqa: E402

from models import build_model                                                # noqa: E402
from training.common import (ROOT, assert_competition_limits, base_arg_parser,  # noqa: E402
                             load_config, load_mae_pretrained)
from training.tasks import TASK_ORDER, TASKS, proxy_score                     # noqa: E402
from training.trainer import MultitaskFinetuner                               # noqa: E402
from utils.dataset import build_task_dataloaders, load_param_arrays           # noqa: E402
from utils.logger import get_logger                                           # noqa: E402
from utils.param_codec import ParamCodec                                      # noqa: E402
from utils.train_utils import (autocast_context, get_device, init_distributed,  # noqa: E402
                               seed_all, teardown_distributed)


def prepare_codec(args, tc, logger, is_main):
    """参数编解码器：out_dir 已有则复用（保证续训/推理一致），否则在 train 划分上拟合。"""
    path = Path(args.out_dir) / "param_codec.json"
    reuse = path.exists() and not args.refit_codec
    if dist.is_initialized():
        dist.barrier()                     # 各卡先统一判定是否复用，再由主卡写文件（防读到半写文件）
    if reuse:
        codec = ParamCodec.load(path)
        logger.info("loaded param codec: %s", path)
    else:
        try:
            p, v = load_param_arrays(args.data_dir, "train", tc.get("seq_len", 1024))
            cc = tc.get("codec", {})
            codec = ParamCodec.fit(p, v, max_classes=cc.get("max_classes", 32),
                                   log_ratio=cc.get("log_ratio", 20.0), force=cc.get("force"),
                                   decode=cc.get("decode", "expectation"))
            logger.info("fitted param codec on %d train samples", len(p))
        except FileNotFoundError:
            codec = ParamCodec.default()
            logger.warning("no parameter train data found; using default codec")
        if is_main:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            codec.save(path)
    logger.info("param codec:\n%s", codec.describe())
    return codec


def evaluate_on_test(trainer, data_dir, logger):
    """用导出的 best 权重在 test 划分上做一次无偏评估（test 从不参与训练期决策）。"""
    from safetensors.torch import load_file
    model = trainer.eval_model()
    sd = load_file(str(trainer.out_dir / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    tc = trainer.tcfg
    loaders = build_task_dataloaders(data_dir, seq_len=tc.get("seq_len", 1024),
                                     splits=("test",), num_workers=tc.get("num_workers", 4),
                                     eval_batch_size=tc.get("eval_batch_size", 1024))
    if len(loaders["test"]) < len(TASK_ORDER):
        logger.warning("[test] test split incomplete, skipped")
        return
    res = {}
    with autocast_context(trainer.device, tc.get("amp", True)):
        for t in TASK_ORDER:
            res[t] = TASKS[t].evaluate(model, loaders["test"][t], trainer.device, tc)
            logger.info("[test] %s: %s", t, {k: round(v, 6) for k, v in res[t].items()})
    logger.info("[test] proxy score=%.5f", proxy_score(res, tc.get("score_weights")))


def main():
    parser = base_arg_parser("Single-backbone multi-task fine-tuning")
    parser.add_argument("--from_mae", default=None,
                        help="MAE 预训练权重路径（默认: out_dir/mae_pretrained.pt）")
    parser.add_argument("--eval_test", action="store_true",
                        help="训练结束后在 test 划分上无偏评估（加载导出的 best 权重）")
    parser.add_argument("--refit_codec", action="store_true",
                        help="强制重新拟合 param_codec.json（数据划分变化后使用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tc = cfg["training"]
    tc["lr"] = tc.get("lr_finetune", tc["lr"])
    tc["batch_size"] = tc.get("batch_size_finetune", tc["batch_size"])

    seed_all(tc.get("seed", 42))
    device, rank, world_size, is_main = init_distributed()
    if device is None:
        device = get_device()
    logger = get_logger("finetune", ROOT / "logs", quiet=not is_main)

    codec = prepare_codec(args, tc, logger, is_main)
    cfg["model"]["param_head_spec"] = codec.head_spec()
    cfg["model"]["param_codec"] = codec.spec

    loaders = build_task_dataloaders(
        args.data_dir, batch_size=tc["batch_size"], seq_len=tc.get("seq_len", 1024),
        num_workers=tc.get("num_workers", 4), bucket=tc.get("length_bucketing", True),
        eval_batch_size=tc.get("eval_batch_size", 1024), rank=rank, world_size=world_size,
        seed=tc.get("seed", 42))
    missing = [t for t in TASK_ORDER if t not in loaders["train"]]
    if missing:
        raise RuntimeError(f"missing required train tasks: {missing}")

    model = build_model(cfg["model"], with_decoder=False).to(device)
    model.codec = codec
    shared, total = assert_competition_limits(model)
    logger.info("device=%s world_size=%d params=%.2fM shared_ratio=%.2f%% backbone=%s neck=%s",
                device, world_size, total / 1e6, shared * 100, model.backbone.layer_types,
                model.neck.layer_types if model.neck is not None else None)

    if not args.resume:                    # 续训时沿用断点权重，不重复加载 MAE 初始化
        load_mae_pretrained(model, args.from_mae or str(Path(args.out_dir) / "mae_pretrained.pt"),
                            logger)

    trainer = MultitaskFinetuner(cfg, model, device, logger, out_dir=args.out_dir,
                                 loaders=loaders, codec=codec, resume=args.resume,
                                 smoke_steps=args.smoke_steps, rank=rank, world_size=world_size)
    trainer.fit()
    if args.eval_test and is_main:
        evaluate_on_test(trainer, args.data_dir, logger)
    teardown_distributed()


if __name__ == "__main__":
    main()
