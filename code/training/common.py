"""两阶段训练脚本（pretrain_mae / finetune_multitask）共享的脚手架：

  - base_arg_parser: 统一命令行参数
  - load_config:      读取 JSON 配置
  - load_mae_pretrained: 加载 MAE 预训练共享权重（微调用）
  - ROOT / CODE_ROOT: 目录常量
"""

import argparse
import json
from pathlib import Path

import torch

# code/training/common.py -> parents[1]=code/, parents[2]=仓库根
CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent


def base_arg_parser(description):
    """预训练/微调共用的命令行参数（脚本可在此基础上追加专属参数）。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(CODE_ROOT / "configs" / "default.json"),
                        help="模型/训练 JSON 配置路径")
    parser.add_argument("--data_dir", default=str(ROOT / "datasets"),
                        help="数据目录（官方预处理 memmap 或模拟 parquet 布局）")
    parser.add_argument("--out_dir", default=str(ROOT / "weights"),
                        help="权重与导出文件输出目录")
    parser.add_argument("--resume", default=None,
                        help="断点文件（mae_resume.pt / finetune_resume.pt），续训用")
    parser.add_argument("--smoke_steps", type=int, default=0,
                        help=">0 时每 epoch 只跑指定步数（冒烟测试，跳过评估）")
    return parser


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mae_pretrained(model, path, logger=None):
    """将 MAE 预训练的共享权重加载进模型（仅加载 checkpoint 中存在且模型有的模块）。

    任务头与 MAE decoder 不在预训练导出范围内，保持随机初始化。

    Returns:
        bool: 是否成功加载。
    """
    if not Path(path).exists():
        if logger:
            logger.info("no MAE checkpoint at %s, training from scratch", path)
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    loaded = []
    for key in ("tokenizer", "backbone", "frequency_tokenizer", "cross_domain_fusion"):
        if key in state and hasattr(model, key):
            getattr(model, key).load_state_dict(state[key])
            loaded.append(key)
    if logger:
        logger.info("loaded MAE pretrained weights from %s [%s]", path, ", ".join(loaded))
    return True
