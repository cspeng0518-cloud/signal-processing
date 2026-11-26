"""两阶段训练脚本共享的脚手架

  - base_arg_parser:     统一命令行参数
  - normalize_config:    配置格式归一化 -> {"model": ..., "training": ...}
  - load_config:         读取 JSON 配置（自动归一化）
  - load_mae_pretrained: 把 MAE 预训练共享权重加载进微调模型
  - assert_competition_limits: 赛题红线校验（<1B 参数、共享率 >80%）
  - ROOT / CODE_ROOT:    目录常量
"""

import argparse
import json
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent

MAX_TOTAL_PARAMS = 1_000_000_000   # 赛题红线：总参数量 < 1B
MIN_SHARED_RATIO = 0.80            # 赛题红线：权重共用率 > 80%

# 旧扁平配置格式中归属 model 部分的键（新格式 {"model","training"} 直接透传）
_MODEL_KEYS = {
    "backbone", "d_model", "n_layers", "nhead", "ffn_hidden", "patch_size", "seq_len",
    "mask_ratio", "mae_decoder_layers", "mae_decoder_dim", "layer_types", "hybrid_layers",
    "mla_kv_lora_rank", "mla_q_lora_rank", "neck_layers", "neck_layer_types",
    "head_config", "dropout", "drop_path",
}

# 预训练时从 MAE checkpoint 加载的模块前缀（neck / 任务头 / 任务提示不在预训练范围）
PRETRAINED_PREFIXES = ("tokenizer.", "freq_stream.", "fusion.", "backbone.")
_NOT_PRETRAINED = ("tokenizer.task_embed",)


def normalize_config(cfg):
    """输出恒为 {"model": {...}, "training": {...}}（兼容旧扁平格式）。"""
    if "model" in cfg or "training" in cfg:
        return {"model": dict(cfg.get("model", {})), "training": dict(cfg.get("training", {}))}
    return {"model": {k: v for k, v in cfg.items() if k in _MODEL_KEYS},
            "training": {k: v for k, v in cfg.items() if k not in _MODEL_KEYS}}


def base_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "default.json"),
                        help="模型/训练 JSON 配置路径")
    parser.add_argument("--data_dir", default=str(ROOT / "datasets"),
                        help="数据目录（官方预处理 memmap 或模拟 parquet 布局）")
    parser.add_argument("--out_dir", default=str(ROOT / "weights"),
                        help="权重与导出文件输出目录")
    parser.add_argument("--resume", default=None,
                        help="断点文件（mae_resume.pt / finetune_resume.pt），续训用")
    parser.add_argument("--smoke_steps", type=int, default=0,
                        help=">0 时每 epoch 只跑指定步数且只跑 1 个 epoch（冒烟测试）")
    return parser


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return normalize_config(json.load(f))


def assert_competition_limits(model):
    """赛题红线校验（按部署参数计，不含仅预训练的 MAE 解码器），返回 (shared_ratio, total)。"""
    total = model.deploy_parameter_count()
    shared_ratio = model.shared_ratio()
    if total >= MAX_TOTAL_PARAMS:
        raise ValueError(f"总参数量 {total} 超过 1B 红线")
    if shared_ratio <= MIN_SHARED_RATIO:
        raise ValueError(f"权重共用率 {shared_ratio:.2%} 未达 80% 红线")
    return shared_ratio, total


def load_mae_pretrained(model, path, logger=None):
    """把 MAE 预训练的共享权重（tokenizer/频域流/融合/主干）加载进微调模型。

    修复旧版缺陷：旧实现按 state["tokenizer"] 这种“模块名”键查找，而导出的是扁平
    state_dict（tokenizer.xxx / backbone.xxx），结果一个张量也没加载——预训练白做。
    现按前缀筛选扁平键、核对形状，并打印实际加载数量；一个都没加载上时直接报错。

    Returns:
        int: 实际加载的张量数（路径不存在时返回 0，从头训练）。
    """
    if not Path(path).exists():
        if logger:
            logger.warning("no MAE checkpoint at %s, finetuning from scratch", path)
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    flat = {}
    for k, v in state.items():          # 兼容 {模块名: 子 state_dict} 的更早格式
        if isinstance(v, dict):
            flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            flat[k] = v
    own = model.state_dict()
    cand = {k: v for k, v in flat.items()
            if k.startswith(PRETRAINED_PREFIXES) and not k.startswith(_NOT_PRETRAINED)}
    picked = {k: v for k, v in cand.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(cand) - set(picked))
    if cand and not picked:
        raise RuntimeError(f"MAE checkpoint {path} 与当前模型结构完全不匹配（0 个张量可加载），"
                           "请确认预训练与微调使用同一份 model 配置")
    model.load_state_dict(picked, strict=False)
    expected = [k for k in own if k.startswith(PRETRAINED_PREFIXES)
                and not k.startswith(_NOT_PRETRAINED)]
    if logger:
        logger.info("loaded MAE pretrained weights from %s: %d/%d tensors (%d skipped)",
                    path, len(picked), len(expected), len(skipped))
        if skipped:
            logger.warning("  skipped (shape/key mismatch) e.g. %s", skipped[:5])
    return len(picked)
