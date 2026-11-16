"""训练断点（checkpoint）保存/加载，预训练与微调脚本共用。"""

from pathlib import Path

import torch

_STATE_KEYS = ("epoch", "model", "optimizer", "scheduler", "scaler")


def save_checkpoint(path, *, epoch, model, optimizer, scheduler, scaler, **extra):
    """保存可恢复训练所需的全部状态。

    extra 中的键值（如 finetune 的 best/bad）会一并写入 checkpoint，
    并在 load_checkpoint 时原样返回。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        **extra,
    }
    torch.save(state, path)


def load_checkpoint(path, *, model, optimizer, scheduler, scaler, device="cpu"):
    """恢复训练状态。

    Returns:
        (start_epoch, extra): start_epoch 为断点 epoch+1；extra 为
        save_checkpoint 时传入的附加字段（无则空 dict）。
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    extra = {k: v for k, v in state.items() if k not in _STATE_KEYS}
    return state["epoch"] + 1, extra
