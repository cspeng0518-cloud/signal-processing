"""训练断点（checkpoint）保存/加载，预训练与微调共用。"""

from pathlib import Path

import torch

_STATE_KEYS = ("epoch", "model", "optimizer", "scheduler")


def save_checkpoint(path, *, epoch, model, optimizer, scheduler, **extra):
    """保存可恢复训练所需的全部状态；extra（如 best/bad/ema）原样写入并在加载时返回。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                **extra}, tmp)
    tmp.replace(path)                      # 原子替换，写断点时被打断不会损坏旧断点


def load_checkpoint(path, *, model, optimizer, scheduler, device="cpu"):
    """Returns (start_epoch, extra)：start_epoch = 断点 epoch + 1。"""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    extra = {k: v for k, v in state.items() if k not in _STATE_KEYS and k != "scaler"}
    return state["epoch"] + 1, extra
