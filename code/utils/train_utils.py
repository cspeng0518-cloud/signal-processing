"""可复现训练通用组件。"""
import math
import random
import numpy as np
import torch
from tqdm import tqdm  # 进度条（训练可视化，固定依赖）


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device, enabled=True):
    """bf16 混合精度 autocast（预训练后量化友好，无 CUDA 专有算子）。

    bf16 动态范围与 fp32 同阶，无需梯度缩放；NPU/CUDA 均原生支持。
    """
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=bool(enabled and device.type in ("cuda", "npu")))


def build_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    warmup = int(total_steps * warmup_ratio)
    def fn(step):
        if step < warmup:
            return max(1e-6, (step+1)/max(1,warmup))
        t = (step-warmup)/max(1,total_steps-warmup)
        return max(0.01, 0.5*(1+math.cos(math.pi*min(t,1.0))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def optimizer_step(loss, optimizer, model, scaler, clip):
    """反向 + 裁剪 + 优化器步进，返回裁剪前的全局梯度范数（供监控）。"""
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
    return float(grad_norm)
