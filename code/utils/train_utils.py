"""可复现训练通用组件。"""
import math
import random
import numpy as np
import torch


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device, enabled=True):
    return torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
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
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
