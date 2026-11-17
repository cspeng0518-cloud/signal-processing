"""可复现训练通用组件。"""
import math
import os
import random
import numpy as np
import torch
import torch.distributed as dist
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


# ================================================================ 分布式
def init_distributed():
    """torchrun 多卡启动时初始化进程组；单卡直接 python 运行返回 (None, 0, 1, True)。

    返回 (device, rank, world_size, is_main)。device 已绑定本进程的 LOCAL_RANK
    对应设备（GPU 用 nccl，昇腾 NPU 用 hccl）。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None, 0, 1, True
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % world_size)))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        device, backend = torch.device("npu", local_rank), "hccl"
    else:
        torch.cuda.set_device(local_rank)
        device, backend = torch.device("cuda", local_rank), "nccl"
    dist.init_process_group(backend=backend)
    return device, rank, world_size, rank == 0


def teardown_distributed():
    """训练结束后销毁进程组（未初始化时为空操作）。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_grads(model, world_size):
    """手动梯度同步：逐参数 all-reduce 求和后取平均。

    不用 DDP 封装的原因：模型有 forward_mae / encode 等多个自定义前向入口，
    且 MAE 预训练时任务头不参与反传（DDP 需开 find_unused_parameters）。
    各卡同 seed 初始化 + 梯度平均，保证权重始终同步。
    """
    if world_size < 2:
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad.div_(world_size)


def broadcast_flag(flag, device, world_size):
    """主卡广播布尔标志（如早停决策），保证各卡 epoch 循环步调一致。"""
    if world_size < 2:
        return bool(flag)
    t = torch.tensor([int(bool(flag))], dtype=torch.int64, device=device)
    dist.broadcast(t, src=0)
    return bool(t.item())


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


def optimizer_step(loss, optimizer, model, scaler, clip, world_size=1):
    """反向 + 多卡梯度同步 + 裁剪 + 优化器步进，返回裁剪前的全局梯度范数。

    多卡时在 unscale 之后、裁剪之前做梯度 all-reduce（平均），各卡看到
    完全一致的梯度，权重更新保持同步。
    """
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    all_reduce_grads(model, world_size)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
    return float(grad_norm)
