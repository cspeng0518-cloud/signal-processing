"""可复现训练通用组件。"""
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


def all_reduce_grads(model, world_size, bucket_mb=32):
    """手动梯度同步：按 dtype 分桶拼平后 all-reduce 求平均（替代逐参数 all-reduce）。

    不用 DDP 封装的原因：模型有 forward_pretrain / head_outputs 等多个自定义前向入口，
    且预训练时任务头不参与反传。各卡同 seed 初始化 + 梯度平均 => 权重始终同步。
    分桶（默认 32MB）把上千次小通信合并为几十次，16 卡 HCCL/NCCL 下显著提速。
    """
    if world_size < 2:
        return
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    limit = bucket_mb * 1024 * 1024
    buckets, cur, cur_bytes, cur_dtype = [], [], 0, None
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        nbytes = g.numel() * g.element_size()
        if cur and (g.dtype != cur_dtype or cur_bytes + nbytes > limit):
            buckets.append(cur)
            cur, cur_bytes = [], 0
        cur.append(g)
        cur_bytes += nbytes
        cur_dtype = g.dtype
    if cur:
        buckets.append(cur)
    for grads in buckets:
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat.div_(world_size)
        for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)


def all_reduce_sum(t, world_size):
    """张量 all-reduce 求和（单卡直接返回）。评估用：混淆矩阵 / 误差平方和精确汇总。"""
    if world_size < 2:
        return t
    dist.all_reduce(t)
    return t


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


def optimizer_step(loss, optimizer, model, clip, world_size=1, params=None):
    """反向 + 多卡梯度同步 + 裁剪 + 优化器步进，返回裁剪前的全局梯度范数。

    bf16 autocast 动态范围与 fp32 同阶，无需 GradScaler（已移除 fp16 loss scaling 分支）。
    """
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    return clip_and_step(optimizer, model, clip, world_size, params)


def clip_and_step(optimizer, model, clip, world_size=1, params=None):
    """梯度已就位后：多卡同步 -> 裁剪 -> step，返回裁剪前梯度范数。"""
    all_reduce_grads(model, world_size)
    params = params if params is not None else [p for p in model.parameters()
                                                if p.grad is not None]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
    optimizer.step()
    return float(grad_norm)
