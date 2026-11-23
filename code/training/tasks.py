"""任务抽象：损失与评估按任务类型派生，调用方零分支。

BaseTaskModule                 统一 loss / evaluate 契约
├── ClassificationTask         CE 损失；acc / macro_recall（protocol、modulation 参数化复用）
└── ParameterTask              有效维 smooth L1 + pattern 辅助 CE；precision / rmse

TASKS 注册表在初始化时完成实现选择，训练/评估代码直接查表分发。
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import accuracy, macro_recall

# 任务名 -> 任务模块（注册表）
TASK_ORDER = ("protocol", "modulation", "parameter")


class BaseTaskModule(ABC):
    """任务抽象基类：统一 loss / evaluate 契约。"""

    @abstractmethod
    def loss(self, model, feat, batch, cfg):
        """单任务损失。feat 为 model.encode 的输出，头由任务模块自选。"""

    @abstractmethod
    def evaluate(self, model, loader, device, param_stats, ref_scale):
        """验证集指标 dict（键进入 metrics jsonl，如 val_protocol_acc）。"""

    @staticmethod
    def _iter_outputs(model, loader, task, device):
        """公共评估骨架：逐 batch 编码 + 任务头，yield (batch, 输出)。"""
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                feat = model.encode(batch["iq"].to(device), batch["sr"].to(device))
                yield batch, model.task_heads[task](feat)


class ClassificationTask(BaseTaskModule):
    """分类任务（protocol / modulation 参数化复用）。"""

    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes

    def loss(self, model, feat, batch, cfg):
        return F.cross_entropy(model.task_heads[self.name](feat), batch["label"])

    def evaluate(self, model, loader, device, param_stats, ref_scale):
        gt, pred = [], []
        for batch, out in self._iter_outputs(model, loader, self.name, device):
            gt.append(batch["label"].numpy())
            pred.append(out.argmax(-1).cpu().numpy())
        y, p = np.concatenate(gt), np.concatenate(pred)
        return {"acc": float(accuracy(y, p)),
                "macro_recall": float(macro_recall(y, p, self.n_classes))}


class ParameterTask(BaseTaskModule):
    """参数回归任务：损失含 pattern 辅助 CE；评估与 conformer 分支同一口径
    （SI 空间 + 预测 pattern 硬掩码 + 逐参数 nRMSE，见 evaluate 文档）。"""

    def loss(self, model, feat, batch, cfg):
        pred = model.task_heads["parameter"](feat)
        reg = F.smooth_l1_loss(pred, batch["params"], reduction="none") * batch["valid"]
        reg = reg.sum() / batch["valid"].sum().clamp_min(1.0)
        pattern_logits = model.task_heads["parameter_pattern"](feat)
        aux = F.cross_entropy(pattern_logits, batch["pattern"])
        return reg + cfg.get("pattern_loss_weight", 0.3) * aux

    def evaluate(self, model, loader, device, param_stats, ref_scale):
        """SI 空间评估，与 conformer 分支逐项一致。

        流程：归一化预测 -> 反归一化到 SI（裁剪到训练范围）-> 用预测 pattern
        硬掩码（模式判错的样本该族参数预测置 0，与官方推理一致）-> 每个参数
        计算 RMSE / 训练取值范围，7 个参数取均值得到 nrmse（无量纲）。
        precision = 1 / (1 + nrmse / ref_scale)，对应赛题 P=1/(1+RMSE/参考尺度)
        的同形公式；两分支 ref_scale 均取 0.5，分数可直接横向比较。
        """
        mn = np.asarray(param_stats["min"], dtype=np.float64)
        mx = np.asarray(param_stats["max"], dtype=np.float64)
        rng = np.maximum(mx - mn, 1e-12)
        gt, pred, valid, pattern = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                feat = model.encode(batch["iq"].to(device), batch["sr"].to(device))
                gt.append(batch["params"].numpy())
                valid.append(batch["valid"].numpy())
                pred.append(model.task_heads["parameter"](feat).float().cpu().numpy())
                pattern.append(
                    model.task_heads["parameter_pattern"](feat).argmax(-1).cpu().numpy())
        y_norm, p_norm = np.concatenate(gt), np.concatenate(pred)
        v = np.concatenate(valid).astype(bool)
        pat = np.concatenate(pattern)
        # 无效维真值在归一化空间已置 0，反归一化后被 valid 掩码排除，不参与误差
        y = y_norm * rng + mn
        p = np.clip(p_norm * rng + mn, mn, mx)
        keep = np.zeros_like(p)
        keep[:, :3] = (pat == 0)[:, None]   # 预测通信：保留前 3 个参数
        keep[:, 3:] = (pat == 1)[:, None]   # 预测雷达：保留后 4 个参数
        p = p * keep
        per = []
        for j in range(y.shape[1]):
            m = v[:, j]
            if not m.any():
                continue
            per.append(np.sqrt(np.mean((p[m, j] - y[m, j]) ** 2)) / rng[j])
        nrmse = float(np.mean(per)) if per else 0.0
        return {"nrmse": nrmse,
                "precision": float(1.0 / (1.0 + nrmse / ref_scale))}


TASKS = {
    "protocol": ClassificationTask("protocol", 8),
    "modulation": ClassificationTask("modulation", 23),
    "parameter": ParameterTask(),
}
