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

from utils.metrics import accuracy, macro_recall, precision_score

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
    """参数回归任务：损失含 pattern 辅助 CE；指标用 ground-truth valid 掩码
    （仅本地监控；官方推理由 pattern 头硬掩码，见 infer.py）。"""

    def loss(self, model, feat, batch, cfg):
        pred = model.task_heads["parameter"](feat)
        reg = F.smooth_l1_loss(pred, batch["params"], reduction="none") * batch["valid"]
        reg = reg.sum() / batch["valid"].sum().clamp_min(1.0)
        pattern_logits = model.task_heads["parameter_pattern"](feat)
        aux = F.cross_entropy(pattern_logits, batch["pattern"])
        return reg + cfg.get("pattern_loss_weight", 0.3) * aux

    def evaluate(self, model, loader, device, param_stats, ref_scale):
        """指标在归一化 [0,1] 空间计算：ref_scale=0.5 是归一化空间的参考尺度，
        若先用 SI 单位反归一化再算 precision，rmse 会比 ref_scale 大数百倍，
        指标对任何非完美模型都趋近 0（SI 空间 rmse≈175 -> precision≈0.003）。
        官方提交仍由 infer.py 输出 SI 原始单位，此处仅作模型选择监控。"""
        gt, pred, valid = [], [], []
        for batch, out in self._iter_outputs(model, loader, "parameter", device):
            pred.append(out.float().cpu().numpy())
            gt.append(batch["params"].numpy())
            valid.append(batch["valid"].numpy())
        y, p = np.concatenate(gt), np.concatenate(pred)
        v = np.concatenate(valid)
        return {
            "precision": float(precision_score(y, p, ref_scale, valid=v)),
            "rmse": float(np.sqrt(np.sum((y - p) ** 2 * v) / max(1, v.sum()))),
        }


TASKS = {
    "protocol": ClassificationTask("protocol", 8),
    "modulation": ClassificationTask("modulation", 23),
    "parameter": ParameterTask(),
}
