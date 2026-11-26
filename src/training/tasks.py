"""任务模块：损失 + 评估（调用方查 TASKS 表分发，零分支）

分类（protocol / modulation）
  L = CE(label_smoothing) + λ_fam·CE(调制族, 仅 modulation) + λ_sup·SupCon(可选)
参数（parameter）
  L = mean_j L_j + λ_pat·CE(pattern)
      L_j: 连续参数 -> 编码空间 Huber/MSE（按有效样本平均）
           离散参数 -> CE（按有效样本平均）
      7 个参数等权（不受通信 3 个 / 雷达 4 个的数量影响）

评估（模型选择与监控；官方分数以主办方评测为准）
  分类: acc、macro_recall —— 由混淆矩阵 all-reduce 后精确计算（多卡结果与单卡一致）
  参数: pattern_acc；每参数 SI 空间 RMSE；按训练集取值范围归一化的 nRMSE；
        precision = 1/(1 + mean nRMSE / ref_scale)
        使用“预测的” pattern 硬掩码（与官方推理一致：模式判错的参数输出 0 计入误差）
        —— 误差平方和/计数 all-reduce 后精确计算
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from utils.augment import augment_iq
from utils.train_utils import all_reduce_sum

TASK_ORDER = ("protocol", "modulation", "parameter")


def supcon_loss(z, y, temperature=0.1):
    """监督对比损失（Khosla et al., NeurIPS 2020），单视角 batch 内同类为正样本。"""
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.t() / temperature
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    npos = pos.sum(1)
    valid = npos > 0
    loss = -(log_prob * pos).sum(1)[valid] / npos[valid]
    return loss.mean() if valid.any() else z.sum() * 0


class BaseTask(ABC):
    name = ""

    @abstractmethod
    def loss(self, model, batch, cfg):
        """返回 (loss, logs: dict[str, tensor])。batch 已在设备上。"""

    @abstractmethod
    def evaluate(self, model, loader, device, cfg, world_size=1):
        """返回指标 dict（多卡时内部 all-reduce，各卡结果一致）。"""

    @staticmethod
    def _aug(batch, name, cfg):
        if cfg.get("augment", True):
            return augment_iq(batch["iq"], name, cfg.get("augment_policy"))
        return batch["iq"]


class ClassificationTask(BaseTask):
    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes

    def loss(self, model, batch, cfg):
        iq = self._aug(batch, self.name, cfg)
        out = model.head_outputs(iq, self.name, return_all=True)
        y = batch["label"]
        ls = (cfg.get("label_smoothing") or {}).get(self.name, 0.0)
        loss = F.cross_entropy(out["logits"].float(), y, label_smoothing=ls)
        logs = {"ce": loss.detach()}
        if out.get("aux") is not None:
            l_fam = F.cross_entropy(out["aux"].float(), model.mod_family[y])
            loss = loss + cfg.get("family_loss_weight", 0.2) * l_fam
            logs["fam"] = l_fam.detach()
        w_sup = (cfg.get("supcon_weight") or {}).get(self.name, 0.0)
        if w_sup > 0 and out.get("proj") is not None:
            l_sup = supcon_loss(out["proj"], y, cfg.get("supcon_temperature", 0.1))
            loss = loss + w_sup * l_sup
            logs["supcon"] = l_sup.detach()
        return loss, logs

    @torch.no_grad()
    def evaluate(self, model, loader, device, cfg, world_size=1):
        model.eval()
        C = self.n_classes
        cm = torch.zeros(C * C, dtype=torch.float64, device=device)
        for batch in loader:
            logits = model.head_outputs(batch["iq"].to(device, non_blocking=True), self.name)
            y = batch["label"].to(device)
            p = logits.argmax(-1)
            cm += torch.bincount(y * C + p, minlength=C * C).double()
        cm = all_reduce_sum(cm, world_size).view(C, C).cpu().numpy()
        n = cm.sum()
        support = cm.sum(1)
        recalls = np.diag(cm)[support > 0] / support[support > 0]
        return {"acc": float(np.trace(cm) / max(n, 1)),
                "macro_recall": float(recalls.mean()) if len(recalls) else 0.0,
                "n": float(n)}


class ParameterTask(BaseTask):
    name = "parameter"

    @staticmethod
    def _reg_loss(pred, target, kind):
        if kind == "mse":
            return (pred - target) ** 2
        if kind == "huber":
            return F.huber_loss(pred, target, reduction="none", delta=1.0)
        return F.smooth_l1_loss(pred, target, reduction="none", beta=0.5)

    def loss(self, model, batch, cfg):
        codec = model.codec
        iq = self._aug(batch, "parameter", cfg)
        out = model.head_outputs(iq, "parameter")
        valid = batch["valid"].float()
        z, cls_t = codec.encode(batch["raw_params"])
        kind = cfg.get("param_loss", "huber")
        per_param = []
        for k, j in enumerate(codec.reg_idx):
            v = valid[:, j]
            l = self._reg_loss(out["reg"][:, k].float(), z[:, k], kind)
            per_param.append((l * v).sum() / v.sum().clamp_min(1.0))
        for k, j in enumerate(codec.cls_idx):
            v = valid[:, j]
            l = F.cross_entropy(out["cls"][k].float(), cls_t[:, k], reduction="none")
            per_param.append((l * v).sum() / v.sum().clamp_min(1.0))
        reg = torch.stack(per_param).mean()
        pat = F.cross_entropy(out["pattern"].float(), batch["pattern"])
        loss = reg + cfg.get("pattern_loss_weight", 0.3) * pat
        return loss, {"param": reg.detach(), "pattern": pat.detach()}

    @torch.no_grad()
    def predict_batch(self, model, iq):
        """-> (SI 预测 [B,7]（已按预测 pattern 置零）, 预测 pattern [B])。"""
        out = model.head_outputs(iq, "parameter")
        si = model.codec.decode(out)
        radar = out["pattern"].argmax(-1)
        mask = torch.zeros_like(si)
        mask[:, :3] = (radar == 0).float()[:, None]
        mask[:, 3:] = (radar == 1).float()[:, None]
        return si * mask, radar

    @torch.no_grad()
    def evaluate(self, model, loader, device, cfg, world_size=1):
        model.eval()
        n_p = len(model.codec.names)
        # [SSE_0..6, count_0..6, pattern_correct, n]，float64 精确累加
        acc = torch.zeros(2 * n_p + 2, dtype=torch.float64, device=device)
        for batch in loader:
            pred, radar = self.predict_batch(model, batch["iq"].to(device, non_blocking=True))
            y = batch["raw_params"].to(device).double()
            v = batch["valid"].to(device).double()
            e2 = (pred.double() - y) ** 2 * v
            acc[:n_p] += e2.sum(0)
            acc[n_p:2 * n_p] += v.sum(0)
            acc[2 * n_p] += (radar == batch["pattern"].to(device)).double().sum()
            acc[2 * n_p + 1] += y.size(0)
        acc = all_reduce_sum(acc, world_size).cpu().numpy()
        sse, cnt = acc[:n_p], acc[n_p:2 * n_p]
        n = acc[2 * n_p + 1]
        rng = model.codec.vmax - model.codec.vmin
        res = {"pattern_acc": float(acc[2 * n_p] / max(n, 1)), "n": float(n)}
        nrmse = []
        for j, name in enumerate(model.codec.names):
            if cnt[j] <= 0:
                continue
            rm = float(np.sqrt(sse[j] / cnt[j]))
            res[f"rmse_{name}"] = rm
            nrmse.append(rm / max(rng[j], 1e-30))
        mean_n = float(np.mean(nrmse)) if nrmse else 0.0
        res["nrmse"] = mean_n
        res["precision"] = 1.0 / (1.0 + mean_n / cfg.get("ref_scale", 0.5))
        return res


TASKS = {
    "protocol": ClassificationTask("protocol", 8),
    "modulation": ClassificationTask("modulation", 23),
    "parameter": ParameterTask(),
}


def proxy_score(results, weights=None):
    """本地代理总分：0.3·protocol_acc + 0.3·modulation_acc + 0.4·parameter_precision。"""
    w = weights or {"protocol": 0.3, "modulation": 0.3, "parameter": 0.4}
    return (w["protocol"] * results["protocol"]["acc"]
            + w["modulation"] * results["modulation"]["acc"]
            + w["parameter"] * results["parameter"]["precision"])
