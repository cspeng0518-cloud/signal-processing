"""训练器：生命周期基类 + 两阶段派生实现

BaseTrainer               fit() 封装完整生命周期（optimizer/scheduler 构建、resume、
                          epoch 循环、断点保存、指标记录、出图），子类按钩子填充差异
├── MAEPretrainer         MAE 预训练（只编码可见 token）+ val 重建损失早停，导出 mae_pretrained.pt
└── MultitaskFinetuner    三任务梯度累加单步更新 + PCGrad/不确定性加权(可选) + EMA + val 早停，
                          导出官方推理格式（config.json + model.safetensors + param_codec.json）

“epoch” 统一定义为 steps_per_epoch 个优化步（0 = 遍历一次最长的训练 loader）。
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data.distributed import DistributedSampler

from training.common import PRETRAINED_PREFIXES, ROOT
from training.tasks import TASK_ORDER, TASKS, proxy_score
from utils.augment import augment_iq
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import (plot_metrics, plot_step_metrics, record_metrics, record_step,
                          reset_metrics)
from utils.optim import ModelEMA, build_optimizer, build_scheduler
from utils.train_utils import (all_reduce_sum, autocast_context, broadcast_flag,
                               clip_and_step, optimizer_step, tqdm)

try:                                       # TensorBoard 可选（缺失时不影响训练）
    from torch.utils.tensorboard import SummaryWriter
except Exception:                          # pragma: no cover
    SummaryWriter = None


# ================================================================ 导出
def export_pretrained(model, out_dir):
    """导出 MAE 预训练共享权重（只保留会被微调加载的模块，体积小）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k.startswith(PRETRAINED_PREFIXES)}
    torch.save(state, out_dir / "mae_pretrained.pt")


def export_model(model, cfg, codec, out_dir):
    """导出官方推理格式：config.json（内嵌 param_codec）+ model.safetensors + param_codec.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(cfg))
    cfg["model"]["param_codec"] = codec.spec
    cfg["model"]["param_head_spec"] = codec.head_spec()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    sd = model.deploy_state_dict()
    save_file({k: v.detach().contiguous().clone().cpu() for k, v in sd.items()},
              str(out_dir / "model.safetensors"))
    codec.save(out_dir / "param_codec.json")


class _Cycle:
    """无限循环迭代器（steps_per_epoch 与 loader 长度解耦）。"""

    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


# ================================================================ 基类
class BaseTrainer(ABC):
    tag = "train"
    resume_name = None

    def __init__(self, cfg, model, device, logger, out_dir, resume=None, smoke_steps=0,
                 rank=0, world_size=1):
        self.cfg = cfg
        self.tcfg = cfg["training"]
        self.model = model
        self.device = device
        self.logger = logger
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.smoke_steps = smoke_steps
        self.rank, self.world_size = rank, world_size
        self.is_main = rank == 0
        self.start_epoch = 0
        self.global_step = 0
        self._epoch = 0
        self.optimizer = build_optimizer(self.train_named_parameters(), self.tcfg)
        self.scheduler = build_scheduler(self.optimizer, self.total_steps(),
                                         self._warmup_steps(), self.tcfg.get("min_lr_ratio", 0.05))
        self.writer = (SummaryWriter(log_dir=str(ROOT / "logs" / "tb" / self.tag))
                       if self.is_main and SummaryWriter is not None else None)
        if resume:
            self.start_epoch, extra = load_checkpoint(
                resume, model=model, optimizer=self.optimizer, scheduler=self.scheduler)
            self.global_step = int(extra.get("global_step", 0))
            self.on_resume(extra)
            logger.info("resumed from %s at epoch %d", resume, self.start_epoch)

    # ---------------- 调度 ----------------
    def steps_per_epoch(self):
        spe = self._spe
        return min(spe, self.smoke_steps) if self.smoke_steps else spe

    def total_steps(self):
        return max(1, self.steps_per_epoch() * self.n_epochs())

    def _warmup_steps(self):
        """warmup_steps>0 优先，否则 warmup_ratio×总步数；上限为总步数的 20%。

        旧 conformer 分支固定 warmup_steps=2000：只跑 300 步对比实验时学习率始终
        停留在峰值的 ~15%，是那次对比分数偏低的原因之一——此处加上限并打印提示。
        """
        total = self.total_steps()
        warm = int(self.tcfg.get("warmup_steps", 0) or total * self.tcfg.get("warmup_ratio", 0.05))
        cap = max(1, total // 5)
        if warm > cap:
            self.logger.warning("warmup %d > 20%% of total steps %d, clamped to %d",
                                warm, total, cap)
            warm = cap
        return max(1, warm)

    # ---------------- 钩子 ----------------
    def train_named_parameters(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    @abstractmethod
    def train_one_epoch(self):
        """一个 epoch 的训练，返回指标 dict。"""

    @abstractmethod
    def n_epochs(self):
        """总 epoch 数。"""

    def on_epoch_end(self, epoch):
        return {}

    def checkpoint_extra(self):
        return {}

    def on_resume(self, extra):
        pass

    def should_stop(self):
        return False

    def on_finish(self):
        pass

    def _train_loaders(self):
        return []

    # ---------------- 日志 ----------------
    def _log_scalars(self, values):
        if not self.is_main:
            return
        if self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(k, v, self.global_step)
        record_step(ROOT / "logs", self.tag, self.global_step, self._epoch + 1, values)

    def _log_epoch(self, values, epoch):
        if not self.is_main or self.writer is None:
            return
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"epoch/{k}", v, epoch)
        self.writer.flush()

    def _set_sampler_epoch(self, epoch):
        """DistributedSampler 需显式 set_epoch；LengthGroupedBatchSampler 每次迭代自动推进。"""
        for loader in self._train_loaders():
            if isinstance(getattr(loader, "sampler", None), DistributedSampler):
                loader.sampler.set_epoch(epoch)

    # ---------------- 生命周期 ----------------
    def fit(self):
        if self.is_main and self.start_epoch == 0:
            reset_metrics(ROOT / "logs", self.tag)
        try:
            for epoch in range(self.start_epoch, self.n_epochs()):
                self._epoch = epoch
                self._set_sampler_epoch(epoch)
                values = self.train_one_epoch()
                values["lr"] = self.scheduler.get_last_lr()[0]
                values.update(self.on_epoch_end(epoch))
                if self.is_main:
                    record_metrics(ROOT / "logs", self.tag, epoch + 1, values)
                    self.logger.info("epoch=%d %s", epoch + 1,
                                     " ".join(f"{k}={v:.5f}" for k, v in values.items()))
                    self._log_epoch(values, epoch + 1)
                    save_checkpoint(self.out_dir / self.resume_name, epoch=epoch,
                                    model=self.model, optimizer=self.optimizer,
                                    scheduler=self.scheduler, global_step=self.global_step,
                                    **self.checkpoint_extra())
                if self.smoke_steps:
                    break
                if broadcast_flag(self.should_stop(), self.device, self.world_size):
                    self.logger.info("early stopping at epoch %d", epoch + 1)
                    break
        finally:
            if self.writer is not None:
                self.writer.close()
        if self.is_main:
            try:
                for png in plot_metrics(ROOT / "logs" / f"{self.tag}_metrics.jsonl",
                                        ROOT / "logs" / "plots", self.tag):
                    self.logger.info("plot saved: %s", png)
                for png in plot_step_metrics(ROOT / "logs" / f"{self.tag}_steps.jsonl",
                                             ROOT / "logs" / "plots", self.tag,
                                             ROOT / "logs" / f"{self.tag}_metrics.jsonl"):
                    self.logger.info("step plot saved: %s", png)
            except Exception as e:                       # 出图失败不影响导出
                self.logger.warning("plotting failed: %s", e)
            self.on_finish()


# ================================================================ 阶段一：MAE 预训练
class MAEPretrainer(BaseTrainer):
    tag = "pretrain"
    resume_name = "mae_resume.pt"

    def __init__(self, cfg, model, device, logger, out_dir, loader, val_loader=None,
                 resume=None, smoke_steps=0, rank=0, world_size=1):
        self.loader, self.val_loader = loader, val_loader
        self.best_loss, self.bad = float("inf"), 0
        self._spe = cfg["training"].get("steps_per_epoch_pretrain", 0) or len(loader)
        super().__init__(cfg, model, device, logger, out_dir, resume, smoke_steps,
                         rank, world_size)
        self.cycle = _Cycle(loader)

    def train_named_parameters(self):
        return [(n, p) for n, p in self.model.named_parameters()
                if p.requires_grad and not n.startswith(("task_heads.", "neck.",
                                                          "neck_freq_type",
                                                          "tokenizer.task_embed"))]

    def _train_loaders(self):
        return [self.loader]

    def n_epochs(self):
        return self.tcfg["epochs_pretrain"]

    def train_one_epoch(self):
        self.model.train()
        n = self.steps_per_epoch()
        tot, cnt, norms = 0.0, 0, []
        parts_sum = {"loss_time": 0.0, "loss_stft": 0.0}
        log_every = self.tcfg.get("log_every", 50)
        params = [p for _, p in self.train_named_parameters()]
        for step in tqdm(range(n), desc=f"pretrain epoch {self._epoch + 1}/{self.n_epochs()}",
                         leave=False, dynamic_ncols=True, disable=not self.is_main):
            iq = self.cycle.next()["iq"].to(self.device, non_blocking=True)
            iq = augment_iq(iq, "pretrain", self.tcfg.get("augment_policy"))
            with autocast_context(self.device, self.tcfg.get("amp", True)):
                loss, parts = self.model.forward_pretrain(iq)
            gn = optimizer_step(loss, self.optimizer, self.model,
                                self.tcfg.get("grad_clip", 1.0), self.world_size, params)
            self.scheduler.step()
            self.global_step += 1
            if step % log_every == 0 or step == n - 1:
                lv = float(loss)
                tot, cnt = tot + lv, cnt + 1
                norms.append(gn)
                for k in parts_sum:
                    parts_sum[k] += float(parts[k])
                self._log_scalars({"train/loss": lv, "train/grad_norm": gn,
                                   "train/lr": self.scheduler.get_last_lr()[0]})
        self._epoch_loss = tot / max(1, cnt)
        out = {"recon_loss": self._epoch_loss, "grad_norm": sum(norms) / max(1, len(norms))}
        out.update({k: v / max(1, cnt) for k, v in parts_sum.items()})
        return out

    @torch.no_grad()
    def _val_loss(self):
        """固定随机种子掩码下的 val 重建损失（多卡 all-reduce 汇总）。"""
        if self.val_loader is None:
            return None
        self.model.eval()
        n_max = self.tcfg.get("val_batches_pretrain", 50)
        acc = torch.zeros(2, dtype=torch.float64, device=self.device)
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1234 + self.rank)
            for i, batch in enumerate(self.val_loader):
                if i >= n_max:
                    break
                with autocast_context(self.device, self.tcfg.get("amp", True)):
                    loss, _ = self.model.forward_pretrain(batch["iq"].to(self.device))
                acc[0] += float(loss)
                acc[1] += 1
        acc = all_reduce_sum(acc, self.world_size)
        self.model.train()
        return float(acc[0] / acc[1].clamp_min(1))

    def on_epoch_end(self, epoch):
        if self.smoke_steps:
            return {}
        v = self._val_loss()
        metric = v if v is not None else self._epoch_loss
        if metric < self.best_loss * (1 - self.tcfg.get("min_delta_pretrain", 1e-3)):
            self.best_loss, self.bad = metric, 0
            if self.is_main:
                export_pretrained(self.model, self.out_dir)
                self.logger.info("pretrain best %s=%.5f -> mae_pretrained.pt",
                                 "val_recon" if v is not None else "train_recon", metric)
        else:
            self.bad += 1
        return {"val_recon_loss": v} if v is not None else {}

    def checkpoint_extra(self):
        return {"best_loss": self.best_loss, "bad": self.bad}

    def on_resume(self, extra):
        self.best_loss = extra.get("best_loss", float("inf"))
        self.bad = extra.get("bad", 0)

    def should_stop(self):
        return self.bad >= self.tcfg.get("patience_pretrain", 5)

    def on_finish(self):
        if self.smoke_steps or not (self.out_dir / "mae_pretrained.pt").exists():
            export_pretrained(self.model, self.out_dir)
        self.logger.info("best pretrained weights -> %s", self.out_dir / "mae_pretrained.pt")


# ================================================================ 阶段二：多任务微调
class MultitaskFinetuner(BaseTrainer):
    """一次迭代 = 三任务各一个 batch 的梯度累加 -> 一次优化步（共享参数看到联合梯度）。

    可选 PCGrad（Yu et al., NeurIPS 2020）消解冲突梯度；可选不确定性加权
    （Kendall et al., CVPR 2018）；EMA 权重评估与导出；val 代理分数早停。
    """

    tag = "finetune"
    resume_name = "finetune_resume.pt"

    def __init__(self, cfg, model, device, logger, out_dir, loaders, codec, resume=None,
                 smoke_steps=0, rank=0, world_size=1):
        self.loaders = loaders
        self.codec = codec
        tc = cfg["training"]
        self.weights = tc.get("task_weights", {"protocol": 0.5, "modulation": 1.0,
                                               "parameter": 1.0})
        self.log_vars = None
        if tc.get("task_weighting", "fixed") == "uncertainty":
            self.log_vars = torch.nn.Parameter(torch.zeros(len(TASK_ORDER), device=device))
        self._spe = tc.get("steps_per_epoch", 0) or max(len(loaders["train"][t])
                                                        for t in TASK_ORDER)
        self.best, self.bad = -float("inf"), 0
        self.ema = ModelEMA(model, tc["ema_decay"]) if tc.get("ema_decay", 0) > 0 else None
        super().__init__(cfg, model, device, logger, out_dir, resume, smoke_steps,
                         rank, world_size)
        self.cycles = {t: _Cycle(loaders["train"][t]) for t in TASK_ORDER}
        self.pcgrad = bool(tc.get("pcgrad", False))
        priv = {id(p) for p in model.tokenizer.private_parameters()}
        heads = {id(p) for p in model.task_heads.parameters()}
        self.shared_params = [p for p in model.parameters()
                              if p.requires_grad and id(p) not in priv and id(p) not in heads]
        self.all_params = [p for _, p in self.train_named_parameters()]

    def train_named_parameters(self):
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if self.log_vars is not None:
            named.append(("log_vars.bias", self.log_vars))      # ".bias" -> 不做权重衰减
        return named

    def n_epochs(self):
        return self.tcfg["epochs_finetune"]

    def _train_loaders(self):
        return list(self.loaders["train"].values())

    def _weighted(self, i, task, loss):
        if self.log_vars is not None:
            s = self.log_vars[i]
            return 0.5 * torch.exp(-s) * loss + 0.5 * s
        return float(self.weights.get(task, 1.0)) * loss

    def _flat_shared_grad(self):
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in self.shared_params])

    def _apply_pcgrad(self, grads):
        out = []
        for i, g in enumerate(grads):
            g = g.clone()
            for j in torch.randperm(len(grads)).tolist():
                if j == i:
                    continue
                dot = torch.dot(g, grads[j])
                if dot < 0:
                    g -= dot / (grads[j].pow(2).sum() + 1e-12) * grads[j]
            out.append(g)
        total = torch.stack(out).sum(0)
        off = 0
        for p in self.shared_params:
            n = p.numel()
            p.grad = total[off:off + n].view_as(p).to(p.dtype)
            off += n

    def train_one_epoch(self):
        self.model.train()
        n = self.steps_per_epoch()
        sums = {t: 0.0 for t in TASK_ORDER}
        cnt, norms = 0, []
        log_every = self.tcfg.get("log_every", 50)
        amp = self.tcfg.get("amp", True)
        for step in tqdm(range(n), desc=f"finetune epoch {self._epoch + 1}/{self.n_epochs()}",
                         leave=False, dynamic_ncols=True, disable=not self.is_main):
            self.optimizer.zero_grad(set_to_none=True)
            grads, losses = [], {}
            for i, task in enumerate(TASK_ORDER):
                batch = self.cycles[task].next()
                batch = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with autocast_context(self.device, amp):
                    loss, _ = TASKS[task].loss(self.model, batch, self.tcfg)
                    wl = self._weighted(i, task, loss)
                wl.backward()                     # 逐任务反传：峰值显存 = 单任务前反向
                losses[task] = loss.detach()
                if self.pcgrad:
                    grads.append(self._flat_shared_grad())
                    for p in self.shared_params:
                        p.grad = None
            if self.pcgrad:
                self._apply_pcgrad(grads)
            gn = clip_and_step(self.optimizer, self.model, self.tcfg.get("grad_clip", 1.0),
                               self.world_size, self.all_params)
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)
            self.global_step += 1
            if step % log_every == 0 or step == n - 1:
                cnt += 1
                norms.append(gn)
                vals = {f"train/loss_{t}": float(v) for t, v in losses.items()}
                for t, v in losses.items():
                    sums[t] += float(v)
                vals.update({"train/grad_norm": gn, "train/lr": self.scheduler.get_last_lr()[0]})
                if self.log_vars is not None:
                    vals.update({f"train/logvar_{t}": float(self.log_vars[i])
                                 for i, t in enumerate(TASK_ORDER)})
                self._log_scalars(vals)
        out = {f"train_{t}_loss": sums[t] / max(1, cnt) for t in TASK_ORDER}
        out["grad_norm"] = sum(norms) / max(1, len(norms))
        return out

    def eval_model(self):
        m = self.ema.ema if self.ema is not None else self.model
        m.codec = self.codec
        return m

    def evaluate(self, split="val"):
        model = self.eval_model()
        results = {}
        with autocast_context(self.device, self.tcfg.get("amp", True)):
            for t in TASK_ORDER:
                results[t] = TASKS[t].evaluate(model, self.loaders[split][t], self.device,
                                               self.tcfg, self.world_size)
        self.model.train()
        return results

    def on_epoch_end(self, epoch):
        if self.smoke_steps or (epoch + 1) % self.tcfg.get("eval_every", 1) != 0:
            return {}
        missing = [t for t in TASK_ORDER if t not in self.loaders.get("val", {})]
        if missing:
            raise RuntimeError(f"missing validation split for tasks: {missing}")
        results = self.evaluate("val")
        values = {f"val_{t}_{k}": v for t, m in results.items() for k, v in m.items()
                  if k != "n"}
        score = proxy_score(results, self.tcfg.get("score_weights"))
        values["val_proxy_score"] = score
        if score > self.best + self.tcfg.get("min_delta", 1e-4):
            self.best, self.bad = score, 0
            if self.is_main:
                export_model(self.eval_model(), self.cfg, self.codec, self.out_dir)
                self.logger.info("new best proxy score=%.5f (exported %s weights)", score,
                                 "EMA" if self.ema is not None else "raw")
        else:
            self.bad += 1
        return values

    def checkpoint_extra(self):
        extra = {"best": self.best, "bad": self.bad}
        if self.ema is not None:
            extra["ema"] = self.ema.state_dict()
        if self.log_vars is not None:
            extra["log_vars"] = self.log_vars.detach().cpu()
        return extra

    def on_resume(self, extra):
        self.best = extra.get("best", -float("inf"))
        self.bad = extra.get("bad", 0)
        if self.ema is not None and "ema" in extra:
            self.ema.load_state_dict(extra["ema"])
        if self.log_vars is not None and "log_vars" in extra:
            with torch.no_grad():
                self.log_vars.copy_(extra["log_vars"].to(self.log_vars.device))

    def should_stop(self):
        return self.bad >= self.tcfg.get("patience", 8)

    def on_finish(self):
        if self.smoke_steps or not (self.out_dir / "model.safetensors").exists():
            export_model(self.eval_model(), self.cfg, self.codec, self.out_dir)
        self.logger.info("finished; official inference weights: %s",
                         self.out_dir / "model.safetensors")
