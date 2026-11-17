"""统一文本/JSONL 指标日志，避免重复添加 handler。"""
import json
import logging
from pathlib import Path


def get_logger(name, log_dir, file_name="train.log", quiet=False):
    """quiet=True 时返回无 handler 的静默 logger（多卡训练的非主卡进程用，
    避免多进程并发写同一日志文件）。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR if quiet else logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    if quiet:
        return logger
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / file_name, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def record_metrics(log_dir, stage, epoch, metrics):
    path = Path(log_dir) / f"{stage}_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"stage": stage, "epoch": epoch, **{k:float(v) for k,v in metrics.items()}}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def record_step(log_dir, stage, step, epoch, metrics):
    """step 级指标落盘（<stage>_steps.jsonl），带 epoch 字段供出图时标注分界。"""
    path = Path(log_dir) / f"{stage}_steps.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"stage": stage, "step": step, "epoch": epoch,
           **{k: float(v) for k, v in metrics.items()}}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def reset_metrics(log_dir, stage):
    """新运行开始时清空该阶段的 jsonl（刷新而非追加）。

    训练脚本在"从头训练"时调用；断点续训不调用（保留历史继续追加，
    曲线延续同一次运行）。
    """
    for name in (f"{stage}_metrics.jsonl", f"{stage}_steps.jsonl"):
        path = Path(log_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _last_run(records, key):
    """取追加写 jsonl 中最近一次运行的数据。

    单次运行内 key（epoch 或 step）严格递增；新一次运行从开头重新编号，
    必然出现 value <= 前一行，以此为分界。不能用 "epoch==1" 判定运行起点：
    冒烟模式只有 1 个 epoch，每行都是 epoch=1，会误截断到最后一行。
    """
    start = 0
    for i in range(1, len(records)):
        if records[i].get(key, 0) <= records[i - 1].get(key, 0):
            start = i
    return records[start:]


def plot_metrics(jsonl_path, out_dir, stage):
    """训练结束后从 jsonl 指标自动绘制曲线图，保存为 out_dir/<stage>_<group>.png。

    分组出图（每组一张，同组曲线叠加便于对比）:
      loss       训练损失 / recon_mse / train_*_weighted_loss
      grad_norm  梯度范数
      lr         学习率
      val        验证指标（val_* / val_proxy_score）
    标签用英文避免训练机缺中文字体；返回生成的图片路径列表。
    """
    import matplotlib
    matplotlib.use("Agg")  # 纯保存无显示（训练机无 GUI 也能出图）
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in
               Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return []
    # jsonl 追加写会累积多次运行，只取最近一次（key 回退检测，见 _last_run；
    # 断点续训的 epoch 延续编号，会完整保留在同一 run 内）
    records = _last_run(records, "epoch")
    epochs = [r.get("epoch", i + 1) for i, r in enumerate(records)]

    def group_of(key):
        if "grad_norm" in key:
            return "grad_norm"
        if "lr" in key:
            return "lr"
        if "loss" in key or "mse" in key:
            return "loss"
        if key.startswith("val_"):
            return "val"
        return "other"

    groups = {}
    for r in records:
        for k, v in r.items():
            if k not in ("stage", "epoch"):
                groups.setdefault(group_of(k), {}).setdefault(k, []).append(float(v))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, series in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, v in series.items():
            ax.plot(epochs[: len(v)], v, marker="o", markersize=3, label=k)
        ax.set_xlabel("epoch")
        ax.set_title(f"{stage} | {name}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out / f"{stage}_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_step_metrics(jsonl_path, out_dir, stage, epoch_jsonl_path=None):
    """step 级指标出图：所有 epoch 的 step 汇总在一张图，x 轴为全局 step，
    并用竖直虚线 + 顶部文字标注 epoch 分界。

    数据源为 record_step 落盘的 <stage>_steps.jsonl；分组规则与 plot_metrics
    一致（loss / grad_norm / lr），保存为 out_dir/<stage>_step_<group>.png。

    传入 epoch_jsonl_path（<stage>_metrics.jsonl）时，val_* 指标（epoch 级）
    会折算到该 epoch 最后一个 step 的位置，以 "x" 标记叠加在 loss 图的右侧
    副轴上——train 曲线与 val 指标汇总在同一张图便于对照。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(jsonl_path)
    if not path.exists():
        return []
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return []
    # 与 plot_metrics 相同：jsonl 追加写累积多次运行，只取最近一次
    # （step 回退检测——冒烟等单 epoch 运行里每行 epoch 都相同，不能用 epoch==1 判定）
    records = _last_run(records, "step")

    # epoch 分界（step, epoch 列表，遇到 epoch 变化即一条边界）
    boundaries, last_epoch = [], None
    for r in records:
        if r.get("epoch") != last_epoch:
            boundaries.append((r["step"], r["epoch"]))
            last_epoch = r["epoch"]

    # val 指标（epoch 级）折算到该 epoch 最后一个 step 的位置，
    # 以 (step, value) 序列收集，稍后叠加到 loss 图副轴
    val_series = {}
    if epoch_jsonl_path and Path(epoch_jsonl_path).exists():
        erecs = [json.loads(line) for line in
                 Path(epoch_jsonl_path).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        erecs = _last_run(erecs, "epoch")
        epoch_last_step = {}
        for r in records:
            epoch_last_step[r["epoch"]] = r["step"]
        for r in erecs:
            s = epoch_last_step.get(r.get("epoch"))
            if s is None:
                continue
            for k, v in r.items():
                if k.startswith("val_"):
                    val_series.setdefault(k, []).append((s, float(v)))

    def group_of(key):
        if "grad_norm" in key:
            return "grad_norm"
        if "lr" in key:
            return "lr"
        if "loss" in key or "mse" in key:
            return "loss"
        return "other"

    # 按 (step, value) 成对收集：微调时三任务交替写记录，各键的 step 不连续，
    # 不能假设与全局 steps 列表前缀对齐
    groups = {}
    for r in records:
        for k, v in r.items():
            if k not in ("stage", "step", "epoch"):
                groups.setdefault(group_of(k), {}).setdefault(k, []).append(
                    (r["step"], float(v)))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, series in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, pts in series.items():
            xs, ys = zip(*pts)
            # marker：单个/稀疏数据点也可见（折线在 1 个点时不可见）
            ax.plot(xs, ys, linewidth=0.8, marker=".", markersize=3, label=k)
        # val 指标仅在 loss 图叠加（右副轴，"x" 标记），避免污染 grad_norm/lr 图
        ax2 = None
        if name == "loss" and val_series:
            ax2 = ax.twinx()
            for k, pts in val_series.items():
                xs, ys = zip(*pts)
                ax2.plot(xs, ys, linestyle="None", marker="x",
                         markersize=7, label=k)
            ax2.set_ylabel("val metrics")
            ax2.grid(False)
        # epoch 分界：竖直虚线 + 顶部编号（曲线太密时不加 marker）
        y_top = ax.get_ylim()[1]
        for s, e in boundaries:
            ax.axvline(s, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.text(s, y_top, f" ep{e}", fontsize=7, color="gray",
                    va="top", ha="left")
        ax.set_xlabel("global step")
        ax.set_title(f"{stage} | {name} (per-step, epoch boundaries marked)")
        ax.grid(alpha=0.3)
        # 合并主轴（train 曲线）与副轴（val 指标）的图例
        if ax2 is not None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2)
        else:
            ax.legend()
        fig.tight_layout()
        png = out / f"{stage}_step_{name}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        paths.append(png)
    return paths
