"""统一文本/JSONL 指标日志，避免重复添加 handler。"""
import json
import logging
from pathlib import Path


def get_logger(name, log_dir, file_name="train.log"):
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
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
    # jsonl 追加写会累积多次运行，只取最近一次（从最后一个 epoch==1 截断；
    # 断点续训的 epoch 延续编号，会完整保留在同一 run 内）
    starts = [i for i, r in enumerate(records) if r.get("epoch") == 1]
    if starts:
        records = records[starts[-1]:]
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
