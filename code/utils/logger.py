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
