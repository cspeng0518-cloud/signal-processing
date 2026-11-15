"""MAE 掩码重建预训练（对应赛题任务1的时域表征学习 + EMind 预训练阶段）

用法:
    python code/training/pretrain_mae.py --config code/configs/default.json [--skip]

流程:
  1. 读取三任务全部训练数据（MAE 不需要标签，所有 IQ 混用）
  2. 随机掩码 75% patch，backbone 只编码可见 token，轻量 decoder 重建
  3. 保存 backbone + tokenizer 权重（供微调初始化）
"""

import os
import sys
import json
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_ROOT)  # models/utils 包根（code/）

from models import EMFoundation
from utils.dataset import EMSignalDataset, collate_fn, PARAM_NAMES, FS


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(
            os.path.join(log_dir, "train.log"), encoding="utf-8")],
    )
    return logging.getLogger("pretrain")


class MixedMAEDataset(EMSignalDataset):
    """MAE 用：忽略标签，只取 IQ"""

    def __getitem__(self, i):
        row = self.df.iloc[i]
        from utils.dataset import bytes_to_iq
        iq = bytes_to_iq(row["iq"], self.seq_len)
        return {"iq": torch.from_numpy(iq),
                "sr": torch.tensor(FS, dtype=torch.float32)}


def mae_collate(batch):
    return {"iq": torch.stack([b["iq"] for b in batch]),
            "sr": torch.stack([b["sr"] for b in batch])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "code", "configs", "default.json"))
    parser.add_argument("--data_dir", default=os.path.join(ROOT, "datasets"))
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "weights"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    logger = setup_logger(os.path.join(ROOT, "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}, backbone: {cfg['backbone']}")

    # ---- 数据：三任务训练 IQ 混用 ----
    ds_list = []
    for task in ("system", "modulation", "parameter"):
        p = os.path.join(args.data_dir, f"train_{task}.parquet")
        ds_list.append(MixedMAEDataset(p, task, seq_len=cfg["seq_len"]))
    dataset = torch.utils.data.ConcatDataset(ds_list)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=mae_collate, num_workers=cfg["num_workers"])
    logger.info(f"MAE pretraining on {len(dataset)} unlabeled IQ samples")

    # ---- 模型 ----
    model = EMFoundation(cfg).to(device)
    logger.info(f"params: shared={model.shared_parameter_count()/1e6:.2f}M "
                f"heads={model.head_parameter_count()/1e3:.1f}K "
                f"shared_ratio={model.shared_ratio():.1%}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                              weight_decay=cfg["weight_decay"])
    criterion = nn.MSELoss()
    epochs = cfg["epochs_pretrain"]

    model.train()
    for ep in range(epochs):
        total, n = 0.0, 0
        for batch in loader:
            iq = batch["iq"].to(device)  # [B,N,2]
            sr = batch["sr"].to(device)
            recon, target = model.forward_mae(iq, sr)
            loss = criterion(recon, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * iq.size(0)
            n += iq.size(0)
        logger.info(f"epoch {ep+1}/{epochs}  recon_mse={total/n:.6f}")

    # ---- 保存 ----
    os.makedirs(args.out_dir, exist_ok=True)
    state = {
        "tokenizer": model.tokenizer.state_dict(),
        "backbone": model.backbone.state_dict(),
        "mae_decoder": model.mae_decoder.state_dict(),
    }
    torch.save(state, os.path.join(args.out_dir, "mae_pretrained.pt"))
    logger.info(f"saved MAE weights -> {args.out_dir}/mae_pretrained.pt")


if __name__ == "__main__":
    main()
