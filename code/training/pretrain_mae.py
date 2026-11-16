"""阶段一：时频双流 MAE 自监督预训练。"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from models import EMFoundation
from utils.dataset import make_task_dataset
from utils.logger import get_logger, record_metrics
from utils.train_utils import seed_all, get_device, autocast_context, build_scheduler, optimizer_step


class UnlabeledIQ(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, i):
        item = self.dataset[i]
        return item["iq"], item["sr"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=str(ROOT/"code/configs/default.json"))
    p.add_argument("--data_dir",default=str(ROOT/"datasets"))
    p.add_argument("--out_dir",default=str(ROOT/"weights"))
    p.add_argument("--resume",default=None)
    p.add_argument("--smoke_steps",type=int,default=0)
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed_all(cfg.get("seed",42)); device=get_device()
    logger=get_logger("pretrain",ROOT/"logs")
    datasets=[]
    for task in ("protocol","modulation","parameter"):
        try:
            datasets.append(UnlabeledIQ(make_task_dataset(args.data_dir,task,"train",cfg["seq_len"])))
        except FileNotFoundError:
            logger.warning("Missing training data for %s",task)
    if not datasets: raise FileNotFoundError("No training data")
    loader=DataLoader(ConcatDataset(datasets),batch_size=cfg["batch_size"],shuffle=True,
                      num_workers=cfg.get("num_workers",0),pin_memory=(device.type=="cuda"))
    model=EMFoundation(cfg).to(device)
    optim=torch.optim.AdamW(list(model.tokenizer.parameters()) + list(model.backbone.parameters()) +
                             list(model.mae_decoder.parameters()) + list(model.mask_token for _ in [0]) +
                             (list(model.frequency_tokenizer.parameters())+list(model.cross_domain_fusion.parameters()) if model.use_frequency else []),
                             lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    steps=min(len(loader),args.smoke_steps) if args.smoke_steps else len(loader)
    scheduler=build_scheduler(optim,max(1,steps*cfg["epochs_pretrain"]),cfg.get("warmup_ratio",0.1))
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and cfg.get("amp",True)))
    start=0
    if args.resume:
        state=torch.load(args.resume,map_location="cpu",weights_only=False)
        model.load_state_dict(state["model"]);optim.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"])
        start=state["epoch"]+1
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    logger.info("device=%s samples=%s shared_ratio=%.3f",device,len(loader.dataset),model.shared_ratio())
    for epoch in range(start,cfg["epochs_pretrain"]):
        model.train();total=0; count=0
        for i,(iq,sr) in enumerate(loader):
            if args.smoke_steps and i>=args.smoke_steps: break
            iq=iq.to(device);sr=sr.to(device)
            with autocast_context(device,cfg.get("amp",True)):
                recon,target=model.forward_mae(iq,sr)
                loss=(recon.float()-target.float()).square().mean()
            optimizer_step(loss,optim,model,scaler,cfg.get("grad_clip",1.0))
            scheduler.step();total+=loss.item()*len(iq);count+=len(iq)
        avg=total/max(1,count)
        record_metrics(ROOT/"logs","pretrain",epoch+1,{"recon_mse":avg,"lr":scheduler.get_last_lr()[0]})
        logger.info("epoch=%d masked_MSE=%.6f",epoch+1,avg)
        torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":optim.state_dict(),
                    "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict()},
                   Path(args.out_dir)/"mae_resume.pt")
    torch.save({"tokenizer":model.tokenizer.state_dict(),"backbone":model.backbone.state_dict(),
                **({"frequency_tokenizer":model.frequency_tokenizer.state_dict(),
                    "cross_domain_fusion":model.cross_domain_fusion.state_dict()} if model.use_frequency else {})},
               Path(args.out_dir)/"mae_pretrained.pt")
    logger.info("saved MAE pretrained backbone")

if __name__=="__main__":main()
