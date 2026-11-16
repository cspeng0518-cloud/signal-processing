"""阶段二：单基座三任务联合微调，校验后保存最佳 checkpoint。"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from models import EMFoundation
from utils.dataset import build_task_dataloaders, save_param_stats
from utils.metrics import accuracy, macro_recall, precision_score
from utils.logger import get_logger,record_metrics
from utils.train_utils import seed_all,get_device,autocast_context,build_scheduler,optimizer_step
TASKS=("protocol","modulation","parameter")


def parameter_loss(pred,target,valid):
    """逐元素 masked smooth L1，归一化于有效目标数。"""
    return (F.smooth_l1_loss(pred,target,reduction="none")*valid).sum()/valid.sum().clamp_min(1.0)


def evaluate(model,loader,task,device,stats,reference):
    model.eval(); gt=[];pred=[];valid=[]
    with torch.inference_mode():
        for b in loader:
            iq=b["iq"].to(device);sr=b["sr"].to(device)
            feat=model.encode(iq,sr)
            out=model.task_heads[task](feat)
            if task=="parameter":
                raw=out.float().cpu().numpy()
                raw=raw*(stats["max"]-stats["min"])+stats["min"]
                truth=b["params"].numpy()*(stats["max"]-stats["min"])+stats["min"]
                gt.append(truth);pred.append(raw);valid.append(b["valid"].numpy())
            else:
                gt.append(b["label"].numpy());pred.append(out.argmax(-1).cpu().numpy())
    y=np.concatenate(gt); p=np.concatenate(pred)
    if task=="parameter":
        v=np.concatenate(valid)
        # 仅 valid=true 的维度评估；无效分量不参与本地指标
        return {"precision":float(precision_score(y,p,reference,valid=v)),
                "rmse":float(np.sqrt(np.sum((y-p)**2*v)/max(1,v.sum())))}
    n_cls=8 if task=="protocol" else 23
    return {"acc":float(accuracy(y,p)),"macro_recall":float(macro_recall(y,p,n_cls))}


def export_model(model,cfg,stats,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tensors={f"model.{k}":v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    save_file(tensors,str(out/"model.safetensors"))
    (out/"config.json").write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8")
    save_param_stats(stats,out/"param_stats.json")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=str(ROOT/"code/configs/default.json"))
    p.add_argument("--data_dir",default=str(ROOT/"datasets"))
    p.add_argument("--out_dir",default=str(ROOT/"weights"))
    p.add_argument("--from_mae",default=None)
    p.add_argument("--resume",default=None)
    p.add_argument("--smoke_steps",type=int,default=0)
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"));seed_all(cfg.get("seed",42))
    device=get_device(); logger=get_logger("finetune",ROOT/"logs")
    loaders,stats=build_task_dataloaders(args.data_dir,batch_size=cfg["batch_size"],seq_len=cfg["seq_len"])
    missing=[t for t in TASKS if t not in loaders["train"]]
    if missing:raise RuntimeError(f"Missing required train tasks: {missing}")
    model=EMFoundation(cfg).to(device)
    if model.shared_ratio()<=.8:raise RuntimeError("Shared parameter ratio must exceed 80%")
    if sum(p.numel() for p in model.parameters())>=1_000_000_000:raise RuntimeError("Model exceeds 1B parameters")
    mae=args.from_mae or str(Path(args.out_dir)/"mae_pretrained.pt")
    if Path(mae).exists() and not args.resume:
        sd=torch.load(mae,map_location="cpu",weights_only=False)
        for key in ("tokenizer","backbone","frequency_tokenizer","cross_domain_fusion"):
            if key in sd and hasattr(model,key):getattr(model,key).load_state_dict(sd[key])
        logger.info("loaded pretrained backbone from %s",mae)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    n_steps=max(len(loaders["train"][t]) for t in TASKS)
    if args.smoke_steps:n_steps=min(n_steps,args.smoke_steps)
    scheduler=build_scheduler(optimizer,n_steps*len(TASKS)*cfg["epochs_finetune"],cfg.get("warmup_ratio",.1))
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and cfg.get("amp",True)))
    start=0;best=-float("inf");bad=0
    if args.resume:
        state=torch.load(args.resume,map_location="cpu",weights_only=False)
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"])
        start=state["epoch"]+1;best=state["best"];bad=state.get("bad",0)
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    logger.info("device=%s shared=%.2f%% model_params=%d",device,model.shared_ratio()*100,sum(p.numel() for p in model.parameters()))
    weights=cfg.get("task_weights",{"protocol":.3,"modulation":.3,"parameter":.4})
    for epoch in range(start,cfg["epochs_finetune"]):
        model.train();accum={t:0 for t in TASKS};iters={t:iter(loaders["train"][t]) for t in TASKS}
        for step in range(n_steps):
            for task in TASKS:
                try:b=next(iters[task])
                except StopIteration:
                    iters[task]=iter(loaders["train"][task]);b=next(iters[task])
                iq=b["iq"].to(device);sr=b["sr"].to(device)
                with autocast_context(device,cfg.get("amp",True)):
                    feat=model.encode(iq,sr)
                    out=model.task_heads[task](feat)
                    if task=="parameter":
                        target=b["params"].to(device); valid=b["valid"].to(device)
                        loss=parameter_loss(out,target,valid)
                        pattern=model.task_heads["parameter_pattern"](feat)
                        loss=loss+cfg.get("pattern_loss_weight",.3)*F.cross_entropy(pattern,b["pattern"].to(device))
                    else:loss=F.cross_entropy(out,b["label"].to(device))
                    loss=loss*float(weights[task])
                optimizer_step(loss,optimizer,model,scaler,cfg.get("grad_clip",1.0))
                scheduler.step();accum[task]+=loss.item()
        values={f"train_{t}_weighted_loss":accum[t]/max(1,n_steps) for t in TASKS}
        values["lr"]=scheduler.get_last_lr()[0]
        if not args.smoke_steps and (epoch+1)%cfg.get("eval_every",1)==0:
            if any(t not in loaders.get("val",{}) for t in TASKS):
                raise RuntimeError("Missing validation split for one or more tasks")
            results={t:evaluate(model,loaders["val"][t],t,device,stats,cfg.get("ref_scale",.5)) for t in TASKS}
            for t,metrics in results.items():
                for key,val in metrics.items():values[f"val_{t}_{key}"]=val
            # 与初赛 30/30/40 权重一致；参数精度仅为非官方参考尺度的本地近似
            score=.3*results["protocol"]["acc"]+.3*results["modulation"]["acc"]+.4*results["parameter"]["precision"]
            values["val_proxy_score"]=score
            if score>best:
                best=score;bad=0;export_model(model,cfg,stats,args.out_dir)
                logger.info("new best proxy score=%.5f",best)
            else:bad+=1
        record_metrics(ROOT/"logs","finetune",epoch+1,values)
        logger.info("epoch=%d %s",epoch+1," ".join(f"{k}={v:.5f}" for k,v in values.items()))
        torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"best":best,"bad":bad},
                   Path(args.out_dir)/"finetune_resume.pt")
        if args.smoke_steps:break
        if bad>=cfg.get("patience",7):
            logger.info("early stopping at epoch %d",epoch+1);break
    if args.smoke_steps:export_model(model,cfg,stats,args.out_dir)
    logger.info("finished. official inference weights: %s",Path(args.out_dir)/"model.safetensors")

if __name__=="__main__":main()
