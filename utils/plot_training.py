"""读取 JSONL 训练日志生成 PNG/PDF，无需 TensorBoard 服务。"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="logs/finetune_metrics.jsonl")
    p.add_argument("--output_dir",default="figures/training")
    args=p.parse_args()
    lines=[json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines: raise ValueError("empty log")
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    keys=sorted(set().union(*(row.keys() for row in lines)) - {"epoch","stage"})
    for key in keys:
        pairs=[(row["epoch"],row[key]) for row in lines if key in row]
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot([a for a,b in pairs],[b for a,b in pairs],marker="o",markersize=3)
        ax.set(xlabel="Epoch",ylabel=key,title=key);ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(out/f"{key}.png",dpi=180);fig.savefig(out/f"{key}.pdf")
        plt.close(fig)
    print(f"Saved {len(keys)} metric charts to {out}")

if __name__=="__main__":main()
