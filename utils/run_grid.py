"""grid.json 批量训练 runner：逐配置跑两阶段训练, 只保留历史最佳权重。

用法:
  python utils/run_grid.py --dry                 # 预览将跑哪些配置
  python utils/run_grid.py                       # 全量批量训练（顺序执行）
  python utils/run_grid.py --only 1024d_27l      # 只跑名字含该子串的配置
  python utils/run_grid.py --limit 2 --smoke_steps 2   # 冒烟验证流水线
  python utils/run_grid.py --retry               # 重跑上次失败的配置

每个配置的产物归档到 logs/grid_runs/<name>/:
  config.json            实际训练用的完整配置（可复现）
  pretrain.log/finetune.log   两阶段完整训练日志
  pretrain_metrics.jsonl / finetune_metrics.jsonl  epoch 级指标
  *_steps.jsonl          step 级指标
  plots/                 trainer 自动生成的指标曲线图（每配置独立归档）

全局产物:
  weights/best/          历史最佳一组权重（model.safetensors + config.json +
                         param_stats.json, 官方推理格式）——非最佳配置的权重训完即清理
  logs/grid_runs/best.json          最佳配置汇总（名称/得分/完整配置条目）
  logs/grid_runs/state.json         批量进度（断点续跑: done 的自动跳过）
  logs/grid_runs/summary.jsonl      每配置一行成绩流水
  logs/grid_runs/plots/proxy_curves.png   各配置 val 曲线对比
  logs/grid_runs/plots/best_bar.png       各配置最佳得分排名

说明:
  - 收敛判据沿用 trainer 的 val_proxy_score（0.3*protocol_acc +
    0.3*modulation_acc + 0.4*parameter_precision, 早停 patience）。
  - 单卡顺序执行；多卡需自行 torchrun, 本脚本不托管。
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID_DEFAULT = ROOT / "code" / "configs" / "grid.json"


# ---------------------------------------------------------------- 工具
def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in
            p.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_score_from_metrics(jsonl_records):
    """从 finetune_metrics.jsonl 取本次运行的最佳 val_proxy_score。"""
    scores = [r["val_proxy_score"] for r in jsonl_records
              if "val_proxy_score" in r]
    return max(scores) if scores else None


def detect_card_count():
    """探测本机可用加速卡数（优先昇腾 NPU，其次 CUDA GPU），探测失败返回 1。"""
    probe = ("import torch\n"
             "n = (torch.npu.device_count() if hasattr(torch, 'npu') "
             "and torch.npu.is_available() else torch.cuda.device_count())\n"
             "print(max(n, 1))")
    try:
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                             text=True, timeout=120)
        return int(out.stdout.strip().splitlines()[-1])
    except Exception:
        return 1


def run_stage(stage_script, cfg_path, data_dir, out_dir, log_path,
              smoke_steps, nproc=1):
    """跑一个训练阶段, stdout/stderr 全量落 log 文件, 返回进程返回码。

    nproc > 1 时用 torchrun（GPU 走 nccl, 昇腾 NPU 走 hccl）拉起多卡 DDP。
    """
    script = str(ROOT / "code" / "training" / stage_script)
    if nproc > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={nproc}", "--standalone",
               script,
               "--config", str(cfg_path), "--data_dir", str(data_dir),
               "--out_dir", str(out_dir)]
    else:
        cmd = [sys.executable, script,
               "--config", str(cfg_path), "--data_dir", str(data_dir),
               "--out_dir", str(out_dir)]
    if smoke_steps:
        cmd += ["--smoke_steps", str(smoke_steps)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                              ).returncode


def archive_logs(run_dir, logs_dir):
    """把 trainer 写到全局 logs/ 的本配置指标/图归档进 run_dir。"""
    copied = []
    for name in ("pretrain_metrics.jsonl", "finetune_metrics.jsonl",
                 "pretrain_steps.jsonl", "finetune_steps.jsonl"):
        src = logs_dir / name
        if src.exists() and src.read_text(encoding="utf-8").strip():
            shutil.copy2(src, run_dir / name)
            copied.append(name)
    plots_dir = logs_dir / "plots"
    if plots_dir.is_dir():
        # 只归档本 runner 两阶段的图（pretrain_*/finetune_*），历史手动训练的
        # 其他图不混入
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        for png in plots_dir.glob("*.png"):
            if png.name.startswith(("pretrain", "finetune")):
                shutil.copy2(png, run_dir / "plots" / png.name)
    return copied


def update_best(best_json_path, best_dir, name, score, entry, weights_run_dir):
    """新最佳: 覆盖 weights/best/ 与 best.json; 返回是否刷新。"""
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True)
    for f in ("model.safetensors", "config.json", "param_stats.json"):
        src = weights_run_dir / f
        if src.exists():
            shutil.copy2(src, best_dir / f)
    # 单独输出可直接喂回训练入口的最佳配置
    (best_dir / "best_config.json").write_text(
        json.dumps({"model": entry["model"], "training": entry["training"]},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    best_json_path.write_text(
        json.dumps({"name": name, "val_proxy_score": score,
                    "params_M": entry.get("params_M"),
                    "vram_est_gb": entry.get("vram_est_gb"),
                    "config": {"model": entry["model"],
                               "training": entry["training"]}},
                   indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- 批量绘图
def plot_batch(runs_dir):
    """汇总图: 各配置 val_proxy_score 曲线 + 最佳得分排名。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = runs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        recs = read_jsonl(run_dir / "finetune_metrics.jsonl")
        curve = [(r["epoch"], r["val_proxy_score"]) for r in recs
                 if "val_proxy_score" in r]
        if curve:
            entries.append((run_dir.name, curve))

    # 1) 曲线对比
    if entries:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, curve in entries:
            xs, ys = zip(*curve)
            best_y = max(ys)
            ax.plot(xs, ys, linewidth=1.0, label=f"{name} (best {best_y:.4f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val_proxy_score")
        ax.set_title("grid search | val proxy score curves")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "proxy_curves.png", dpi=150)
        plt.close(fig)

    # 2) 排名
    if entries:
        ranked = sorted(entries, key=lambda e: -max(y for _, y in e[1]))
        names = [n for n, _ in ranked]
        bests = [max(y for _, y in c) for _, c in ranked]
        fig, ax = plt.subplots(figsize=(10, 0.4 * len(ranked) + 2))
        colors = ["#d62728" if i == 0 else "#4c72b0" for i in range(len(ranked))]
        ax.barh(range(len(ranked)), bests, color=colors)
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        for i, v in enumerate(bests):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
        ax.set_xlabel("best val_proxy_score")
        ax.set_title("grid search | ranking (red = current best)")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(plots_dir / "best_bar.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="grid.json 批量训练 runner")
    ap.add_argument("--grid", default=str(GRID_DEFAULT), help="grid.json 路径")
    ap.add_argument("--data_dir", default=str(ROOT / "datasets"))
    ap.add_argument("--weights_dir", default=str(ROOT / "weights"))
    ap.add_argument("--runs_dir", default=str(ROOT / "logs" / "grid_runs"))
    ap.add_argument("--only", default=None, help="只跑名字含该子串的配置")
    ap.add_argument("--limit", type=int, default=0, help="本次最多跑 N 个配置")
    ap.add_argument("--retry", action="store_true", help="重跑 failed 的配置")
    ap.add_argument("--skip_pretrain", action="store_true",
                    help="跳过阶段一（需 weights/<name>/mae_pretrained.pt 已存在）")
    ap.add_argument("--keep_weights", action="store_true",
                    help="保留每个配置的权重目录（默认只留全局最佳一组）")
    ap.add_argument("--smoke_steps", type=int, default=0, help="冒烟步数透传")
    ap.add_argument("--nproc", default="auto",
                    help="每组配置的训练进程数（卡数）: auto=探测全部 NPU/GPU, "
                         "或指定整数如 16; 默认 auto")
    ap.add_argument("--dry", action="store_true", help="只列出计划, 不执行")
    args = ap.parse_args()

    grid = json.load(open(args.grid, encoding="utf-8"))
    configs = grid["configs"] if isinstance(grid, dict) else grid

    nproc = (detect_card_count() if args.nproc == "auto"
             else max(1, int(args.nproc)))
    if nproc > 1:
        print(f"[nproc] 每组配置 {nproc} 卡 DDP "
              f"(batch_size 为单卡批量, 总批量 = batch_size × {nproc}; "
              f"lr 是否随总批量放大请自行决定)")
    runs_dir = Path(args.runs_dir)
    logs_dir = ROOT / "logs"
    weights_dir = Path(args.weights_dir)
    state_path = runs_dir / "state.json"
    summary_path = runs_dir / "summary.jsonl"
    best_json_path = runs_dir / "best.json"
    best_dir = weights_dir / "best"

    state = json.loads(state_path.read_text(encoding="utf-8")) \
        if state_path.exists() else {}
    best_info = json.loads(best_json_path.read_text(encoding="utf-8")) \
        if best_json_path.exists() else None
    global_best = best_info["val_proxy_score"] if best_info else None

    # 计划: 未跑 + （可选）failed 重试
    plan = []
    for entry in configs:
        name = entry["name"]
        if args.only and args.only not in name:
            continue
        st = state.get(name, {}).get("status")
        if st == "done":
            continue
        if st == "failed" and not args.retry:
            print(f"[skip] {name}: 上次 failed（--retry 重跑）")
            continue
        plan.append(entry)

    n_total = len(configs)
    print(f"[plan] grid={args.grid} 共 {n_total} 配置; 待跑 {len(plan)} "
          f"(done={n_total - len(plan) if not args.only else '?'})")
    for entry in plan[:args.limit or None]:
        print(f"  - {entry['name']}")
    if args.dry:
        return
    if args.limit:
        plan = plan[:args.limit]

    runs_dir.mkdir(parents=True, exist_ok=True)

    for idx, entry in enumerate(plan, 1):
        name = entry["name"]
        t0 = time.time()
        print(f"\n[run {idx}/{len(plan)}] {name} "
              f"(params={entry.get('params_M', 0):.0f}M)")

        run_dir = runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.json"
        cfg_path.write_text(
            json.dumps({"model": entry["model"], "training": entry["training"]},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        w_run = weights_dir / name

        # ---- 两阶段顺序执行 ----
        rc = 0
        if not args.skip_pretrain:
            rc = run_stage("pretrain_mae.py", cfg_path, args.data_dir, w_run,
                           run_dir / "pretrain.log", args.smoke_steps, nproc)
        if rc == 0:
            rc = run_stage("finetune_multitask.py", cfg_path, args.data_dir, w_run,
                           run_dir / "finetune.log", args.smoke_steps, nproc)

        # ---- 成绩提取 + 日志归档 ----
        fin_recs = read_jsonl(logs_dir / "finetune_metrics.jsonl")
        score = best_score_from_metrics(fin_recs) if rc == 0 else None
        archive_logs(run_dir, logs_dir)
        status = "done" if rc == 0 else "failed"

        # ---- 全局最佳权重管理: 只保留最高一组 ----
        refreshed = False
        if score is not None and (global_best is None or score > global_best):
            global_best = score
            update_best(best_json_path, best_dir, name, score, entry, w_run)
            refreshed = True
            print(f"[best] 新全局最佳 {name}: {score:.5f} -> {best_dir}")

        # 清理非最佳配置的权重（mae_pretrained/断点/best 导出全清, 只留归档日志）
        if not args.keep_weights and w_run.exists():
            shutil.rmtree(w_run)

        state[name] = {"status": status, "score": score,
                       "elapsed_min": round((time.time() - t0) / 60, 1)}
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"name": name, "status": status,
                                "val_proxy_score": score,
                                "is_best": refreshed,
                                "elapsed_min": round((time.time() - t0) / 60, 1)},
                               ensure_ascii=False) + "\n")

        print(f"[done] {name}: status={status} score="
              f"{score if score is None else round(score, 5)} "
              f"({(time.time() - t0) / 60:.1f} min)")

    # ---- 收尾: 批量对比图 + 最佳配置输出 ----
    plot_batch(runs_dir)
    if global_best is not None:
        print(f"\n[final] 历史最佳配置见 {best_json_path}")
        print(f"[final] 最佳权重（官方推理格式）: {best_dir}")
        print(f"[final] 可直接训练的最佳配置: {best_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
