#!/usr/bin/env python3
"""main(FA+MLA) vs conformer 的 NPU 完整 A/B 对比（16 卡可用）。

全流程自动，无需手工改配置：
  1. git worktree 拉起两个分支目录，datasets 软链过去；
  2. 各分支自带 utils/gen_configs.py 搜参，参数量自动对齐（默认 100M），
     并按该分支显存模型取预算内最大 batch；
  3. 写 pretrain / finetune 阶段配置，除模型结构外训练超参完全一致；
  4. 每分支每阶段 1 步显存探针，OOM 自动对半降 batch，得出实测可用 batch；
  5. 完整训练：pretrain（两分支并行或串行）-> finetune（--eval_test）；
  6. 汇总 test 指标 + 耗时 + 日志路径，输出对比表并落盘 report.json。

用法（NPU 机器仓库根目录，用 NPU 环境的 python）:
  python utils/compare_branches.py --plan-only            # 只出参数/batch/lr 方案
  python utils/compare_branches.py                       # 完整跑，8+8 卡并行
  python utils/compare_branches.py --nproc 16 --sequential   # 串行各占 16 卡
  python utils/compare_branches.py --params 150 --epochs 50
  python utils/compare_branches.py --steps 300            # 冒烟模式（每 epoch 300 iter）

产物: <repo>/logs/compare/run_<时间戳>/
  main_pretrain.log / main_finetune.log / conformer_*.log / probe_*.log
  main/、conformer/  各分支 metrics jsonl 与曲线图副本
  report.json        参数量 / batch / lr / 耗时 / val / test 指标
"""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

META_KEYS = ("name", "params_M", "shared_ratio",
             "vram_est_gb", "vram_pretrain_gb", "vram_finetune_gb")
STAGES = ("pretrain", "finetune")
BRANCHES = ("main", "conformer")
# 训练日志里值得实时回显到控制台的行（其余只进日志文件，避免 tqdm 刷屏）
ECHO_RE = re.compile(r"(epoch=|\[test\]|Traceback|Error|error:|plot saved|"
                     r"loaded MAE|device=|early stopping|params=)")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[compare {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd, env, log_path, tail=40):
    """短命令（gen_configs）：执行完一次性取输出。"""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        log(f"  -> exit {proc.returncode}; 末尾输出:")
        for line in (proc.stdout or "").splitlines()[-tail:]:
            log(f"     {line}")
    return proc.returncode == 0, proc.stdout or ""


def is_oom(text):
    t = (text or "").lower()
    return any(k in t for k in ("out of memory", "npu out of memory",
                                "memoryerror", "alloc_fail", "oom"))


_VIS_RE = re.compile(r"^\d+(,\d+)*$")


def base_env():
    """清理掉非法的 ASCEND_RT_VISIBLE_DEVICES（如 "0-7" 区间写法）。

    CANN 只认逗号列表；非法值时各版本行为不一致，可能只枚举到部分卡，
    导致 set_device 报 107001 Invalid device ID。
    """
    env = dict(os.environ)
    vis = env.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if vis and not _VIS_RE.match(vis.replace(" ", "")):
        log(f"!! ASCEND_RT_VISIBLE_DEVICES={vis!r} 不是逗号列表，忽略它")
        env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    return env


def accel_env(cards):
    env = base_env()
    # 环境里已有合法的可见卡列表就尊重它（容器/调度器分配），否则用脚本分配的卡
    if "ASCEND_RT_VISIBLE_DEVICES" not in env:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(cards)
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", "8")
    return env


def card_range(base, nproc, idx):
    """并行时按分支切卡：base=0, nproc=8 -> main=0..7, conformer=8..15。

    用逗号列表而不是 "0-7" 区间：ASCEND_RT_VISIBLE_DEVICES 对区间语法的
    支持随 CANN 版本而异，逗号列表全版本可靠。
    """
    start = int(base) + idx * nproc
    return ",".join(str(i) for i in range(start, start + nproc))


def detect_npus():
    """当前环境可见的 NPU 数量（None = 检测失败）。"""
    r = subprocess.run(
        [sys.executable, "-c",
         "import torch, torch_npu; print(torch.npu.device_count())"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=base_env())
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        log(f"!! NPU 检测失败: {tail[-1] if tail else 'unknown'}")
        return None
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 子进程流式执行（训练几小时，日志要实时落盘）
# ---------------------------------------------------------------------------
def spawn(job):
    """启动一个训练子进程，独立线程把输出实时写日志并按需回显。"""
    Path(job["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    job["fh"] = open(job["log_path"], "w", encoding="utf-8")
    job["tail"] = deque(maxlen=4000)
    job["t0"] = time.monotonic()
    job["proc"] = subprocess.Popen(
        [str(c) for c in job["cmd"]], cwd=str(job["cwd"]), env=job["env"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace", bufsize=1)
    job["thread"] = threading.Thread(target=_pump, args=(job,), daemon=True)
    job["thread"].start()
    return job


def _pump(job):
    prefix = f"[{job['branch']}/{job['stage']}]"
    for line in job["proc"].stdout:
        job["fh"].write(line)
        job["fh"].flush()
        job["tail"].append(line.rstrip("\r\n"))
        if ECHO_RE.search(line):
            print(f"{prefix} {line.rstrip()}", flush=True)


def wait(job):
    rc = job["proc"].wait()
    job["thread"].join()
    job["fh"].close()
    job["elapsed"] = time.monotonic() - job["t0"]
    text = "\n".join(job["tail"])
    if rc != 0:
        log(f"[{job['branch']}/{job['stage']}] exit {rc} "
            f"({job['elapsed']:.0f}s)，日志: {job['log_path']}")
        for line in list(job["tail"])[-25:]:
            print(f"     {line}", flush=True)
    else:
        log(f"[{job['branch']}/{job['stage']}] 完成，耗时 {job['elapsed'] / 60:.1f} min，"
            f"日志: {job['log_path']}")
    return rc == 0, text


def run_jobs(jobs, parallel):
    """并行跑同一阶段的多个分支；串行模式逐个跑。"""
    if not jobs:
        return {}
    out = {}
    if parallel and len(jobs) > 1:
        log(f"并行启动 {len(jobs)} 个任务（每分支 {jobs[0]['nproc']} 卡）")
        for j in jobs:
            spawn(j)
        for j in jobs:
            out[j["branch"]] = wait(j)
    else:
        for j in jobs:
            spawn(j)
            out[j["branch"]] = wait(j)
    return out


# ---------------------------------------------------------------------------
# 分支准备
# ---------------------------------------------------------------------------
def checked_out_path(repo, ref):
    """返回 ref 已被 git 检出的目录（没有则 None）。

    主分支通常就在 repo 根目录检出，不能再 `git worktree add main`，
    直接复用现有目录即可。
    """
    r = subprocess.run(["git", "-C", str(repo), "worktree", "list", "--porcelain"],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cur = None
    for line in r.stdout.splitlines():
        if line.startswith("worktree "):
            cur = Path(line[len("worktree "):].strip())
        elif line.startswith("branch "):
            branch = line[len("branch "):].strip().replace("refs/heads/", "")
            if branch == ref:
                return cur
    return None


def prepare_branch(repo, ref, given_dir, wt_root, data_dir):
    if given_dir:
        d = Path(given_dir).resolve()
        if not (d / "utils" / "gen_configs.py").exists():
            sys.exit(f"目录不像项目根（缺 utils/gen_configs.py）: {d}")
        return d

    found = checked_out_path(repo, ref)
    if found:
        log(f"  {ref} 已在 {found} 检出，直接复用（不新建 worktree）")
        link_data(found, data_dir)
        return found.resolve()

    wt_root.mkdir(parents=True, exist_ok=True)
    d = wt_root / ref
    if not d.exists():
        r = subprocess.run(["git", "-C", str(repo), "worktree", "add",
                            str(d), ref], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"git worktree add {ref} 失败:\n{r.stdout}")
    link_data(d, data_dir)
    return d


def link_data(branch_dir, data_dir):
    """worktree 不含 gitignore 的 datasets，软链过去。"""
    src = Path(data_dir)
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    dst = branch_dir / "datasets"
    if dst.exists() or dst.is_symlink() or not src.exists():
        return
    try:
        os.symlink(str(src), str(dst), target_is_directory=True)
        log(f"  datasets 软链: {dst} -> {src}")
    except OSError as e:
        log(f"  !! datasets 软链失败（{e}），请手动放数据到 {dst}")


# ---------------------------------------------------------------------------
# 1) 参数对齐 + batch 估算（调用各分支自己的 gen_configs）
# ---------------------------------------------------------------------------
def gen_args(branch_dir, args, out_path):
    """按分支实际支持的参数拼 gen_configs 命令行（两版 CLI 略有差异）。"""
    src = (branch_dir / "utils" / "gen_configs.py").read_text(encoding="utf-8")
    cmd = [sys.executable, "utils/gen_configs.py",
           "--d_model", *map(str, args.d_model),
           "--n_layers", *map(str, args.n_layers),
           "--ffn_hidden", *map(str, args.ffn_hidden),
           "--shared_ratio", str(args.shared_ratio),
           "--batch_size", "auto",            # 各分支按自己的显存模型取预算内最大
           "--vram", str(args.budget_gb),
           "--min_params", "0", "--max_configs", "500",
           "--out", str(out_path)]
    if "--headroom" in src:
        cmd += ["--headroom", str(args.headroom)]
    if "--min_bs_pretrain" in src:
        cmd += ["--min_bs_pretrain", "8", "--min_bs_finetune", "8"]
    elif "--min_batch_size" in src:
        cmd += ["--min_batch_size", "8"]
    return cmd


def probe_branch(branch_dir, args):
    out = branch_dir / "logs" / "compare" / "probe_grid.json"
    if out.exists():
        out.unlink()
    ok, _ = run(gen_args(branch_dir, args, out), branch_dir,
                accel_env(card_range(args.card, 1, 0)),
                branch_dir / "logs" / "compare" / "gen_configs.log")
    if not ok or not out.exists():
        sys.exit(f"{branch_dir.name}: gen_configs 失败，见 logs/compare/gen_configs.log")
    return load_json(out)["configs"]


def entry_vram(entry, stage):
    if stage == "pretrain" and "vram_pretrain_gb" in entry:
        return float(entry["vram_pretrain_gb"])
    if stage == "finetune" and "vram_finetune_gb" in entry:
        return float(entry["vram_finetune_gb"])
    if "vram_est_gb" in entry:
        return float(entry["vram_est_gb"])
    return max(float(entry.get("vram_pretrain_gb") or 0),
               float(entry.get("vram_finetune_gb") or 0))


def pick_config(entries, args):
    lo, hi = args.params_min * 1e6, args.params_max * 1e6
    cand = [e for e in entries if lo <= e["params_M"] * 1e6 <= hi]
    if not cand:
        sys.exit(f"没有落在 [{args.params_min}, {args.params_max}]M 的配置，"
                 f"请调整 --d_model/--n_layers/--ffn_hidden")
    target = args.params * 1e6
    return min(cand, key=lambda e: abs(e["params_M"] * 1e6 - target))


def stage_bs(chosen, stage):
    t = chosen["training"]
    return int(t.get(f"batch_size_{stage}") or t["batch_size"])


def scaled_lr(bs, nproc, override):
    """lr 随全局 batch 缩放；两分支用同一个值。"""
    return override or 1e-4 * (bs * nproc / 16) ** 0.5


# ---------------------------------------------------------------------------
# 2) 写阶段配置
# ---------------------------------------------------------------------------
def make_stage_cfg(chosen, bs, lr, args, stage, branch_name):
    cfg = json.loads(json.dumps(chosen))
    for k in META_KEYS:
        cfg.pop(k, None)
    t = cfg["training"]
    t["seq_len"] = args.seq_len
    t["batch_size"] = bs
    for k in ("batch_size_pretrain", "batch_size_finetune"):
        if k in t:
            t[k] = bs
    if "eval_batch_size" in t:
        t["eval_batch_size"] = min(int(t["eval_batch_size"]), max(bs, 256))
    t["lr"] = lr
    for k in ("lr_pretrain", "lr_finetune"):
        if k in t:
            t[k] = lr
    t["min_lr_ratio"] = 0.01
    t["eval_every"] = 1
    t["amp"] = True
    t["amp_dtype"] = "bf16"
    t["num_workers"] = min(int(t.get("num_workers", 4)), 8)
    if args.epochs:  # 0 = 沿用各分支配置的 epochs_pretrain / epochs_finetune
        t["epochs_pretrain"] = t["epochs_finetune"] = args.epochs
    n_epochs = args.epochs or int(
        t.get("epochs_pretrain" if stage == "pretrain" else "epochs_finetune", 200))
    if args.steps:  # 冒烟模式：关早停，warmup 固定占该阶段 5%
        t["patience"] = t["patience_pretrain"] = 10 ** 6
        t["min_delta"] = t["min_delta_pretrain"] = 0.0
        stage_total = args.steps * n_epochs * (3 if stage == "finetune" else 1)
        if branch_name == "main":   # main 用 epoch 口径的 warmup_epochs
            t.pop("warmup_steps", None)
            t.pop("warmup_ratio", None)
            t["warmup_epochs"] = 0.05
        else:                       # conformer 用绝对 warmup_steps
            t.pop("warmup_epochs", None)
            t.pop("warmup_ratio", None)
            t["warmup_steps"] = max(1, int(stage_total * 0.05))
    # 完整模式沿用各分支配置的 warmup/patience（main: 1 epoch; conformer: 2000 steps）
    return cfg


def write_cfg(branch_dir, cfg, stage):
    p = branch_dir / "configs" / f"compare_{stage}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 3) 训练 / 指标
# ---------------------------------------------------------------------------
def stage_script(stage):
    return "code/training/pretrain_mae.py" if stage == "pretrain" else \
           "code/training/finetune_multitask.py"


def launch(branch, branch_dir, stage, cfg_path, out_dir, args, log_path, cards,
           port, idx, steps, eval_test=False):
    """构造一个训练 job（多卡时用 torchrun 拉起 --nproc 个进程）。"""
    script = stage_script(stage)
    argv = [script,
            "--config", str(cfg_path), "--data_dir", str(args.data_abs),
            "--out_dir", str(out_dir), "--smoke_steps", str(steps)]
    if stage == "finetune":
        mae = Path(out_dir) / "mae_pretrained.pt"
        if mae.exists():
            argv += ["--from_mae", str(mae)]
        if eval_test:
            argv.append("--eval_test")
    if args.nproc > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               "--nnodes=1", "--node_rank=0",
               "--nproc_per_node", str(args.nproc),
               "--master_port", str(port), *argv]
    else:
        cmd = [sys.executable, *argv]
    return {"branch": branch, "stage": stage, "cmd": cmd,
            "cwd": branch_dir, "env": accel_env(cards), "log_path": Path(log_path),
            "nproc": args.nproc, "idx": idx}


def parse_test_metrics(text):
    """解析 finetune --eval_test 的日志: [test] protocol: {'acc': .9}。"""
    out = {}
    for line in (text or "").splitlines():
        m = re.search(r"\[test\]\s+(protocol|modulation|parameter):\s*(\{.*\})", line)
        if m:
            try:
                out.update({f"test_{m.group(1)}_{k}": float(v)
                            for k, v in ast.literal_eval(m.group(2)).items()})
            except (ValueError, SyntaxError):
                pass
        m = re.search(r"\[test\]\s+proxy score=([\d.]+)", line)
        if m:
            out["test_proxy_score"] = float(m.group(1))
    return out


def last_metrics(branch_dir, stage):
    p = Path(branch_dir) / "logs" / f"{stage}_metrics.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-1] if rows else {}


def metric_rows(branch_dir, stage):
    p = Path(branch_dir) / "logs" / f"{stage}_metrics.jsonl"
    if not p.exists():
        return 0
    return len([l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])


def collect(branch_dir, run_dir, branch):
    """把分支 metrics jsonl 和曲线图复制到本次 run 目录（worktree 可能被清）。"""
    dst = Path(run_dir) / branch
    dst.mkdir(parents=True, exist_ok=True)
    logs = Path(branch_dir) / "logs"
    for f in logs.glob("*.jsonl"):
        shutil.copy2(f, dst / f.name)
    plots = logs / "plots"
    if plots.exists():
        shutil.copytree(plots, dst / "plots", dirs_exist_ok=True)
    return str(dst)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="main(FA+MLA) vs conformer NPU 完整 A/B")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--main-dir", default=None)
    ap.add_argument("--conformer-dir", default=None)
    ap.add_argument("--main-ref", default="main")
    ap.add_argument("--conformer-ref", default="conformer")
    # 参数量对齐：默认 100M（大参数量下 conformer 搜参空间很窄）
    ap.add_argument("--params", type=float, default=100, help="目标参数量 M")
    ap.add_argument("--params-min", type=float, default=90)
    ap.add_argument("--params-max", type=float, default=110)
    ap.add_argument("--d_model", nargs="+", type=int, default=[384, 512])
    ap.add_argument("--n_layers", nargs="+", type=int, default=[18, 20, 24])
    ap.add_argument("--ffn_hidden", nargs="+", type=int, default=[1024, 1536, 2048])
    ap.add_argument("--shared_ratio", type=float, default=0.90)
    ap.add_argument("--budget-gb", type=float, default=61.0, help="单卡可用 HBM")
    ap.add_argument("--headroom", type=float, default=0.90)
    ap.add_argument("--data_dir", default="datasets")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=0,
                    help="0=沿用各分支配置的 epochs（200）")
    ap.add_argument("--steps", type=int, default=0,
                    help="0=完整训练；>0 为冒烟模式（每 epoch 的迭代数）")
    ap.add_argument("--lr", type=float, default=0.0,
                    help="0=按共享全局 batch 缩放（1e-4×sqrt(bs×nproc/16)）")
    ap.add_argument("--nproc", type=int, default=8, help="每分支进程数（卡数）")
    ap.add_argument("--parallel", dest="parallel", action="store_true", default=True,
                    help="两分支同时跑（默认，每分支 --nproc 卡）")
    ap.add_argument("--sequential", dest="parallel", action="store_false",
                    help="两分支串行跑（各占 --nproc 卡）")
    ap.add_argument("--card", type=int, default=0, help="起始卡号")
    ap.add_argument("--port", type=int, default=29500, help="torchrun master port 基址")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    args.data_abs = (repo / args.data_dir).resolve() \
        if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    wt_root = repo.parent / "signal_compare_wt"
    args.n_layers = sorted(set(args.n_layers))
    args.ffn_hidden = sorted(set(args.ffn_hidden))
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    total = detect_npus()
    if total is not None:
        log(f"可见 NPU: {total} 张" + (f"（ASCEND_RT_VISIBLE_DEVICES={vis}）" if vis else ""))
        if total <= 0:
            sys.exit("未检测到可用 NPU，请检查 npu-smi info / 容器卡分配")
        if total < args.nproc:
            log(f"  提示: 本 shell 只看到 {total} 张卡。若机器确实有更多卡，"
                f"请执行 unset ASCEND_RT_VISIBLE_DEVICES 后重跑")
        if args.nproc > total:
            log(f"!! --nproc {args.nproc} 超过可见卡数 {total}，自动改为 {total}")
            args.nproc = total
        if args.parallel and len(BRANCHES) * args.nproc > total:
            log(f"!! 并行需要 {len(BRANCHES) * args.nproc} 卡，可见只有 {total}，改为串行")
            args.parallel = False
        if vis and args.parallel:
            log("!! 环境已设 ASCEND_RT_VISIBLE_DEVICES，脚本无法再切分卡，改为串行")
            args.parallel = False
    else:
        log(f"!! 未检测到 NPU 数量，按 --nproc={args.nproc} 继续；"
            f"若报 Invalid device ID 请检查 npu-smi info / ASCEND_RT_VISIBLE_DEVICES")
    if args.steps and not args.epochs:
        args.epochs = 1  # 冒烟默认 1 epoch，warmup/调度才有实际形状

    log("准备两个分支目录 ...")
    main_dir = prepare_branch(repo, args.main_ref, args.main_dir, wt_root, args.data_abs)
    conf_dir = prepare_branch(repo, args.conformer_ref, args.conformer_dir, wt_root,
                              args.data_abs)
    dirs = {"main": main_dir, "conformer": conf_dir}

    plan = {}
    for name in BRANCHES:
        d = dirs[name]
        log(f"[{name}] 搜索参数配置 ({d}) ...")
        chosen = pick_config(probe_branch(d, args), args)
        plan[name] = {"dir": d, "chosen": chosen,
                      "bs": {s: stage_bs(chosen, s) for s in STAGES},
                      "vram": {s: entry_vram(chosen, s) for s in STAGES}}
        m = chosen["model"]
        log(f"[{name}] d={m['d_model']} L={m['n_layers']} F={m['ffn_hidden']} "
            f"params={chosen['params_M']:.1f}M "
            f"bs(pre/ft)={plan[name]['bs']['pretrain']}/{plan[name]['bs']['finetune']}")

    p0, p1 = plan["main"]["chosen"]["params_M"], plan["conformer"]["chosen"]["params_M"]

    print("\n" + "=" * 84)
    print("参数对齐方案（NPU，单卡预算 %.0fGB × %.2f，%d 卡/分支）"
          % (args.budget_gb, args.headroom, args.nproc))
    print("=" * 84)
    print(f"{'branch':<11}{'d':>6}{'layers':>8}{'ffn':>7}{'shared':>8}{'params':>10}"
          f"{'bs_pre':>8}{'bs_ft':>7}{'vram_pre':>10}{'vram_ft':>9}")
    for name, item in plan.items():
        m = item["chosen"]["model"]
        print(f"{name:<11}{m['d_model']:>6}{m['n_layers']:>8}{m['ffn_hidden']:>7}"
              f"{item['chosen']['shared_ratio']:>8.2f}{item['chosen']['params_M']:>9.1f}M"
              f"{item['bs']['pretrain']:>8}{item['bs']['finetune']:>7}"
              f"{item['vram']['pretrain']:>9.1f}G{item['vram']['finetune']:>8.1f}G")
    print(f"\n参数量差异: {abs(p0 - p1) / max(p0, p1) * 100:.1f}%")
    print(f"训练模式: {'冒烟 ' + str(args.steps) + ' iter/epoch' if args.steps else '完整训练'}"
          f"，epochs={'各分支配置' if not args.epochs else args.epochs}")

    if args.plan_only:
        for s in STAGES:
            cbs = min(plan[b]["bs"][s] for b in BRANCHES)
            print(f"{s:<9} 共同 per-card batch={cbs:>4}  全局 batch={cbs * args.nproc:>5}"
                  f"  lr={scaled_lr(cbs, args.nproc, args.lr):.2e}")
        log("--plan-only: 不启动训练")
        return

    run_dir = repo / "logs" / "compare" / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"本次运行目录: {run_dir}")

    # ---- 显存探针：每分支每阶段 1 步，OOM 对半降 ----
    for i, name in enumerate(BRANCHES):
        item = plan[name]
        out_dir = item["dir"] / "weights" / "compare_probe"
        for s in STAGES:
            bs = item["bs"][s]
            while True:
                cfg = make_stage_cfg(item["chosen"], bs, 1e-4, args, s, name)
                j = launch(name, item["dir"], s,
                           write_cfg(item["dir"], cfg, f"probe_{s}"),
                           out_dir, args, run_dir / f"probe_{name}_{s}.log",
                           card_range(args.card, args.nproc, i if args.parallel else 0),
                           args.port, i, steps=1)
                ok, text = run_jobs([j], parallel=False)[j["branch"]]
                if ok or not is_oom(text) or bs <= 8:
                    break
                log(f"[{name}/{s}] bs={bs} OOM，降到 {bs // 2}")
                bs = max(8, bs // 2)
            if not ok:
                hint = "显存不足（bs 已降到 8）" if is_oom(text) else "非显存错误"
                sys.exit(f"[{name}/{s}] 探针失败：{hint}，停止。日志: {j['log_path']}")
            item.setdefault("bs_verified", {})[s] = bs
            log(f"[{name}/{s}] 实测可用 per-card batch = {bs}")

    common_bs = {s: min(plan[b]["bs_verified"][s] for b in BRANCHES) for s in STAGES}
    lr = {s: scaled_lr(common_bs[s], args.nproc, args.lr) for s in STAGES}
    for s in STAGES:
        log(f"{s}: 公平 batch={common_bs[s]}（全局 {common_bs[s] * args.nproc}），"
            f"lr={lr[s]:.2e}")

    # ---- 完整训练：pretrain -> finetune ----
    results, elapsed, logs = {}, {}, {}
    for stage in STAGES:
        jobs = []
        for i, name in enumerate(BRANCHES):
            item = plan[name]
            out_dir = item["dir"] / "weights" / "compare"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = make_stage_cfg(item["chosen"], common_bs[stage], lr[stage],
                                 args, stage, name)
            jobs.append(launch(
                name, item["dir"], stage, write_cfg(item["dir"], cfg, stage), out_dir,
                args, run_dir / f"{name}_{stage}.log",
                card_range(args.card, args.nproc, i if args.parallel else 0),
                args.port + (i if args.parallel else 0), i,
                steps=args.steps, eval_test=(stage == "finetune")))
        done = run_jobs(jobs, parallel=args.parallel)
        results[stage] = {b: done[b][0] for b in BRANCHES}
        elapsed[stage] = {b: next(j["elapsed"] for j in jobs if j["branch"] == b)
                          for b in BRANCHES}
        logs[stage] = {b: str(run_dir / f"{b}_{stage}.log") for b in BRANCHES}
        if stage == "pretrain" and not all(results[stage].values()):
            log("!! pretrain 有分支失败，finetune 仍会跑（没有权重时从头训练）")

    # ---- 汇总 ----
    metrics = {}
    for name in BRANCHES:
        d = dirs[name]
        row = parse_test_metrics(
            (run_dir / f"{name}_finetune.log").read_text(encoding="utf-8",
                                                         errors="replace"))
        val = last_metrics(d, "finetune")
        row.update({k: v for k, v in val.items()
                    if isinstance(v, (int, float)) and k.startswith("val_")})
        pre = last_metrics(d, "pretrain")
        if isinstance(pre.get("recon_mse"), (int, float)):
            row["pretrain_recon_mse"] = float(pre["recon_mse"])
        row["epochs_pretrain_done"] = metric_rows(d, "pretrain")
        row["epochs_finetune_done"] = metric_rows(d, "finetune")
        metrics[name] = row
        collect(d, run_dir, name)

    print("\n" + "=" * 84)
    print(f"对比结果（test 划分；同数据/同 epoch/同 batch={common_bs['finetune']}/"
          f"同 lr={lr['finetune']:.1e}，两分支各 {args.nproc} 卡）")
    print("=" * 84)
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "smoke" if args.steps else "full",
        "nproc": args.nproc, "parallel": args.parallel,
        "params_M": {"main": p0, "conformer": p1},
        "common_bs": common_bs, "lr": lr,
        "bs_verified": {b: plan[b]["bs_verified"] for b in BRANCHES},
        "elapsed_sec": elapsed, "logs": logs,
        "ok": {s: {b: results[s][b] for b in BRANCHES} for s in STAGES},
        "metrics": metrics,
    }
    all_keys = {k for k in set().union(*[set(m) for m in metrics.values()])
                if not k.startswith("_")}
    pref = ["test_proxy_score"]
    keys = sorted(all_keys, key=lambda k: (k not in pref, k))
    print(f"\n{'metric':<30}{'main':>14}{'conformer':>14}{'Δ(conf-main)':>16}")
    for k in keys:
        a, b = metrics["main"].get(k), metrics["conformer"].get(k)
        delta = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        fmt = lambda x: "-" if x is None else (
            f"{x:d}" if isinstance(x, int) else
            (f"{x:.5f}" if abs(x) < 10 else f"{x:.3f}"))
        print(f"{k:<30}{fmt(a):>14}{fmt(b):>14}"
              f"{('-' if delta is None else f'{delta:+.5f}'):>16}")

    out = run_dir / "report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"报告: {out}")
    log(f"权重: {dirs['main'] / 'weights' / 'compare'}  "
        f"{dirs['conformer'] / 'weights' / 'compare'}")
    ok_all = all(results[s][b] for s in STAGES for b in BRANCHES)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
