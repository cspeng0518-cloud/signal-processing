"""显存估算校准探针：真实跑每个配置的冒烟 step, 记录进程内真实显存峰值。

背景: gen_configs 的 vram_estimate_gb 只在 d512/27L 一个锚点上校准过
(docs/gpu_memory_analysis.md), 外推到 1024/1536 宽度可能漂移。本脚本用
真实训练进程校验/校准:
  1. 逐配置以 --smoke_steps 跑两阶段(预训练+微调)冒烟子进程
  2. 子进程结束后在同进程读 torch 的 max_memory_allocated/reserved 峰值
     (allocated 口径与估算公式一致; optimizer 态在第一个 step 后即齐)
  3. 对比 估算 vs 实测, 输出误差表 + logs/vram_probe/report.json
  4. --apply: 把最大 实测/估算 比写成 code/configs/vram_calibration.json,
     gen_configs 之后生成会自动乘该校准系数(闭环修正)——重跑
     python utils/gen_configs.py 即可得到修正后的 grid.json

用法:
  python utils/probe_vram.py --dry                  # 只列探测计划
  python utils/probe_vram.py                         # 全部配置探测
  python utils/probe_vram.py --only 1536d_24l        # 只探名字含子串的
  python utils/probe_vram.py --limit 2 --steps 3     # 前 2 个, 每阶段 3 step
  python utils/probe_vram.py --apply                 # 探完写校准系数
  python utils/probe_vram.py --only 1024d_39l --apply  # 两者可组合

注意:
  - 探测用真实训练子进程, 单卡(与估算口径一致), 冒烟 2-3 step 足以达到
    显存峰值(静态权重+梯度+优化器态在首个 step 内全部就位)
  - 与 run_grid 一样是串行逐配置; 期间不要并行跑其他训练(共享 metrics 文件)
  - 冒烟子进程若直接 CUDA OOM 崩溃, 恰好说明该配置在真实卡上放不下
    (估算低估), 记为 CRASH 并继续探下一个
  - 每个配置的临时权重目录(~4GB)探测后自动清理, --keep 保留
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
CALIBRATION = ROOT / "code" / "configs" / "vram_calibration.json"

# 探测 wrapper: runpy 在本进程内执行训练脚本本体(不动训练代码),
# 结束后读进程内 torch 显存峰值写 json
WRAPPER_SRC = '''
import sys, json, runpy, torch
script, out_json, *train_args = sys.argv[1:]
sys.argv = [script] + train_args
runpy.run_path(script, run_name="__main__")
peak = {}
try:
    if torch.cuda.is_available():
        peak["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        peak["max_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        peak["device"] = "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            peak["max_allocated_gb"] = torch.npu.max_memory_allocated() / 1e9
            peak["max_reserved_gb"] = torch.npu.max_memory_reserved() / 1e9
            peak["device"] = "npu"
        except Exception as e:
            peak["npu_error"] = repr(e)
except Exception as e:
    peak["error"] = repr(e)
json.dump(peak, open(out_json, "w"))
'''


def run_stage_probed(stage_script, cfg_path, data_dir, out_dir, log_path,
                     peak_json, steps):
    """wrapper 子进程跑一个训练阶段(冒烟), 返回 (returncode, 峰值 dict)。"""
    script = str(ROOT / "code" / "training" / stage_script)
    cmd = [sys.executable, str(out_dir.parent / "_wrapper.py"),
           script, str(peak_json),
           "--config", str(cfg_path), "--data_dir", str(data_dir),
           "--out_dir", str(out_dir), "--smoke_steps", str(steps)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    peak = {}
    if peak_json.exists():
        try:
            peak = json.loads(peak_json.read_text(encoding="utf-8"))
        except Exception:
            peak = {}
    return rc, peak


def main():
    ap = argparse.ArgumentParser(description="grid.json 显存估算实测校准探针")
    ap.add_argument("--grid", default=str(GRID_DEFAULT), help="grid.json 路径")
    ap.add_argument("--data_dir", default=str(ROOT / "datasets"))
    ap.add_argument("--probe_dir", default=str(ROOT / "logs" / "vram_probe"))
    ap.add_argument("--only", default=None, help="只探名字含该子串的配置")
    ap.add_argument("--limit", type=int, default=0, help="本次最多探 N 个配置")
    ap.add_argument("--steps", type=int, default=2,
                    help="每阶段冒烟 step 数（默认 2, 显存峰值首个 step 即就位）")
    ap.add_argument("--keep", action="store_true",
                    help="保留每个配置的临时运行目录（含冒烟权重, ~4GB/个）")
    ap.add_argument("--apply", action="store_true",
                    help="探测结束后把最大 实测/估算 比写入 vram_calibration.json")
    ap.add_argument("--dry", action="store_true", help="只列计划不执行")
    args = ap.parse_args()

    grid = json.load(open(args.grid, encoding="utf-8"))
    entries = (grid["configs"] if isinstance(grid, dict) else grid)[:]

    if args.only:
        entries = [e for e in entries if args.only in e["name"]]
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        print("[done] 无匹配配置")
        return

    budget = float(grid.get("vram_budget_gb", 74.0)) if isinstance(grid, dict) \
        else 74.0

    if args.dry:
        hdr = f'{"name":<52}{"params":>9}{"est":>9}'
        print(hdr)
        print("-" * len(hdr))
        for e in entries:
            print(f"{e['name']:<52}{e.get('params_M', 0):>8.0f}M"
                  f"{e.get('vram_est_gb', 0):>9.1f}G")
        print(f"\n[dry] 待探测 {len(entries)} 个配置（预算 {budget:.0f}G）")
        return

    probe_dir = Path(args.probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    wrapper = probe_dir / "_wrapper.py"
    wrapper.write_text(WRAPPER_SRC.strip() + "\n", encoding="utf-8")

    rows = []
    for i, e in enumerate(entries, 1):
        name = e["name"]
        est = float(e.get("vram_est_gb", 0.0))
        t0 = time.time()
        run_dir = probe_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.json"
        cfg_path.write_text(json.dumps({"model": e["model"],
                                        "training": e["training"]},
                                       indent=2, ensure_ascii=False),
                            encoding="utf-8")
        # 阶段一预训练冒烟（结束时导出 mae_pretrained.pt 供阶段二加载）
        rc_pre, pre = run_stage_probed(
            "pretrain_mae.py", cfg_path, args.data_dir, run_dir,
            run_dir / "pretrain.log", run_dir / "peak_pretrain.json", args.steps)
        # 阶段二微调冒烟（同 out_dir, 默认加载阶段一导出的 MAE 权重）
        rc_ft, ft = run_stage_probed(
            "finetune_multitask.py", cfg_path, args.data_dir, run_dir,
            run_dir / "finetune.log", run_dir / "peak_finetune.json", args.steps)

        peaks = [p.get("max_allocated_gb") for p in (pre, ft)
                 if p.get("max_allocated_gb")]
        reserved = max([p.get("max_reserved_gb", 0) for p in (pre, ft)] or [0])
        crashed = [s for s, rc in (("pretrain", rc_pre), ("finetune", rc_ft))
                   if rc != 0]
        if crashed or not peaks:
            real, ratio, verdict = None, None, \
                f"CRASH({'+'.join(crashed) or 'no-peak'}; 疑似OOM/低估)"
        else:
            real = max(peaks)
            ratio = real / est if est else None
            if real > budget:
                verdict = f"超预算(>{budget:.0f}G)"
            elif ratio > 1.10:
                verdict = f"低估{(ratio - 1) * 100:.0f}%"
            elif ratio < 0.85:
                verdict = "偏保守(可挖batch)"
            else:
                verdict = "准确"
        row = {
            "name": name, "params_M": e.get("params_M"), "est_gb": est,
            "pretrain_alloc_gb": pre.get("max_allocated_gb"),
            "finetune_alloc_gb": ft.get("max_allocated_gb"),
            "real_peak_gb": real, "reserved_gb": reserved,
            "ratio_real_over_est": ratio, "verdict": verdict,
            "rc_pretrain": rc_pre, "rc_finetune": rc_ft,
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        real_s = f"{real:.1f}G" if real else "CRASH"
        ratio_s = f"{ratio:.2f}" if ratio else "-"
        print(f"[{i}/{len(entries)}] {name}: est {est:.1f}G vs real {real_s} "
              f"(ratio {ratio_s}) -> {verdict}  [{row['seconds']}s]")

        if not args.keep:
            shutil.rmtree(run_dir, ignore_errors=True)

    report_path = probe_dir / "report.json"
    report_path.write_text(json.dumps(
        {"vram_budget_gb": budget, "steps": args.steps, "configs": rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {len(rows)} 条探测记录 -> {report_path}")

    # 汇总表
    hdr = f'{"name":<52}{"est":>7}{"real":>8}{"ratio":>7}  verdict'
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        real_s = f"{r['real_peak_gb']:.1f}" if r["real_peak_gb"] else "CRASH"
        ratio_s = f"{r['ratio_real_over_est']:.2f}" \
            if r["ratio_real_over_est"] else "-"
        print(f"{r['name']:<52}{r['est_gb']:>7.1f}{real_s:>8}"
              f"{ratio_s:>7}  {r['verdict']}")

    if args.apply:
        ratios = [r["ratio_real_over_est"] for r in rows
                  if r["ratio_real_over_est"]]
        if not ratios:
            print("\n[apply] 无有效实测(全部 CRASH?), 不写校准系数——"
                  "先降低 batch/宽度再探")
            return
        factor = round(max(max(ratios), 1.0), 2)
        CALIBRATION.write_text(json.dumps(
            {"factor": factor,
             "source": "utils/probe_vram.py --apply (实测峰值/估算值 的最大比)",
             "probed_configs": len(ratios),
             "observed_ratios": sorted(round(x, 3) for x in ratios)},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[apply] 校准系数 {factor:.2f} -> {CALIBRATION}\n"
              f"        重跑 `python utils/gen_configs.py` 重新生成 grid.json, "
              f"估算即乘该系数（闭环生效）")


if __name__ == "__main__":
    main()
