"""消融实验配置生成脚本。

用法：
    python utils/gen_ablation.py                         # 生成全部消融配置
    python utils/gen_ablation.py --out src/configs/ablation
    python utils/gen_ablation.py --filter B               # 只生成 B 组
    python utils/gen_ablation.py --dry                    # 只打印不落盘

生成逻辑：每个消融 = deep_copy(default.json) + patch(diff dict)。
"""

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "src", "configs", "default.json")

ABLATIONS = [
    # === A: 任务 1 时-频双流 / 融合消融 ===
    ("A0_time_only",   "仅时域单流（关闭 STFT 门控、频域流、统计 token）",
     {"model": {"use_stft": False, "use_spec": False, "use_acf": False, "use_const": False,
                "use_caf": False, "use_dechirp": False, "use_hist": False, "use_stats": False}}),
    ("A1_no_stft",     "去掉通道级 STFT 门控融合",   {"model": {"use_stft": False}}),
    ("A2_no_spec",     "频域流去掉非线性谱 token",     {"model": {"use_spec": False}}),
    ("A3_no_acf",      "频域流去掉包络/复自相关 token",   {"model": {"use_acf": False}}),
    ("A4_no_const",    "频域流去掉星座图 token",       {"model": {"use_const": False}}),
    ("A5_no_stats",    "去掉统计量 token",             {"model": {"use_stats": False}}),
    ("A7_no_caf",      "频域流去掉非共轭循环自相关 token", {"model": {"use_caf": False}}),
    ("A8_no_dechirp",  "关闭去斜谱（前端不计算，统计量置 0）", {"model": {"use_dechirp": False}}),
    ("A9_no_hist",     "频域流去掉分布直方图 token",   {"model": {"use_hist": False}}),
    ("A10_no_ccaf",    "spec 不叠加共轭循环谱通道",    {"model": {"ccaf_lags": []}}),
    ("A6_freq_enc0",   "频域流不加自注意力编码（仅投影）", {"model": {"freq_layers": 0}}),

    # === B: 主干 / neck 消融 ===
    ("B1_all_fa",      "主干全部为 full attention",    {"model": {"layer_types": ["full_attention"]}}),
    ("B2_all_mla",     "主干全部为 MLA",               {"model": {"layer_types": ["mla"]}}),
    ("B3_no_neck",     "去掉 Conformer neck",          {"model": {"neck_layers": 0}}),
    ("B4_neck4",       "Conformer neck 4 层",          {"model": {"neck_layers": 4}}),
    ("B5_neck_nofreq", "neck 不拼接频域 token",         {"model": {"neck_with_freq": False}}),

    # === C: 任务 2 多任务策略消融 ===
    ("C1_pcgrad",          "PCGrad 冲突梯度投影",     {"training": {"pcgrad": True}}),
    ("C2_uncertainty",     "Kendall 不确定性加权",    {"training": {"task_weighting": "uncertainty"}}),
    ("C3_no_supcon",       "去掉调制 SupCon",         {"training": {"supcon_weight": {}}}),
    ("C4_no_task_prompt",  "去掉任务提示",            {"model": {"task_prompt": False}}),

    # === D: 优化 / 预训练策略消融 ===
    ("D1_muon",        "Muon 优化器",               {"training": {"optimizer": "muon"}}),
    ("D2_param_mse",   "参数回归改用 MSE",           {"training": {"param_loss": "mse"}}),
    ("D3_no_aug",      "关闭数据增强",               {"training": {"augment": False}}),
    ("D4_mask50",      "MAE 掩码率 0.5",             {"model": {"mask_ratio": 0.5}}),
]


def deep_merge(base, patch):
    """递归合并 patch 到 base（patch 覆盖 base，不删 key）。"""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def main():
    ap = argparse.ArgumentParser(description="生成消融实验 JSON 配置")
    ap.add_argument("--out", default=os.path.join(ROOT, "src", "configs", "ablation"),
                    help="输出目录（默认 src/configs/ablation/）")
    ap.add_argument("--filter", default="",
                    help="按编号前缀过滤，如 A/B/C/D 或 A0/C3")
    ap.add_argument("--dry", action="store_true", help="只打印不落盘")
    args = ap.parse_args()

    base = json.load(open(DEFAULT, encoding="utf-8"))

    to_gen = ABLATIONS
    if args.filter:
        to_gen = [(n, d, p) for n, d, p in ABLATIONS
                  if n.startswith(args.filter.upper())]

    print(f"Base: {DEFAULT}")
    print(f"Output: {args.out}")
    print(f"Plan: {len(to_gen)} configs\n")

    if not args.dry:
        os.makedirs(args.out, exist_ok=True)

    for name, desc, patch in to_gen:
        cfg = copy.deepcopy(base)
        deep_merge(cfg, patch)
        cfg["_note"] = desc

        if args.dry:
            print(f"  [dry] {name}.json  ({desc})")
        else:
            fp = os.path.join(args.out, f"{name}.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(f"  [ok ] {name}.json  ({desc})")

    if args.dry:
        print(f"\n[dry] 共 {len(to_gen)} 个（加 --dry 才不落盘）")
    else:
        print(f"\n[done] {len(to_gen)} 个 -> {args.out}")


if __name__ == "__main__":
    main()
