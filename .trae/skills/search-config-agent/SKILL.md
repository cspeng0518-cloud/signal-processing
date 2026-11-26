---
name: search-config-agent
description: Use when measuring GPU/NPU memory for EMFoundation configs, choosing batch sizes, comparing default.json variants, or regenerating src/configs/grid.json (at most 20 configs).
---

# Search Config Agent

Measure first, never estimate VRAM with formulas. **本 skill 的目标是生成 ≥300M 参数、batch ≥256、显存 ≤76GB 的可运行配置组，并写入 grid.json。**

## 前置知识

### grad_checkpoint（可选）

PyTorch 原生技术：**前向传播不保存中间激活，反向传播时重新计算**。显存降到约 1/3，速度慢约 30%。gen_configs.py 默认不开，如需启用加 `--auto_grad_checkpoint 1024`（d≥1024 自动开），或直接改 grid.json 里的 `model.grad_checkpoint: true`。

### measure_vram 口径

- 用合成数据真跑两阶段各 2 步（预训练带 MAE 解码器、微调三任务梯度累加）
- 返回 `pretrain_reserved_gb` / `finetune_reserved_gb`（GB=1e9，reserved=torch.cuda.memory_reserved）
- `status=ok 且 fits_budget=true` 才算通过；OOM 时 `status="OOM"`，不中断，继续测下一个
- 写 `logs/vram/report.json`（每次运行覆盖，需留存先备份）

## Tools

### 枚举候选

```bash
python utils/gen_configs.py \
    --d_model 768 1024 1536 \
    --n_layers 24 30 \
    --neck 2 4 \
    --bs_pretrain 512 --bs_finetune 256 \
    --min_params 300 \
    --max_configs 20 \
    --auto_grad_checkpoint 1024 \
    --out src/configs/grid.json \
    [--dry]               # 只枚举不写文件
```

行为：
- 在 meta 设备上精确统计参数量/共享率（非公式估算）
- 过滤 <1B 红线 / 共享率 ≤80% / d_model % 64 ≠ 0 / < min_params
- 按参数量降序取前 max_configs 组
- 默认不开 grad_checkpoint；显存不够时加 `--auto_grad_checkpoint 1024` 让 d≥1024 自动开

### 实测显存

```bash
python utils/measure_vram.py --grid src/configs/grid.json --budget 76
python utils/measure_vram.py --config <cfg> --find_max_bs   # 单配置找最大 batch
```

### 约束清单

| 项 | 值 | 强制来源 |
|---|---|---|
| 部署参数 < 1B | 是 | 赛题红线，gen_configs 过滤 |
| 共享率 > 80% | 是 | 赛题红线，gen_configs 过滤 |
| 参数量 ≥ min_params | 默认 300M | 用户需求 / 赛题竞争力 |
| batch ≥ min_bs | 默认 256 | 用户需求 |
| reserved ≤ budget | 默认 76GB | 80G 卡留 4GB 余量 |
| Conformer 只放 neck | 是 | 代码硬校验 |
| 最多 max_configs | 默认 20 | 用户需求 |

## Workflow（按步骤执行，任一步失败自动重试）

### 0. Preflight

```bash
nvidia-smi                       # GPU 必须空闲，没有训练进程
git status --short               # 无未提交改动
python utils/gen_configs.py --dry --min_params 300 --max_configs 20  # 先 dry 看候选
```

### 1. Generate

```bash
python utils/gen_configs.py \
    --d_model 768 1024 1536 --n_layers 24 30 --neck 2 4 \
    --bs_pretrain 512 --bs_finetune 256 \
    --min_params 300 --max_configs 20
# 显存不够时加 --auto_grad_checkpoint 1024
```

**失败处理**（最多重试 3 轮）：
- 候选为空 → 降 `--min_params` 到 250 或扩 `--d_model`（加 640）/ `--n_layers`（加 36）
- 候选过多 → 升 `--min_params` 或降 `--max_configs`

### 2. Measure

```bash
python utils/measure_vram.py --grid src/configs/grid.json --budget 76
```

读 `logs/vram/report.json`，按 `fits_budget=true` 筛出通过的配置。

**失败处理**（对每个未通过的配置，逐一下调 batch 重试）：
```bash
# 对 OOM / 超 budget 的配置，将 batch 降到 0.75× → 0.5× → 0.375×（下限 min_bs）
# 每个 batch 档跑一次 measure_vram --config，通过即停
# 三档都失败才剔除
```

### 3. Commit grid.json

只把通过的配置写回 `src/configs/grid.json`：

```python
# agent 用 Python 执行
import json
report = json.load(open("logs/vram/report.json"))
grid = json.load(open("src/configs/grid.json"))
verified = []
for cfg in grid["configs"]:
    entry = next((e for e in report if e.get("name") == cfg["name"]), None)
    if entry and entry["status"] == "ok" and entry.get("fits_budget"):
        cfg["training"]["batch_size_pretrain"] = entry.get("pretrain_bs", cfg["training"]["batch_size_pretrain"])
        cfg["training"]["batch_size"] = entry.get("batch_size", cfg["training"]["batch_size"])
        verified.append(cfg)
json.dump({"configs": verified}, open("src/configs/grid.json", "w"), indent=2, ensure_ascii=False)
```

**不要**直接覆盖 `src/configs/default.json`，要改 default 时先 copy 一份改名。

**如果 verified 为空**（全部 OOM / 超 budget）：
- 降 batch 下限到 192 试一轮
- 还是 OOM 再降到 128
- 超过 3 轮请停下来检查模型结构

### 4. 追加 gpu_memory_analysis.md

在 `docs/gpu_memory_analysis.md` 末尾追加新小节：

```markdown
## N. 本次实测（YYYY-MM-DD HH:MM）

GPU: A800 80GB  torch: X.Y+cuXYZ  命令: gen_configs.py + measure_vram.py --budget 76

| 日期 | 硬件 | 配置 | 参数量 | 预训练 bs / reserved | 微调 bs / reserved |
|---|---|---|---|---|---|
| 2026-11-24 15:00 | A800 80G | d1024_l30_ffn4096_neck4_ckpt | 637.8M | 512 / XX.X | 256 / XX.X |
| ... | ... | ... | ... | ... | ... |
```

日期、硬件、torch 版本、确切命令必须齐全；reserved 数值从 report.json 读取。

## Rules

- Conformer layers 放 `neck_layer_types`，**绝不**放 `layer_types`（会 raise）
- 实验期间**永不覆盖** `src/configs/default.json`，需要新 default 就 copy 一份改名
- gen_configs 的参数量是 meta 设备上精确计数，**不是**公式估算——和旧 probe_vram 流程的根本区别
- measure_vram 的 `fits_budget` 字段是最终判定，不要自己算 reserved 和 budget
- 重试最多 3 轮，超过请停下来检查模型结构（batch 是否下限太低）

## 调参速查

| 场景 | 调整 |
|---|---|
| 候选太少 | 降 `--min_params` 到 250，或扩 `--d_model`（加 640）/ `--n_layers`（加 36） |
| 候选太多 | 升 `--min_params` 到 400，或降 `--max_configs` 到 10 |
| 全部超显存 | 降 batch 下限到 192；还不够改 grid.json 里 `grad_checkpoint: true` |
| 全部 OOM（非超 budget） | 检查 `nvidia-smi` GPU 是否真空闲，再降 batch 下限到 128 |
