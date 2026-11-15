# SignalDetection — EMind 式电磁信号通用感知基座（赛题代码框架）

单一共享基座模型（IQ Tokenizer + Transformer/Mamba Backbone）+ 三个轻量任务头，
完成体制识别（8 类）、调制识别（23 类）、参数估计（7 参数）三项任务。

**权重共用率 95.6%（共享 2.38M vs 任务头 108.8K），满足赛题 >80% 红线。**

## 目录结构

```
SignalDetection/
├── code/
│   ├── infer.py                  # 统一推理入口（官方评测直接调用）
│   ├── configs/
│   │   └── default.json          # 模型/训练超参
│   ├── models/
│   │   ├── tokenizer.py          # IQ patch tokenizer + CLS/采样率 token
│   │   ├── backbone.py           # Transformer/Mamba 架构选择入口
│   │   ├── heads.py              # 分类/回归任务头
│   │   └── foundation.py         # EMFoundation 总装（含 MAE decoder）
│   ├── training/
│   │   ├── pretrain_mae.py       # 阶段1：MAE 掩码重建预训练
│   │   └── finetune_multitask.py # 阶段2：三任务联合微调
│   └── utils/
│       ├── dataset.py            # Parquet 读取、IQ bytes 反序列化、归一化
│       └── metrics.py           # 官方评分指标（准确率/宏召回/RMSE 精准度）
├── datasets/                     # 模拟数据（train/val/test × 3 任务 Parquet）
├── weights/                      # 交付物：model.safetensors + config.json + param_stats.json
├── out/                          # 推理输出 JSONL
└── logs/
```

## 快速开始

```bash
# 0. 模拟数据（已生成 780 条；重跑会覆盖）
python code/utils/data_gen.py

# 1. MAE 预训练（掩码 75%，CLS/采样率 token 不掩码）
python code/training/pretrain_mae.py

# 2. 多任务联合微调（加载 MAE 权重，三任务轮转训练共享 backbone）
python code/training/finetune_multitask.py

# 3. 推理（官方格式：Parquet 进 -> JSONL 出）
python code/infer.py --input datasets/test_system.parquet    --output out/pred_system.jsonl
python code/infer.py --input datasets/test_modulation.parquet --output out/pred_modulation.jsonl
python code/infer.py --input datasets/test_parameter.parquet  --output out/pred_parameter.jsonl
```

## 数据格式（对齐官方 schema）

| 字段 | 说明 |
|---|---|
| task | system / modulation / parameter |
| sample_id | 唯一 ID |
| iq | [N,2] I/Q 序列，存为扁平 float32 bytes（Arrow 不支持嵌套数组列） |
| label | 体制/调制类别名（仅训练集） |
| param_* / valid_* | 7 个物理参数 + 有效性位（仅 parameter 任务训练集） |

## 关键设计（EMind 技术路线）

1. **两阶段训练**：MAE 掩码重建预训练学通用时域表征 → 多任务联合微调。
2. **架构选择入口**：`configs/default.json` 的 `backbone` 字段，`"transformer"`（纯 PyTorch，无 CUDA 专有算子）/ `"mamba"`（可选依赖 `mamba-ssm`）。
3. **按样本掩码 MAE**：backbone 仅前向可见 token（`keep_ids`），训练提速。
4. **参数归一化**：min-max 到 [0,1] 训练，推理时用 `weights/param_stats.json` 反归一化回原始单位。
5. **参数损失带 valid 掩码**：仅对有效参数维度计算 MSE。

## 接入真实赛题数据

把官方 Parquet（含真实 iq 列）替换 `datasets/` 下的文件即可——若官方 iq 列为
原生 [N,2] 数组，请把 `utils/dataset.py` 的 `bytes_to_iq` 换成直接读取逻辑，
其余流程不变。

## 环境锁定

见 `requirements_lock.txt`。
