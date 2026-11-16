# 电磁信号基座模型初步训练指导

> 适用仓库：`signal-processing`（电磁信号认知赛题：protocol / modulation / parameter 三任务）
> 训练机环境：Linux + GPU（Ampere 及以上，需支持 bf16；昇腾 NPU 同样可跑）

---

## 1. 环境准备

```bash
# Python 3.8+ / PyTorch 2.x
pip install torch safetensors pandas pyarrow numpy tqdm tensorboard matplotlib
```

**无需安装任何 CUDA 专有算子库**：attention / MLA / GDA / Mamba 全部为纯 torch 原生实现，
不依赖 `mamba-ssm`、`causal-conv1d`、`flash-attn`，国产卡迁移友好。
`tqdm`（进度条）与 `tensorboard`（训练曲线）为固定依赖，训练机务必安装。
推荐使用锁定版本：

```bash
pip install -r requirements_lock.txt
```

---

## 2. 代码结构与数据布局

```
signal-processing/
├── code/
│   ├── configs/default.json      # 模型/训练配置（model + training 两分区）
│   ├── models/                   # 赛题相关：foundation / tokenizer / heads / frequency / backbone
│   │   └── layers/               # 通用组件：attention / gated_delta / mamba / hybrid
│   ├── training/
│   │   ├── pretrain_mae.py       # 阶段一入口：MAE 自监督预训练
│   │   ├── finetune_multitask.py # 阶段二入口：多任务微调
│   │   ├── trainer.py            # 训练生命周期（调度/断点/导出）
│   │   ├── tasks.py              # 任务损失/评估（多态实现）
│   │   └── common.py             # 共享脚手架（配置归一化/MAE 加载）
│   ├── utils/                    # 数据集/日志/断点/指标
│   └── infer.py                  # 官方推理入口
├── datasets/                     # 官方预处理数据（train_*.parquet / val_*.parquet）
├── weights/                      # 输出目录（断点、导出权重）
└── logs/                         # 训练指标（jsonl）
```

数据集兼容新旧命名：`train_protocol.parquet`（新）与 `train_system.parquet`(旧) 自动回退。

---

## 3. 模型配置说明（configs/default.json）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `model.backbone` | `hybrid` | 混合架构（GLM-5.2 式逐层排布） |
| `model.d_model / nhead` | 512 / 8 | head_dim = 64 |
| `model.n_layers` | 27 | 层数控制在 20-30 |
| `model.ffn_hidden` | 2048 | 各层 SwiGLU FFN 通道维（默认 4*d_model） |
| `model.layer_types` | 27 项列表 | `full_attention : mla = 1 : 2`（每 3 层一组 F,M,M） |
| `model.pos_encoding` | `rope` | hybrid 骨干必须配 rope |
| `model.mla_kv/q_lora_rank` | 128 / 128 | MLA KV/Q 低秩压缩秩 |
| `training.amp` | `true` | bf16 混合精度（无需梯度缩放，量化友好） |

- FFN 为 **SwiGLU**（Llama/GLM 式门控：`down(silu(gate(x)) * up(x))`，无 bias）
- 总参数量约 **118M**（bf16 约 240MB），远低于赛题 1B 红线
- 预训练/微调共享主干，权重共用率 > 95%，满足 > 80% 红线（微调入口自动校验）

---

## 4. 训练流程

### 4.0 冒烟测试（正式训练前先跑通）

```bash
cd signal-processing

# 阶段一冒烟：每 epoch 只跑 2 步，跳过评估
python code/training/pretrain_mae.py --data_dir datasets --out_dir weights --smoke_steps 2

# 阶段二冒烟（无 MAE 权重时自动从头初始化）
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights --smoke_steps 2
```

冒烟通过的标准：两个脚本无报错退出，`weights/` 下生成断点文件，
日志末尾出现 `recon_mse=...` / `train_*_weighted_loss=...`。

### 4.1 阶段一：MAE 自监督预训练（任务无关）

```bash
python code/training/pretrain_mae.py \
    --data_dir datasets \
    --out_dir weights \
    --config code/configs/default.json
```

- 全部任务训练集混用，仅用 iq/sr（无标签）
- 每个 epoch 结束自动保存断点 `weights/mae_resume.pt`
- 结束后导出共享权重 `weights/mae_pretrained.pt`

**断点续训**（中断后接着跑）：

```bash
python code/training/pretrain_mae.py --data_dir datasets --out_dir weights \
    --resume weights/mae_resume.pt
```

### 4.2 阶段二：单骨干多任务微调

```bash
# 自动加载 weights/mae_pretrained.pt 作为初始化
python code/training/finetune_multitask.py \
    --data_dir datasets \
    --out_dir weights
```

- 三任务轮转训练（每步依次更新 protocol / modulation / parameter）
- 每 epoch 评估验证集，代理得分 = 0.3*protocol_acc + 0.3*modulation_acc + 0.4*parameter_precision
- 得分创新高时自动导出官方推理格式到 `weights/`：`model.safetensors` + `config.json` + `param_stats.json`
- 早停：连续 `patience`(默认7) 个 epoch 无提升则停止

**指定 MAE 权重 / 断点续训**：

```bash
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights \
    --from_mae weights/mae_pretrained.pt        # 指定预训练权重
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights \
    --resume weights/finetune_resume.pt         # 续训（沿用原权重，不重复加载 MAE）
```

### 4.3 推理（官方格式）

```bash
python code/infer.py \
    --input datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet \
    --output out/pred.jsonl \
    --weights_dir weights \
    --batch_size 32
```

输出 JSONL 每行 `{task, sample_id, prediction}`，严格保持输入行序；
parameter 样本由 pattern 头硬掩码（不适用参数填 0.0）。

---

## 5. 关键参数调整建议

| 现象 | 调整 |
|---|---|
| 显存不足（OOM） | `training.batch_size` 减半（32→16→8）；再降 `model.ffn_hidden`（2048→1536）或 `model.d_model`（512→384） |
| 想更大模型 | 提高 `model.d_model`（512→768）与 `model.ffn_hidden`（保持约 4 倍关系），参数量仍远低于 1B |
| 预训练 loss 下降过慢 | 增大 `epochs_pretrain` 或 `training.lr`（1e-4 → 2e-4） |
| 梯度范数持续尖峰 | 减小 `training.grad_clip`（1.0→0.5）或 `training.lr` |
| 微调过拟合 | 增大 `training.weight_decay`（0.05→0.1）或减小 `epochs_finetune` |
| 数据加载慢 | `training.num_workers` 设为 CPU 核数（如 8） |
| 老 GPU 不支持 bf16 | `training.amp` 设为 `false`（回退 fp32 训练） |
| 想调整层排布 | 修改 `model.layer_types` 列表（如再加 mamba：`["full_attention","mla","mamba"]` 短列表会自动循环铺满） |

**训练指标查看**（三套可视化并行）：

```bash
# 1) 终端进度条：训练时 tqdm 自动显示（loss 无法实时显示在条上，看下面两处）

# 2) jsonl 逐 epoch 指标（每行一个 epoch）
tail -f logs/pretrain_metrics.jsonl    # recon_mse / grad_norm / lr
tail -f logs/finetune_metrics.jsonl    # train_*_loss / val_* / val_proxy_score

# 3) TensorBoard 曲线（step 级 loss / 梯度范数 / lr + epoch 级验证指标）
tensorboard --logdir logs/tb --port 6006
# 浏览器打开 http://<训练机IP>:6006

# 4) 训练结束自动出图：logs/plots/ 下 PNG（loss / grad_norm / lr / val 分组），
#    无需手动运行任何命令，fit() 结束时自动生成并写入日志
ls logs/plots/
```

TensorBoard scalar 标签一览：

| 标签 | 级别 | 含义 |
|---|---|---|
| `train/loss`、`train/loss_{task}` | step | 训练损失（微调按任务分曲线） |
| `train/grad_norm` | step | 裁剪前全局梯度范数（尖峰=不稳定，持续偏小=收敛） |
| `train/lr` | step | 学习率（warmup + cosine） |
| `epoch/recon_mse`、`epoch/train_*_loss` | epoch | 逐 epoch 平均 |
| `epoch/val_*`、`epoch/val_proxy_score` | epoch | 验证指标（微调） |

---

## 6. 赛题红线自检

微调入口会自动校验并打印，超限直接报错：

- 总参数量 < 1B（`MAX_TOTAL_PARAMS`）
- 任务间权重共用率 > 80%（`MIN_SHARED_RATIO`）

---

## 7. 常见问题

1. **报错 `no train_*.parquet under ...`**：`--data_dir` 路径不对，确认指向含
   `train_protocol.parquet` 等文件的目录。
2. **报错 `未知层类型`**：`layer_types` 里只能用 `full_attention` / `mla` / `gda` / `mamba`
   （旧写法 `attn` 也可，自动归一）。
3. **hybrid 骨干报错 `请设置 pos_encoding='rope'`**：设计如此，MLA/注意力层依赖旋转位置，
   把 `model.pos_encoding` 改为 `rope` 即可。
4. **续训后学习率异常**：断点内含 scheduler 状态，恢复位置即断点处，属正常行为。
