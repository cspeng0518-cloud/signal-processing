# C01 多类型电磁信号感知 — 时-频双流电磁基座模型

单一共享基座同时完成**信号体制识别（8 类）/ 调制识别（23 类）/ 7 个物理参数估计**。

- **预训练主干**：FA : MLA = 1 : 2 混合 Transformer（RoPE，SDPA/NPU 融合注意力），MAE 只编码可见 token（省显存）
- **微调主干**：预训练主干 + Conformer neck（默认 2 层，补充局部波形建模）
- **时-频融合**：物理先验前端 → 时域流（ConvStem ⊕ 门控 STFT）+ 频域流（非线性谱/包络 ACF/星座）→ 跨注意力门控融合
- **多任务优化**：任务提示 + 任务专属注意力池化；三任务梯度累加单步更新；PCGrad / 不确定性加权可选；EMA
- **国产化/效率**：纯 PyTorch，无 CUDA 专有依赖；融合算子自检回退（CUDA SDPA/Triton、昇腾 npu_fusion_attention/npu_swiglu）；DFT 用实数矩阵乘（NPU/ONNX 友好）；长度分桶 + 补零裁剪；bf16

方案说明见 **[docs/training_guide.md](docs/training_guide.md)**，显存实测见 **[docs/gpu_memory_analysis.md](docs/gpu_memory_analysis.md)**。

---

## 架构

```
IQ [B,1024,2]（512 点样本右侧补零）
  │
  SignalFrontEnd（dsp.py，物理先验，无参数，fp32）
  ├─ 时域多通道(I,Q,校正I/Q,包络,瞬时频率) + 对齐 STFT + 16 维统计量
  │     └─ TimeTokenizer：ConvStem ⊕ 门控 STFT（通道级融合） -> [CLS(+任务提示)][STAT][TIME×T]
  │                                                        (512 点 T=64, 1024 点 T=128)
  └─ |X|,|X²|,|X⁴| 非线性谱 + 包络 ACF + 星座密度
        └─ FrequencyStream：SpectralTokenizer + 2 层自注意力 -> [FREQ×64]（固定长度）
  │
  CrossDomainFusion：时域 query ← 频域 key/value，门控残差
  │                 （频域 token 用完即弃，不进主干序列）
  │
  backbone：HybridEncoder，24 层 FA:MLA=1:2，RoPE，SDPA        ← MAE 预训练 + 微调
  │
  neck：HybridEncoder，2 层 Conformer（序列尾部拼接 FREQ token） ← 仅微调/推理
  │
  task_heads：注意力池化(CLS ⊕ AttnPool) + MLP
     protocol(8) / modulation(23, +调制族辅助 +SupCon) / parameter(pattern + ParamCodec 输出)
```

### MAE 预训练流程（`EMFoundation.forward_pretrain`）

1. 按每个样本的有效长度采样可见 patch，padding patch 永远排在最后、不会被选为可见。512/1024 混批时逐样本保持 75% 掩码率。
2. **被掩码 patch 对应的采样点先置零，再进前端**——频域 token、STAT、STFT 都只看到可见部分，修复旧版"整段 FFT 泄漏被掩码内容"的缺陷。
3. 只把 CLS、STAT 和可见 token 送入主干，`pos_ids` 携带原始下标给 RoPE。
4. 轻量解码器（256 维 × 2 层）在全长序列上插入 mask token，重建被掩码 patch 的单位功率 IQ + 对应 STFT 帧（时域、时频双目标）。

### 赛题约束核算（`model.shared_ratio()`，部署参数计）

| 部分 | 模块 |
|---|---|
| **共享** | tokenizer（除任务提示）、频域流、融合模块、backbone、neck |
| **私有** | task_heads、任务提示嵌入 |
| **不计入** | mae_decoder（仅预训练构建，不导出） |

默认配置共享率约 96.8%。微调入口自动校验总参数 < 1B 且共享率 > 80%。

---

## 快速开始

```bash
pip install -r environment/requirements.txt
python -m pytest tests -q                                    # 单元测试（CPU，约 1 分钟）
python utils/prepare_official.py                             # 官方数据 -> memmap（只需一次）
python utils/measure_vram.py --config code/configs/default.json --find_max_bs  # 实测显存
python code/training/pretrain_mae.py --smoke_steps 2         # 冒烟预训练
python code/training/finetune_multitask.py --smoke_steps 2   # 冒烟微调
python code/training/pretrain_mae.py --data_dir datasets --out_dir weights
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights --eval_test
python code/infer.py --input datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet \
                     --output out/test_predictions.jsonl
python utils/benchmark_latency.py --weights_dir weights
```

### 常用变体

- **多卡**：`torchrun --nproc_per_node=8 code/training/pretrain_mae.py ...`（GPU 用 nccl，昇腾用 hccl，长度分桶多卡同样生效）
- **续训**：`--resume weights/mae_resume.pt` / `weights/finetune_resume.pt`
- **跳过预训练**：微调加 `--from_mae weights/mae_pretrained.pt`
- **批量网格搜索**：`python utils/gen_configs.py --d_model 512 768 1024 --n_layers 24 30` → `python utils/measure_vram.py --grid code/configs/grid.json --budget 76` → `python utils/run_grid.py`
- **300 step 对比**：覆盖 `epochs_pretrain=1, steps_per_epoch_pretrain=300, epochs_finetune=1, steps_per_epoch=300, ema_decay=0`（EMA 在 300 步内追不上原始权重）

### 重点核对的日志行

- 微调日志 `loaded MAE pretrained weights ... N/M tensors`，N 应等于 M（或只差很少几个）
- `shared_ratio` > 80%，`backbone=[full_attention, mla, mla, ...]`，`neck=['conformer', 'conformer']`
- `measure_vram` 默认配置预训练峰值与旧 MLA 版同量级；微调因 Conformer neck 峰值略高

---

## 配置说明（`code/configs/default.json` 常用键）

| 键 | 默认 | 说明 |
|---|---|---|
| `model.d_model` / `model.n_layers` | 512 / 24 | 主干宽度与层数 |
| `model.layer_types` | `[full_attention, mla, mla]` | 逐层类型，短列表循环铺满。**不能含 conformer** |
| `model.neck_layers` / `model.neck_layer_types` | 2 / `[conformer]` | 微调 neck，0 即关闭 |
| `model.neck_with_freq` | true | neck 序列尾部是否拼接频域 token |
| `model.grad_checkpoint` | false | 逐层激活重计算，显存约降到 1/3，速度约慢 30% |
| `model.freq_layers` | 2 | 频域流自注意力层数，0 = 只做线性投影 |
| `model.mae_decoder_dim` / `model.mae_decoder_layers` | 256 / 2 | MAE 解码器规模，仅预训练 |
| `model.mask_ratio` | 0.75 | MAE 掩码率 |
| `training.batch_size_pretrain` / `training.batch_size` | 512 / 256 | 预训练/微调 batch |
| `training.patience_pretrain` / `training.patience` | 5 / 8 | 早停容忍 epoch 数（预训练 val_recon / 微调 val_proxy_score） |
| `training.min_delta_pretrain` / `training.min_delta` | 1e-3 / 1e-4 | 早停阈值（预训练相对提升 / 微调绝对提升） |
| `training.warmup_ratio` / `training.warmup_steps` | 0.05 / 0 | `warmup_steps>0` 时优先生效；两者都受"总步数 20%"上限约束 |
| `training.pretrain_lr` / `training.lr` | 1.5e-4 / 1.5e-4 | 微调 lr 按 `pretrain_lr × batch/256 × 0.66` 自动缩放 |
| `training.score_weights` | protocol:0.3, modulation:0.3, parameter:0.4 | proxy_score 加权（早停与最佳权重导出的监控指标） |

---

## 任务与代理分数

三个任务量纲不同，统一映射到 [0,1] 且方向一致后加权：

| 任务 | 指标 | 归一化 | 权重 |
|---|---|---|---|
| protocol（8 类） | acc | 天然 [0,1] | 0.3 |
| modulation（23 类） | acc | 天然 [0,1] | 0.3 |
| parameter（7 参数 + 雷达模式） | precision | `1 / (1 + mean_nrmse / ref_scale)`，其中 `mean_nrmse` = 各参数 RMSE 除以自身值域后的平均，`ref_scale` 默认 0.5 | 0.4 |

`proxy_score` 是本地代理总分，用于早停和导出最佳权重。组织方参数参考尺度、掩码聚合细则未正式核对，**不能将其称为官方分数**。

---

## 目录

```
code/
  infer.py                 统一推理入口（评测程序调用）
  configs/default.json     默认配置（约 126M 参数）；grid.json 起始网格
  models/                  foundation(组装/MAE) backbone(HybridEncoder) tokenizer(时域流)
                           frequency(频域流+融合) dsp(物理前端) heads layers/ ops/(融合算子)
  training/                pretrain_mae.py finetune_multitask.py trainer.py tasks.py common.py
  utils/                   dataset augment param_codec optim metrics logger ...
                           diagnose_val.py / bad_case_analysis.py（诊断）
utils/                     prepare_official measure_vram gen_configs gen_ablation run_grid
                           benchmark_latency profile_model export_onnx fit_param_codec
weights/                   config.json model.safetensors param_codec.json（训练后生成）
weights/best/              run_grid 批量训练的全局最佳（由 run_grid.py 自动覆盖）
docker/Dockerfile          推理镜像
docs/                      training_guide gpu_memory_analysis data_analysis_report ...
```

---

## 赛题约束

总参数 < 1B；共享权重 > 80%（默认约 97%，微调入口自动校验）；单一基座、无集成、无外挂知识库。
