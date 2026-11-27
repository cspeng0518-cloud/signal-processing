# C01 多类型电磁信号感知 — 时-频双流电磁基座模型

单一共享基座同时完成**信号体制识别（8 类）/ 调制识别（23 类）/ 7 个物理参数估计**。

- **预训练主干**：FA : MLA = 1 : 2 混合 Transformer（RoPE，SDPA/NPU 融合注意力），MAE 只编码可见 token（省显存）
- **微调主干**：预训练主干 + Conformer neck（默认 2 层，补充局部波形建模）
- **时-频融合**：物理先验前端 → 时域流（10 通道 ConvStem ⊕ 门控 STFT）+ 频域流（非线性谱+共轭循环谱 / 循环自相关 CAF / 去斜谱 / 复自相关 / 三通道星座 / 分布直方图）→ 跨注意力门控融合
- **符号级物理先验**：白化循环谱估码元速率 → Oerder–Meyr 定时 → M 次方精频偏去旋转 → 符号同步星座、符号域高阶累积量（详见 [docs/frontend_upgrade.md](docs/frontend_upgrade.md)）
- **多任务优化**：任务提示 + 任务专属注意力池化；三任务梯度累加单步更新；PCGrad / 不确定性加权可选；EMA
- **国产化/效率**：纯 PyTorch，无 CUDA 专有依赖；融合算子自检回退（CUDA SDPA/Triton、昇腾 npu_fusion_attention/npu_swiglu）；DFT 用实数矩阵乘（NPU/ONNX 友好）；长度分桶 + 补零裁剪；bf16

方案说明见 **[docs/training_guide.md](docs/training_guide.md)**，显存实测见 **[docs/gpu_memory_analysis.md](docs/gpu_memory_analysis.md)**。

---

## 架构

```
IQ [B,1024,2]（512 点样本右侧补零）
  │
  SignalFrontEnd（dsp.py，物理先验，无参数，fp32）
  ├─ 码元速率(白化循环谱) -> 符号同步(Oerder–Meyr) -> M 次方精频偏/相位去旋转
  ├─ 时域 10 通道(I,Q,精校正I/Q,包络,瞬时频率,平滑瞬时频率,调频率,延迟共轭积I/Q)
  │  + 对齐 STFT + 40 维统计量（含 ρ2/4/8、符号域 C40~C80、观测符号数、去斜增益、PSL）
  │     └─ TimeTokenizer：ConvStem ⊕ 门控 STFT（通道级融合） -> [CLS(+任务提示)][STAT][TIME×T]
  │                                                        (512 点 T=64, 1024 点 T=128)
  └─ SPEC(|X|,|X²|,|X⁴|,共轭循环谱) ×32 + CAF ×16 + 去斜谱 ×16 + ACF(包络+复自相关) ×16
     + 三通道星座(全部/稳定点/符号点 64×64) ×16 + 直方图 ×6
        └─ FrequencyStream：SpectralTokenizer + 2 层自注意力 -> [FREQ×102]（固定长度）
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
pip install -r requirements.txt                              # 依赖（训练用的 torch 等见文件末尾注释）
python -m pytest tests -q                                    # 单元测试（CPU，约 1 分钟；含前端物理特征回归）
python utils/probe_features.py --task modulation             # 前端特征可分性探针（不训练主干，几分钟）
python utils/prepare_official.py                             # 官方数据 -> memmap（只需一次）
python utils/measure_vram.py --config src/configs/default.json --find_max_bs  # 实测显存
python src/training/pretrain_mae.py --smoke_steps 2         # 冒烟预训练
python src/training/finetune_multitask.py --smoke_steps 2   # 冒烟微调
python src/training/pretrain_mae.py --data_dir datasets --out_dir weights
python src/training/finetune_multitask.py --data_dir datasets --out_dir weights --eval_test
bash run.sh datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet out/pred.jsonl
python utils/benchmark_latency.py --weights_dir weights
```

`run.sh` 是评测服务固定入口（`bash run.sh <input.parquet> <output.jsonl>`，也支持 `INPUT/OUTPUT`
环境变量与无参自动探测），输出 JSONL 行序与输入严格一致；镜像用根目录 `Dockerfile` 构建。

### 常用变体

- **多卡**：`torchrun --nproc_per_node=8 src/training/pretrain_mae.py ...`（GPU 用 nccl，昇腾用 hccl，长度分桶多卡同样生效）
- **续训**：`--resume weights/mae_resume.pt` / `weights/finetune_resume.pt`
- **跳过预训练**：微调加 `--from_mae weights/mae_pretrained.pt`
- **批量网格搜索**：`python utils/gen_configs.py --d_model 512 768 1024 --n_layers 24 30` → `python utils/measure_vram.py --grid src/configs/grid.json --budget 76` → `python utils/run_grid.py`
- **300 step 对比**：覆盖 `epochs_pretrain=1, steps_per_epoch_pretrain=300, epochs_finetune=1, steps_per_epoch=300, ema_decay=0`（EMA 在 300 步内追不上原始权重）

### 重点核对的日志行

- 微调日志 `loaded MAE pretrained weights ... N/M tensors`，N 应等于 M（或只差很少几个）
- `shared_ratio` > 80%，`backbone=[full_attention, mla, mla, ...]`，`neck=['conformer', 'conformer']`
- `measure_vram` 默认配置预训练峰值与旧 MLA 版同量级；微调因 Conformer neck 峰值略高

---

## 配置说明（`src/configs/default.json` 常用键）

| 键 | 默认 | 说明 |
|---|---|---|
| `model.d_model` / `model.n_layers` | 512 / 24 | 主干宽度与层数 |
| `model.layer_types` | `[full_attention, mla, mla]` | 逐层类型，短列表循环铺满。**不能含 conformer** |
| `model.neck_layers` / `model.neck_layer_types` | 2 / `[conformer]` | 微调 neck，0 即关闭 |
| `model.neck_with_freq` | true | neck 序列尾部是否拼接频域 token |
| `model.grad_checkpoint` | false | 逐层激活重计算，显存约降到 1/3，速度约慢 30% |
| `model.freq_layers` | 2 | 频域流自注意力层数，0 = 只做线性投影 |
| `model.use_caf` / `use_dechirp` / `use_hist` | true | 循环自相关 / 去斜谱 / 直方图视图开关（消融 A7~A9） |
| `model.caf_lags` / `ccaf_lags` | `[0,1,2,4,8,16,32]` / `[1,2,4,8]` | 非共轭 / 共轭循环谱滞后集合（`ccaf_lags: []` 关闭共轭循环谱） |
| `model.dechirp_rates` | 19 路 ±1/16~±2 | 去斜候选调频率（N 点内扫过的 cycles/sample），自动含 0 |
| `model.const_bins` / `const_patch` | 64 / 16 | 三通道星座图分辨率与 patch |
| `model.fine_cfo_bins` | 12 | 符号域精频偏搜索范围（bin，另封顶 0.25 cycles/symbol） |
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
Dockerfile                推理镜像（CUDA 12.8 + torch 2.9.0）
run.sh                    评测服务固定入口（Parquet -> JSONL）
requirements.txt          推理依赖；wheels/ 为离线安装包
src/
  infer/infer.py          统一推理入口（评测程序调用）
  models/                 foundation(组装/MAE) backbone(HybridEncoder) tokenizer(时域流)
                          frequency(频域流+融合) dsp(物理前端) heads layers/ ops/(融合算子)
  training/               pretrain_mae.py finetune_multitask.py trainer.py tasks.py common.py
  utils/                  dataset augment param_codec optim metrics logger ...
                          diagnose_val.py / bad_case_analysis.py（诊断：混淆块熵、acc vs 符号数）
  configs/                default.json（默认配置，约 126M 参数）；grid.json 起始网格
config/                   运行环境配置（infer.json：设备/精度/批大小/权重目录）
utils/                    prepare_official measure_vram gen_configs gen_ablation run_grid probe_features
                          benchmark_latency profile_model export_onnx fit_param_codec
weights/                  config.json model.safetensors param_codec.json（提交必需）
datasets/                 官方数据与预处理产物（readme.txt 说明来源）
docs/                     training_guide gpu_memory_analysis data_analysis_report ...
figures/                  数据分析图（对应 docs/data_analysis_report.md）
tests/                    单元测试（CPU 可跑：python -m pytest tests -q）
logs/ output/             运行产物（训练日志、推理输出），不参与提交
```

## 提交与补充材料

| 路径 | 用途 | 是否必需 |
|---|---|---|
| `weights/model.safetensors` | 微调后最终权重（唯一推理权重） | 必需 |
| `weights/config.json` | 该权重对应的模型配置（结构、seq_len、head 参数） | 必需 |
| `weights/param_codec.json` | 参数归一化/解码统计量，缺失会导致参数输出失真 | 必需 |
| `weights/mae_pretrained.pt` `mae_resume.pt` `finetune_resume.pt` | 训练中间产物（可复现训练用） | 非必需，可删 |
| `datasets/*.zip`、`datasets/*.parquet`、`datasets/{task}/*.memmap` | 官方原始数据与预处理缓存 | 非必需，见 `datasets/readme.txt` |
| `figures/` | `docs/data_analysis_report.md` 的配图 | 补充材料 |
| `tests/` | 单元测试，验证算子/指标/数据管线 | 补充材料 |
| `utils/` | 数据准备、显存实测、配置搜索、导出、时延基准等脚本 | 补充材料 |

打包提交时排除 `logs/`、`output/`、`datasets/`（体积大且可重新生成）、`**/__pycache__`。

## 权重与运行环境

| 项 | 说明 |
|---|---|
| 权重文件 | `weights/model.safetensors`（safetensors，fp32，约 483 MB，126.53M 参数） |
| 键格式 | 与 `build_model(cfg["model"], with_decoder=False)` 的 state_dict 一致 |
| 配套文件 | `weights/config.json`（模型/训练配置）、`weights/param_codec.json`（参数编解码，缺失会导致参数输出单位错误） |
| 加载方式 | `safetensors.torch.load_file` → `model.load_state_dict(..., strict=False)`，缺键/多键直接报错（`src/infer/infer.py:load_model`） |
| 模型结构 | d_model=512、24 层（FA:MLA=1:2）、ffn=2048、neck=2、seq_len=1024、patch=8 |
| 运行环境配置 | `config/infer.json`：`device`（auto/cuda/cpu）、`precision`（bf16/fp16/fp32）、`batch_size`、`weights_dir`；环境变量 `BATCH_SIZE` / `PRECISION` 可覆盖 |

`src/configs/` 放的是**训练/模型超参**，`config/` 放的是**推理运行环境**，两者不要混用。

---

## 赛题约束

总参数 < 1B；共享权重 > 80%（默认约 97%，微调入口自动校验）；单一基座、无集成、无外挂知识库。
