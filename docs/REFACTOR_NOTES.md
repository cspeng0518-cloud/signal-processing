# 重构说明：预训练 FA/MLA 主干 + 微调 Conformer neck（单一实现）

> 适用分支：在 `conformer` 分支基础上重构，目标合入 `main`。
> 本次重构后仓库只有**一套**模型/训练/推理实现，不再区分 v1/v2，旧配置键做了兼容或给出明确报错。

---

## 0. TL;DR

| 项 | 旧 main（MLA 版） | 旧 conformer 分支 | **本次重构** |
|---|---|---|---|
| 预训练主干 | FA:MLA=1:2，MAE 只编码可见 token | 全 Conformer，SimMIM 全序列 | **FA:MLA=1:2，MAE 只编码可见 token**（保留省显存路径） |
| 微调主干 | 同预训练主干 | 同预训练主干 | **预训练主干 + 2 层 Conformer neck**（只在微调/推理时存在） |
| 注意力实现 | eager `q@kᵀ`，物化 [B,H,L,L] | SDPA / NPU 融合 | **FA 与 MLA 都走 SDPA / NPU 融合算子** |
| 输入表征 | 峰值归一化 + 线性 patch + 整段 FFT token | 物理先验前端 + 多视角 token 全部塞进主干 | **物理先验前端 + 时-频双流**：频域 token 经独立编码器后跨注意力融合，**不进主干序列** |
| 预训练泄漏 | 频域分支看到完整未掩码信号（泄漏） | 已修复 | **已修复**（被掩码采样点先置零再进前端） |
| MAE 权重加载 | **从未成功加载**（键名不匹配） | 已修复 | **已修复 + 加载 0 个张量时直接报错** |
| 参数估计目标 | min-max + Sigmoid | ParamCodec（分类/对数/标准化自动选择） | **ParamCodec** |
| 多任务优化 | 三任务各自 step | 梯度累加单步 + PCGrad/不确定性(可选) + EMA | **同 conformer 分支** |
| 补零样本 | 无掩码 | 有效长度掩码 + 分桶裁剪（多卡时关闭分桶） | **有效长度掩码 + 分桶裁剪，多卡同样分桶** |
| 多卡梯度同步 | 逐参数 all-reduce | 逐参数 all-reduce | **按 32MB 分桶拼平 all-reduce** |
| 默认规模 | ~110M | ~35M（显存限制） | **约 126M（解析估算）**，可开 `grad_checkpoint` 继续加大 |

---

## 1. 为什么这样改：对两次实验的复盘

用户实测（同 ~100M 参数，预训练+微调各 300 step）：MLA 版 0.52 分、显存 14 GB；Conformer 版 0.30 分、显存 43 GB，单机 A800 上参数量只能维持在 100M 左右。

### 1.1 Conformer 版显存高的根因（按影响从大到小）

1. **预训练路径从 MAE 退化为 SimMIM。** MAE（mask 75%）只把约 25% 的可见 token 送进主干，即 1024 点样本约 2+32 个 token。Conformer 的深度卷积要求时间序列连续，不能 gather 可见 token，只能把被掩码 token 换成 mask token 后整段送入，即约 194 个 token。预训练阶段主干的激活量因此差了约 5～6 倍。
2. **多视角 token 直接拼进主干序列。** 32 个 SPEC、16 个 ACF、16 个 CONST 和 1 个 STAT，每层都要多算约 65 个 token。
3. **Conformer 每层更重。** Macaron 结构每层有 2 个 FFN 和一个卷积模块（PW→GLU→DW→LN→PW），外加 5 个 LayerNorm；autocast 下 LayerNorm 输出为 fp32，激活量进一步放大。

### 1.2 Conformer 版 300 step 分数低，不全是结构的问题

- **warmup 没走完。** `warmup_steps=2000` 固定，只跑 300 步时学习率始终停在峰值的约 15%。本版改为 `warmup_ratio`（默认 5%），并把 warmup 上限设为总步数的 20%，超限时打印 warning。
- **两版本地分数的口径不同。** MLA 版的 parameter precision 用**真值** valid 掩码，在 [0,1] 归一化空间计算。Conformer 版用**预测的** pattern 做硬掩码，在 SI 空间算 nRMSE。后者更接近官方，但数值会偏低。所以 0.52 和 0.30 不是同一口径，本版统一采用后者。

### 1.3 结论

- 预训练用 FA/MLA 加 MAE gather，这是显存效率的关键，必须保留。
- Conformer 的价值在于对局部波形结构（符号跳变、脉冲沿）建模，放在微调阶段作为 neck 即可。代价只是 neck 层数那几层的全序列激活。
- Conformer 分支中与主干无关的改进（前端、损失、优化、数据）全部保留。

---

## 2. 新架构

```
IQ [B,1024,2]
  │
  SignalFrontEnd（dsp.py，物理先验，无参数，fp32；有效长度按 512 取整）
  ├─ 时域多通道(I,Q,校正I/Q,包络,瞬时频率) + 对齐 STFT + 16 维统计量
  │     └─ TimeTokenizer：ConvStem ⊕ 门控 STFT（通道级融合） -> [CLS(+任务提示)][STAT][TIME×T]
  └─ |X|,|X²|,|X⁴| 非线性谱 + 包络 ACF + 星座密度
        └─ FrequencyStream：SpectralTokenizer + 2 层自注意力 -> [FREQ×64]
  │
  CrossDomainFusion：时域 query ← 频域 key/value，门控残差（注意力级融合）
  │
  backbone：HybridEncoder，24 层 FA:MLA=1:2，RoPE，SDPA        ← MAE 预训练 + 微调
  │
  neck：HybridEncoder，2 层 Conformer（序列尾部拼接 FREQ token） ← 仅微调/推理
  │
  task_heads：注意力池化(CLS ⊕ AttnPool) + MLP
     protocol(8) / modulation(23, +调制族辅助 +SupCon) / parameter(pattern + ParamCodec 输出)
```

**MAE 预训练流程**（`EMFoundation.forward_pretrain`）：

1. 按每个样本的有效长度采样可见 patch，padding patch 永远排在最后、不会被选为可见。512/1024 混批时逐样本保持 75% 掩码率。
2. 被掩码 patch 对应的采样点先置零，再进前端，因此频域 token、STAT、STFT 都只看到可见部分（防泄漏）。
3. 只把 CLS、STAT 和可见 token 送入主干，`pos_ids` 携带原始下标给 RoPE。
4. 轻量解码器（256 维 × 2 层）在全长序列上插入 mask token，重建被掩码 patch 的单位功率 IQ，以及对应的 STFT 帧（时域、时频双目标）。

**为什么 Conformer 不能放进预训练主干**：`EMFoundation.__init__` 中会检查，若 `layer_types` 里含 `conformer` 直接报错，提示改配到 `neck_layer_types`。

**赛题约束核算**（`model.shared_ratio()`，按部署参数计）：

- 共享部分：tokenizer（除任务提示）、频域流、融合模块、backbone、neck。
- 私有部分：task_heads 和任务提示嵌入。
- `mae_decoder` 只在预训练中构建，不导出，也不计入核算。
- 默认配置解析估算的共享率约 96.8%。微调入口会自动校验总参数 < 1B 且共享率 > 80%。

---

## 3. 逐文件改动清单

### 3.1 新增

| 文件 | 内容 |
|---|---|
| `code/models/backbone.py` | `HybridEncoder`：主干与 neck 共用同一个类。支持按层类型列表循环铺满、DropPath 线性递增、可选 RoPE，以及 `grad_checkpoint` 逐层激活重计算。 |
| `code/models/tokenizer.py` | 时域流：ConvStem、门控 STFT 融合、CLS/STAT 前缀和任务提示。 |
| `code/models/frequency.py` | 频域流：SpectralTokenizer、FrequencyEncoder 和 CrossDomainFusion。 |
| `code/models/layers/conformer.py` | ConformerLayer 与 ConvModule（取自 conformer 分支，接口统一）。 |
| `tests/test_model_pipeline.py` | 覆盖 MAE 前向/反向（含混合长度）、主干配 Conformer 时报错、三任务损失、共享率、**补零不变性**、**MAE 权重确实加载**、导出→推理往返。 |
| `docs/REFACTOR_NOTES.md` | 本文。 |
| `docs/self_test_report.md` | 赛题要求的本地资源自测报告模板。 |

### 3.2 重写

| 文件 | 改动要点 |
|---|---|
| `code/models/foundation.py` | 新架构组装与 MAE 流程（见 §2）。新增 `with_decoder` 参数：预训练为 True，微调/推理为 False，不再白占解码器显存和优化器状态。保留 `encode / head_outputs / predict / forward_pretrain / deploy_state_dict` 接口，工具脚本无需改动。 |
| `code/models/layers/attention.py` | `MultiHeadAttention`（合并原 RoPEAttention 与 FullAttention）、`MLAAttention`、`CrossAttention` 全部改走 `ops.fused.attention`。MLA 的共享 k_rope 改为先旋转再 expand，少 H 倍旋转计算。`DropPath` 和 `RoPEPositionalEncoding` 也移到这里。 |
| `code/models/layers/hybrid.py` | 统一层契约 `forward(x, rope, pos_ids, key_padding_mask, time_slice)`，层类型为 full_attention/mla/gda/mamba/conformer，所有层都支持 DropPath。 |
| `code/training/trainer.py` | 合并两版：生命周期骨架取 main，MAE 流程取本版，微调流程（梯度累加、PCGrad、不确定性加权、EMA）取 conformer 分支。去掉重复定义的 `export_model`，去掉 GradScaler（bf16 不需要）。warmup 设上限；断点保存 `global_step` 并原子写入；TensorBoard 缺失时不影响训练。 |
| `code/training/tasks.py` | 损失取自 conformer 分支。**评估改为精确的多卡汇总**：分类先 all-reduce 混淆矩阵，参数先 all-reduce 误差平方和与计数，再计算指标，替代原来按样本数加权平均比率的近似做法。 |
| `code/training/common.py` | `load_mae_pretrained` 按前缀加载并核对形状；路径存在却加载 0 个张量时**报错**，不再静默从头训练。`assert_competition_limits` 移到这里。 |
| `code/training/pretrain_mae.py`, `finetune_multitask.py` | 入口精简。微调多卡时 codec 文件先 barrier 再写，避免其他卡读到写了一半的文件。 |
| `code/utils/dataset.py` | 去掉 conformer 分支里**重复定义**的 `build_task_dataloaders`。`LengthGroupedBatchSampler` 支持多卡（同种子生成相同 batch 列表，再按 rank 交错切分）。去掉不再使用的 min-max 参数归一化字段。 |
| `code/utils/diagnose_val.py` | 旧版引用已不存在的 API，按新接口重写。 |
| `code/infer.py` | `load_model` 严格校验键，IQ 只解析一次；其余沿用 conformer 分支的按(任务, 长度)分组、bf16、预热计时。 |
| `code/configs/default.json`, `grid.json` | 新配置见 §4。grid 为 4 档起始网格（约 126M / 275M / 481M / 638M）。 |
| `utils/gen_configs.py` | 改为在 meta 设备上实例化模型，得到精确参数量和共享率，不再用公式估显存。显存一律用 `measure_vram.py` 实测。 |
| `utils/gen_ablation.py` | 消融项更新为新结构：双流/融合、FA vs MLA、neck 层数和是否拼频域、多任务策略、MAE 掩码率等。 |

### 3.3 小改

- `code/models/ops/fused.py`：融合算子自检的缓存键加入形状签名（E/Ev/是否有 mask）。修复前，FA 自检通过后，MLA（Ev≠E）会直接调用 NPU 融合算子而出错；NPU 融合算子现在只在 Ev==E 时启用。
- `code/models/dsp.py`：有效长度改为按 `length_quantum`（默认 512）向上取整，规则与数据侧分桶完全一致，避免尾部恰好为 0 的脉冲雷达样本被截成奇怪的长度。
- `code/utils/train_utils.py`：梯度 all-reduce 改为分桶；新增 `all_reduce_sum` 和 `clip_and_step`；删除未使用的 `all_reduce_mean` 和旧的 `build_scheduler`（调度器统一用 `utils/optim.py`）。
- `code/utils/checkpoint.py`：去掉 scaler，原子写入。
- `utils/measure_vram.py`：预训练用带解码器的模型、微调用不带解码器的模型分别实测；`--find_max_bs` 两个阶段分别找最大 batch。
- `utils/export_onnx.py`、`benchmark_latency.py`、`run_grid.py`：适配新接口（`load_model` 返回两项、`model.trim_padding`、`param_codec.json`）。
- `environment/requirements_infer.txt`：补 `tqdm`（infer 间接依赖），否则 Docker 推理镜像启动即报错。

### 3.4 删除

- `code/models/multiview.py`：拆分为 `tokenizer.py` 和 `frequency.py`。
- 原 `code/models/layers/conformer.py` 中的 `UnifiedEncoder`、`AttnBlock`、`ConvBlock`：由 `HybridEncoder` 加统一层工厂替代。
- `utils/probe_vram.py`：由 `measure_vram.py` 实测替代。
- `docs/CODE_WALKTHROUGH.md`：描述的是被替换的架构，由本文替代。
- main 中的 `SinTransformerEncoder`、`RoPETransformerEncoder`、`MambaEncoder`：分别等价于 `layer_types=["full_attention"]` 和 `["mamba"]`。

### 3.5 兼容性

- **旧权重不兼容。** 两个旧分支导出的 `model.safetensors` 和 `mae_pretrained.pt` 都无法加载，需要重新预训练。微调加载不匹配的 MAE 权重时会直接报错。
- 旧配置键 `backbone`、`pos_encoding`、`use_frequency`、`sr_bins`、`head_hidden` 已不再读取（被忽略）。`hybrid_layers` 仍可作为 `layer_types` 的别名使用。
- 数据格式不变，`utils/prepare_official.py` 产出的 memmap 可直接复用。

---

## 4. 配置说明（`code/configs/default.json` 新增/变化的键）

| 键 | 默认 | 说明 |
|---|---|---|
| `layer_types` | `[full_attention, mla, mla]` | 预训练主干的逐层类型，短列表循环铺满。**不能含 conformer。** |
| `neck_layers` / `neck_layer_types` | 2 / `[conformer]` | 微调 neck。设为 0 即关闭 neck（消融用）。 |
| `neck_with_freq` | true | neck 序列尾部是否拼接频域 token，用于晚期联合注意力。 |
| `grad_checkpoint` | false | 逐层激活重计算，显存约降到 1/3，速度约慢 30%，适合大模型。 |
| `length_quantum` | 512 | 有效长度取整粒度。 |
| `freq_layers` | 2 | 频域流自注意力层数，设为 0 表示只做线性投影。 |
| `mae_decoder_dim` / `mae_decoder_layers` | 256 / 2 | MAE 解码器规模，仅预训练使用。 |
| `mask_ratio` | 0.75 | MAE 掩码率。 |
| `warmup_ratio` / `warmup_steps` | 0.05 / 0 | `warmup_steps>0` 时优先生效；两者都受“总步数 20%”上限约束。 |
| `steps_per_epoch_pretrain` / `steps_per_epoch` | 0 | 0 表示遍历一次 loader。做“300 step 对比”时设为 300，并把 epochs 设为 1。 |

做公平的 300 step 对比时，推荐覆盖为：`epochs_pretrain=1, steps_per_epoch_pretrain=300, epochs_finetune=1, steps_per_epoch=300, ema_decay=0`（EMA 在 300 步内追不上原始权重）。

---

## 5. 运行与验证顺序

> ⚠️ **重构环境中没有 PyTorch，也无法联网。** 代码通过了语法检查、全仓库导入/未定义名的 AST 静态检查和 numpy DSP 测试，**但尚未实际跑过前向和训练**。请在训练机上按下面顺序验证，任何一步报错请把日志发回来。

```bash
pip install -r environment/requirements.txt
python -m pytest tests -q                                    # 1. 单元测试（CPU，约 1 分钟）
python utils/measure_vram.py --config code/configs/default.json --find_max_bs   # 2. 实测显存
python code/training/pretrain_mae.py --smoke_steps 2         # 3. 冒烟（需已 prepare_official）
python code/training/finetune_multitask.py --smoke_steps 2
python code/infer.py --input datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet \
                     --output out/pred.jsonl                 # 4. 推理链路
# 5. 300 step 对比（与之前同口径）
python code/training/pretrain_mae.py --out_dir weights_cmp   # 配置按 §4 覆盖
python code/training/finetune_multitask.py --out_dir weights_cmp
```

重点核对的日志行：

- 微调日志中 `loaded MAE pretrained weights ... N/M tensors`，N 应等于 M，或只差很少几个。
- `shared_ratio` > 80%，`backbone=[full_attention, mla, mla, ...]`，`neck=['conformer', 'conformer']`。
- `measure_vram`：默认配置的预训练峰值应与旧 MLA 版同量级。微调阶段因为有 Conformer neck，峰值会略高于旧 MLA 版。

---

## 6. 后续建议

1. **扩大规模**：先用 `measure_vram --find_max_bs` 确定 d512 配置的最大 batch，再试 grid 中 d768 和 d1024 两档（后者已开 `grad_checkpoint`）。
2. **消融**（技术报告用）：运行 `python utils/gen_ablation.py` 生成配置。优先跑 B3（无 neck）对 默认 对 B4（4 层 neck），量化 neck 的收益；再跑 A0（单流）对 默认，支撑“时-频双流”的创新点。
3. **复赛时延**：neck 层数和 `neck_with_freq` 是推理时延的主要可调项，用 `utils/benchmark_latency.py` 量化。
