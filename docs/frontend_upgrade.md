# 信号前端升级说明：符号级物理先验 + 循环平稳 / 去斜 / 分布特征

> 适用分支：在原 `signal-processing-conformer` 上直接迭代（无 v1/v2 并存）。
> 影响范围：`src/models/{dsp,frequency,tokenizer,foundation}.py`、`src/configs/default.json`、
> `src/utils/bad_case_analysis.py`、`src/infer/infer.py`（报错提示）、`utils/gen_ablation.py`；
> 新增 `utils/probe_features.py`、`tests/test_frontend_features.py`、`figures/frontend_upgrade/`。
> **旧权重不兼容**：前端输入维度变化，需重新 MAE 预训练 + 微调。

---

## 1. 现有结果诊断：主要矛盾不是"几对类相像"

对 `modulation` 验证集（n=20000，acc=0.6906）的混淆矩阵按"错误是否均匀撒开"重新统计（每类错误在
其所在块内的归一化熵，1.0 = 完全均匀），结论与自动生成的 summary 不同。

### 1.1 23 类天然分成三个互不交叉的块

| 块 | 类别 | 跨块错误 | 块内错误熵 | 错分样本中位置信度 |
|---|---|---|---|---|
| A（14 类数字通信，1024 点） | OOK…AQPSK | 0 | **0.91 ~ 0.99** | 0.11 ~ 0.17（≈ 1/14 的 2 倍） |
| B（4 类） | CSS / GFSK / π/4-DQPSK / FM | 0 | 0.71 ~ 0.99 | 0.59 ~ 0.65 |
| C（5 类雷达，512 点） | Radar/* | 0 | 0.87 ~ 0.98 | 0.29 ~ 0.37 |

块 A 的召回只有 0.44 ~ 0.78，且错误几乎均匀分给块内其余 13 类。2FSK / OOK 精确率低（0.46 / 0.54），
是模型对"看不出来"样本的默认猜测。也就是说，块 A 约一半样本在当前表示下**几乎不含可分信息**。

### 1.2 Top 混淆对大多是均匀噪声

以 64QAM→32QAM（8.2%）为例：64QAM 错误率 56.4%，均匀分给 13 类的期望就是 4.3%，超出部分很小。
真正显著高于均匀期望的成对混淆只有：

- 4FSK→2FSK（13.6%，均匀期望约 3.7%）
- π/4-DQPSK→GFSK（块内熵 0.71）
- 若干相邻 QAM 阶

### 1.3 "高 SNR 却分不对"的物理原因

估计 SNR 15–20 dB 桶 acc 只有 0.37（n=5528）。看 bad case 图，带外 SNR 30 dB 以上，星座却是一团。
综合判断有三个原因：

1. **观测窗内符号数少。** 数据 `symbol_rate` 低至 62.5 Hz，样本被高度过采样，1024 点里可能只有几十个符号。
2. **没有符号同步。** 原星座图把符号间过渡点也统计进去，结构被抹平；每格 0.156，理想条件下也分不出 256QAM。
3. **粗频偏有偏。** 原前端用谱质心估计 CFO。本次验证中在 256 符号 QPSK 上实测偏差约 **6 个 bin**（周期图起伏使质心有偏），
   1024 点内相位转出数圈，星座必然糊掉。

雷达块的错误则全部落在估计 SNR < −5 dB，是**低信噪比**问题，而不是"缺相位特征"。

### 1.4 由此导出的策略

- 块 A：恢复符号级信息，即码元速率 → 定时 → 精频偏 → 符号星座 / 符号域累积量。
  同时把"观测符号数、速率估计置信度"作为显式输入，让模型知道哪些样本本身不可分（校准）。
- 块 B：提供 π/4-DQPSK 的符号相位跳变分布、FSK 的频率电平分布。
- 块 C：提供带积分增益的视图，即复自相关与去斜谱。
  这里的预期收益是处理增益，不是"补上缺失信息"。

---

## 2. 对原提案三项特征的评审

| 原提案 | 评审 | 本次实现 |
|---|---|---|
| ① 雷达脉内相位视图（复自相关 + 相位差序列 + 去斜谱） | 方向对，但前提要修正：相位差序列**已存在**（原 `ifreq` 通道即 angle(x[n]x*[n-1])/π）。真正缺的是带积分增益的视图。另外雷达错误集中在低 SNR，应按处理增益来期待收益。 | 复自相关 R(τ)（幅度 + 归一化实 / 虚部）、19 路去斜谱、平滑瞬时频率、调频率、延迟共轭积通道、PSL 统计量 |
| ② 符号同步高分辨率星座图 | **最值得做。** 但光有星座不够，必须配套码元速率估计、定时恢复和精频偏去旋转，否则同步不上。 | 白化循环谱估速 → Oerder–Meyr 定时 → M 次方精频偏 / 相位 → 64×64 符号星座 + 符号幅度环直方图 + 符号相位直方图 |
| ③ 相位差直方图 | 合理且便宜。π/4-DQPSK↔GFSK 是少数真实的成对混淆之一。 | 符号间相位差直方图（同步后）+ 延迟 D 相位差直方图 |

---

## 3. 新增特征一览

### 3.1 时域通道：6 → 10

| 通道 | 含义 / 针对 |
|---|---|
| i, q, i_c, q_c, amp, ifreq | 原有。i_c / q_c 现为**精频偏校正后**的 xf |
| ifreq_smooth | 复域滑动平均（5 点）后的瞬时频率，FSK 电平与相位码跳变更清晰 |
| chirp | 平滑瞬时频率的差分（调频率），LFM 为常数，Frank 呈阶梯 |
| dcmp_i, dcmp_q | 延迟共轭积 xf[n]·xf*[n−4]，对 CFO 不敏感的相位差视图 |

### 3.2 频域流视图：3 类 64 token → 6 类 102 token

| 视图 | token 数 | 形状（N=1024） | 物理含义 / 针对的混淆 |
|---|---|---|---|
| SPEC | 32 | 7×1024 | 原 \|X\|、\|X²\|、\|X⁴\|，**新增共轭循环谱** \|FFT(x[n]x[n−d])\|，d=1,2,4,8：OQPSK / MSK 在 ±Rs/2 处有谱线，QPSK 没有 |
| **CAF**（新，高维） | 16 | 7×512 | 非共轭循环自相关相干度 c_d(α)，d=0..32。线性调制在 α=Rs 处有谱线，雷达在 1/PRI 处 |
| **DECHIRP**（新，高维） | 16 | 19×512 | 去斜谱：调频率 m ∈ {0, ±1/16 … ±2} × 频率（离散 chirplet / FrFT 近似）。LFM / CSS 聚焦，Frank 部分聚焦，Barker 不聚焦 |
| ACF | 16 | 4×512 | 包络 ACF（原）+ **复自相关** \|R\|/R0、Re R/R0、Im R/R0（保留相位编码） |
| CONST | 16 | **3×64×64** | 三通道：全部采样点 / **稳定点加权**（抑制过渡点）/ **符号同步点** |
| **HIST**（新） | 6 | 6×64 | 平滑瞬时频率、延迟相位差、**符号间相位差**、采样幅度、**符号幅度环**、符号相位 |

### 3.3 统计量：16 → 40 维（`dsp.STAT_NAMES`）

新增 24 维：

- **速率与观测量：** `sym_rate`、`rate_conf`、`log2_sps`、`log_nsym`（观测窗内符号数）、`bw_over_rate`（FSK 带宽远大于 Rs，PSK 约为 Rs(1+β)）
- **M 次方相干度：** `rho2/4/8_sym`（符号域）、`rho2/4_smp`（采样域）。BPSK 的 ρ2≈1，QPSK 的 ρ4≈0.93 且 ρ2≈0.1，8PSK 的 ρ8≈0.8，QAM 的 ρ4≈0.4–0.5
- **同步状态：** `sync_gate`、`fine_cfo`
- **符号域累积量：** `C40/C42/C60/C63/C80_sym`、`sym_amp_std`（QAM 阶数）
- **频率与雷达：** `ifs_std`、`ifs_kurt`（2FSK≈1.2，4FSK≈2.3）、`dechirp_gain/best`、`acf_psl`（Barker 13 ≈ 0.13）、`acf_r1`

### 3.4 符号同步处理链（`SignalFrontEnd._symbol_rate / _symbol_sync / _fine_correct`）

1. **码元速率**
   - 在白化循环域剖面上取最高峰，搜索区间 α∈[4/L, 0.5)。
   - 白化：CDP(α) = mean_{相干 d} c_d(α) / 局部均值_d(α)。
   - 若半频处也有 ≥80% 的峰，回退到半频（防锁谐波）；再做抛物线插值到亚 bin。
2. **定时**：Oerder–Meyr 闭式解 τ = −sps/(2π)·arg Σ|x|²e^{−j2πn/sps}，再线性插值抽取符号 y_k。
3. **精频偏 / 相位**
   - 对 p=2/4/8 计算受限范围内的谱峰相干度 ρ_p，选最强的 p。
   - 得到残余频偏与相位，同时作用于符号域和采样域（xf）。
   - p=4 时补 π，使 QAM 网格与坐标轴对齐。
   - 仅当 ρ 显著（大于 3/√K）时才去旋转。

全部是无参数的 fp32 张量运算，DFT 仍为实数矩阵乘（NPU / ONNX 友好），所有路并入 batch 维做一次大矩阵乘。

---

## 4. 验证（合成信号，含 RRC 成形 / 载频偏移 / 定时偏移 / AWGN）

开发环境没有 PyTorch，因此用一个 numpy 实现的 torch 子集替身**执行真实的 `dsp.py` 代码**进行验证（替身不随代码提交）。
`tests/test_frontend_features.py` 的 11 个用例在替身下全部通过；在有 torch 的环境请再跑一次 `pytest tests -q`。

### 4.1 码元速率估计（210 个随机样本，7 种调制，sps 2.2–40，SNR 5–30 dB）

| 观测窗内符号数 | 估计正确率（相对误差 <2%） |
|---|---|
| ≥ 128 | 97% |
| 64 – 128 | 98% |
| 32 – 64 | 76% |
| < 32 | 30%（此时 `rate_conf` 明显偏低：中位 0.88 vs 正确样本 1.39，模型可据此降权） |

### 4.2 其他视图

| 项 | 结果 |
|---|---|
| 符号同步星座 | 原始采样是一团；同步后 16QAM 网格、8PSK 环、QPSK 四点清晰可见（`figures/frontend_upgrade/symbol_sync_constellation.png`） |
| π/4-DQPSK | 符号相位差集中于 ±π/4、±3π/4（`figures/frontend_upgrade/hist_and_dechirp.png`） |
| LFM vs Barker | 去斜增益：LFM 1.10，Barker 明显更低 |
| 雷达 PSL | Barker 0.13 / Frank 0.15 / 多相 Barker 0.28 / LFM 0.31 |
| 2FSK vs 4FSK | `ifs_kurt` 1.17 vs 2.31 |
| 鲁棒性 | 全零、极弱（1e-4）、恒定、47 倍幅度、MAE 掩码路径：无 NaN / Inf |
| 批不变性 | 单样本推理与混批推理差异 < 5e-3（软分箱保证特征对输入连续） |

### 4.3 验证中发现并修复的问题

这些问题如果不修，新特征会悄悄失效：

1. **循环谱"假峰"。**
   - 现象：大滞后（d ≥ 符号长度）的乘积在低 α 有宽带随机连续谱，直接取 max 会锁到错误速率（sps=8 时估成 186）。
   - 修复：逐滞后局部白化（窗口只在搜索带内统计）+ 只用相干滞后（|R(d)|/R0 > 0.2）+ 跨滞后平均。
2. **π/4-DQPSK 退化成 QPSK。**
   - 原因：π/4-DQPSK 的 y⁴ 在 0.5 cycles/symbol 有一根"交替"谱线。不限范围的精频偏搜索会把 π/4 交替旋转当成频偏去掉。
   - 修复：搜索范围封顶 0.25 cycles/symbol。
3. **粗频偏偏差大于预期。**
   - 原因：谱质心在 256 符号 QPSK 上偏差约 6 个 bin。
   - 修复：精搜索范围设为 ±12 bin（`fine_cfo_bins`）。这一项同时说明，**原星座图模糊的一个直接原因就是粗频偏**。
4. **硬分箱导致跳变。** 改为线性软分箱，相位直方图首尾循环相连，越界样本并入边缘箱。

---

## 5. 代价

| 项 | 原 | 新 | 说明 |
|---|---|---|---|
| 前端 DFT 量（按 N 点 DFT 等效） | 约 12 | 约 50 | 每样本约 0.4 GFLOP；主干约 33 GFLOP/样本，前端占比约 1% |
| 频域 token | 64 | 102 | 频域编码器 2 层 + neck 序列由 194 → 232 |
| 参数量 | 126.5M | 约 127.5M | SpectralTokenizer 约 1.1M；共享率仍 >96%，红线校验自动执行 |
| 去斜 19 路 | — | 前端最大的一项 | 复赛如需压时延：`dechirp_rates` 减到 9 路或 `use_dechirp: false`（A8 消融先量化收益） |

---

## 6. 使用流程（推荐顺序）

```bash
python -m pytest tests -q                                   # 先跑：含 11 个前端物理特征回归
python utils/probe_features.py --task modulation            # 可分性探针（只跑前端 + GBDT，几分钟）
python src/training/pretrain_mae.py --data_dir datasets --out_dir weights      # 必须重训
python src/training/finetune_multitask.py --data_dir datasets --out_dir weights --eval_test
python src/utils/bad_case_analysis.py --task modulation     # 新增 error_structure / acc_vs_nsym
python utils/gen_ablation.py --filter A                     # A7~A10：新视图逐项消融
```

### 6.1 探针结果的读法

- **组间差值：** `probe_features` 输出 base16 → stats40 → +hist → +const → all 的 acc，组间差值就是新特征的边际信息量。
- **主干利用不足：** 若 `all` 的探针 acc 已接近或超过当前模型 val acc（0.69），说明主干还没把已有信息用好。
  可考虑增大 `family_loss_weight`，或给 stats token 更大的学习率。
- **数据天花板：** 若块 A 某类探针也只有约 50%，且 `acc_vs_nsym` 单调上升，这部分就是数据本身的上限。
  继续堆特征只能逼近上限，不能突破。

### 6.2 bad case 分析新增输出

- **`error_structure.csv`**：每类的块内归一化熵，块由混淆连通分量自动发现。熵 >0.9 表示缺信息，<0.8 表示成对混淆。
- **`acc_vs_nsym.csv` / `acc_vs_rateconf.csv` / `acc_vs_bw.csv`**：按模型自己前端估计的观测符号数、速率置信度、占用带宽分桶。
- **`bad_cases.csv`**：新增 `nsym_est`、`rate_conf`、`occupied_bw` 三列。

---

## 7. 兼容性与注意事项

- **权重：** 旧 `model.safetensors` / `mae_pretrained.pt` 与新结构不匹配。
  - `infer.py` 会给出明确报错。
  - `--from_mae` 加载旧 MAE 时，主干等形状一致的层仍会加载，前端相关投影层被跳过并在日志列出。
  - 建议直接重新预训练，因为前端输入分布已经变了。
- **配置：** `default.json` 新增 `use_caf / use_dechirp / use_hist / caf_tokens / dechirp_tokens / caf_lags / ccaf_lags / dechirp_rates / dechirp_pool / hist_bins / dcmp_lag / if_smooth / min_symbols / cyc_whiten / fine_cfo_bins / const_range`。
  `const_bins` 由 32 改为 64，`const_patch` 由 8 改为 16。旧配置缺少的键使用代码默认值。
- **ONNX / 昇腾：**
  - 新增算子包括 `atan2 / remainder / gather / scatter_add / avg_pool1d / max_pool1d`，均为常见算子。
  - 未引入复数 dtype 或 `torch.fft`（`dft_impl="fft"` 仅作对照）。
  - 导出前请在目标平台用 `utils/export_onnx.py` 实测一次。
- **数据增强：** 现有增强（随机相位、加噪、可选共轭）与新特征相容。前端在增强之后计算，特征随之变化，不需要额外修改。

---

## 8. 后续可选方向（本次未实现）

- **盲均衡（批量 CMA，少量迭代，向量化）。** 若探针与 `error_structure` 显示块 A 在符号数充足时仍不可分，多径信道是下一个嫌疑。
  当前没有数据证据，暂不引入。
- **雷达码模板匹配滤波器组**（Barker-13 / 多相 Barker / Frank 在若干码片宽度上）。
  这是低 SNR 下的最优检测器，但需要码片宽度先验。等 A8（去斜）、A3（复自相关）消融显示雷达块确有提升空间后再做。
- **双谱 / 小波。** 计算量大，且与循环谱、去斜谱信息高度重叠，暂不建议。
