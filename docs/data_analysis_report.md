# 官方真实数据集分析报告（rf_signal_benchmark_train_v1.0）

> 分析工具：`utils/analyze_real.py`（全量扫描 1,179,000 行）
> 统计明细：`docs/real_data_stats.json`
> 统计图：`figures/real_analysis/`，样本图：`figures/real_samples/`
> 日期：2026-11-15

---

## 一、数据集文件清单

下载的官方数据集放在 `datasets/01`、`datasets/02`、`datasets/03` 三个目录：

| 文件 | 大小级别 | 内容 | 对应任务 |
|---|---|---|---|
| `01/rf_signal_benchmark_train_v1.0/rf_signal_benchmark_train_v1.0_part_001_of_003.parquet` | 346,264 行 | 全部为 modulation 任务样本 | 任务1：调制方式识别（23 类） |
| `02/.../part_002_of_003.parquet` | 414,336 行 | 少量 protocol + 大量 parameter 样本 | 任务2 协议识别（8 类）+ 任务3 参数估计 |
| `03/.../part_003_of_003.parquet` | 418,400 行 | 全部为 parameter 任务样本 | 任务3：7 参数回归估计 |
| `01/rf_signal_benchmark_train_v1.0/task_schema.json` | — | 官方任务定义（23 调制类 / 8 协议类 / 7 参数目标） | 全部任务 |
| `01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet` | 6 行 | 推理输入样例（每任务 2 条，含 512/1024 两种长度） | 评测接口演示 |
| `01/rf_signal_sample_io_v1.0/rf_signal_sample_io_expected_output_v1.0.jsonl` | 6 行 | 期望输出格式样例（官方评测答案格式） | 评测接口演示 |
| `01/rf_signal_sample_io_v1.0/rf_signal_sample_io_truth_v1.0.json` | — | 6 条样例的真值（验证评测格式用） | 评测接口演示 |
| `01/rf_signal_sample_io_v1.0/sample_preview.json` | — | 数据预览说明 | — |
| `01/使用说明.txt` | — | 官方说明（解压全部三个 ZIP 到同一目录） | — |

**训练数据总计 1,179,000 行**，按行组（1024 行/组）分块存储：part_01 含 339 个行组、part_02 含 405 个、part_03 含 409 个。

## 二、每行数据的字段结构（官方 Schema）

| 字段 | 类型 | 含义 |
|---|---|---|
| `task` | string | 该行属于哪个任务：`modulation` / `protocol` / `parameter` |
| `sample_id` | string | 32 位十六进制随机 ID（不编码任何来源/标签信息） |
| `iq` | list<list<float>> | **原生嵌套数组**：N×2（N=512 或 1024），第 0 列=同相分量 I、第 1 列=正交分量 Q |
| `modulation_label` | string | 任务1 真值（23 类；仅 modulation 行有效，其余行为 None） |
| `modulation_valid` | bool | 该真值是否适用 |
| `protocol_label` | string | 任务2 真值（8 类；仅 protocol 行有效） |
| `protocol_valid` | bool | 同上 |
| `symbol_rate` ~ `number_of_pulses` | float | 任务3 的 7 个参数真值（仅 parameter 行适用） |
| `symbol_rate_valid` ~ `number_of_pulses_valid` | bool | 每参数独立的有效位（false = 该参数不适用此样本） |

注意：**官方真实数据的 iq 是原生 `list<list<float>>` 嵌套列**（非 bytes），我们现有 `code/utils/dataset.py` 的 bytes 反序列化路径需适配这一点（我们模拟数据是 bytes 存储，读取层做双格式兼容即可）。

## 三、任务与数据对应关系

三份数据文件实际是**一个大表按任务分块**：

- **任务1 调制识别（modulation，207,000 行）**：给一段 IQ，从 23 类中识别调制方式。
  - 通信波形 18 类：OOK、BPSK、QPSK、8PSK、16QAM、32QAM、64QAM、128QAM、256QAM、OQPSK、AQPSK、pi_4DQPSK、2FSK、4FSK、MSK、GFSK、CSS、FM
  - 雷达波形 5 类：Radar/LFM、Radar/Barker Code、Radar/Polyphase Barker Code、Radar/Frank Code、Radar/Coherent Pulse Train
  - **各类完全均衡：每类 9,000 条**
- **任务2 协议识别（protocol，144,000 行）**：从 8 类协议中识别信号来源：WiFi、Bluetooth、LoRa、5G、ZigBee、FM broadcast、Radar、Idle background（空闲底噪）。**每类 18,000 条，完全均衡**
- **任务3 参数估计（parameter，828,000 行）**：回归 7 个物理参数——通信 3 参数（symbol_rate 码元速率、bandwidth 带宽、frequency_shift 频偏）+ 雷达 4 参数（pulse_width 脉宽、pri 脉冲重复间隔、time_delay 时延、number_of_pulses 脉冲数）

## 四、全量统计发现

### 4.1 IQ 长度
- 1024 点：710,970 条（60.3%）；512 点：468,030 条（39.7%）
- 框架已支持混合长度（tokenizer 按 patch 对齐），无需改动

### 4.2 参数有效位呈严格二分（全量证实）
828,000 条 parameter 样本中，7 个 valid 位的组合**只有两种**：
- `1110000`（通信 3 参数有效）：414,000 条
- `0001111`（雷达 4 参数有效）：414,000 条

即数据集是**两个参数子集的并**：一半样本是"通信参数题"（symbol_rate/bandwidth/frequency_shift 有真值，雷达 4 参数不适用），另一半是"雷达参数题"（反之）。每参数 valid 率均为 50%。
→ **训练含义**：参数任务的损失必须用 valid 掩码逐位屏蔽（我们 finetune 的逐维加权平均已实现该逻辑，直接兼容）；推理输出对不适用参数填 0（官方 expected_output 样例确认此约定：通信样本的 pulse_width 等填 0.0）。

### 4.3 参数数值范围（valid=True 样本，全量）
| 参数 | 单位 | min | 中位 | max |
|---|---|---|---|---|
| symbol_rate | Hz | 62.5 | 300 | 2000 |
| bandwidth | Hz | 65 | 675 | 3500 |
| frequency_shift | Hz | -50 | 100 | 1000 |
| pulse_width | µs | 10.0 | 13.0 | 16.0 |
| pri | µs | 17.0 | 20.0 | 23.0 |
| time_delay | µs | 1.0 | 5.5 | 10.0 |
| number_of_pulses | 个 | 2 | 4 | 6 |

→ **训练含义**：回归头 Sigmoid 输出 [0,1] 后的反归一化范围需更新为此表（与模拟数据假设不同，例如官方 pulse_width 集中在 10~16µs 很窄区间、无 µs 级以上的宽脉宽；模拟数据生成器的范围假设需修正或直接以真值统计为准）。

### 4.4 幅度动态范围极大
抽样峰值幅度从 0.0005 到 47.6，跨约 5 个数量级：
- Idle background（空闲底噪）信号极弱（0.001 量级）
- 强协议信号可达 40+ 量级
→ **训练含义**：印证 tokenizer 的按样本峰值归一化设计是必须的；归一化同时天然抹掉绝对幅度信息——若任务需要幅度线索（如区分 Idle background），模型可从归一化后残留的噪声结构学习，但建议保留 RMS 作为附加 token 输入（后续实验决定）。

### 4.5 类别分布完全均衡
调制 23 类每类 9,000、协议 8 类每类 18,000——**无需类别重采样/加权**，直接均匀训练即可。

### 4.6 sample_id 为不透明哈希
32 位十六进制、不编码来源信息（官方 schema 说明确认），不能从 ID 泄漏标签，无捷径可走。

## 五、图形产出索引

**统计图（`figures/real_analysis/`）**：
| 文件 | 内容 |
|---|---|
| 01_task_distribution.png | 三任务样本量分布（mod 207k / proto 144k / param 828k） |
| 02_modulation_distribution.png | 23 类调制分布（橙=雷达 5 类，蓝=通信 18 类，每类 9k） |
| 03_protocol_distribution.png | 8 类协议分布（每类 18k） |
| 04_param_histograms.png | 7 参数直方图网格 + 有效位组合模式（快速模式抽样） |
| 05_amplitude_distribution.png | 峰值幅度 log 分布（三任务分色，跨 5 数量级） |
| 06_valid_rates.png | 各参数 valid=True 比例（全部 50%） |
| 07_valid_combo_patterns.png | 有效位组合全量统计（仅两种模式，严格二分） |

**样本图（`figures/real_samples/`，每张 2x2：时域/星座/频谱/时频）**：
- `modulation/`：5 类雷达波形 + 6 类通信波形（LFM、Barker、Frank、相干脉冲串、多相 Barker、64QAM、QPSK、MSK、GFSK、FM、CSS）
- `protocol/`：8 类协议全类各 1 张（含 Idle background 底噪样本）
- `parameter/`：通信参数样本 + 雷达参数样本各 1 张

## 六、对我们框架的影响（适配清单）

| 项 | 现状 | 需要的动作 |
|---|---|---|
| iq 列格式 | dataset.py 按 bytes 读 | 增加原生 list 列分支（`np.asarray(row.iq)` 直接成 [N,2]） |
| 任务2 名称 | system（模拟数据） | 改为 protocol；类别表改为官方 8 类（含 Idle background） |
| 任务1 类别表 | 模拟数据 23 类名近似 | 替换为官方 23 类精确名（如 pi_4DQPSK、AQPSK、Radar/Coherent Pulse Train） |
| 参数反归一化范围 | 模拟数据假设范围 | 更新为第四节 4.3 全量统计范围 |
| valid 掩码 | 逐维加权已支持 | 直接兼容，无需改 |
| 数据规模 | 780 条模拟 | 120 万条真值；DataLoader 需行组流式读取（避免一次性载入内存） |
| 评测接口 | infer.py JSONL | 与官方 sample_io 格式核对：分类输出字符串、参数输出 dict（不适用项填 0.0）——**已核对一致** |

## 七、结论

官方数据集 = **一张 1,179,000 行大表按任务切成三个分块文件**，三个任务的真值列同表共存（靠 `task` 列 + 各自 `*_valid` 位区分适用性）。数据完全类别均衡、参数有效位严格二分、幅度跨 5 数量级。训练框架的适配点集中在：iq 原生列读取、protocol 8 类映射、参数范围更新、行组流式加载四处，均为局部改动，不影响已验证的 tokenizer/backbone/多任务头架构。
