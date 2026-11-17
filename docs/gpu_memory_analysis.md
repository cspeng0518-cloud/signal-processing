# GPU 显存占用分析

> 基准配置：`code/configs/default.json` —— hybrid backbone（9 层 full_attention + 18 层 MLA），
> d_model=512，n_layers=27，nhead=8，ffn_hidden=2048，patch_size=8，mask_ratio=0.75，
> bf16 autocast（权重保持 fp32），模型总参数量 **117.38M**。
>
> 实测校准：B=512 占 23–24 GB，B=1024 占 ~80 GB（nvidia-smi reserved 口径）。

---

## 一、序列长度事实

### 1.1 官方数据长度分布（全量 1,179,000 行扫描，docs/real_data_stats.json）

| IQ 长度 | 样本数 | 占比 |
|---|---|---|
| 1024 点 | 710,970 | 60.3% |
| 512 点 | 468,030 | 39.7% |

### 1.2 对齐方式

`utils/prepare_official.py` 将所有样本**右侧补零到 1024**，落盘为固定形状
memmap `[N, 1024, 2]`（float32）。训练时样本等长、无需动态 padding；
代价是 39.7% 的样本拖着 512 个零点参与全量计算。

### 1.3 各阶段 token 序列长度（patch_size=8 → 每 patch 8 个复数点）

| 阶段 | 张量序列长度 L | 说明 |
|---|---|---|
| Tokenizer / 频域分支 | 128 patch + 2 特殊位 = 130 | 全序列 |
| MAE backbone（keep 25%） | 34 | 96 个 patch 被掩码跳过 |
| MAE decoder | 130 | 全序列（掩码位填 mask token） |
| 微调 backbone（无掩码） | 130 | 全序列 |

**关键结论**：MAE 的掩码加速只作用于 backbone 段；decoder 与频域分支始终
跑全序列 130 token，是显存账目中容易被漏算的部分。

---

## 二、显存分桶总账（B=1024）

| 桶 | 大小 | 占比 |
|---|---|---|
| 权重 + 梯度 + 优化器状态 | 1.9 GB | 2.5% |
| 激活（前向保存） | 41 GB | **55%** |
| backward 峰值放大（中间梯度叠加） | +9 GB | 12% |
| cuBLAS/cuFFT workspace + allocator 碎片 | 5–10 GB | 15% |
| CUDA context | ~0.5 GB | 1% |
| **合计** | **75–80 GB** | 与实测一致 |

B=512 时激活项全部减半：20.5 GB 激活 + 5 GB 峰值 + 1.9 GB 固定 + ~3 GB 碎片
≈ **23–26 GB**，与实测 23.5 GB 吻合。

**全程对 batch size 严格线性**。B=512→1024 显存增长约 3.4 倍，超出"翻倍"
直觉的原因：固定项占比小（8%），实际增长倍数 ≈ (2A + f + peak) / (A + f + peak)
≈ 2.4，再叠 allocator 保留块尺寸翻倍导致的 reserved 跳涨与计量口径。

---

## 三、固定部分（与 batch size 无关）

autocast bf16 **不改变参数精度**：模型权重保持 fp32，仅算子执行时转 bf16，
backward 后的 `param.grad` 与 AdamW 的 m/v 全为 fp32。

| 项 | 计算 | 大小 |
|---|---|---|
| 权重 fp32 | 117.38M × 4B | 0.47 GB |
| 梯度 fp32 | 117.38M × 4B | 0.47 GB |
| AdamW 状态（exp_avg + exp_avg_sq） | 117.38M × 4B × 2 | 0.94 GB |
| RoPE cos/sin 表、mask_token 等缓冲 | <1M 参数 | 忽略 |
| **小计** | | **1.88 GB** |

各模块参数量分布：backbone 106.84M / mae_decoder 8.41M /
heads+fusion 2.12M / tokenizer 0.01M。

---

## 四、激活值（随 B 线性，B=1024 口径）

单位换算：bf16 = 2B/元素；autocast 下 **LayerNorm/RMSNorm/Softmax 的
输出张量强制 fp32 = 4B/元素**（易漏算项）。

### 4.1 Backbone 段（27 层，L=34 可见 token）

| 张量 | 形状 | 单层大小 | 说明 |
|---|---|---|---|
| SwiGLU 保存 4 份 | [B,34,2048]×4 | 0.57 GB | gate/up/silu/mul 各自需输入，**单项最大** |
| LayerNorm×2 输出 fp32 | [B,34,512]×2 | 0.28 GB | autocast fp32 拷贝 |
| MLA 低秩中间路 | [B,34,128~1536] 若干 | ~0.35 GB | q_a/kv_a/b 投影链（18 层） |
| qkv 投影输出 | [B,34,1536] | 0.11 GB | |
| 注意力 scores+probs fp32 | [B,8,34,34]×2 | 0.07 GB | L² 项，此处很小 |

- MLA 层 ~1.3 GB/层 × 18 + FA 层 ~1.0 GB/层 × 9 → **≈ 34 GB**
- 其中 SwiGLU 独占 27 层 × 0.57 GB ≈ **15.4 GB**（全模型第一大户）

### 4.2 MAE Decoder 段（2 层，L=130 全序列）

| 张量 | 形状 | 单层大小 |
|---|---|---|
| SwiGLU 保存 4 份 | [B,130,2048]×4 | 2.2 GB |
| norm fp32 | [B,130,512]×2 | 0.54 GB |
| qkv [B,130,1536] + scores [B,8,130,130] | | 0.42 GB |

每层 ≈ 3.2 GB，**2 层合计 ≈ 6.4 GB** —— 与主干同宽的 decoder 是
第二大消耗源。

### 4.3 频域分支 + CrossDomainFusion（L=128 谱 token，全序列）

| 张量 | 大小 |
|---|---|
| FFT 链 fp32（z/spectrum/spec，[B,1024,2] 级） | 0.17 GB |
| 谱 token proj+norm [B,128,512] | 0.07 GB |
| CrossAttention q/k/v/scores [B,8,128,34] 级 | 0.10 GB |
| gate concat [B,34,1024] + Linear | 0.09 GB |
| **小计** | **≈ 0.5 GB** |

### 4.4 杂项

mask_token expand [B,130,512]、dec 容器、visible gather、
recon/target [B,128,16] fp32 ≈ **0.4 GB**

### 4.5 前向保存合计

**34 + 6.4 + 0.5 + 0.4 ≈ 41 GB**

---

## 五、Backward 峰值

反向传播时"仍在的底层激活 + 已生成的中间梯度"叠加，峰值约为前向保存量的
1.2~1.3 倍：**+9 GB**（B=1024）。逐算子产生的瞬时梯度缓冲由 allocator
缓存复用，不长期累积。

---

## 六、Workspace 与 allocator

| 项 | 说明 |
|---|---|
| cuBLAS workspace | 大矩阵乘切换 split-K 算法，随 B 跳档，几 GB 级 |
| cuFFT workspace | FFT 按序列长度查表申请，~0.5–1 GB |
| allocator 缓存碎片 | nvidia-smi 的 reserved − allocated 差值；块尺寸随 B 翻倍后旧缓存不可复用，reserved 一次性跳涨 |
| WDDM（Windows） | nvidia-smi 含共享内存换页虚高 |

**定位方法**：

```python
torch.cuda.reset_peak_memory_stats()
# ... 训练若干 step ...
print(f"allocated={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
print(f"reserved  ={torch.cuda.max_memory_reserved()/2**30:.2f}GB")
```

- allocated 翻倍、reserved 飙升 → 碎片问题，设
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- allocated 本身逼近物理上限 → 激活真实到顶，需降 B / 梯度检查点

---

## 七、微调阶段的额外注意

微调（全序列 L=130、无掩码）相比 MAE 预训练（backbone L=34）：

- backbone 线性路激活 ×4（34→130 token）
- 注意力 scores ×15（34²→130²，但该项基数小）
- 预估 B=512 微调峰值 55–65 GB，**接近或超出单卡 48 GB**

微调大 batch **必须配合梯度检查点或减 B**。

---

## 八、优化建议（按性价比排序）

| # | 措施 | 预期收益 | 代价 |
|---|---|---|---|
| 1 | **按长度分桶**：512 长度样本独立组 batch（L 130→66） | 平均显存 -25%，算力同步节省 | 数据层按长度分组，模型无需改动 |
| 2 | **decoder 瘦身**：d_model 512→256、层数 2→1（MAE 原文风格：窄而浅） | -5 GB + 预训练提速 | 重建质量略降，通常可接受 |
| 3 | **频域分支只算可见谱**：谱 token 仅作 K/V，可只保留可见位置对应谱 patch | 频域段激活 -80% | 实现改动中等 |
| 4 | **梯度检查点**：27 层逐层 checkpoint | backbone 激活 34→~3 GB | step time +30% |
| 5 | SwiGLU 窄化 F=2048→1536 | SwiGLU 激活 -25% | 容量受损，优先级最低 |

推荐执行顺序：**1 → 2**（纯赚零风险）；3、4 视 B=1024 是否硬需求。
