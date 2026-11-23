# 显存分析与调参

> 历史版本的公式估算（`probe_vram.py` / `vram_estimate_gb`）只在单一配置点校准，对其他宽度系统性
> 低估，已废弃。**一律用 `utils/measure_vram.py` 实测。**

## 1. 显存构成（单卡，bf16 autocast，AdamW）

| 部分 | 量级 | 说明 |
|---|---|---|
| 静态 | ≈16 B/参数 | fp32 权重 + 梯度 + AdamW m/v；微调另加 EMA 副本 4 B/参数 |
| 预训练激活 | ∝ batch × (2 + 0.25·T) × 层数 | MAE 只编码可见 token（T=128 时约 34 个） |
| 预训练解码器 | ∝ batch × (2+T) × 2 层 × 256 维 | 窄解码器，占比小 |
| 微调激活 | ∝ batch × (2+T) × 层数 + neck × batch × (2+T+64) | 逐任务反传：峰值 = 单任务前反向 |
| 前端 | 少量 fp32 张量 | 无梯度，DFT 矩阵为常量 buffer（≈40 MB） |

关键结论（对应用户实测：同 ~100M 参数，MLA 版 14 GB / 全 Conformer 版 43 GB）：
1. **预训练必须走 MAE gather**：全序列 SimMIM 的主干激活约为 MAE 的 5～6 倍；
2. **多视角 token 不进主干**：频域 token 经独立 2 层编码器后跨注意力融合，主干序列长度不变；
3. **Conformer 只做 neck**：每层激活约为 FA/MLA 层的 2～3 倍，只放 2～4 层；
4. **注意力走 SDPA/融合算子**：不物化 [B,H,L,L] 注意力矩阵；
5. `grad_checkpoint: true` 可把激活压到约 1/3（速度约慢 30%），用于 d_model ≥ 1024 或大 batch。

## 2. 实测流程

```bash
python utils/measure_vram.py --config code/configs/default.json            # 两阶段峰值
python utils/measure_vram.py --config code/configs/default.json --find_max_bs
python utils/gen_configs.py --d_model 512 768 1024 --n_layers 24 30 --neck 2 4
python utils/measure_vram.py --grid code/configs/grid.json --budget 76
```

## 3. 实测记录（请在训练机上填写）

| 日期 | 硬件 | 配置 | 参数量 | 预训练 bs / 峰值 | 微调 bs / 峰值 |
|---|---|---|---|---|---|
| | A800 80G | default (d512 L24 neck2) | ~126M | 512 / | 256 / |
| | A800 80G | d768 L24 neck2 | ~275M | | |
| | A800 80G | d1024 L24 neck2 + ckpt | ~481M | | |
