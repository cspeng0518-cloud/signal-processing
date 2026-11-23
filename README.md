# C01 多类型电磁信号感知 — 时-频双流电磁基座模型

单一共享基座同时完成**信号体制识别（8 类）/ 调制识别（23 类）/ 7 个物理参数估计**。

- **预训练主干**：FA : MLA = 1 : 2 混合 Transformer（RoPE，SDPA/NPU 融合注意力），MAE 只编码可见 token（省显存）
- **微调主干**：预训练主干 + Conformer neck（默认 2 层，补充局部波形建模）
- **任务 1 时-频融合**：物理先验前端 → 时域流（ConvStem ⊕ 门控 STFT）+ 频域流（非线性谱/包络 ACF/星座）→ 跨注意力门控融合
- **任务 2 多任务**：任务提示 + 任务专属注意力池化；三任务梯度累加单步更新；PCGrad / 不确定性加权可选；EMA
- **任务 3 国产化/效率**：纯 PyTorch，无 CUDA 专有依赖；融合算子自检回退（CUDA SDPA/Triton、昇腾 npu_fusion_attention/npu_swiglu）；DFT 用实数矩阵乘（NPU/ONNX 友好）；长度分桶 + 补零裁剪；bf16

本次重构说明见 **[docs/REFACTOR_NOTES.md](docs/REFACTOR_NOTES.md)**。

## 快速开始

```bash
pip install -r environment/requirements.txt
python -m pytest tests -q
python utils/prepare_official.py                       # 官方数据 -> memmap（只需一次）
python utils/measure_vram.py --config code/configs/default.json --find_max_bs
python code/training/pretrain_mae.py --data_dir datasets --out_dir weights
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights --eval_test
python code/infer.py --input datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet \
                     --output out/test_predictions.jsonl
python utils/benchmark_latency.py --weights_dir weights
```

多卡：`torchrun --nproc_per_node=8 code/training/pretrain_mae.py ...`（GPU 用 nccl，昇腾用 hccl，长度分桶多卡同样生效）。
冒烟：两阶段脚本加 `--smoke_steps 2`。续训：`--resume weights/mae_resume.pt` / `weights/finetune_resume.pt`。

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
                           benchmark_latency profile_model export_onnx fit_param_codec 可视化/数据分析
weights/                   config.json model.safetensors param_codec.json（训练后生成）
docker/Dockerfile          推理镜像
docs/                      REFACTOR_NOTES training_guide gpu_memory_analysis self_test_report ...
```

## 赛题约束

总参数 < 1B；共享权重 > 80%（默认约 97%，微调入口自动校验）；单一基座、无集成、无外挂知识库。
