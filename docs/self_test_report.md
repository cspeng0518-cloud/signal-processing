# 本地运行资源需求说明（自测报告模板）

> 提交前在训练/推理机上实测后填写。数据来源命令见每节。

## 1. 环境
| 项 | 值 |
|---|---|
| 硬件 | 例：1× A800 80GB / 昇腾 910B |
| 驱动 / CUDA / CANN | |
| Python / torch / torch_npu | |
| 依赖锁定 | `pip freeze > environment/requirements_lock.txt` |

## 2. 模型规模（`finetune` 日志首行）
| 项 | 值 |
|---|---|
| 部署参数量 | |
| 共享权重占比 | |
| 主干 / neck 层类型 | |

## 3. 训练资源（`python utils/measure_vram.py --config code/configs/default.json`）
| 阶段 | batch | 峰值显存 (reserved) | 单步耗时 | 总训练时长 |
|---|---|---|---|---|
| MAE 预训练 | | | | |
| 多任务微调 | | | | |

## 4. 推理资源（`python utils/benchmark_latency.py --weights_dir weights`，日志 `logs/inference.log`）
| 精度 | batch | 长度 | P50 时延 (ms) | 每样本 (µs) | 吞吐 (样本/s) | 峰值显存 |
|---|---|---|---|---|---|---|
| bf16 | 256 | 1024 | | | | |
| bf16 | 256 | 512 | | | | |

## 5. 本地验证指标（`finetune_multitask.py --eval_test`）
| 任务 | 指标 | val | test |
|---|---|---|---|
| protocol | acc / macro recall | | |
| modulation | acc / macro recall | | |
| parameter | precision / pattern acc | | |
| 代理总分 | 0.3/0.3/0.4 加权 | | |
