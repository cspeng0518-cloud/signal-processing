# C01 电磁信号基座项目代码审查与升级说明

> 版本：2026-11-16 工程增强版。说明基于用户上传的赛题任务书及源码静态审查和 CPU 小规模测试，不代表已经完成官方大数据训练或国产算力适配认证。

## 一、比赛硬性条件与对应实现

| 要求 | 当前情况 | 风险/行动 |
|---|---|---|
| 单一共享基座、共享权重 >80% | `EMFoundation` 使用共享 tokenizer / FFT tokenizer / cross attention / backbone，三任务仅小 head；训练启动时检查比例 | 需要对最终提交权重及配置再计算；比例统计不含预训练后不用于推理的 MAE decoder |
| 模型 <1B 参数 | 微调启动时检查所有模块参数数 | 实际提交配置仍需记录 |
| 时-频双流 + 交互 | 新增 frequency.py 的频域 FFT 幅相 token、跨注意力及门控残差 | 频域支路尚未与无频域基线完成消融；FFT 可迁移性须在目标平台验证 |
| 三项任务 | protocol、modulation、parameter / parameter_pattern 辅助头，共享同一 backbone | 参数置信度、模式误分类与不同信号混叠需额外评估 |
| 输入输出 | 官方 Parquet task/sample_id/iq -> JSONL；兼容旧 system 别名 | 官方测试集未知，需正式输入输出回归测试 |
| 国产算力 | 优先纯 PyTorch Transformer；提供 NPU 优先设备选择 | **仅检测设备不代表适配**，需安装对应 torch_npu/框架并验证 FFT、attention 和部署链路 |
| 交付物 | 权重、配置、训练及推理源码、日志、模型说明和 Dockerfile 模板 | **未生成训练好的正式权重、Docker tar、官方评估报告** |

赛题初赛权重为 protocol 30%，modulation 30%，parameter 40%。复赛还额外考核推理延迟；不要只看综合训练损失。项目中 `val_proxy_score` 采用 0.3/0.3/0.4 组合，仅供本地比较：组织方参数参考尺度、掩码聚合细则尚未由代码仓得到正式核对，**不能将其称为官方分数**。

## 二、原仓库存在的问题与修复

### P0: MAE 掩码逻辑错误

原 `foundation.py` 中 `_random_mask` 返回保留的 patch 下标，随后的 `forward_mae` 再用同一索引提取目标与解码 token，实际上是在重建**可见 token**，不是 masked token。升级后编码器只接收 `[CLS]/[SR]` 和可见 patch；解码器从 mask token 和编码后的可见 token 重建全序列，仅对**被掩码** patch 计算误差。`tests/test_model_pipeline.py` 验证重建形状。

### P0: 回归有效性掩码错误

原实现 `mse = nn.MSELoss()` 默认 `reduction='mean'`，使 `mse(out,target)` 先成为一个标量，再乘 `[B,7]` 掩码，无法仅对有效参数归约。新 `parameter_loss()` 采用 `reduction='none'` 的 Smooth L1，每个元素相乘 valid 再除有效元素数量；无效参数对损失贡献严格为零。增加单元测试。

### P0: 无效参数反归一化顺序错误

原推理与验证代码先清零归一化数值再加最小值，造成应该输出 0 的参数出现非零值。新推理 **先将归一化回归预测反归一化到物理单位，再应用有效模式掩码**。推理测试确保真正输出 0。

### P1: 仅时域架构不满足双流的完整描述

新增 `code/models/frequency.py`：`torch.fft.fft(... norm='ortho')` 处理复 IQ，频域特征采用 log(1+|X|) 与相位（/π），基于 patch 投影；`CrossDomainFusion` 使用时域 query 与频域 key/value 做 attention，融合采用可学习 gate 与 LayerNorm。训练默认 `use_frequency=true`。这里是工程上可运行的一种双流实现，**还没有经过精度消融，不应直接称为最优方案**。

### P1: 训练设施缺失

重写 `pretrain_mae.py` / `finetune_multitask.py`：

- 数据源支持 `train_*.parquet` 和 `datasets_official/{task}/meta.json+iq.memap`；MAE 仅在训练子集上自监督，避免验证集信息泄漏。
- 固定 Python/NumPy/PyTorch 随机种子，支持 CUDA 自动混合精度、梯度裁剪、warmup+cosine 学习率调度。
- 支持 `--resume` 断点恢复 optimizer、scheduler、GradScaler、epoch 等状态；预训练输出 `mae_pretrained.pt`，微调保存 `finetune_resume.pt`。
- 三任务轮换、共享同一组 backbone 参数和 optimizer；对训练任务使用可配置的权重；参数模式二分类作为辅助任务。
- 逐 epoch 验证、按本地代理分数保存最佳 `weights/model.safetensors`、早停、保存指标 JSONL。
- `--smoke_steps 1` 为快速冒烟用途，会产生未训练完成的实验权重，**不得提交**。

**局限：**训练步骤目前每个任务分别 optimizer.step()，并非一批次聚合多任务梯度；三任务负迁移、PCGrad/GradNorm、采样不平衡仍需在真实验证集上比较。大规模预训练可能需进一步优化 DataLoader、IO、checkpoint、DDP；当前不具备真正的大规模分布式训练机制。

### P1: 现有 Dataset 的真实任务过滤问题

`EMSignalDataset` 新增按 `task` 列过滤，避免加载混排官方 parquet 时跨任务污染。注意准备好的 memmap 内仍需由 `prepare_official.py` 确保分层切分无泄漏。

### P2: 日志、曲线、配置与部署

- `code/utils/logger.py`：统一控制台 + UTF-8 文件日志；各阶段 JSONL 指标记录。
- `utils/plot_training.py`：把预训练/微调 JSONL 变成多张 PNG/PDF 曲线，适用于无图形界面的远程服务器。
- `code/utils/train_utils.py`：统一 seed、AMP、scheduler、梯度更新、设备选择。
- `environment/`：依赖版本范围和 conda 环境定义，实际硬件上还需锁定精确版本。
- `docker/Dockerfile`：提供在线构建 / 离线部署模板，**不是已经提交的 Docker 镜像**。
- `tests/`：增加 masked MAE、共享比例、masked regression、混合输入顺序、无效参数清零测试。

## 三、最重要的工程验证状态

| 验证项目 | 结论 |
|---|---|
| Python 语法编译 | 已执行 |
| 新增 PyTorch 单元测试 | 已执行 |
| CPU 小配置 MAE 1 step | 已运行，通过 |
| CPU 小配置三任务微调 1 step | 已运行，通过 |
| `safetensors` 模型导出 | 已运行，通过 |
| 官方 Parquet 端到端推理 | 未运行：当前验证容器未安装 pyarrow，采用 DataFrame 仿真测试推理行为 |
| 官方 120 万数据训练、性能、资源占用 | 未运行 |
| CUDA/NPU 完整训练与耗时 | 未运行 |
| 工业级可靠性 / Docker tar / 国产平台验收 | 未验证 |

## 四、建议运行顺序

请从项目根目录执行，先创建环境：

```bash
conda create -n c01 python=3.10 -y
conda activate c01
pip install -r environment/requirements_lock.txt
python -m pytest tests/test_model_pipeline.py tests/test_infer_outputs.py -q
```

**正式数据**：按 `utils/prepare_official.py --help` 查看官方准备流程，确保按 task 分组并输出训练/验证划分。检查 `datasets_official/` 的 meta.json 与 `iq.memap`，并且只用训练集计算回归参数统计。当前默认内置参数范围基于旧仓库统计，正式赛前需要复核。随后执行：

```bash
python code/training/pretrain_mae.py --config code/configs/default.json --data_dir datasets_official --out_dir weights
python code/training/finetune_multitask.py --config code/configs/default.json --data_dir datasets_official --out_dir weights
python utils/plot_training.py --input logs/pretrain_metrics.jsonl --output_dir figures/training/pretrain
python utils/plot_training.py --input logs/finetune_metrics.jsonl --output_dir figures/training/finetune
python code/infer.py --input path/to/official_test.parquet --output out/predictions.jsonl --weights_dir weights
```

如果从现有旧权重升级到新频域模型，**不能直接加载旧版 `model.safetensors`**（网络结构改变）。应重新运行预训练与微调；或者把 `use_frequency` 设为 false，保持纯时域基线作消融，仍需对权重结构兼容性单独检查。

## 五、下一轮优化优先级

1. 在真实数据上核对 23 个调制与 8 个体制标签以及参数有效性标注模式（不能仅依赖旧项目写死的假设）。
2. 按真实官方隐式评分规则确认参数 RMSE 的具体聚合方式和参考尺度；选择合适参数变换（符号率/带宽对数，脉冲宽度标准化）并按验证集比较。
3. 三个消融：时域 vs 频域 vs 时频双流；MAE vs 随机初始化；hard pattern mask vs 不做 mask。
4. 大数据加载优化：以 mmap+索引/Parquet shard 训练，监控 IO 和内存；增加 DDP、梯度累积和分布式采样。
5. 国产平台做真实推理移植：测试 FFT、attention、LayerNorm 算子可用性与 fallback，测吞吐、P50/P95 推理延迟和精度漂移。
6. 按赛题目录补齐正式训练权重、来源说明、详细运行日志、Docker tar、技术报告与资源报告，并确认评测调用规范。

## 六、文件改动概览

**新增**：`code/models/frequency.py`、`code/utils/logger.py`、`code/utils/train_utils.py`、`utils/plot_training.py`、`tests/test_model_pipeline.py`、`tests/test_infer_outputs.py`、`environment/*`、`docker/Dockerfile`、`docs/CHANGELOG_AND_REVIEW.md`、`docs/self_test_report.md`。

**修改**：`code/models/foundation.py`、`code/training/pretrain_mae.py`、`code/training/finetune_multitask.py`、`code/infer.py`、`code/utils/dataset.py`、`code/configs/default.json`，以及根目录 README。保留原数据分析脚本、合成样本、可视化资源与赛题资料。
