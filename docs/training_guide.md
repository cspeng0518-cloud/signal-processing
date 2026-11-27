# 训练指导

## 1. 数据准备（只需一次）

```bash
python utils/prepare_official.py                 # 全量 1,179,000 行 -> datasets/{task}/iq.memap + meta.json
python utils/prepare_official.py --limit 3000    # 冒烟：每任务前 N 行
```
train/val/test 按 80/10/10 分层切分（protocol/modulation 按类别，parameter 按通信/雷达模式）。
val 用于早停与导出，test 只在训练结束 `--eval_test` 时用一次。首次训练会在各任务目录下缓存
`lengths.npy`（512/1024 长度，供分桶）。

## 2. 两阶段训练

训练放 tmux 里跑（裸终端断开后 tqdm 写 pty 会阻塞僵死）；日志已由代码写入 `logs/train.log`，无需重定向：

```bash
# 阶段一：MAE 自监督预训练（三任务数据混用；FA/MLA 主干；val 重建损失早停）
tmux new-session -d -s pretrain 'python src/training/pretrain_mae.py --data_dir datasets --out_dir weights'
# 阶段二：多任务微调（加载 mae_pretrained.pt；叠加 Conformer neck；val 代理分数早停）
tmux new-session -d -s finetune 'python src/training/finetune_multitask.py --data_dir datasets --out_dir weights --eval_test'
# 断点续训：--resume weights/finetune_resume.pt（僵死进程先 kill）
```

- 查看：`tmux attach -t <会话名>`；退出按 `Ctrl-b` `d`，训练继续。
- 多卡（GPU 走 nccl，昇腾 NPU 自动切 hccl，无需额外参数）：

```bash
# 4 卡微调（tmux 中）
tmux new-session -d -s finetune 'torchrun --nproc_per_node=4 \
    src/training/finetune_multitask.py --data_dir datasets --out_dir weights --eval_test'
# 只用指定卡：CUDA_VISIBLE_DEVICES=0,1 配合 --nproc_per_node=2
# 多个训练并行时错开 rendezvous 端口：--rdzv_endpoint=localhost:29501
```

  `batch_size` 为单卡值（总 batch = 单卡值 × 卡数）；日志、checkpoint、TensorBoard 只由
  主进程（rank 0）写，其他卡不落盘。断点续训同样加 `--resume`。

产物（`weights/`）：
| 文件 | 阶段 | 用途 |
|---|---|---|
| `mae_pretrained.pt` | 预训练 | 共享权重（tokenizer/频域流/融合/主干），微调加载 |
| `mae_resume.pt` / `finetune_resume.pt` | 两阶段 | 断点续训 `--resume` |
| `param_codec.json` | 微调 | 7 参数编解码规则（train 划分拟合；`--refit_codec` 重拟合） |
| `config.json` + `model.safetensors` | 微调 | 官方推理格式（best 时刷新；config 内嵌 codec） |

日志：`logs/train.log`、`logs/{pretrain,finetune}_metrics.jsonl`、`logs/{...}_steps.jsonl`、
`logs/plots/*.png`、`logs/tb/`（TensorBoard）。

### TensorBoard

```bash
tensorboard --logdir logs/tb --port 6006 --bind_all   # 注意是 --bind_all（下划线），不是 --bind-all
```

- 写入内容：每 50 步（`log_every`）写 step 级 `train/loss_*`、`train/grad_norm`、`train/lr`；
  每 epoch 写 `epoch/<指标>` 前缀的验证指标。TensorBoard 软依赖，未安装不影响训练。
- 日志目录按阶段分开：`logs/tb/pretrain/`、`logs/tb/finetune/`，同一目录下多个事件文件会被
  TensorBoard 按 global_step 自动合并。
- 断点续训无缝衔接：checkpoint 保存了 `global_step`，resume 后新事件文件从该值继续计数，
  曲线连续。注意 resume 回退重跑的 epoch 会有同一 step 区间重复的两个数据点，曲线上表现为轻微毛刺。


**必须核对的日志行**
- 微调：`loaded MAE pretrained weights ... N/M tensors`（N≈M；路径存在但 0 个可加载会直接报错）
- 微调：`shared_ratio > 80%`，`backbone=[full_attention, mla, mla, ...] neck=['conformer', ...]`
- 两阶段：如果出现 `warmup ... clamped`，说明 warmup 步数超过了总步数的 20%

## 3. 关键超参

| 键 | 默认 | 建议 |
|---|---|---|
| `d_model / n_layers / ffn_hidden` | 512 / 24 / 2048 | 先测 `measure_vram`，再试 768、1024（开 `grad_checkpoint`） |
| `layer_types` | FA:MLA=1:2 | 消融 `["full_attention"]` / `["mla"]` |
| `neck_layers` | 2 | 0（消融）/ 4；层数越多推理越慢 |
| `batch_size_pretrain / lr_pretrain` | 512 / 3e-4 | MAE 线性缩放：1.5e-4 × bs/256 |
| `batch_size / lr_finetune` | 256（每任务）/ 2e-4 | 每步 = 三任务各一个 batch |
| `mask_ratio` | 0.75 | 0.5～0.8 |
| `task_weights` | 0.5 / 1 / 1 | 或 `task_weighting: uncertainty` |
| `ema_decay` | 0.999 | 短实验（<2k 步）设 0 |

**短步数对比实验**（例如 300 步）：设 `epochs_*=1`、`steps_per_epoch_pretrain=300`、
`steps_per_epoch=300`、`ema_decay=0`。

## 4. 诊断与分析

```bash
python src/utils/diagnose_val.py --weights_dir weights          # val 链路逐级诊断
python src/utils/bad_case_analysis.py --task modulation          # 混淆矩阵 / SNR 分桶 / 错例图
python utils/gen_ablation.py && ls src/configs/ablation          # 消融配置（技术报告）
python utils/benchmark_latency.py --weights_dir weights           # 推理时延（复赛 30 分）
python utils/profile_model.py --weights_dir weights               # 算子热点
python utils/export_onnx.py --weights_dir weights                 # ONNX -> ATC / TensorRT
```


### 前端升级后的推荐诊断顺序

1. `python utils/probe_features.py --task modulation`：只跑无参数前端 + 梯度提升树，按特征组
   （base16 / stats40 / +hist / +const / all）对比可分性。组间差值即新特征的边际信息量；
   若某类探针也只有 ~50% 且随观测符号数单调上升，说明是数据天花板。
2. 重新 MAE 预训练 + 微调（前端输入维度已变化，旧权重不可复用）。
3. `python src/utils/bad_case_analysis.py --task modulation`：看 `error_structure.csv`
   （块内熵≈1 = 缺信息型错误，<0.8 = 成对混淆）与 `acc_vs_nsym.csv`。
4. 消融：`python utils/gen_ablation.py --filter A` 生成 A7~A10（去掉 CAF / 去斜 / 直方图 / 共轭循环谱）。

## 5. 批量网格

```bash
python utils/gen_configs.py --d_model 512 768 1024 --n_layers 24 30
python utils/measure_vram.py --grid src/configs/grid.json --budget 76
python utils/run_grid.py --dry && python utils/run_grid.py
```

## 6. 常见问题

1. **`主干 layer_types 不能包含 conformer`**：Conformer 只能放在 `neck_layer_types`（MAE gather 与卷积不兼容）。
2. **`MAE checkpoint ... 0 个张量可加载`**：预训练与微调用了不同的 model 配置；用同一份 config 重跑。
3. **OOM**：先降 batch 或开 `grad_checkpoint`；用 `measure_vram --find_max_bs` 定位。
4. **NPU 融合算子回退**：`logs/inference.log` 的 `fused ops:` 行会给出回退原因；`EM_FUSED_OPS=0` 可全局关闭融合算子。
