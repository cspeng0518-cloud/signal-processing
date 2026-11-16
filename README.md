# C01 多类型电磁信号感知 — 时频双流共享基座

本仓库在原 `signal-processing-main` 上升级。包含两个训练阶段、三任务共享 Transformer、时域/FFT 双流跨注意力融合、统一 Parquet→JSONL 推理、日志与绘图、单元测试。**仅为工程增强的可运行原型，不代表已达到官方数据分数或工业级交付认证。**

## 最快入口

```bash
pip install -r environment/requirements_lock.txt
python -m pytest tests/test_model_pipeline.py tests/test_infer_outputs.py -q
python code/training/pretrain_mae.py --data_dir datasets --out_dir weights
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights
python utils/plot_training.py --input logs/finetune_metrics.jsonl
python code/infer.py --input datasets/test_modulation.parquet --output out/test_predictions.jsonl
```

现有 `datasets/` 是旧仓库附带的**模拟示例**。正式比赛必须换成赛事真实数据并完整训练，详见 [docs/CHANGELOG_AND_REVIEW.md](docs/CHANGELOG_AND_REVIEW.md)。

`code/configs/default.json` 控制结构和训练；训练参数设定为建议起点，不是调参最优值。仅调试时可给两阶段脚本加 `--smoke_steps 1`。训练恢复使用 `--resume weights/mae_resume.pt` 或 `--resume weights/finetune_resume.pt`。

`weights/model.safetensors`、`weights/config.json`、`weights/param_stats.json` 在微调结束/最佳模型选择后产生，ZIP 内**不包含已经训练完成的比赛模型**。

## 文档

- [详细审查、改动与验证状态](docs/CHANGELOG_AND_REVIEW.md)
- [本地自测报告](docs/self_test_report.md)
- [官方任务书](files/)

## 关键限制

模型需要共享权重 >80%、总参数 <1B；不允许三个独立基座或集成。复赛包含国产硬件适配及推理延迟。此代码优先实现纯 PyTorch 路径，NPU FFT/attention 的实际部署行为尚未验证。
