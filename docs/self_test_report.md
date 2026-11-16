# Self-test report (pending actual hardware validation)

- Unit tests: `python -m pytest tests/test_model_pipeline.py -q`.
- CPU smoke training: use reduced config and `--smoke_steps 1`.
- CUDA/NPU, official 1.2M sample throughput and memory: **not measured**.
- Docker tar: **not built**. Build from docker/Dockerfile in target environment and export image.
- Organizer's hidden scoring parity: **not verified**.
