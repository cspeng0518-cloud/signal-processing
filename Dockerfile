# C01 多类型电磁信号通用感知 —— 推理镜像
# 基础环境：Ubuntu 22.04 + CUDA 12.8 + PyTorch 2.9.0(+cu128)，与赛题指定环境一致
# 构建（离线也可）：docker build -t t001370_c01:v1 .
# 运行：docker run --gpus all --rm -v $PWD/data:/workspace/data t001370_c01:v1 \
#           /workspace/data/test.parquet /workspace/output/pred.jsonl
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    OMP_NUM_THREADS=8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace

# 依赖：优先使用随包提供的离线 wheels（wheels/），缺失时回退到在线安装
COPY requirements.txt /tmp/requirements.txt
COPY wheels/ /tmp/wheels/
RUN if ! pip install --no-cache-dir --no-index --find-links /tmp/wheels -r /tmp/requirements.txt; then \
        echo "[build] offline wheels 不可用，回退在线安装"; \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi && rm -rf /tmp/wheels /tmp/requirements.txt /root/.cache/pip

COPY src/ /workspace/src/
COPY config/ /workspace/config/
COPY weights/ /workspace/weights/
COPY run.sh /workspace/run.sh
COPY docs/ /workspace/docs/

RUN chmod +x /workspace/run.sh && mkdir -p /workspace/logs /workspace/output /workspace/data

ENTRYPOINT ["bash", "/workspace/run.sh"]
