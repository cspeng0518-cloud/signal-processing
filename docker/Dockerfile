# Build on networked builder then export using docker save; offline evaluation requires embedded dependencies.
FROM python:3.10-slim
WORKDIR /workspace
COPY environment/requirements_lock.txt /tmp/requirements_lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements_lock.txt
COPY code/ /workspace/code/
COPY weights/ /workspace/weights/
ENTRYPOINT ["python", "/workspace/code/infer.py"]
