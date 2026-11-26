#!/usr/bin/env bash
# 评测服务固定调用入口。
# 支持三种调用方式（按优先级）：
#   1) bash run.sh <input.parquet> <output.jsonl>
#   2) INPUT=... OUTPUT=... bash run.sh
#   3) bash run.sh                       # 自动探测常见挂载路径
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG_DIR"

INPUT="${1:-${INPUT:-}}"
OUTPUT="${2:-${OUTPUT:-}}"

if [[ -z "$INPUT" ]]; then
  for cand in /workspace/data/test.parquet /data/test.parquet ./data/test.parquet ./test.parquet; do
    [[ -f "$cand" ]] && INPUT="$cand" && break
  done
fi
if [[ -z "$INPUT" ]]; then
  INPUT="$(find /workspace/data /data ./data -maxdepth 2 -name '*.parquet' -type f 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$OUTPUT" ]]; then
  for d in /workspace/output /data/output ./output; do
    [[ -d "$d" ]] && OUTPUT="$d/pred.jsonl" && break
  done
  [[ -z "$OUTPUT" ]] && OUTPUT="$PKG_DIR/output/pred.jsonl"
fi

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "[run.sh] ERROR: 找不到输入 Parquet（可用 bash run.sh <input> <output> 指定）" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUTPUT")" "$PKG_DIR/logs"

export PYTHONPATH="$PKG_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# 运行环境参数默认读 config/infer.json；此处仅在显式设置环境变量时覆盖
EXTRA=()
[[ -n "${BATCH_SIZE:-}" ]] && EXTRA+=(--batch_size "$BATCH_SIZE")
[[ -n "${PRECISION:-}" ]] && EXTRA+=(--precision "$PRECISION")

echo "[run.sh] pkg=$PKG_DIR"
echo "[run.sh] input=$INPUT"
echo "[run.sh] output=$OUTPUT"
echo "[run.sh] env_config=$PKG_DIR/config/infer.json override=${EXTRA[*]:-none}"

python "$PKG_DIR/src/infer/infer.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  "${EXTRA[@]}" \
  2>&1 | tee -a "$PKG_DIR/logs/inference.log"

echo "[run.sh] done: $OUTPUT ($(wc -l < "$OUTPUT" 2>/dev/null || echo 0) lines)"
