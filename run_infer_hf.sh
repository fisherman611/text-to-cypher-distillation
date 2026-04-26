#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for infer.py with Hugging Face model/checkpoint support.
# Override any setting via env vars before running this script.
#
# Example:
#   MODEL=Qwen/Qwen3-0.6B \
#   CKPT_PATH=your-org/your-finetuned-ckpt \
#   CKPT_REVISION=main \
#   BENCHMARK=Cypherbench \
#   DB=full \
#   bash run_infer_hf.sh

BENCHMARK="${BENCHMARK:-Cypherbench}"
DB="${DB:-full}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
CKPT_PATH="${CKPT_PATH:-}"
CKPT_REVISION="${CKPT_REVISION:-}"
DEVICE="${DEVICE:-auto}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"
LIMIT="${LIMIT:-}"

CMD=(
  python infer.py
  --benchmark "$BENCHMARK"
  --db "$DB"
  --model "$MODEL"
  --device "$DEVICE"
  --max-length "$MAX_LENGTH"
  --batch-size "$BATCH_SIZE"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
)

if [[ -n "$CKPT_PATH" ]]; then
  CMD+=(--ckpt_path "$CKPT_PATH")
fi

if [[ -n "$CKPT_REVISION" ]]; then
  CMD+=(--ckpt_revision "$CKPT_REVISION")
fi

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
