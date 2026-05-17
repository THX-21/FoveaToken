#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
export HF_HUB_OFFLINE=1

BASE_MODEL="${EVAL_BASE_MODEL:-Qwen/Qwen3.5-9B}"
CHECKPOINT_MODE="${EVAL_CHECKPOINT_MODE:-full}"
LORA_CHECKPOINT="${EVAL_LORA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/fovea-visual-query-replay/checkpoint-200}"
FULL_CHECKPOINT="${EVAL_FULL_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/fovea-visual-query-replay/checkpoint-200}"
TASKS="${EVAL_TASKS:-xlrs-lite}"
OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/logs}"
ATTN_IMPLEMENTATION="${EVAL_ATTN_IMPLEMENTATION:-sdpa}"
DEVICE_MAP="${EVAL_DEVICE_MAP:-cuda:0}"
DEVICE="${EVAL_DEVICE:-${DEVICE_MAP}}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
if [[ "${CHECKPOINT_MODE}" == "full" ]]; then
  RESOLVED_CHECKPOINT="${FULL_CHECKPOINT}"
  LOG_SUFFIX_DEFAULT="$(basename "${FULL_CHECKPOINT}")"
  MODEL_ARGS="pretrained=${FULL_CHECKPOINT},device=${DEVICE},device_map=${DEVICE_MAP},attn_implementation=${ATTN_IMPLEMENTATION},enable_thinking=False,max_image_tokens=512"
else
  RESOLVED_CHECKPOINT="${LORA_CHECKPOINT}"
  LOG_SUFFIX_DEFAULT="$(basename "${LORA_CHECKPOINT}")"
  MODEL_ARGS="pretrained=${BASE_MODEL},peft=${LORA_CHECKPOINT},device=${DEVICE},device_map=${DEVICE_MAP},attn_implementation=${ATTN_IMPLEMENTATION},enable_thinking=False,max_image_tokens=512"
fi
LOG_SUFFIX="${EVAL_LOG_SUFFIX:-${LOG_SUFFIX_DEFAULT}}"
if command -v accelerate >/dev/null 2>&1; then
  ACCELERATE_BIN="accelerate"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/accelerate" ]]; then
  ACCELERATE_BIN="${PROJECT_ROOT}/.venv/bin/accelerate"
else
  echo "[eval] accelerate not found in PATH or ${PROJECT_ROOT}/.venv/bin" >&2
  exit 127
fi

echo "[eval] BASE_MODEL=${BASE_MODEL}"
echo "[eval] CHECKPOINT_MODE=${CHECKPOINT_MODE}"
echo "[eval] RESOLVED_CHECKPOINT=${RESOLVED_CHECKPOINT}"
echo "[eval] LORA_CHECKPOINT=${LORA_CHECKPOINT}"
echo "[eval] FULL_CHECKPOINT=${FULL_CHECKPOINT}"
echo "[eval] TASKS=${TASKS}"
echo "[eval] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[eval] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[eval] DEVICE=${DEVICE}"
echo "[eval] DEVICE_MAP=${DEVICE_MAP}"
echo "[eval] BATCH_SIZE=${BATCH_SIZE}"
echo "[eval] LOG_SUFFIX=${LOG_SUFFIX}"
echo "[eval] ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "[eval] MODEL_ARGS=${MODEL_ARGS}"

"${ACCELERATE_BIN}" launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 12345 \
  -m lmms_eval \
  --model fovea \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}"
