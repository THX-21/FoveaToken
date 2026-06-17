#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
# export HF_HUB_OFFLINE=1

FULL_CHECKPOINT="${EVAL_FULL_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/fovea-vgr-qwen/checkpoint-666}"
TASKS="${EVAL_TASKS:-mmstar}"
OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/logs}"
ATTN_IMPLEMENTATION="${EVAL_ATTN_IMPLEMENTATION:-sdpa}"
DEVICE_MAP="${EVAL_DEVICE_MAP:-cuda:1}"
DEVICE="${EVAL_DEVICE:-${DEVICE_MAP}}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_IMG_TOKENS="${EVAL_MAX_IMG_TOKENS:-2048}"
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-10240}"
DISABLE_FOVEA_RETRIEVAL="${EVAL_DISABLE_FOVEA_RETRIEVAL:-false}"

RESOLVED_CHECKPOINT="${FULL_CHECKPOINT}"
MODEL_ARGS="pretrained=${FULL_CHECKPOINT},device=${DEVICE},device_map=${DEVICE_MAP},attn_implementation=${ATTN_IMPLEMENTATION},enable_thinking=True,disable_fovea_retrieval=${DISABLE_FOVEA_RETRIEVAL},max_image_tokens=${MAX_IMG_TOKENS}"
LOG_SUFFIX_DEFAULT="$(basename "${RESOLVED_CHECKPOINT}")"
LOG_SUFFIX="${EVAL_LOG_SUFFIX:-${LOG_SUFFIX_DEFAULT}}"
ACCELERATE_BIN="$(command -v accelerate || true)"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PROJECT_ROOT}/.venv/bin/accelerate}"
[[ -x "${ACCELERATE_BIN}" ]] || { echo "[eval] accelerate not found: ${ACCELERATE_BIN}" >&2; exit 127; }

echo "[eval] RESOLVED_CHECKPOINT=${RESOLVED_CHECKPOINT}"
echo "[eval] FULL_CHECKPOINT=${FULL_CHECKPOINT}"
echo "[eval] TASKS=${TASKS}"
echo "[eval] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[eval] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[eval] DEVICE=${DEVICE}"
echo "[eval] DEVICE_MAP=${DEVICE_MAP}"
echo "[eval] BATCH_SIZE=${BATCH_SIZE}"
echo "[eval] MAX_IMG_TOKENS=${MAX_IMG_TOKENS}"
echo "[eval] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[eval] DISABLE_FOVEA_RETRIEVAL=${DISABLE_FOVEA_RETRIEVAL}"
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
  --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS}" \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" \
  --limit 100
