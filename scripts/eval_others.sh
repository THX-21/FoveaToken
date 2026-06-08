#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

BASE_MODEL="${EVAL_BASE_MODEL:-Qwen/Qwen3.5-9B}"
MODEL="${EVAL_MODEL:-qwen3_5}"
TASKS="${EVAL_TASKS:-chartqa}"
OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/logs}"
ATTN_IMPLEMENTATION="${EVAL_ATTN_IMPLEMENTATION:-sdpa}"
DEVICE_MAP="${EVAL_DEVICE_MAP:-cuda:0}"
DEVICE="${EVAL_DEVICE:-${DEVICE_MAP}}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
LIMIT="${EVAL_LIMIT:-500}"
FORCE_SIMPLE="${EVAL_FORCE_SIMPLE:-1}"

MODEL_ARGS_DEFAULT="pretrained=${BASE_MODEL},device=${DEVICE},device_map=${DEVICE_MAP},attn_implementation=${ATTN_IMPLEMENTATION}"
MODEL_ARGS="${EVAL_MODEL_ARGS:-${MODEL_ARGS_DEFAULT}}"
LOG_SUFFIX="${EVAL_LOG_SUFFIX:-$(basename "${BASE_MODEL}")}"
ACCELERATE_BIN="$(command -v accelerate || true)"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PROJECT_ROOT}/.venv/bin/accelerate}"
[[ -x "${ACCELERATE_BIN}" ]] || { echo "[eval_others] accelerate not found: ${ACCELERATE_BIN}" >&2; exit 127; }

EXTRA_ARGS=()
[[ -n "${LIMIT}" ]] && EXTRA_ARGS+=(--limit "${LIMIT}")
[[ "${FORCE_SIMPLE}" == "1" ]] && EXTRA_ARGS+=(--force_simple)

echo "[eval_others] MODEL=${MODEL}"
echo "[eval_others] BASE_MODEL=${BASE_MODEL}"
echo "[eval_others] TASKS=${TASKS}"
echo "[eval_others] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[eval_others] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[eval_others] DEVICE=${DEVICE}"
echo "[eval_others] DEVICE_MAP=${DEVICE_MAP}"
echo "[eval_others] BATCH_SIZE=${BATCH_SIZE}"
echo "[eval_others] LIMIT=${LIMIT}"
echo "[eval_others] FORCE_SIMPLE=${FORCE_SIMPLE}"
echo "[eval_others] LOG_SUFFIX=${LOG_SUFFIX}"
echo "[eval_others] ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "[eval_others] MODEL_ARGS=${MODEL_ARGS}"

"${ACCELERATE_BIN}" launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 12345 \
  -m lmms_eval \
  --model "${MODEL}" \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" \
  "${EXTRA_ARGS[@]}"
