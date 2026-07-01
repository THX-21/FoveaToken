#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
# export HF_HUB_OFFLINE=1

FULL_CHECKPOINT="${EVAL_FULL_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/fovea-vgr-qwen-9b/checkpoint-400}"
TASKS="${EVAL_TASKS:-textvqa}"
OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/logs}"
ATTN_IMPLEMENTATION="${EVAL_ATTN_IMPLEMENTATION:-sdpa}"
DEVICE_MAP="${EVAL_DEVICE_MAP:-cuda:4}"
DEVICE="${EVAL_DEVICE:-${DEVICE_MAP}}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_IMG_TOKENS="${EVAL_MAX_IMG_TOKENS:-2048}"
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-10240}"
DISABLE_FOVEA_RETRIEVAL="${EVAL_DISABLE_FOVEA_RETRIEVAL:-false}"
FOVEA_USE_AUX_HEAD="${EVAL_FOVEA_USE_AUX_HEAD:-true}"
TASK_BUDGET="${EVAL_TASK_BUDGET:-160}"

RESOLVED_CHECKPOINT="${FULL_CHECKPOINT}"
MODEL_ARGS="pretrained=${FULL_CHECKPOINT},device=${DEVICE},device_map=${DEVICE_MAP},attn_implementation=${ATTN_IMPLEMENTATION},enable_thinking=True,disable_fovea_retrieval=${DISABLE_FOVEA_RETRIEVAL},fovea_use_aux_head=${FOVEA_USE_AUX_HEAD},max_image_tokens=${MAX_IMG_TOKENS}"
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
echo "[eval] FOVEA_USE_AUX_HEAD=${FOVEA_USE_AUX_HEAD}"
echo "[eval] TASK_BUDGET=${TASK_BUDGET}"
echo "[eval] LOG_SUFFIX=${LOG_SUFFIX}"
echo "[eval] ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "[eval] MODEL_ARGS=${MODEL_ARGS}"

IFS=',' read -r -a RAW_TASK_LIST <<< "${TASKS}"
EXPANDED_TASKS=()
EXPANDED_LIMITS=()

expand_eval_task() {
  local task_name="$1"
  case "${task_name}" in
    textvqa)
      EXPANDED_TASKS+=("textvqa_val" "textvqa_test")
      EXPANDED_LIMITS+=("$((TASK_BUDGET / 2))" "$((TASK_BUDGET - TASK_BUDGET / 2))")
      ;;
    vstar_bench)
      EXPANDED_TASKS+=("vstar_bench_direct_attributes" "vstar_bench_relative_position")
      EXPANDED_LIMITS+=("$((TASK_BUDGET / 2))" "$((TASK_BUDGET - TASK_BUDGET / 2))")
      ;;
    vsibench)
      EXPANDED_TASKS+=("vsibench" "vsibench_debiased" "vsibench_pruned")
      EXPANDED_LIMITS+=("$((TASK_BUDGET / 3))" "$((TASK_BUDGET / 3))" "$((TASK_BUDGET - 2 * (TASK_BUDGET / 3)))")
      ;;
    chartqa|textvqa_val|textvqa_test|hrbench8k|mmstar|vsibench_debiased|vsibench_pruned|vstar_bench_direct_attributes|vstar_bench_relative_position)
      EXPANDED_TASKS+=("${task_name}")
      EXPANDED_LIMITS+=("${TASK_BUDGET}")
      ;;
    *)
      EXPANDED_TASKS+=("${task_name}")
      EXPANDED_LIMITS+=("${TASK_BUDGET}")
      ;;
  esac
}

for raw_task in "${RAW_TASK_LIST[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  [[ -n "${task}" ]] || continue
  expand_eval_task "${task}"
done

for idx in "${!EXPANDED_TASKS[@]}"; do
  CURRENT_TASK="${EXPANDED_TASKS[$idx]}"
  CURRENT_LIMIT="${EXPANDED_LIMITS[$idx]}"
  CURRENT_SUFFIX="${LOG_SUFFIX}-${CURRENT_TASK}"
  echo "[eval] RUN_TASK=${CURRENT_TASK} LIMIT=${CURRENT_LIMIT} LOG_SUFFIX=${CURRENT_SUFFIX}"

  "${ACCELERATE_BIN}" launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port 12345 \
    -m lmms_eval \
    --model fovea \
    --model_args "${MODEL_ARGS}" \
    --tasks "${CURRENT_TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS}" \
    --log_samples \
    --log_samples_suffix "${CURRENT_SUFFIX}" \
    --output_path "${OUTPUT_PATH}" \
    --limit "${CURRENT_LIMIT}"
done
