#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
# export HF_HUB_OFFLINE=1

ENV_FILE="${PROJECT_ROOT}/.env"
[[ -f "${ENV_FILE}" ]] || { echo "[eval] missing ${ENV_FILE}" >&2; exit 1; }
set -a
source "${ENV_FILE}"
set +a

FULL_CHECKPOINT="${EVAL_FULL_CHECKPOINT:-Qwen/Qwen2.5-VL-7B-Instruct}"
LORA_CHECKPOINT="${EVAL_LORA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/fovea-vlmr3-json-qwen2.5-vl-7b/checkpoint-1445}"
# Common eval task names:
#   hrbench8k      -> single task
#   xlrs-lite      -> single task
#   textvqa        -> group task, expands to textvqa_val + textvqa_test
#   chartqa        -> single task
#   vstar_bench    -> single task with internal categories: direct_attributes / relative_position
#   mmstar         -> single task
#   docvqa         -> group task, expands to docvqa_val + docvqa_test
# MathVista has multiple variants; pick one explicitly before running:
#   mathvista                -> group, expands to mathvista_testmini + mathvista_test
#   mathvista_testmini       -> group, expands to mathvista_testmini_cot / _solution / _format
#   mathvista_test           -> submission-style test split
#   mathvista_testmini_cot   -> public mini eval, step-by-step prompt
#   mathvista_testmini_solution
#   mathvista_testmini_format
# Note: `charvqa` was not found in the current repo. If you meant chart QA, use `chartqa`.
# TASKS="${EVAL_TASKS:-hrbench8k,xlrs-lite,textvqa_val,chartqa,vstar_bench,mmstar,mathvista_testmini_solution}"
TASKS="${EVAL_TASKS:-chartqa}"
OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/logs}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python"
ATTN_IMPLEMENTATION="${EVAL_ATTN_IMPLEMENTATION:-sdpa}"
DEVICE_MAP="${EVAL_DEVICE_MAP:-cuda:0}"
DEVICE="${EVAL_DEVICE:-${DEVICE_MAP}}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_PROCESSES="${EVAL_NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${EVAL_MAIN_PROCESS_PORT:-12345}"
BALANCED_LIMIT="${EVAL_BALANCED_LIMIT:-true}"
MAX_IMG_TOKENS="${EVAL_MAX_IMG_TOKENS:-2048}"
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
DISABLE_FOVEA_RETRIEVAL="${EVAL_DISABLE_FOVEA_RETRIEVAL:-false}"
FOVEA_CROP_THRESHOLD="${EVAL_FOVEA_CROP_THRESHOLD:-0.25}"
FOVEA_CROP_REGION_SCALE="${EVAL_FOVEA_CROP_REGION_SCALE:-1.2}"
PREFILL_THINK="${EVAL_PREFILL_THINK:-true}"
DEFAULT_FOVEA_REASONING_PROMPT='\nYou need to first think about the reasoning process in your mind and then provide the answer. When thinking you should call the "fovea" tool (format: {"fovea"}) to focus on key areas in the image. The reasoning process and the answer are included in the <think> </think> and <answer> </answer> tags respectively.'
FOVEA_REASONING_PROMPT="${EVAL_FOVEA_REASONING_PROMPT:-${DEFAULT_FOVEA_REASONING_PROMPT}}"
TASK_BUDGET="${EVAL_TASK_BUDGET:-600}"
EVAL_SUBSET_SEED="${EVAL_SUBSET_SEED:-42}"

RESOLVED_CHECKPOINT="${LORA_CHECKPOINT:-${FULL_CHECKPOINT}}"
MODEL_ARGS="pretrained=${RESOLVED_CHECKPOINT},device=${DEVICE},device_map=${DEVICE_MAP},attn_implementation=${ATTN_IMPLEMENTATION},disable_fovea_retrieval=${DISABLE_FOVEA_RETRIEVAL},fovea_crop_threshold=${FOVEA_CROP_THRESHOLD},fovea_crop_region_scale=${FOVEA_CROP_REGION_SCALE},prefill_think=${PREFILL_THINK},max_image_tokens=${MAX_IMG_TOKENS},reasoning_prompt=${FOVEA_REASONING_PROMPT}"
LOG_SUFFIX_DEFAULT="$(basename "${RESOLVED_CHECKPOINT}")"
LOG_SUFFIX="${EVAL_LOG_SUFFIX:-${LOG_SUFFIX_DEFAULT}}"
ACCELERATE_BIN="$(command -v accelerate || true)"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PROJECT_ROOT}/.venv/bin/accelerate}"
[[ -x "${ACCELERATE_BIN}" ]] || { echo "[eval] accelerate not found: ${ACCELERATE_BIN}" >&2; exit 127; }

echo "[eval] RESOLVED_CHECKPOINT=${RESOLVED_CHECKPOINT}"
echo "[eval] FULL_CHECKPOINT=${FULL_CHECKPOINT}"
echo "[eval] LORA_CHECKPOINT=${LORA_CHECKPOINT}"
echo "[eval] TASKS=${TASKS}"
echo "[eval] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[eval] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[eval] DEVICE=${DEVICE}"
echo "[eval] DEVICE_MAP=${DEVICE_MAP}"
echo "[eval] BATCH_SIZE=${BATCH_SIZE}"
echo "[eval] NUM_PROCESSES=${NUM_PROCESSES}"
echo "[eval] MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "[eval] BALANCED_LIMIT=${BALANCED_LIMIT}"
echo "[eval] MAX_IMG_TOKENS=${MAX_IMG_TOKENS}"
echo "[eval] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[eval] DISABLE_FOVEA_RETRIEVAL=${DISABLE_FOVEA_RETRIEVAL}"
echo "[eval] FOVEA_CROP_THRESHOLD=${FOVEA_CROP_THRESHOLD}"
echo "[eval] FOVEA_CROP_REGION_SCALE=${FOVEA_CROP_REGION_SCALE}"
echo "[eval] PREFILL_THINK=${PREFILL_THINK}"
echo "[eval] FOVEA_REASONING_PROMPT=${FOVEA_REASONING_PROMPT}"
echo "[eval] TASK_BUDGET=${TASK_BUDGET}"
echo "[eval] EVAL_SUBSET_SEED=${EVAL_SUBSET_SEED}"
echo "[eval] LOG_SUFFIX=${LOG_SUFFIX}"
echo "[eval] ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "[eval] PYTHON_BIN=${PYTHON_BIN}"
echo "[eval] MODEL_ARGS=${MODEL_ARGS}"

case "${BALANCED_LIMIT}" in
  true|TRUE|1|yes|YES)
    export LMMS_EVAL_BALANCED_LIMIT=1
    export LMMS_EVAL_BALANCED_SEED="${EVAL_SUBSET_SEED}"
    ;;
  *)
    unset LMMS_EVAL_BALANCED_LIMIT
    unset LMMS_EVAL_BALANCED_SEED
    ;;
esac

IFS=',' read -r -a RAW_TASK_LIST <<< "${TASKS}"
EXPANDED_TASKS=()
EXPANDED_LIMITS=()

resolve_group_tasks() {
  "${PYTHON_BIN}" - "$1" "${PROJECT_ROOT}/lmms-eval/lmms_eval/tasks" <<'PY'
from pathlib import Path
import sys

import yaml


def flatten_tasks(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_tasks(item))
        return out
    if isinstance(value, dict):
        return flatten_tasks(value.get("task", []))
    return []


group_name = sys.argv[1]
tasks_root = Path(sys.argv[2])

for path in sorted(tasks_root.rglob("*.yaml")):
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        continue
    if not isinstance(data, dict) or data.get("group") != group_name:
        continue
    leaf_tasks = flatten_tasks(data.get("task", []))
    if leaf_tasks:
        print("\n".join(leaf_tasks))
        break
PY
}

expand_eval_task() {
  local task_name="$1"
  mapfile -t group_tasks < <(resolve_group_tasks "${task_name}")
  if ((${#group_tasks[@]} > 1)); then
    echo "[eval] EXPAND_TASK=${task_name} LEAF_TASKS=${#group_tasks[@]} BUDGET_PER_LEAF=${TASK_BUDGET}"
    local idx
    for idx in "${!group_tasks[@]}"; do
      echo "[eval]   LEAF[$idx]=${group_tasks[$idx]} LIMIT=${TASK_BUDGET}"
      EXPANDED_TASKS+=("${group_tasks[$idx]}")
      EXPANDED_LIMITS+=("${TASK_BUDGET}")
    done
    return
  fi

  echo "[eval] EXPAND_TASK=${task_name} LEAF_TASKS=1 BUDGET_PER_LEAF=${TASK_BUDGET}"
  EXPANDED_TASKS+=("${task_name}")
  EXPANDED_LIMITS+=("${TASK_BUDGET}")
}

for raw_task in "${RAW_TASK_LIST[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  [[ -n "${task}" ]] || continue
  expand_eval_task "${task}"
done

echo "[eval] TOTAL_EXPANDED_TASKS=${#EXPANDED_TASKS[@]}"

for idx in "${!EXPANDED_TASKS[@]}"; do
  CURRENT_TASK="${EXPANDED_TASKS[$idx]}"
  CURRENT_LIMIT="${EXPANDED_LIMITS[$idx]}"
  CURRENT_SUFFIX="${LOG_SUFFIX}-${CURRENT_TASK}"
  echo "[eval] RUN_TASK=${CURRENT_TASK} LIMIT=${CURRENT_LIMIT} LOG_SUFFIX=${CURRENT_SUFFIX}"

  "${ACCELERATE_BIN}" launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port "${MAIN_PROCESS_PORT}" \
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
