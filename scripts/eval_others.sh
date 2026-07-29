#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
# export HF_HUB_OFFLINE=1
# MathVista tasks use the OpenAI-backed answer extractor / judge in lmms-eval.
ENV_FILE="${PROJECT_ROOT}/.env"
[[ -f "${ENV_FILE}" ]] || { echo "[eval_others] missing ${ENV_FILE}" >&2; exit 1; }
set -a
source "${ENV_FILE}"
set +a

BASE_MODEL="${EVAL_BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL="${EVAL_MODEL:-qwen2_5_vl}"
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
TASKS="${EVAL_TASKS:-hrbench8k,xlrs-lite,textvqa_val,chartqa,vstar_bench,mmstar,mathvista_testmini_solution}"
# TASKS="${EVAL_TASKS:-mathvista_testmini_solution}"
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
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
TASK_BUDGET="${EVAL_TASK_BUDGET:-600}"
EVAL_SUBSET_SEED="${EVAL_SUBSET_SEED:-42}"

MODEL_ARGS_DEFAULT="pretrained=${BASE_MODEL},device=${DEVICE},device_map=${DEVICE_MAP},attn_implementation=${ATTN_IMPLEMENTATION}"
MODEL_ARGS="${EVAL_MODEL_ARGS:-${MODEL_ARGS_DEFAULT}}"
LOG_SUFFIX_DEFAULT="$(basename "${BASE_MODEL}")"
LOG_SUFFIX="${EVAL_LOG_SUFFIX:-${LOG_SUFFIX_DEFAULT}}"
ACCELERATE_BIN="$(command -v accelerate || true)"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PROJECT_ROOT}/.venv/bin/accelerate}"
[[ -x "${ACCELERATE_BIN}" ]] || { echo "[eval_others] accelerate not found: ${ACCELERATE_BIN}" >&2; exit 127; }

echo "[eval_others] MODEL=${MODEL}"
echo "[eval_others] BASE_MODEL=${BASE_MODEL}"
echo "[eval_others] TASKS=${TASKS}"
echo "[eval_others] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[eval_others] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[eval_others] DEVICE=${DEVICE}"
echo "[eval_others] DEVICE_MAP=${DEVICE_MAP}"
echo "[eval_others] BATCH_SIZE=${BATCH_SIZE}"
echo "[eval_others] NUM_PROCESSES=${NUM_PROCESSES}"
echo "[eval_others] MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "[eval_others] BALANCED_LIMIT=${BALANCED_LIMIT}"
echo "[eval_others] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[eval_others] TASK_BUDGET=${TASK_BUDGET}"
echo "[eval_others] EVAL_SUBSET_SEED=${EVAL_SUBSET_SEED}"
echo "[eval_others] LOG_SUFFIX=${LOG_SUFFIX}"
echo "[eval_others] ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "[eval_others] PYTHON_BIN=${PYTHON_BIN}"
echo "[eval_others] MODEL_ARGS=${MODEL_ARGS}"

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
    echo "[eval_others] EXPAND_TASK=${task_name} LEAF_TASKS=${#group_tasks[@]} BUDGET_PER_LEAF=${TASK_BUDGET}"
    local idx
    for idx in "${!group_tasks[@]}"; do
      echo "[eval_others]   LEAF[$idx]=${group_tasks[$idx]} LIMIT=${TASK_BUDGET}"
      EXPANDED_TASKS+=("${group_tasks[$idx]}")
      EXPANDED_LIMITS+=("${TASK_BUDGET}")
    done
    return
  fi

  echo "[eval_others] EXPAND_TASK=${task_name} LEAF_TASKS=1 BUDGET_PER_LEAF=${TASK_BUDGET}"
  EXPANDED_TASKS+=("${task_name}")
  EXPANDED_LIMITS+=("${TASK_BUDGET}")
}

for raw_task in "${RAW_TASK_LIST[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  [[ -n "${task}" ]] || continue
  expand_eval_task "${task}"
done

echo "[eval_others] TOTAL_EXPANDED_TASKS=${#EXPANDED_TASKS[@]}"

for idx in "${!EXPANDED_TASKS[@]}"; do
  CURRENT_TASK="${EXPANDED_TASKS[$idx]}"
  CURRENT_LIMIT="${EXPANDED_LIMITS[$idx]}"
  CURRENT_SUFFIX="${LOG_SUFFIX}-${CURRENT_TASK}"
  echo "[eval_others] RUN_TASK=${CURRENT_TASK} LIMIT=${CURRENT_LIMIT} LOG_SUFFIX=${CURRENT_SUFFIX}"

  "${ACCELERATE_BIN}" launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${CURRENT_TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS}" \
    --log_samples \
    --log_samples_suffix "${CURRENT_SUFFIX}" \
    --output_path "${OUTPUT_PATH}" \
    --limit "${CURRENT_LIMIT}"
done
