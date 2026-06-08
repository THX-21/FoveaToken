#!/bin/bash
# set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUN_NAME="${FT3_RUN_NAME:-fovea-fixed-token}"
OUTPUT_DIR="${FT3_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${RUN_NAME}}"
STEP_INTERVAL="${TRAIN_EVAL_STEP_INTERVAL:-200}"
SAVE_INTERVAL="${FT3_SAVE_STEPS:-200}"
NUM_TRAIN_EPOCHS="${FT3_NUM_TRAIN_EPOCHS:-3}"
LOG_PREFIX="${TRAIN_EVAL_LOG_PREFIX:-fovea_xlrs_lite}"
DATA_PATH="${FT3_DATA_PATH:-${PROJECT_ROOT}/data/vgr/preprocessed}"
IMAGE_FOLDER="${FT3_IMAGE_FOLDER:-${PROJECT_ROOT}/data/vgr/llava_next_raw_format}"
CKPT_PATH="${FT3_CKPT_PATH:-Qwen/Qwen3.5-4B}"
BASE_MODEL="${EVAL_BASE_MODEL:-Qwen/Qwen3.5-4B}"
TASKS="${EVAL_TASKS:-xlrs-lite}"
OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/logs}"

latest_checkpoint_step() {
    find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | sed -E 's/.*checkpoint-([0-9]+)$/\1/' \
        | sort -n \
        | tail -n 1
}

abort_script() {
    local code="$1"
    shift
    if [[ $# -gt 0 ]]; then
        echo "$*" >&2
    fi
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return "${code}"
    fi
    exit "${code}"
}

exit_on_failure() {
    local code="$1"
    shift
    if [[ $# -gt 0 ]]; then
        echo "$*" >&2
    fi
    exit "${code}"
}

while true; do
    current_step="$(latest_checkpoint_step)"
    if [[ -z "${current_step}" ]]; then
        current_step=0
    fi

    target_step=$(( ((current_step / STEP_INTERVAL) + 1) * STEP_INTERVAL ))

    echo "Training until global step ${target_step} or epoch ${NUM_TRAIN_EPOCHS} end..."
    FT3_RUN_NAME="${RUN_NAME}" \
    FT3_OUTPUT_DIR="${OUTPUT_DIR}" \
    FT3_DATA_PATH="${DATA_PATH}" \
    FT3_IMAGE_FOLDER="${IMAGE_FOLDER}" \
    FT3_CKPT_PATH="${CKPT_PATH}" \
    FT3_SAVE_STEPS="${SAVE_INTERVAL}" \
    FT3_NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
    FT3_MAX_STEPS="-1" \
    FT3_STOP_STEP="${target_step}" \
    bash "${SCRIPT_DIR}/ft3.sh"
    status="$?"
    if (( status != 0 )); then
        exit_on_failure "${status}" "Training failed before reaching step ${target_step}."
    fi

    trained_step="$(latest_checkpoint_step)"
    if [[ -z "${trained_step}" ]]; then
        abort_script 1 "No checkpoint found after training."
    fi
    checkpoint="${OUTPUT_DIR}/checkpoint-${trained_step}"
    if [[ ! -d "${checkpoint}" ]]; then
        abort_script 1 "Expected checkpoint directory not found: ${checkpoint}"
    fi

    echo "Evaluating ${checkpoint}..."
    EVAL_BASE_MODEL="${BASE_MODEL}" \
    EVAL_LORA_CHECKPOINT="${checkpoint}" \
    EVAL_TASKS="${TASKS}" \
    EVAL_OUTPUT_PATH="${OUTPUT_PATH}" \
    EVAL_LOG_SUFFIX="${LOG_PREFIX}_ckpt${trained_step}" \
    bash "${SCRIPT_DIR}/eval.sh"
    status="$?"
    if (( status != 0 )); then
        exit_on_failure "${status}" "Evaluation failed for ${checkpoint}."
    fi

    if (( trained_step < target_step )); then
        echo "Training finished at checkpoint-${trained_step} before reaching target step ${target_step}."
        break
    fi
done
