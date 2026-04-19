#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUN_NAME="fovea-ft3-imgslot"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
STEP_INTERVAL=1000
SAVE_INTERVAL=500
TRAIN_EVAL_MAX_STEPS=20030
LOG_PREFIX="fovea_xlrs_lite"
JSON_PATH="/mnt/data/GeoLLaVA-Data/ft3_whole_shuffle.json"
IMAGE_FOLDER="/mnt/data/GeoLLaVA-Data/jpg_images"
CKPT_PATH="Qwen/Qwen3.5-9B"
BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TASKS="xlrs-lite"
OUTPUT_PATH="${PROJECT_ROOT}/logs"

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

break_loop() {
    if [[ $# -gt 0 ]]; then
        echo "$*" >&2
    fi
    break
}

while true; do
    current_step="$(latest_checkpoint_step)"
    if [[ -z "${current_step}" ]]; then
        current_step=0
    fi
    if (( current_step >= TRAIN_EVAL_MAX_STEPS )); then
        echo "Training already reached checkpoint-${current_step}; target is ${TRAIN_EVAL_MAX_STEPS}."
        break
    fi

    target_step=$(( ((current_step / STEP_INTERVAL) + 1) * STEP_INTERVAL ))
    if (( target_step > TRAIN_EVAL_MAX_STEPS )); then
        target_step="${TRAIN_EVAL_MAX_STEPS}"
    fi

    echo "Training until global step ${target_step}..."
    FT3_RUN_NAME="${RUN_NAME}" \
    FT3_OUTPUT_DIR="${OUTPUT_DIR}" \
    FT3_JSON_PATH="${JSON_PATH}" \
    FT3_IMAGE_FOLDER="${IMAGE_FOLDER}" \
    FT3_CKPT_PATH="${CKPT_PATH}" \
    FT3_SAVE_STEPS="${SAVE_INTERVAL}" \
    FT3_MAX_STEPS="${TRAIN_EVAL_MAX_STEPS}" \
    FT3_STOP_STEP="${target_step}" \
    bash "${SCRIPT_DIR}/ft3.sh" || break_loop "Training failed before reaching step ${target_step}."

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
    bash "${SCRIPT_DIR}/eval.sh" || break_loop "Evaluation failed for ${checkpoint}."
done
