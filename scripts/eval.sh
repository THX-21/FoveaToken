#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
export HF_HUB_OFFLINE=1

BASE_MODEL="${EVAL_BASE_MODEL:-Qwen/Qwen3.5-9B}"
LORA_CHECKPOINT="${EVAL_LORA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/fovea-ft3-imgslot/checkpoint-10015}"
TASKS="${EVAL_TASKS:-xlrs-lite}"
OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/logs}"
LOG_SUFFIX="${EVAL_LOG_SUFFIX:-fovea_xlrs_lite_ckpt10015}"

echo "[eval] BASE_MODEL=${BASE_MODEL}"
echo "[eval] LORA_CHECKPOINT=${LORA_CHECKPOINT}"
echo "[eval] TASKS=${TASKS}"
echo "[eval] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[eval] LOG_SUFFIX=${LOG_SUFFIX}"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 12345 \
  -m lmms_eval \
  --model fovea \
  --model_args "pretrained=${BASE_MODEL},peft=${LORA_CHECKPOINT},device_map=auto,attn_implementation=sdpa,enable_thinking=False,max_image_tokens=8196" \
  --tasks "${TASKS}" \
  --batch_size 4 \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}"
