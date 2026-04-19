#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
export HF_HUB_OFFLINE=1

BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LORA_CHECKPOINT="${PROJECT_ROOT}/checkpoints/fovea-ft3-imgslot/checkpoint-2000"
TASKS="xlrs-lite"
OUTPUT_PATH="${PROJECT_ROOT}/logs"
LOG_SUFFIX="fovea_xlrs_lite_ckpt2000"

if [[ -v EVAL_BASE_MODEL ]]; then
  BASE_MODEL="${EVAL_BASE_MODEL}"
fi
if [[ -v EVAL_LORA_CHECKPOINT ]]; then
  LORA_CHECKPOINT="${EVAL_LORA_CHECKPOINT}"
fi
if [[ -v EVAL_TASKS ]]; then
  TASKS="${EVAL_TASKS}"
fi
if [[ -v EVAL_OUTPUT_PATH ]]; then
  OUTPUT_PATH="${EVAL_OUTPUT_PATH}"
fi
if [[ -v EVAL_LOG_SUFFIX ]]; then
  LOG_SUFFIX="${EVAL_LOG_SUFFIX}"
fi

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 12345 \
  -m lmms_eval \
  --model fovea \
  --model_args "pretrained=${BASE_MODEL},peft=${LORA_CHECKPOINT},device_map=auto,attn_implementation=flash_attention_2,enable_thinking=False,max_image_tokens=8196" \
  --tasks "${TASKS}" \
  --batch_size 4 \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" 
