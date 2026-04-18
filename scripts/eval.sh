#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
export HF_HUB_OFFLINE=1

BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LORA_CHECKPOINT="${PROJECT_ROOT}/checkpoints/Qwen3.5-ft3-imgslot/checkpoint-4000"
TASKS="xlrs-lite"
OUTPUT_PATH="${PROJECT_ROOT}/logs"
LOG_SUFFIX="qwen35_xlrs_lite_ckpt4000"
IMG_SLOT_TILE_SIZE=1024

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 12345 \
  -m lmms_eval \
  --model qwen35_hf \
  --model_args "pretrained=${BASE_MODEL},peft=${LORA_CHECKPOINT},device_map=auto,attn_implementation=flash_attention_2,enable_thinking=False,max_image_tokens=8196,img_slot_enable=True,img_slot_m=8,img_slot_k=128,img_slot_delta=128,img_slot_beta=0.3,img_slot_lambda=0.9,img_slot_tile_size=${IMG_SLOT_TILE_SIZE}" \
  --tasks "${TASKS}" \
  --batch_size 4 \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" 
