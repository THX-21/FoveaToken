#!/bin/bash
# set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/Qwen3.5-ft3-anyres24/checkpoint-1000}"
  # --model_args "pretrained=${BASE_MODEL},peft=${LORA_CHECKPOINT},device_map=auto,attn_implementation=flash_attention_2,enable_thinking=False,processor_backend=local,image_aspect_ratio=anyres,max_image_tokens=128" \
TASKS="${TASKS:-xlrs-lite}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/logs}"
LOG_SUFFIX="${LOG_SUFFIX:-qwen35_xlrs_lite_ckpt400}"
IMG_SLOT_TILE_SIZE="${IMG_SLOT_TILE_SIZE:?IMG_SLOT_TILE_SIZE must be set when ImgSlot is enabled}"

accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
  --model qwen35_hf \
  --model_args "pretrained=${BASE_MODEL},peft=${LORA_CHECKPOINT},device_map=auto,attn_implementation=flash_attention_2,enable_thinking=False,processor_backend=local,image_aspect_ratio=normal,max_image_tokens=2048,img_slot_enable=True,img_slot_m=4,img_slot_k=64,img_slot_delta=8,img_slot_beta=0.3,img_slot_lambda=0.9,img_slot_tile_size=${IMG_SLOT_TILE_SIZE}" \
  --tasks "${TASKS}" \
  --batch_size 4 \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" \
  --limit 500
