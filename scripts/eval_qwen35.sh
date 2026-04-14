#!/bin/bash
# set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"

BASE_MODEL="Qwen/Qwen3.5-9B"
TASKS="xlrs-lite"
OUTPUT_PATH="${PROJECT_ROOT}/logs"
LOG_SUFFIX="qwen35_xlrs_lite"

accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
  --model qwen3_5 \
  --model_args "pretrained=${BASE_MODEL},device_map=auto,attn_implementation=flash_attention_2,enable_thinking=False,max_pixels=2097152" \
  --tasks "${TASKS}" \
  --batch_size 4 \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" \
  --force_simple
