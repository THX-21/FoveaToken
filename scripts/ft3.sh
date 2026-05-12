#!/bin/bash
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
# export DS_IGNORE_CUDA_DETECTION=1
# export DS_SKIP_CUDA_CHECK=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export NNODES=1
export NUM_GPUS=1
export MASTER_ADDR="127.0.0.1"
if [[ -v FT3_MASTER_PORT ]]; then
    export MASTER_PORT="${FT3_MASTER_PORT}"
else
    export MASTER_PORT=29599
fi
export WORLD_SIZE=$((NNODES * NUM_GPUS))
export RANK=0

NUM_TRAIN_EPOCHS="${FT3_NUM_TRAIN_EPOCHS:-3}"
RUN_NAME="${FT3_RUN_NAME:-fovea-ft3-imgslot}"
JSON_PATH="${FT3_JSON_PATH:-${PROJECT_ROOT}/data/ft3_whole_shuffle.json}"
# JSON_PATH="${FT3_JSON_PATH:-${PROJECT_ROOT}/data/debug/ft3_step144_nearby.json}"
IMAGE_FOLDER="${FT3_IMAGE_FOLDER:-${PROJECT_ROOT}/data/jpg_images}"
CKPT_PATH="${FT3_CKPT_PATH:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${FT3_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${RUN_NAME}}"
SAVE_STEPS="${FT3_SAVE_STEPS:-200}"
MAX_STEPS="${FT3_MAX_STEPS:--1}"
ATTN_IMPLEMENTATION="${FT3_ATTN_IMPLEMENTATION:-sdpa}"

echo "[ft3] NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[ft3] RUN_NAME=${RUN_NAME}"
echo "[ft3] JSON_PATH=${JSON_PATH}"
echo "[ft3] IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "[ft3] CKPT_PATH=${CKPT_PATH}"
echo "[ft3] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ft3] SAVE_STEPS=${SAVE_STEPS}"
echo "[ft3] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"

export PYTHONPATH="${PROJECT_ROOT}/src"

if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)
PY
then
    PRECISION_ARGS=(--bf16 true --fp16 false --tf32 true)
elif python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
    PRECISION_ARGS=(--bf16 false --fp16 true --tf32 true)
else
    PRECISION_ARGS=(--bf16 false --fp16 false --tf32 false)
fi
# PRECISION_ARGS=(--bf16 false --fp16 false --tf32 true)

LAUNCHER=(
    torchrun
    --nproc_per_node="${NUM_GPUS}"
    --nnodes="${NNODES}"
    --node_rank="${RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    -m qwen35_hf.train.sft
)

RESUME_ARGS=()
LATEST_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1)"
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    echo "Resuming from checkpoint: ${LATEST_CHECKPOINT}"
    RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
fi

ACCELERATE_CPU_AFFINITY=1 "${LAUNCHER[@]}" \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_tp2.json" \
    --model_name_or_path "${CKPT_PATH}" \
    --data_path "${JSON_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --max_image_tokens 8196 \
    --lora_enable true \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --unfreeze_vision true \
    "${PRECISION_ARGS[@]}" \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 2 \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --remove_unused_columns false \
    --logging_nan_inf_filter false \
    "${RESUME_ARGS[@]}"
