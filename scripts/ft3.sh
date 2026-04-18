#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"

export NNODES=1
export NUM_GPUS=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29599
export WORLD_SIZE=$((NNODES * NUM_GPUS))
export RANK=0

NUM_TRAIN_EPOCHS=1
RUN_NAME="Qwen3.5-ft3-imgslot"
JSON_PATH="/mnt/data/GeoLLaVA-Data/ft3_whole_shuffle.json"
IMAGE_FOLDER="/mnt/data/GeoLLaVA-Data/jpg_images"
CKPT_PATH="Qwen/Qwen3.5-9B"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
IMG_SLOT_TILE_SIZE=1024

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
    --img_slot_enable true \
    --img_slot_m 8 \
    --img_slot_k 128 \
    --img_slot_delta 128 \
    --img_slot_beta 0.3 \
    --img_slot_lambda 0.5 \
    --img_slot_tile_size "${IMG_SLOT_TILE_SIZE}" \
    "${PRECISION_ARGS[@]}" \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --remove_unused_columns false \
    --logging_nan_inf_filter false \
    "${RESUME_ARGS[@]}"
