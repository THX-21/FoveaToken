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
if [[ -n "${FT3_MASTER_PORT:-}" ]]; then
    export MASTER_PORT="${FT3_MASTER_PORT}"
else
    export MASTER_PORT=29599
fi
export WORLD_SIZE=$((NNODES * NUM_GPUS))
export RANK=0

NUM_TRAIN_EPOCHS="${FT3_NUM_TRAIN_EPOCHS:-1}"
RUN_NAME="${FT3_RUN_NAME:-fovea-visual-query-replay}"
DATA_PATH="${FT3_DATA_PATH:-${PROJECT_ROOT}/data/vgr/preprocessed}"
IMAGE_FOLDER="${FT3_IMAGE_FOLDER:-${PROJECT_ROOT}/data/vgr/llava_next_raw_format}"
CKPT_PATH="${FT3_CKPT_PATH:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${FT3_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${RUN_NAME}}"
SAVE_STEPS="${FT3_SAVE_STEPS:-100}"
MAX_STEPS="${FT3_MAX_STEPS:--1}"
ATTN_IMPLEMENTATION="${FT3_ATTN_IMPLEMENTATION:-sdpa}"
UNFREEZE_VISION="${FT3_UNFREEZE_VISION:-true}"
FREEZE_EMBED_BASE="${FT3_FREEZE_EMBED_BASE:-true}"

echo "[ft3] NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[ft3] RUN_NAME=${RUN_NAME}"
echo "[ft3] DATA_PATH=${DATA_PATH}"
echo "[ft3] IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "[ft3] CKPT_PATH=${CKPT_PATH}"
echo "[ft3] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ft3] SAVE_STEPS=${SAVE_STEPS}"
echo "[ft3] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[ft3] UNFREEZE_VISION=${UNFREEZE_VISION}"
echo "[ft3] FREEZE_EMBED_BASE=${FREEZE_EMBED_BASE}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"

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
    -m fovea_token.train.sft
)

RESUME_ARGS=()
LATEST_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1)"
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    echo "[ft3] RESUME_CHECKPOINT=${LATEST_CHECKPOINT}"
    echo "[ft3] Model weights will initialize from the resume checkpoint because lora_enable=false."
    echo "[ft3] Trainer state will resume from the same checkpoint."
    RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
fi

ACCELERATE_CPU_AFFINITY=1 "${LAUNCHER[@]}" \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_tp2.json" \
    --model_name_or_path "${CKPT_PATH}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --max_image_tokens 512 \
    --retrieve_max_image_tokens "${FT3_RETRIEVE_MAX_IMAGE_TOKENS:-4096}" \
    --lora_enable false \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --unfreeze_vision "${UNFREEZE_VISION}" \
    --freeze_embed_base "${FREEZE_EMBED_BASE}" \
    --visual_query_generated_replay_prob "${FT3_GENERATED_REPLAY_PROB:-0.5}" \
    "${PRECISION_ARGS[@]}" \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --vision_tower_lr 2e-6 \
    --max_grad_norm 50.0 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing true \
    --dataloader_num_workers "${FT3_DATALOADER_NUM_WORKERS:-8}" \
    --report_to tensorboard \
    --remove_unused_columns false \
    --logging_nan_inf_filter false \
    "${RESUME_ARGS[@]}"
