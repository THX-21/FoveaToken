#!/bin/bash
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
# export DS_IGNORE_CUDA_DETECTION=1
# export DS_SKIP_CUDA_CHECK=1

export NCCL_P2P_DISABLE=1   # 分布式训练时禁用P2P通信，避免当前环境下的通信问题
export NCCL_CUMEM_ENABLE=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export NNODES=1
export NUM_GPUS="${FT3_NUM_GPUS:-1}"
export MASTER_ADDR="127.0.0.1"
if [[ -n "${FT3_MASTER_PORT:-}" ]]; then
    export MASTER_PORT="${FT3_MASTER_PORT}"
else
    export MASTER_PORT=29599
fi
export WORLD_SIZE=$((NNODES * NUM_GPUS))
export RANK=0

NUM_TRAIN_EPOCHS="${FT3_NUM_TRAIN_EPOCHS:-1}"
RUN_NAME="${FT3_RUN_NAME:-fovea-vgr-qwen}"
DATA_PATH="${FT3_DATA_PATH:-${PROJECT_ROOT}/data/vgr/preprocessed}"
IMAGE_FOLDER="${FT3_IMAGE_FOLDER:-${PROJECT_ROOT}/data/vgr/llava_next_raw_format}"
CKPT_PATH="${FT3_CKPT_PATH:-checkpoints/fovea-4B}"
OUTPUT_DIR="${FT3_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${RUN_NAME}}"
SAVE_STEPS="${FT3_SAVE_STEPS:-50}"
MAX_STEPS="${FT3_MAX_STEPS:--1}"
ATTN_IMPLEMENTATION="${FT3_ATTN_IMPLEMENTATION:-sdpa}"
MAX_IMG_TOKENS="${FT3_MAX_IMG_TOKENS:-2048}"
CROP_MAX_IMG_TOKENS="${FT3_FOVEA_CROP_MAX_IMG_TOKENS:-1024}"
REPORT_TO="${FT3_REPORT_TO:-tensorboard}"
WORKERS="${FT3_DATALOADER_NUM_WORKERS:-8}"

echo "[ft3] NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[ft3] RUN_NAME=${RUN_NAME}"
echo "[ft3] DATA_PATH=${DATA_PATH}"
echo "[ft3] IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "[ft3] CKPT_PATH=${CKPT_PATH}"
echo "[ft3] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ft3] SAVE_STEPS=${SAVE_STEPS}"
echo "[ft3] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[ft3] MAX_IMG_TOKENS=${MAX_IMG_TOKENS}"
echo "[ft3] FOVEA_CROP_MAX_IMG_TOKENS=${CROP_MAX_IMG_TOKENS}"
echo "[ft3] DATALOADER_NUM_WORKERS=${WORKERS}"
echo "[ft3] REPORT_TO=${REPORT_TO}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/lmms-eval"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi

if "${PYTHON_BIN}" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)
PY
then
    PRECISION_ARGS=(--bf16 true --fp16 false --tf32 true)
elif "${PYTHON_BIN}" - <<'PY'
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
    "${PYTHON_BIN}"
    -m torch.distributed.run
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
    echo "[ft3] Model weights and Trainer state will resume from the same checkpoint."
    RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
fi

ACCELERATE_CPU_AFFINITY=1 "${LAUNCHER[@]}" \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_tp2.json" \
    --model_name_or_path "${CKPT_PATH}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --max_img_tokens "${MAX_IMG_TOKENS}" \
    --fovea_crop_max_img_tokens "${CROP_MAX_IMG_TOKENS}" \
    "${PRECISION_ARGS[@]}" \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${FT3_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${FT3_GRAD_ACCUM:-8}" \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8096 \
    --gradient_checkpointing true \
    --dataloader_num_workers "${WORKERS}" \
    --report_to "${REPORT_TO}" \
    --remove_unused_columns false \
    --logging_nan_inf_filter false \
    "${RESUME_ARGS[@]}"
