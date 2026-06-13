#!/bin/bash
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

export NCCL_P2P_DISABLE=1   # 分布式训练时禁用P2P通信，避免当前环境下的通信问题
export NCCL_CUMEM_ENABLE=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

NUM_TRAIN_EPOCHS="${FT3_NUM_TRAIN_EPOCHS:-1}"
RUN_NAME="${FT3_RUN_NAME:-fovea-fixed-token-lora}"
DATA_PATH="${FT3_DATA_PATH:-${PROJECT_ROOT}/data/vgr/preprocessed}"
IMAGE_FOLDER="${FT3_IMAGE_FOLDER:-${PROJECT_ROOT}/data/vgr/llava_next_raw_format}"
CKPT_PATH="${FT3_CKPT_PATH:-Qwen/Qwen3.5-4B}"
OUTPUT_DIR="${FT3_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${RUN_NAME}}"
SAVE_STEPS="${FT3_SAVE_STEPS:-200}"
MAX_STEPS="${FT3_MAX_STEPS:--1}"
ATTN_IMPLEMENTATION="${FT3_ATTN_IMPLEMENTATION:-sdpa}"
REPORT_TO="${FT3_REPORT_TO:-none}"
WORKERS="${FT3_DATALOADER_NUM_WORKERS:-4}"
UNFREEZE_VISION="${FT3_UNFREEZE_VISION:-true}"
FREEZE_EMBED_BASE="${FT3_FREEZE_EMBED_BASE:-true}"
MAX_IMG_TOKENS="${FT3_MAX_IMG_TOKENS:-2048}"

echo "[ft3_lora] NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[ft3_lora] RUN_NAME=${RUN_NAME}"
echo "[ft3_lora] DATA_PATH=${DATA_PATH}"
echo "[ft3_lora] IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "[ft3_lora] CKPT_PATH=${CKPT_PATH}"
echo "[ft3_lora] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ft3_lora] SAVE_STEPS=${SAVE_STEPS}"
echo "[ft3_lora] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[ft3_lora] DATALOADER_NUM_WORKERS=${WORKERS}"
echo "[ft3_lora] UNFREEZE_VISION=${UNFREEZE_VISION}"
echo "[ft3_lora] FREEZE_EMBED_BASE=${FREEZE_EMBED_BASE}"
echo "[ft3_lora] MAX_IMG_TOKENS=${MAX_IMG_TOKENS}"

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

RESUME_ARGS=()
LATEST_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1)"
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    echo "[ft3_lora] RESUME_CHECKPOINT=${LATEST_CHECKPOINT}"
    RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
fi

"${PYTHON_BIN}" -u -m fovea_token.train.sft \
    --model_name_or_path "${CKPT_PATH}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --max_img_tokens "${MAX_IMG_TOKENS}" \
    --lora_enable true \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --unfreeze_vision "${UNFREEZE_VISION}" \
    --freeze_embed_base "${FREEZE_EMBED_BASE}" \
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
    --max_grad_norm 10.0 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing true \
    --dataloader_num_workers "${WORKERS}" \
    --report_to "${REPORT_TO}" \
    --remove_unused_columns false \
    --logging_nan_inf_filter false \
    "${RESUME_ARGS[@]}"
