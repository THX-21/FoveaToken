# qwen35-hf

`qwen35-hf` 是一个本地 Fovea / Qwen3.5 多模态实验仓库。当前主模型是 `FoveaForConditionalGeneration`，核心实验机制是 `ImgSlot`：先把图像按 block 切分，再把每个 block 的变长视觉 token 软聚合为固定数量的 slot，最后把这些压缩后的视觉表示写回文本侧 placeholder span。

当前仓库的主要使用方式不是发布成标准 Python 包，而是通过 `PYTHONPATH` 直接运行 `src/` 与本地 `lmms-eval/` 代码。

## 当前代码结构

- `src/qwen35_hf/configuration_fovea.py`：`FoveaConfig` / `FoveaTextConfig` / `FoveaVisionConfig`
- `src/qwen35_hf/modeling_fovea.py`：`FoveaForConditionalGeneration` 与 ImgSlot runtime / cache refresh
- `src/qwen35_hf/processing_fovea.py`：`FoveaProcessor` 与视觉 placeholder 展开
- `src/qwen35_hf/tokenization_fovea.py`：`FoveaTokenizer`
- `src/qwen35_hf/train/sft.py`：SFT 训练入口、LoRA、冻结策略、训练 callback
- `src/qwen35_hf/train/data.py`：SFT JSON 数据集、ChatML 编码、VisionPacker、collator
- `src/qwen35_hf/train/image_packing.py`：本地图像 resize / normalize / patch packing / block split
- `scripts/ft3.sh`：训练脚本
- `scripts/eval.sh`：Fovea LoRA 评测脚本
- `scripts/eval_qwen35.sh`：Qwen3.5 baseline 评测脚本
- `scripts/train_eval_loop.sh`：每隔固定 step 分段训练并评测最新 checkpoint，直到训练自然结束
- `lmms-eval/lmms_eval/models/simple/fovea.py`：本地 Fovea 的 lmms-eval adapter

## 环境准备

先安装 PyTorch，并按机器 CUDA 版本选择 wheel，例如 CUDA 12.8：

```bash
pip install --upgrade pip setuptools wheel
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

如果你直接用仓库里的 `requirements.txt`，其中 CUDA 相关版本当前对应：

- `torch==2.7.1+cu128`
- `torchvision==0.22.1+cu128`
- `torchaudio==2.7.1+cu128`
- `nvidia-cublas-cu12==12.8.4.1`
- `nvidia-cuda-cupti-cu12==12.8.90`
- `nvidia-cuda-nvrtc-cu12==12.8.93`
- `nvidia-cuda-runtime-cu12==12.8.90`
- `nvidia-cudnn-cu12==9.8.0.87`
- `nvidia-cufft-cu12==11.3.3.83`
- `nvidia-curand-cu12==10.3.9.90`
- `nvidia-cusolver-cu12==11.7.3.90`
- `nvidia-cusparse-cu12==12.5.8.93`
- `nvidia-nccl-cu12==2.26.2`
- `nvidia-nvtx-cu12==12.8.90`

再安装运行本仓库训练 / 评测所需依赖：

```bash
pip install \
  accelerate==1.7.0 \
  datasets==3.6.0 \
  deepspeed==0.16.7 \
  transformers==4.53.0 \
  tokenizers==0.21.1 \
  huggingface-hub==0.31.2 \
  peft==0.19.1 \
  safetensors==0.7.0 \
  sentencepiece==0.2.1 \
  einops==0.8.2 \
  numpy==1.26.4 \
  pillow==12.1.1 \
  tensorboard==2.20.0
pip install flash-attn --no-build-isolation
pip install -e ./lmms-eval
```

当前默认 attention 后端已切回 `sdpa`；若要尝试 `flash_attention_2`，可通过环境变量改掉。

运行前通常需要：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

## 训练

默认训练入口：

```bash
bash scripts/ft3.sh
```

`scripts/ft3.sh` 当前行为：

- 通过 `torchrun -m qwen35_hf.train.sft` 启动训练
- 默认单节点单卡：`NNODES=1`、`NUM_GPUS=1`
- 使用 `scripts/zero2_tp2_gpu.json` 的 DeepSpeed 配置
- 自动检测 CUDA / bf16 能力，选择 `bf16`、`fp16` 或 `fp32`
- 自动从 `OUTPUT_DIR` 下最新 `checkpoint-*` 续训
- 默认启用 LoRA、ImgSlot、vision tower 解冻、gradient checkpointing
- 固定 `--max_image_tokens 8196`
- 默认使用 `--attn_implementation sdpa`

可覆盖的环境变量：

- `FT3_MASTER_PORT`
- `FT3_NUM_TRAIN_EPOCHS`
- `FT3_RUN_NAME`
- `FT3_JSON_PATH`
- `FT3_IMAGE_FOLDER`
- `FT3_CKPT_PATH`
- `FT3_OUTPUT_DIR`
- `FT3_SAVE_STEPS`
- `FT3_MAX_STEPS`
- `FT3_STOP_STEP`
- `FT3_ATTN_IMPLEMENTATION`
- `FT3_IMG_SLOT_NUM_EXPERTS`
- `FT3_IMG_SLOT_SLOTS_PER_EXPERT`
- `FT3_IMG_SLOT_GATE_TEMP`
- `FT3_IMG_SLOT_ROUTE_TEMP`
- `FT3_IMG_SLOT_AUX_LOSS_COEF`
- `FT3_IMG_SLOT_GATE_SPARSITY_COEF`
- `FT3_IMG_SLOT_EXPERT_BALANCE_COEF`
- `FT3_IMG_SLOT_SLOT_BALANCE_COEF`
- `FT3_IMG_SLOT_ROUTE_ENTROPY_COEF`

其中 `FT3_STOP_STEP` 由 `src/qwen35_hf/train/sft.py` 里的 `StopAtStepCallback` 读取，用于在不改动 epoch 训练计划的前提下按目标 step 分段停训。
`FT3_MAX_STEPS` 仅在你显式设置总 step 上限时生效；`scripts/train_eval_loop.sh` 默认通过 `FT3_NUM_TRAIN_EPOCHS` 控制训练自然结束。

## 训练数据格式

训练集由 `LazySupervisedDataset` 懒加载，JSON 顶层必须是 list。每条记录至少包含：

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this image."},
    {"from": "gpt", "value": "..."}
  ],
  "image": "xxx.jpg"
}
```

约定如下：

- `image` 可以是字符串，也可以是字符串列表
- 图像路径相对 `image_folder`
- 当一条消息里恰好只有一个 `<image>` 且不在开头时，会被移动到消息开头
- 多图样本按 `<image>` 出现顺序依次消费
- 若文本只有一个 `<image>`，但样本提供多张图，则这些图的 placeholder 会顺序拼到同一个占位处

assistant 监督模板当前固定为：

```text
<|im_start|>assistant
<think>

</think>

{answer}<|im_end|>
```

训练时仅监督 answer 内容与 `<|im_end|>`；system / user 与空 thinking scaffold 都被 mask。

## ImgSlot 机制

当前默认配置定义在 `src/qwen35_hf/configuration_fovea.py` 与 `src/qwen35_hf/train/sft.py`：

- `img_slot_enable=True`
- `img_slot_k=128`
- `img_slot_delta=128`
- `img_slot_beta=0.3`
- `img_slot_lambda=0.9`
- `img_slot_max_text_tokens=512`
- `img_slot_tile_size=1280`
- `img_slot_aux_loss_coef=0.01`

数据侧：

- `VisionPacker.pack()` 先对整图应用 `max_image_tokens` 预算
- 若启用 ImgSlot，再按 `ceil(width / tile_size)` 和 `ceil(height / tile_size)` 均匀切 block
- 每个 block 独立做 normal patch packing，得到一行 `image_grid_thw`
- 文本侧不再插入样本级 anchor span
- 每个 block 的文本视觉 span 长度固定为 `k`
- 因此整图文本侧视觉预算是 `num_blocks * k`

模型侧：

- `FoveaForConditionalGeneration` 在 prefill 时会绕过普通 Qwen 多模态 scatter 路径
- 先从 vision tower 得到 `visual_pools`
- 再用一组共享的内部 A/query token 先结合文本、再结合全局视觉池，得到 shared-A state
- 然后用 shared-A 对每个 block 的 visual pool 直接聚合出固定 `k` 个 slot
- 只把每个 block 的压缩 slot span 写回 `inputs_embeds`；A 不进入输入序列
- 若 text context 为空，ImgSlot 会退化为只用共享 A seed + visual aggregation，而不是报错
- decode 时会保留 detach 后的 ImgSlot runtime state，并每隔 `img_slot_delta` 步刷新一次 full-attention 层里的相关 KV cache
- beam / group beam generation 下，首轮 prefill 会由本层接管 packed `pixel_values` / `image_grid_thw` / `mm_token_type_ids` / `image_block_counts`，并按 sample 语义扩展与重排 runtime state 和 `rope_deltas`

辅助统计会通过 `Trainer` 日志暴露为：

- `imgslot/aux_loss`
- `imgslot/attention_entropy`
- `imgslot/num_blocks`：当前 forward 中真实参与 ImgSlot 压缩的 image block 数

## LoRA 与可训练模块

训练入口 `src/qwen35_hf/train/sft.py` 当前策略：

- 先整体冻结模型
- `--unfreeze_vision true` 时解冻 vision tower
- LoRA target 覆盖 attention / MLP / linear-attention 相关投影
- ImgSlot 模块不做 LoRA target，而是放入 `modules_to_save`
- PEFT 包装后，再显式把 `imgslot_*` 参数保持为 trainable

因此 checkpoint 的保存语义是：

- `adapter_model.safetensors`：LoRA 权重 + ImgSlot `modules_to_save`
- DeepSpeed `global_step*/mp_rank_00_model_states.pt`：vision tower 等非 LoRA trainables

## 评测

默认评测入口：

```bash
bash scripts/eval.sh
```

`scripts/eval.sh` 当前行为：

- 设置 `PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/lmms-eval"`
- 默认 `HF_HUB_OFFLINE=1`
- 通过 `accelerate launch -m lmms_eval` 启动；PATH 中找不到 `accelerate` 时自动使用仓库 `.venv/bin/accelerate`
- 使用 `--model fovea`
- 默认 `device` 与 `device_map` 绑定到同一张卡
- 默认 `device_map=cuda:0`
- 默认 `--batch_size 1`
- 默认任务是 `xlrs-lite`
- 默认向 adapter 传 `enable_thinking=False,max_image_tokens=8196`

可覆盖的环境变量：

- `EVAL_BASE_MODEL`
- `EVAL_LORA_CHECKPOINT`
- `EVAL_TASKS`
- `EVAL_OUTPUT_PATH`
- `EVAL_DEVICE`
- `EVAL_DEVICE_MAP`
- `EVAL_BATCH_SIZE`
- `EVAL_LOG_SUFFIX`
- `EVAL_ATTN_IMPLEMENTATION`

`lmms-eval/lmms_eval/models/simple/fovea.py` 当前行为：

- 加载 `FoveaForConditionalGeneration`
- 可选加载 PEFT LoRA adapter
- 会尝试从 Deepspeed checkpoint 恢复 `model.visual*` trainables
- 复用训练侧 `VisionPacker`
- ImgSlot 开启时，processor 不返回 `mm_token_type_ids`
- processor 内部仍用扁平 block 计数展开文本侧 placeholder，但对 generation 额外输出按 sample 分组的 `image_block_counts`
- 当前只支持 image 输入，不支持 video
- `xlrs-lite` 任务当前按数据集全量样本评测，不再在 task 侧按 category 截断

## Qwen3.5 baseline 评测

```bash
bash scripts/eval_qwen35.sh
```

该脚本调用 `lmms-eval` 的 `qwen3_5` adapter，默认：

- `BASE_MODEL="Qwen/Qwen3.5-9B"`
- `TASKS="xlrs-lite"`
- `--limit 200`
- `--force_simple`

## 分段训练 + 评测循环

```bash
bash scripts/train_eval_loop.sh
```

该脚本会：

- 从 `checkpoints/<run_name>/checkpoint-*` 中找当前最新 step
- 每次训练推进到下一个 `STEP_INTERVAL`（默认 200）
- 通过 `FT3_STOP_STEP` 请求训练在目标 step 干净停止并保存
- 每段训练后立即评测最新 checkpoint
- 训练默认按 `FT3_NUM_TRAIN_EPOCHS`（默认 3）自然结束；最后一段不足一个 `STEP_INTERVAL` 时也会正常完成并评测
- 训练或评测失败时，保留原始退出码退出

## 结果汇总

```bash
python scripts/xlrs_eval_report.py <jsonl-or-log-dir>
```

例如：

```bash
python scripts/xlrs_eval_report.py logs/fovea_xlrs_lite_ckpt2000/*_samples_xlrs-lite.jsonl
python scripts/xlrs_eval_report.py logs --latest-only --table-only
```

该脚本会从 `*_samples_xlrs-lite.jsonl` 中读取 `xlrs_micro_score`，按 `lmms-eval` 的 XLRS 子任务口径统计并输出 Markdown 表格。
