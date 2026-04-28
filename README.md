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
- `scripts/train_eval_loop.sh`：分段训练并评测最新 checkpoint
- `lmms-eval/lmms_eval/models/simple/fovea.py`：本地 Fovea 的 lmms-eval adapter

## 环境准备

先安装 PyTorch，并按机器 CUDA 版本选择 wheel，例如 CUDA 12.8：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

再安装运行本仓库训练 / 评测所需依赖：

```bash
pip install transformers huggingface_hub tokenizers numpy pillow accelerate deepspeed peft datasets tensorboard
pip install -e ./lmms-eval
```

如需额外加速，可自行安装与本机环境匹配的 `flash-attn` wheel；当前默认 attention 后端是 `sdpa`，不依赖 `flash_attention_2`。

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
- `FT3_IMG_SLOT_NUM_EXPERTS`
- `FT3_IMG_SLOT_SLOTS_PER_EXPERT`
- `FT3_IMG_SLOT_GATE_TEMP`
- `FT3_IMG_SLOT_ROUTE_TEMP`
- `FT3_IMG_SLOT_AUX_LOSS_COEF`
- `FT3_IMG_SLOT_GATE_SPARSITY_COEF`
- `FT3_IMG_SLOT_EXPERT_BALANCE_COEF`
- `FT3_IMG_SLOT_SLOT_BALANCE_COEF`
- `FT3_IMG_SLOT_ROUTE_ENTROPY_COEF`

其中 `FT3_STOP_STEP` 由 `src/qwen35_hf/train/sft.py` 里的 `StopAtStepCallback` 读取，用于分段训练。

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

当前默认配置定义在 `src/qwen35_hf/configuration_fovea.py`：

- `img_slot_enable=True`
- `img_slot_m=8`
- `img_slot_k=128`
- `img_slot_delta=129`
- `img_slot_beta=0.3`
- `img_slot_lambda=0.9`
- `img_slot_max_text_tokens=512`
- `img_slot_tile_size=1024`
- `img_slot_num_experts=8`
- `img_slot_slots_per_expert=16`
- `img_slot_gate_temperature=1.0`
- `img_slot_route_temperature=1.0`
- `img_slot_aux_loss_coef=0.01`
- `img_slot_gate_sparsity_coef=1.0`
- `img_slot_expert_balance_coef=1.0`
- `img_slot_slot_balance_coef=1.0`
- `img_slot_route_entropy_coef=0.1`
- `img_slot_use_entmax=False`
- `img_slot_enable_hardening=False`
- `img_slot_hardening_schedule="none"`
- `img_slot_min_temperature=0.25`

数据侧：

- `VisionPacker.pack()` 先对整图应用 `max_image_tokens` 预算
- 若启用 ImgSlot，再按 `ceil(width / tile_size)` 和 `ceil(height / tile_size)` 均匀切 block
- 每个 block 独立做 normal patch packing，得到一行 `image_grid_thw`
- 文本侧会先插入一个样本级 anchor span，长度是 `m`
- 每个 block 的文本视觉 span 长度固定为 `k`
- 因此整图文本侧视觉预算是 `m + num_blocks * k`

模型侧：

- `FoveaForConditionalGeneration` 在 prefill 时会绕过普通 Qwen 多模态 scatter 路径
- 先从 vision tower 得到 `visual_pools`
- 再用共享 ImgSlot attention + router，把每个 block 的变长视觉 token 软聚合成固定 `k` 个 slot
- 然后把 anchor span 与压缩后的 slot span 直接写回 `inputs_embeds`
- 若 text context 为空，ImgSlot 会退化为只用 anchor seed + visual routing，而不是报错
- decode 时会保留 detach 后的 ImgSlot runtime state，并每隔 `img_slot_delta` 步刷新一次 full-attention 层里的相关 KV cache
- beam / group beam generation 下，首轮 prefill 会由本层接管 packed `pixel_values` / `image_grid_thw` / `mm_token_type_ids` / `image_block_counts`，并按 sample 语义扩展与重排 runtime state 和 `rope_deltas`

辅助损失会通过 `Trainer` 日志暴露为：

- `imgslot/aux_loss`
- `imgslot/gate_logit_mean`
- `imgslot/gate_logit_std`
- `imgslot/expert_balance`
- `imgslot/slot_balance`
- `imgslot/dispatch_entropy`
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
- 通过 `accelerate launch -m lmms_eval` 启动
- 使用 `--model fovea`
- 默认任务是 `xlrs-lite`
- 默认向 adapter 传 `enable_thinking=False,max_image_tokens=8196`

可覆盖的环境变量：

- `EVAL_BASE_MODEL`
- `EVAL_LORA_CHECKPOINT`
- `EVAL_TASKS`
- `EVAL_OUTPUT_PATH`
- `EVAL_LOG_SUFFIX`

`lmms-eval/lmms_eval/models/simple/fovea.py` 当前行为：

- 加载 `FoveaForConditionalGeneration`
- 可选加载 PEFT LoRA adapter
- 会尝试从 Deepspeed checkpoint 恢复 `model.visual*` trainables
- 复用训练侧 `VisionPacker`
- ImgSlot 开启时，processor 不返回 `mm_token_type_ids`
- processor 内部仍用扁平 block 计数展开文本侧 placeholder，但对 generation 额外输出按 sample 分组的 `image_block_counts`
- 当前只支持 image 输入，不支持 video

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
- 每次训练推进到下一个 `STEP_INTERVAL`
- 通过 `FT3_STOP_STEP` 请求训练在目标 step 干净停止并保存
- 随后评测最新 checkpoint
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
