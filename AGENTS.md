# AGENTS.md

本文件面向 Claude Code 及其他 AI 编程助手，记录当前仓库的真实入口、数据流和协作约束。**修改代码后必须同步更新本文档**，避免后续助手继续依赖过期描述。

---

## 当前定位

当前仓库是本地 `Fovea / Qwen3.5` 多模态实验仓库。对外主入口是：

- `FoveaForConditionalGeneration`
- `FoveaConfig` / `FoveaTextConfig` / `FoveaVisionConfig`
- `FoveaProcessor`
- `FoveaTokenizer`

`Qwen3_5*` 名称仍作为兼容别名导出，但新代码优先使用 `Fovea*` 命名。

核心实验机制是 **ImgSlot**：训练与推理时，先把图像切成 block，再把每个 block 的变长视觉 token 通过共享 attention + soft routing 聚合成固定数量的 slot，最后用一个样本级 anchor span 和每个 block 的固定 slot span 替换文本里的原始视觉 placeholder 区域，从而把视觉信息映射到稳定、可控的文本侧 token 预算上。

仓库根目录当前没有 `pyproject.toml`、`setup.py` 或 `setup.cfg`。不要假设 `pip install -e .` 可用；主训练与评测路径依赖：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

---

## 常用命令

以下命令默认从仓库根目录执行。

**环境准备：**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers huggingface_hub tokenizers numpy pillow accelerate deepspeed peft datasets tensorboard
pip install -e ./lmms-eval
```

如需额外加速，可按本机 CUDA / PyTorch 环境安装兼容的 `flash-attn` wheel；当前默认 attention 后端是 `sdpa`，不是 `flash_attention_2`。

`Qwen3_5VisionAttention` 在全局 `attn_implementation=sdpa` 时会按 `cu_seqlens` 拆分 packed visual tokens，并把每个 q/k/v chunk 先 `.contiguous()` 后送入 SDPA；SDPA wrapper 返回 `[1, chunk_seq, num_heads, head_dim]`，多 chunk 沿 `chunk_seq` 维拼接。当前驱动上单个视觉 chunk 过大时 backward 可能报 CUDA `invalid argument`，当前通过 `FoveaConfig.img_slot_tile_size=768` 控制默认 chunk 长度。

**训练 LoRA + ImgSlot：**
```bash
bash scripts/ft3.sh
```

`scripts/ft3.sh` 当前行为：

- 设置 `PYTHONPATH="$PROJECT_ROOT/src"`
- 使用 `torchrun --nproc_per_node=1 -m qwen35_hf.train.sft`
- 默认单节点单卡
- 自动根据 CUDA 能力选择 `bf16`、`fp16` 或 `fp32`
- 自动从 `FT3_OUTPUT_DIR` / 默认输出目录查找最新 `checkpoint-*` 并续训
- 固定 `--deepspeed scripts/zero2_tp2_gpu.json`
- 固定 `--max_image_tokens 8196`
- 固定 `--lora_enable true`
- 固定 `--unfreeze_vision true`
- 固定 `--gradient_checkpointing true`
- 固定 `--attn_implementation sdpa`
- 固定 `--model_max_length 32768`
- 不再透传任何 `img_slot_*` 结构参数；这些参数统一从 checkpoint / config 读取
- 仍通过脚本透传 ImgSlot 损失权重（`img_slot_aux_loss_coef` / `img_slot_*_coef`）用于训练期调参

可覆盖的训练环境变量：

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
- `FT3_IMG_SLOT_AUX_LOSS_COEF`
- `FT3_IMG_SLOT_GATE_SPARSITY_COEF`
- `FT3_IMG_SLOT_EXPERT_BALANCE_COEF`
- `FT3_IMG_SLOT_SLOT_BALANCE_COEF`
- `FT3_IMG_SLOT_ROUTE_ENTROPY_COEF`

注意：`FT3_STOP_STEP` 由 `src/qwen35_hf/train/sft.py` 中的 `StopAtStepCallback` 读取，用于分段训练；训练总上限仍由 `max_steps` 控制。

**评测 LoRA checkpoint：**
```bash
bash scripts/eval.sh
```

`scripts/eval.sh` 当前通过本地 `lmms-eval` 的 `fovea` adapter 运行，默认：

- `PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/lmms-eval"`
- `HF_HUB_OFFLINE=1`
- `BASE_MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/...`
- `TASKS=xlrs-lite`
- `enable_thinking=False`
- `max_image_tokens=8196`

可覆盖的评测环境变量：

- `EVAL_BASE_MODEL`
- `EVAL_LORA_CHECKPOINT`
- `EVAL_TASKS`
- `EVAL_OUTPUT_PATH`
- `EVAL_LOG_SUFFIX`

**评测 Qwen3.5 baseline：**
```bash
bash scripts/eval_qwen35.sh
```

该脚本走 `lmms-eval` 的 `qwen3_5` adapter，不加载 Fovea LoRA。当前脚本默认带：

- `--limit 200`
- `--force_simple`
- `attn_implementation=sdpa`

**分段训练 + 每段评测：**
```bash
bash scripts/train_eval_loop.sh
```

该脚本会按 checkpoint step 推进训练，再评测最新 checkpoint。训练或评测失败时，脚本保留原始退出码退出。

**评测结果汇总：**
```bash
python scripts/xlrs_eval_report.py logs --latest-only --table-only
```

该脚本从 `*_samples_xlrs-lite.jsonl` 中读取 `xlrs_micro_score`，输出 XLRS 子任务统计表。

**轻量检查：**
```bash
bash -n scripts/ft3.sh
bash -n scripts/eval.sh
bash -n scripts/eval_qwen35.sh
bash -n scripts/train_eval_loop.sh
python -m py_compile src/qwen35_hf/train/sft.py
python -m py_compile src/qwen35_hf/train/data.py
python -m py_compile src/qwen35_hf/modeling_fovea.py
```

---

## 目录地图

```text
src/qwen35_hf/
  __init__.py             # Fovea 公共 API 与 Qwen3_5 兼容别名
  configuration_fovea.py  # FoveaConfig / Text / Vision 配置，含 ImgSlot 默认值
  modeling_qwen3_5.py     # Qwen3.5 text/vision/backbone 基础实现
  modeling_fovea.py       # FoveaForConditionalGeneration 与 ImgSlot runtime/cache 逻辑
  processing_fovea.py     # FoveaProcessor 与视觉占位展开工具
  tokenization_fovea.py   # FoveaTokenizer
  train/
    sft.py                # Trainer 入口、LoRA、冻结策略、训练 callback
    data.py               # SFT JSON 加载、ChatML 编码、VisionPacker、collator
    image_packing.py      # 本地图像 resize / normalize / patch packing / block split

scripts/
  ft3.sh                  # LoRA 训练脚本
  eval.sh                 # Fovea LoRA 评测脚本
  eval_qwen35.sh          # Qwen3.5 baseline 评测脚本
  train_eval_loop.sh      # 分段训练并评测最新 checkpoint
  zero2_tp2.json          # 备用 DeepSpeed 配置
  zero2_tp2_gpu.json      # 当前 ft3.sh 使用的 DeepSpeed 配置

lmms-eval/
  lmms_eval/models/simple/fovea.py      # Fovea 自定义 lmms-eval adapter
  lmms_eval/tasks/xlrs/XLRS-lite.yaml   # 当前默认评测任务
  lmms_eval/tasks/xlrs/mcq_utils.py     # xlrs-lite prompt、解析、聚合逻辑
```

---

## 模型结构

`FoveaConfig`、`FoveaTextConfig`、`FoveaVisionConfig` 都继承 `PreTrainedConfig`；`Qwen3_5Config`、`Qwen3_5TextConfig`、`Qwen3_5VisionConfig` 是对应 Fovea 配置类的兼容别名。

`FoveaConfig.model_type` 是 `fovea`；文本与视觉子配置使用各自独立的 `model_type`：

- `fovea_text`
- `fovea_vision`

`FoveaForConditionalGeneration` 继承 `Qwen3_5PreTrainedModel` 和 `GenerationMixin`，内部结构可概括为：

```text
FoveaForConditionalGeneration
  model: Qwen3_5Model
    visual: Qwen3_5VisionModel
    language_model: Qwen3_5TextModel
      layers: Qwen3_5DecoderLayer
  lm_head
  ImgSlot modules:
    imgslot_a_tokens
    imgslot_text_q_proj / k_proj / v_proj / o_proj
    imgslot_img_q_proj / k_proj / v_proj / o_proj
    imgslot_attn_norm
    imgslot_ffn
    imgslot_ffn_norm
    imgslot_visual_norm
    imgslot_gate_proj
    imgslot_expert_proj
    imgslot_subslot_proj
    imgslot_value_proj
    imgslot_slot_out_proj
```

普通 Qwen 多模态路径中，`Qwen3_5Model` 会把 vision tower 输出 scatter 回 `inputs_embeds`，然后继续进入 text decoder。启用 ImgSlot 时，`FoveaForConditionalGeneration` 会在 prefill 阶段提前改写 embedding，从而绕过普通视觉 scatter 路径。

---

## ImgSlot 约定

默认配置来自 `src/qwen35_hf/configuration_fovea.py` 与 `src/qwen35_hf/train/sft.py`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `img_slot_enable` | `True` | 是否启用 ImgSlot |
| `img_slot_m` | `8` | 每个含图样本的共享 anchor token 数 |
| `img_slot_k` | `128` | 每个 image block 输出的压缩 slot 数 |
| `img_slot_delta` | `129` | decode 时刷新 KV cache 的步间隔 |
| `img_slot_beta` | `0.3` | anchor 更新强度 |
| `img_slot_lambda` | `0.9` | decode refresh 时 slot EMA 平滑系数 |
| `img_slot_max_text_tokens` | `512` | runtime 中保留的文本 token KV 上限 |
| `img_slot_tile_size` | `768` | 图像切块边长 |
| `img_slot_num_experts` | `8` | router expert 数 |
| `img_slot_slots_per_expert` | `16` | 每个 expert 的 subslot 数 |
| `img_slot_gate_temperature` | `1.0` | gate 温度 |
| `img_slot_route_temperature` | `1.0` | route / subslot 温度 |
| `img_slot_topk_experts` | `2` | 每个 visual token 保留的 expert 数 |
| `img_slot_topk_subslots` | `4` | 每个选中 expert 内保留的 subslot 数 |
| `img_slot_use_entmax` | `False` | 预留字段 |
| `img_slot_enable_hardening` | `False` | 预留字段 |
| `img_slot_hardening_schedule` | `"none"` | 预留字段 |
| `img_slot_min_temperature` | `0.25` | gate / route 温度下界 |

额外约束：

- `img_slot_num_experts * img_slot_slots_per_expert` 必须等于 `img_slot_k`
- `img_slot_enable=true` 时，`img_slot_tile_size` 必须非空
- `img_slot_m` 与 `img_slot_k` 都必须为正整数

数据侧 placeholder 约定：

- `VisionPacker.pack()` 先按整图 token budget 缩放，再切 block
- block 数由 `ceil(width / tile_size)` 与 `ceil(height / tile_size)` 决定
- 每个 block 独立生成一行 `image_grid_thw`
- `encode_chatml_example()` 在含图样本前插入一个长度为 `m` 的 anchor span
- 每个 block 的文本视觉 span 长度固定为 `k`
- `image_token_counts` 的形状语义是 `list[list[int]]`：外层按原图，内层按 block

模型侧 runtime 约定：

- `_project_imgslot_kv()` 输入必须是 `[seq, hidden]`，输出是 `[1, num_heads, seq, head_dim]`
- `_build_imgslot_states_for_sample()` 要求 `text_tokens` 非空，且 `len(visual_pools) == len(visual_spans)`
- prefill 阶段会直接把 anchor span 与 slot span 写回 `inputs_embeds`
- generation 首轮通过 `imgslot_first_prefill` 强制走 embedding rewrite
- decode 阶段每 `img_slot_delta` 步仅刷新 full-attention layers 的对应 KV cache
- runtime 缓存里保存的是 detach 后状态，只用于 generation refresh
- top-k sparse routing 中，只有选中的 token-slot pair 能参与 token 维 softmax；未选中的位置必须保持 mask，不得以 0 logit 参与归一化。

辅助统计会保存在 `model._imgslot_aux` 中，并由 `ImgSlotMetricsCallback` 写到训练日志；其中 `imgslot/num_blocks` 表示当前 forward 中真实参与 ImgSlot 压缩的 image block 数。

---

## 训练流程

`scripts/ft3.sh` -> `torchrun -m qwen35_hf.train.sft`。

`src/qwen35_hf/train/sft.py` 当前关键行为：

- 加载 `FoveaTokenizer` 和 `FoveaForConditionalGeneration`
- 把 `img_slot_*` CLI 参数传给 `from_pretrained()`，再写回 `model.config`
- 训练时强制 `model.config.use_cache = False`
- 先冻结全模型；`--unfreeze_vision true` 时解冻 vision tower
- 若启用 LoRA，则对 attention / MLP / linear-attention 投影挂 LoRA
- ImgSlot 模块不作为 LoRA target，而是放入 `modules_to_save`
- PEFT 包装后显式把 `imgslot_*` 参数重新设为 trainable
- 注册 `StopAtStepCallback` 与 `ImgSlotMetricsCallback`

LoRA / checkpoint 注意点：

- `adapter_model.safetensors` 保存 LoRA 权重与 ImgSlot `modules_to_save`
- vision tower 等非 LoRA trainables 保存在 DeepSpeed `global_step*/mp_rank_00_model_states.pt`
- `lmms-eval` 的 `fovea.py` 会尝试从该 DeepSpeed 文件中恢复 `model.visual*` 权重

修改可训练模块列表时，必须同步检查：

- `src/qwen35_hf/train/sft.py`
- `lmms-eval/lmms_eval/models/simple/fovea.py`

---

## 数据与 ChatML

训练数据由 `LazySupervisedDataset` 懒加载：

- JSON 顶层必须是 list
- 每条样本至少包含 `conversations`
- `image` 可选，可为字符串或字符串列表
- 图像路径相对 `image_folder`
- 当消息中恰好一个 `<image>` 且不在开头时，会被移动到 user message 开头
- 多图样本按 `<image>` 顺序消费
- 若文本只有一个 `<image>`，但样本提供多张图，则会把多张图的 block placeholder 拼接到同一个占位处

assistant 监督模板固定为：

```text
<|im_start|>assistant
<think>

</think>

{answer}<|im_end|>
```

label 策略：

- system / user 全部 mask 为 `IGNORE_INDEX=-100`
- assistant role prefix 与空 thinking scaffold 全部 mask
- answer 内容与 `<|im_end|>` 参与监督

`DataCollatorForQwen3_5SFT`：

- 对文本字段右 padding
- 截断到 `model_max_length`
- 对不同样本的 `pixel_values` / `image_grid_thw` 沿视觉 patch / grid 维拼接
- 不做视觉 padding

---

## 评测路径

`scripts/eval.sh` 默认运行：

```text
accelerate launch -m lmms_eval --model fovea --tasks xlrs-lite
```

`lmms-eval/lmms_eval/models/simple/fovea.py` 当前行为：

- 加载 `FoveaForConditionalGeneration`
- 可选加载 PEFT adapter
- 若提供 LoRA checkpoint，则尝试从 DeepSpeed model state 恢复 vision trainables
- `use_cache` 默认允许为 `True`，依赖 ImgSlot prefill + refresh 机制
- `LocalVisionImageProcessor` 复用训练侧 `VisionPacker`
- 通过 `_last_image_block_counts` 追踪一张原图对应多少 block span
- ImgSlot 启用时 processor 不返回 `mm_token_type_ids`
- 当前 adapter 只支持 image inputs；video 会直接报错
- `max_image_tokens` adapter 默认是 `128`，但 `eval.sh` 显式传 `8196`

---

## 修改原则

- 新代码优先使用 `Fovea*` 命名；`Qwen3_5*` 仅作兼容别名
- `modeling_qwen3_5.py` 是基础 text / vision / backbone 实现；`modeling_fovea.py` 是 Fovea + ImgSlot 主实现。不要把 ImgSlot 逻辑塞回基础 backbone
- Fovea 专用评测逻辑放在 `lmms_eval/models/simple/fovea.py`；不要改通用 adapter 去迁就 Fovea
- 训练、评测、processor 的 ImgSlot placeholder 约定必须同步：anchor span、block span、`image_grid_thw` 行数、`visual_pools` 数量都必须一致
- 不要引入未被当前代码路径使用的兼容层或备用分支，除非 THX 明确要求
- 代码优先直接、清晰、可核对
- 涉及 reshape、permute、split、cat、mask、KV cache 写入时，关键变量附近应标注形状
- 注释只解释关键数据流、边界条件和设计意图
- 修改 `img_slot_*` 默认值时，至少同步检查：
  - `src/qwen35_hf/configuration_fovea.py`
  - `src/qwen35_hf/train/sft.py`
  - `src/qwen35_hf/train/data.py`
  - `src/qwen35_hf/train/image_packing.py`
  - `src/qwen35_hf/processing_fovea.py`
  - `lmms-eval/lmms_eval/models/simple/fovea.py`
  - `scripts/ft3.sh`
  - `AGENTS.md`
  - `README.md`
- 修改 LoRA trainable / `modules_to_save` 时，必须同步检查训练保存逻辑与评测加载逻辑
- 修改脚本环境变量时，必须同步更新本文档与 `README.md`

---

## 已知注意事项

- 根目录当前不是标准可编辑 Python 包；主路径依赖 `PYTHONPATH`
- `README.md` 中如出现与代码不一致的旧包安装方式，应以当前脚本和源码入口为准
- `warmup_ratio is deprecated` 是 Transformers 警告，不影响训练
- `NCCL_DEBUG=INFO` 会产生较多日志；需要降噪时可改为 `WARN`
- 参数打印里出现 `device=cpu` 可能发生在 DeepSpeed prepare 之前，不代表最终训练落在 CPU
- `PIL.Image.MAX_IMAGE_PIXELS = None` 已在训练数据路径全局关闭，用于避免超大图触发 PIL 限制
