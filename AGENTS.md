# AGENTS.md

本文件为 Claude Code（claude.ai/code）及其他 AI 编程助手提供项目导航与协作规范。**每次修改代码后必须同步更新本文档**，保持文档与代码一致。

---

## 项目简介

`qwen35-hf` 是将 HuggingFace Transformers 中 `qwen3_5` 实现提取为独立库的最小打包版本，附带 LoRA 微调流程和 lmms-eval 评测适配器。dev 分支新增了 **ImgSlot** 机制：将图像分块后用动态锚点 token + Top-K 视觉 token 替换原始图像占位，减少视觉 token 开销。

---

## 常用命令

所有命令从项目根目录（`/root/wd/FoveaToken/qwen35_hf`）执行。

**安装：**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .              # 基础安装
pip install -e ".[train]"     # 添加训练依赖：accelerate, deepspeed, peft, datasets, tensorboard
pip install -e ".[fast]"      # 添加加速依赖：causal-conv1d, flash-linear-attention
```

**训练（LoRA 微调，启用 ImgSlot）：**
```bash
IMG_SLOT_TILE_SIZE=336 bash scripts/ft3.sh
```
`IMG_SLOT_TILE_SIZE` 是必填环境变量，未设置时脚本报错退出。

**评测 LoRA checkpoint：**
```bash
IMG_SLOT_TILE_SIZE=336 bash scripts/eval.sh
```

**语法检查：**
```bash
bash -n scripts/ft3.sh
bash -n scripts/eval.sh
/root/wd/FoveaToken/.venv/bin/python -m py_compile src/qwen35_hf/train/sft.py
/root/wd/FoveaToken/.venv/bin/python -m py_compile src/qwen35_hf/train/data.py
```

---

## 技术架构

### 整体结构

```
src/qwen35_hf/
  __init__.py                  # 公共 API：导出模型、配置、tokenizer、processor
  configuration_qwen3_5.py     # Qwen3_5Config（含 img_slot_* 字段）/ Text / Vision Config
  modeling_qwen3_5.py          # 模型实现，直接在此修改（不改 modular_qwen3_5.py）
  modular_qwen3_5.py           # 原始模板文件，不修改
  tokenization_qwen3_5.py      # Qwen3_5Tokenizer
  processing_qwen3_vl.py       # Qwen3VLProcessor（视觉-语言预处理、token 构建工具）
  train/
    sft.py                     # 训练入口（ModelArguments / DataArguments / TrainingArguments）
    data.py                    # 数据集加载、ChatML 编码、VisionPacker（含 ImgSlot 路径）
    image_packing.py           # 图像切片：pack_single_image / split_image_into_blocks

scripts/
  ft3.sh                       # LoRA 训练脚本（torchrun + DeepSpeed ZeRO-2，需 IMG_SLOT_TILE_SIZE）
  eval.sh                      # lmms-eval 评测脚本（accelerate launch，需 IMG_SLOT_TILE_SIZE）
  zero2_tp2.json               # DeepSpeed ZeRO-2 配置

lmms-eval/
  lmms_eval/models/simple/qwen35_hf.py   # lmms-eval 自定义模型适配器（含 ImgSlot 参数）
```

### 模型继承链

```
Qwen3_5TextConfig   ← Qwen3NextConfig
Qwen3_5VisionConfig ← Qwen3VLVisionConfig
Qwen3_5Config       ← (组合 Text + Vision 配置，新增 img_slot_* 字段)

Qwen3_5ForConditionalGeneration ← Qwen3VLForConditionalGeneration
  ├── visual: Qwen3VLVisionModel      （vision tower，--unfreeze_vision 时解冻）
  ├── model:  Qwen3_5TextModel
  │     └── layers: Qwen3NextAttention + Qwen3NextGatedDeltaNet（线性注意力）
  └── ImgSlot 专属参数（仅在 img_slot_enable=True 时生效）：
        imgslot_a_tokens.weight   # 可学习锚点 token（Embedding 权重，形状 [m, hidden]）
        imgslot_text_q/k/v/o_proj # 锚点-文本交叉注意力投影
        imgslot_img_q/k_proj      # 视觉 Top-K 评分投影
        imgslot_attn_norm         # LayerNorm（锚点更新后）
        imgslot_ffn + ffn_norm    # 前馈网络（锚点更新后）
```

### ImgSlot 机制（dev 分支核心新功能）

**目标**：用更少的 token 表示图像，减少视觉序列长度。

**流程：**
1. `split_image_into_blocks(image, tile_size)` — 将原图按 `ceil(W/tile_size) × ceil(H/tile_size)` 均分为 blocks。
2. 每个 block 走 `_pack_single_block`（normal 路径），得到 `pixel_values` 和 `image_grid_thw`（多 block 时 grid 为 2D tensor）。
3. 数据编码时，每个 block 对应的 token 数固定为 `img_slot_m + img_slot_k`（不再使用 `image_grid_thw` 计算）。
4. 训练时，`Qwen3_5ForConditionalGeneration` 将图像占位 span 替换为：
   - `img_slot_m` 个动态锚点 token（由 `imgslot_a_tokens.weight` + 文本交叉注意力初始化）
   - `img_slot_k` 个 Top-K 视觉 token（由锚点对视觉池打分后选出）
5. 推理时每隔 `img_slot_delta` 步用动量（`img_slot_lambda`）更新锚点和 Top-K 选择。

**关键参数（`Qwen3_5Config` 中）：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `img_slot_enable` | `False` | 是否启用 ImgSlot |
| `img_slot_m` | 8 | 每个 block 的动态锚点 token 数 |
| `img_slot_k` | 128 | 每个 block 的 Top-K 视觉 token 数 |
| `img_slot_delta` | 8 | 推理时刷新 KV 的解码步间隔 |
| `img_slot_beta` | 0.3 | 锚点动量更新强度 |
| `img_slot_lambda` | 0.9 | 视觉 Top-K 分数动量系数 |
| `img_slot_tile_size` | `None`（必填） | 原图切块边长（像素） |

### 训练流程

1. `ft3.sh` 通过 `torchrun` 启动（即使单卡也走 torch distributed）。
2. `IMG_SLOT_TILE_SIZE` 必须在运行前设置，否则脚本报错。
3. `sft.py` 将 `img_slot_*` 参数写入 `model.config`，并传入 `VisionPacker` 和 `LazySupervisedDataset`。
4. `ft3.sh` 自动查找 `OUTPUT_DIR` 下最新 `checkpoint-*` 并续训。
5. 图像处理固定走 normal 本地路径（`image_aspect_ratio=normal`）。

### 数据编码与图像 token

- 编码入口：`data.py` 的 `encode_chatml_example`。
- `image_token_counts` 类型已从 `list[int]` 改为 `list[list[int]]`（外层按图，内层按 block）。
- ImgSlot 启用时每个 block 的 token 数固定为 `img_slot_m + img_slot_k`；禁用时按 `image_grid_thw.prod() // spatial_merge_size²` 计算。
- ImgSlot 训练侧 prefill 不再对 `visual_pool` 和 `text_tokens` 做 `detach()`：选中的 Top-K 视觉 token 可将梯度传回 vision tower，文本条件化分支也可将梯度传回文本 embedding / LoRA 路径。decode runtime state 中保存的 `A`/`V`/`T`/缓存投影仍使用 `detach()`，仅用于推理刷新状态。

### ChatML 模板

训练和评测共用：

```
<|im_start|>assistant
<think>

</think>

{answer}<|im_end|>
```

### 评测适配器

`lmms_eval/models/simple/qwen35_hf.py`：
- 接受全部 `img_slot_*` 参数（字符串/数值均可，内部做类型转换）。
- `LocalVisionImageProcessor` 记录每张图的 block 数（`_last_image_block_counts`），供 token 对齐使用。
- LoRA 评测：先加载 `BASE_MODEL`，再 `PeftModel.from_pretrained(LORA_CHECKPOINT)`。
- 评测 limit 当前为 500 条（master 为 200）。

---

## 修改原则

- **直接修改 `modeling_qwen3_5.py`**，不修改 `modular_qwen3_5.py`。
- **训练与评测 ImgSlot 参数必须同步**：修改任一侧的参数默认值或逻辑，同步更新另一侧。
- **Qwen3.5 专用评测逻辑放在 `lmms_eval/models/simple/qwen35_hf.py`**，不修改通用 `qwen3_vl.py`。
- **每次修改代码后更新本文档**，重点更新受影响的架构说明、参数约定或命令。
- **遇到不确定的代码设计问题时，必须先询问 THX，不得直接行动。**
- **不能写兼容性代码，除非你明确要求。**
- **代码必须简洁，优先选择直接、清晰的实现，避免不必要的抽象和分支。**
- **代码中应添加必要注释**：重点解释关键数据流、张量变换、边界条件与设计意图，避免无信息量注释。
- **涉及张量或多维结构的关键变量必须标注形状**，尤其是在 reshape、permute、split、cat、scatter、mask 等操作附近。

- ImgSlot 核心实现已收敛为 **sample-batched 主路径**：prefill 初始化与 decode refresh 都优先按 sample 内全部 blocks 批量处理，不再维护独立的单 block helper 分支。
- ImgSlot 文本条件化与视觉打分统一使用 batched helper，避免重复维护单块版和批量版逻辑。

- ImgSlot decode refresh 的空文本分支必须按 **每个 block 各自的 `V`** 构造 fallback text KV，不能错误地复用第一个 block 的视觉池，否则多 block 样本会发生条件化串扰。

- `eval.sh` 通过 `lmms_eval/models/simple/qwen35_hf.py` 间接调用本地 `modeling_qwen3_5.py`；评测入口不是直接执行模型定义文件。
- 当前评测适配器在 ImgSlot 启用时 **允许 `use_cache`**，以便评测路径对齐当前实现的缓存/refresh 行为；但仍不向模型传 `mm_token_type_ids`。

- 评测适配器 `lmms_eval/models/simple/qwen35_hf.py` 目前在 ImgSlot 启用时必须保持 `use_cache=False`：否则 `generate()` 的 prefill 会落回原始 placeholder/image-feature 对齐路径，触发 `image tokens != image features` 错误。

- ImgSlot generation 在 `use_cache=True` 时，首轮 prefill 必须由 `Qwen3_5ForConditionalGeneration.forward()` 的 embedding-rewrite 分支接管；当前实现通过 generation 输入中的 `imgslot_first_prefill` 标记显式触发该分支，随后 decode 再进入 cache / refresh 路径。
- 评测适配器 `lmms_eval/models/simple/qwen35_hf.py` 现在允许 `use_cache=True`，前提是上述首轮 prefill 接管逻辑保持有效。

- `xlrs-lite` 当前在 task 层通过 `process_docs` 预裁剪数据：按 `doc["category"]`（即 task/subtask 组合键）每类最多保留 60 条，而不是用 CLI `--limit` 对整个聚合任务统一截断。

- `Qwen3_5ForConditionalGeneration` 中的 ImgSlot 兼容/兜底逻辑已移除：不再保留 `_sanitize_imgslot_tensor`、空文本 fallback、Top-K padding、legacy tuple/list cache 支持等防御性路径；代码只支持当前仓库明确使用的输入与 cache 结构。

- ImgSlot 配置读取也已收紧：`Qwen3_5ForConditionalGeneration` 不再用 `getattr(..., default)` 提供默认超参，运行时要求 `config` 显式包含 `img_slot_*` 字段，并默认 `_imgslot_runtime`/`layer_types` 等结构始终存在且符合当前仓库约定。

- `imgslot_a_tokens` 是 `nn.Embedding` 子模块，不再是根模块裸 `nn.Parameter`；它通过标准 module 初始化路径初始化，权重名为 `imgslot_a_tokens.weight`。

- LoRA 训练在 `LoraConfig.modules_to_save` 中包含全部 ImgSlot 子模块（`imgslot_a_tokens`、文本/视觉投影、norm、FFN），因此 `adapter_model.safetensors` 会随 LoRA adapter 保存和恢复完整 ImgSlot 权重；不再为 ImgSlot 维护额外 sidecar checkpoint 文件。

## 已知注意事项

- `warmup_ratio is deprecated` 是 Transformers 警告，不影响训练。
- `NCCL_DEBUG=INFO` 产生大量日志属正常噪音，可调低为 `WARN`。
- 参数打印中的 `device=cpu` 出现在 DeepSpeed prepare 之前，不代表训练在 CPU 上执行。
- `xlrs-lite` 评测中 `(A)` 会被标准化为 `A`，两者均算正确。
- `PIL.Image.MAX_IMAGE_PIXELS = None` 已在 `data.py` 中全局关闭，避免超大图报错。
