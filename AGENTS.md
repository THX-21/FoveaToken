# AGENTS.md

本文件为 Claude Code（claude.ai/code）及其他 AI 编程助手提供项目导航与协作规范。**每次修改代码后必须同步更新本文档**，保持文档与代码一致。

---

## 项目简介

`qwen35-hf` 是将 HuggingFace Transformers 中 `qwen3_5` 实现提取为独立库的最小打包版本，附带 LoRA 微调流程和 lmms-eval 评测适配器。

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

**训练（LoRA 微调）：**
```bash
bash scripts/ft3.sh
```

**评测 LoRA checkpoint：**
```bash
bash scripts/eval.sh
```

**语法检查：**
```bash
bash -n scripts/ft3.sh
bash -n scripts/eval.sh
python -m py_compile src/qwen35_hf/train/sft.py
python -m py_compile src/qwen35_hf/train/data.py
```

---

## 技术架构

### 整体结构

```
src/qwen35_hf/
  __init__.py                  # 公共 API：导出模型、配置、tokenizer、processor
  configuration_qwen3_5.py     # Qwen3_5Config / Qwen3_5TextConfig / Qwen3_5VisionConfig
  modeling_qwen3_5.py          # 由 modular_qwen3_5.py 自动生成，不要手动编辑
  modular_qwen3_5.py           # 模型定义源文件（继承 Qwen3Next + Qwen3VL 组件）
  tokenization_qwen3_5.py      # Qwen3_5Tokenizer
  processing_qwen3_vl.py       # Qwen3VLProcessor（视觉-语言预处理、token 构建工具）
  train/
    sft.py                     # 训练入口（ModelArguments / DataArguments / TrainingArguments）
    data.py                    # 数据集加载、ChatML 编码（encode_chatml_example）
    image_packing.py           # anyres 图像切片（pack_anyres_image / pack_single_image）

scripts/
  ft3.sh                       # LoRA 训练脚本（torchrun + DeepSpeed ZeRO-2）
  eval.sh                      # lmms-eval 评测脚本（accelerate launch）
  zero2_tp2.json               # DeepSpeed ZeRO-2 配置
  zero2_tp2_gpu.json           # DeepSpeed ZeRO-2 GPU 配置

lmms-eval/
  lmms_eval/models/simple/qwen35_hf.py   # lmms-eval 自定义模型适配器
```

### 模型继承链

```
Qwen3_5TextConfig   ← Qwen3NextConfig
Qwen3_5VisionConfig ← Qwen3VLVisionConfig
Qwen3_5Config       ← (组合 Text + Vision 配置)

Qwen3_5ForConditionalGeneration ← Qwen3VLForConditionalGeneration
  └── visual: Qwen3VLVisionModel      （vision tower，--unfreeze_vision 时解冻）
  └── model:  Qwen3_5TextModel
        └── layers: 混合 Qwen3NextAttention（标准注意力）
                  + Qwen3NextGatedDeltaNet（线性注意力，需 causal-conv1d / flash-linear-attention 加速）
```

模型逻辑修改直接在 `modeling_qwen3_5.py` 中进行，不修改 `modular_qwen3_5.py`。

### 训练流程

1. `ft3.sh` 通过 `torchrun` 启动（即使单卡也走 torch distributed，避免 DeepSpeed 回退到 MPI）。
2. `sft.py` 解析三组参数：`ModelArguments`、`DataArguments`、`TrainingArguments`。
3. `load_optional_processor` 根据 `processor_backend` 决定是否加载官方 processor：
   - `local` → 不加载，图像编码完全走本地路径
   - `official` / `auto` → 尝试 `AutoProcessor.from_pretrained`
4. `freeze_for_vision_plus_lora`：冻结所有参数，仅在 `--unfreeze_vision true` 时解冻 `model.visual`。
5. `data.py` 的 `LazySupervisedDataset` 惰性编码，`DataCollatorForQwen3_5SFT` 整理 batch。

### 图像处理路径

- **`normal`**：`pack_single_image` — 单图直接 resize 到目标分辨率。
- **`anyres`**：`pack_anyres_image` — 从 `grid_pinpoints` 选合法分辨率（≤ `max_image_tokens`），输出 `image_grid_thw`。
  - `image_aspect_ratio=anyres` 时，训练代码强制走本地路径，无论 `processor_backend` 如何设置。
- 图像最终 token 数计算：`image_tokens = image_grid_thw.prod() // spatial_merge_size²`
  - 例：`image_grid_thw = [1, 84, 84]` → `1764` tokens（对应 1344×1344 canvas）

### ChatML 模板

训练和评测共用的 assistant 输出格式，包含空 thinking scaffold：

```
<|im_start|>assistant
<think>

</think>

{answer}<|im_end|>
```

`<image>` 占位在编码阶段由 `replace_image_tokens_in_conversations` 展开为多个连续 `<|image_pad|>`，最终数量必须与 `image_grid_thw.prod() // spatial_merge_size²` 严格对齐。

### 评测适配器

`lmms_eval/models/simple/qwen35_hf.py` 直接调用本库的三个公共类：
- `qwen35_hf.Qwen3_5ForConditionalGeneration`
- `qwen35_hf.Qwen3_5Tokenizer`
- `qwen35_hf.Qwen3VLProcessor`

LoRA 评测方式：先加载 `BASE_MODEL`，再通过 `PeftModel.from_pretrained` 加载 `LORA_CHECKPOINT`。

---

## 修改原则

- **直接修改 `modeling_qwen3_5.py`**，不要修改 `modular_qwen3_5.py`。
- **训练与评测模板必须同步**：修改 `data.py` 中的 prompt/图像处理逻辑时，同步更新 `lmms_eval/models/simple/qwen35_hf.py`。
- **Qwen3.5 专用评测逻辑放在 `lmms_eval/models/simple/qwen35_hf.py`**，不要修改通用的 `qwen3_vl.py`。
- **每次修改代码后更新本文档**，重点更新受影响的架构说明、参数约定或命令。

## 已知注意事项

- `warmup_ratio is deprecated` 是 Transformers 警告，不影响训练。
- `NCCL_DEBUG=INFO` 产生大量日志属正常噪音，可调低为 `WARN`。
- 参数打印中的 `device=cpu` 出现在 DeepSpeed prepare 之前，不代表训练在 CPU 上执行。
- 如出现 `processor is not defined`，检查 `sft.py` 是否保留了 `load_optional_processor` 调用。
- `xlrs-lite` 评测中 `(A)` 会被标准化为 `A`，两者均算正确。
- `PIL.Image.MAX_IMAGE_PIXELS = None` 已在 `data.py` 中全局关闭，避免超大图报错。
