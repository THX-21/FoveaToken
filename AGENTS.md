# AGENTS.md

本文件面向 Claude Code（claude.ai/code）及其他 AI 编程助手，记录当前仓库的真实入口、数据流和协作约束。**修改代码后必须同步更新本文档**，避免下一位助手沿用过期假设。

---

## 当前定位

`qwen35_hf` 是本地 Fovea / Qwen3.5 多模态实验仓库。当前对外主入口是：

- `FoveaForConditionalGeneration`
- `FoveaConfig` / `FoveaTextConfig` / `FoveaVisionConfig`
- `FoveaProcessor`
- `FoveaTokenizer`

`Qwen3_5*` 名称仍作为兼容别名导出，但新代码优先使用 `Fovea*` 命名。核心实验功能是 **ImgSlot**：训练和推理时先把图像切成 block，再用一个样本级 anchor span + 每个 block 的 Top-K 视觉 token span 替代原始全量视觉占位，从而控制文本侧视觉 token 数。

仓库根目录当前没有 `pyproject.toml`、`setup.py` 或 `setup.cfg`。不要假设 `pip install -e .` 可用；训练和评测脚本通过 `PYTHONPATH` 指向 `src` / `lmms-eval` 运行本地代码。

---

## 常用命令

所有命令从项目根目录 `/root/wd/FoveaToken/qwen35_hf` 执行。

**环境准备：**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers huggingface_hub tokenizers numpy pillow accelerate deepspeed peft datasets tensorboard
pip install -e ./lmms-eval
pip install ./flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

如需本地导入包：
```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

**训练 LoRA + ImgSlot：**
```bash
bash scripts/ft3.sh
```

`ft3.sh` 固定走 `torchrun -m qwen35_hf.train.sft`、DeepSpeed ZeRO-2、单节点单卡默认配置。脚本不再显式传 `img_slot_*` 参数，训练入口使用 `ModelArguments` 默认值并写回 `model.config`。

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
- `FT3_STOP_STEP`（由 `sft.py` 的 `StopAtStepCallback` 读取，用于分段训练）

**评测 LoRA checkpoint：**
```bash
bash scripts/eval.sh
```

`eval.sh` 通过本地 `lmms-eval` 的 `fovea` model adapter 运行，默认任务是 `xlrs-lite`，默认 `HF_HUB_OFFLINE=1`。

可覆盖的评测环境变量：

- `EVAL_BASE_MODEL`
- `EVAL_LORA_CHECKPOINT`
- `EVAL_TASKS`
- `EVAL_OUTPUT_PATH`
- `EVAL_LOG_SUFFIX`

**评测结果汇总脚本：**
```bash
python scripts/xlrs_eval_report.py logs/fovea-ft3-imgslot__checkpoint-6000/20260421_205615_samples_xlrs-lite.jsonl
python scripts/xlrs_eval_report.py logs --latest-only --table-only
```

该脚本从 `*_samples_xlrs-lite.jsonl` 读取 `xlrs_micro_score` 字段，按 `lmms_eval/tasks/xlrs/mcq_utils.py` 的 13 个子任务口径统计准确率，并输出可直接粘贴到论文/表格里的 Markdown 表；末尾同时给出 `Micro Avg.`（整体准确率）和 `Macro Avg.`（13 个子任务准确率均值）。

**分段训练 + 每段评测：**
```bash
bash scripts/train_eval_loop.sh
```

该脚本按 checkpoint step 推进训练，训练到目标 step 后评测最新 checkpoint。训练或评测失败时保留原始 stderr/traceback，随后用子命令的原始退出码退出；脚本启用 `set -uo pipefail`。

**Qwen3.5 baseline 评测：**
```bash
bash scripts/eval_qwen35.sh
```

该脚本走 `lmms-eval` 的 `qwen3_5` adapter，不加载 Fovea LoRA。

**轻量检查：**
```bash
bash -n scripts/ft3.sh
bash -n scripts/eval.sh
bash -n scripts/eval_qwen35.sh
bash -n scripts/train_eval_loop.sh
/root/wd/FoveaToken/.venv/bin/python -m py_compile src/qwen35_hf/train/sft.py
/root/wd/FoveaToken/.venv/bin/python -m py_compile src/qwen35_hf/train/data.py
/root/wd/FoveaToken/.venv/bin/python -m py_compile src/qwen35_hf/modeling_fovea.py
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
    sft.py                # Trainer 入口、LoRA、冻结策略、分段停止 callback
    data.py               # SFT JSON 加载、ChatML 编码、VisionPacker、collator
    image_packing.py      # 本地图像 resize / normalize / patch packing / block split

scripts/
  ft3.sh                  # LoRA 训练脚本
  eval.sh                 # Fovea LoRA 评测脚本
  eval_qwen35.sh          # Qwen3.5 baseline 评测脚本
  train_eval_loop.sh      # 分段训练并评测最新 checkpoint
  zero2_tp2.json          # 当前 ft3.sh 使用的 DeepSpeed ZeRO-2 配置
  zero2_tp2_gpu.json      # 备用 DeepSpeed 配置

lmms-eval/
  lmms_eval/models/simple/fovea.py      # Fovea 自定义 lmms-eval adapter
  lmms_eval/tasks/xlrs/XLRS-lite.yaml   # 当前默认评测任务
  lmms_eval/tasks/xlrs/mcq_utils.py     # xlrs-lite prompt、裁剪、解析、聚合逻辑
```

`modular_qwen3_5.py` 当前不在仓库中。不要再引用或修改这个旧模板文件。

---

## 模型结构

`FoveaConfig`、`FoveaTextConfig`、`FoveaVisionConfig` 都继承 `PreTrainedConfig`；`Qwen3_5Config`、`Qwen3_5TextConfig`、`Qwen3_5VisionConfig` 是对应 Fovea 配置类的别名。

`FoveaForConditionalGeneration` 继承 `Qwen3_5PreTrainedModel` 和 `GenerationMixin`，内部结构是：

```text
FoveaForConditionalGeneration
  model: Qwen3_5Model
    visual: Qwen3_5VisionModel
    language_model: Qwen3_5TextModel
      layers: Qwen3_5DecoderLayer
        self_attn: Qwen3_5Attention 或 Qwen3_5GatedDeltaNet
  lm_head
  ImgSlot modules:
    imgslot_a_tokens
    imgslot_text_q_proj / k_proj / v_proj / o_proj
    imgslot_img_q_proj / k_proj / v_proj / o_proj
    imgslot_attn_norm
    imgslot_ffn
    imgslot_ffn_norm
```

`Qwen3_5Model` 负责普通多模态路径：视觉 tower 输出 features，按 placeholder mask 写回 `inputs_embeds`，再构造 MRoPE position ids 并调用 `Qwen3_5TextModel`。如果普通多模态路径收到 `image_grid_thw` / `video_grid_thw`，必须有 `mm_token_type_ids`，否则会报错。

`FoveaForConditionalGeneration` 在 ImgSlot 启用时绕过普通视觉 scatter：先计算 `visual_pools`，再把文本中的 image placeholder spans 改写为 anchor tokens 和 Top-K visual tokens，随后把 `pixel_values` / `image_grid_thw` / `mm_token_type_ids` 清空并继续走 text decoder。

---

## ImgSlot 约定

默认配置来自 `FoveaConfig` 和 `ModelArguments`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `img_slot_enable` | `True` | 是否启用 ImgSlot |
| `img_slot_m` | `8` | 每个含图样本的共享 anchor token 数 |
| `img_slot_k` | `128` | 每个 image block 保留的 Top-K 视觉 token 数 |
| `img_slot_delta` | `129` | decode 时刷新 KV cache 的步间隔 |
| `img_slot_beta` | `0.3` | anchor 动量更新强度 |
| `img_slot_lambda` | `0.9` | Top-K 分数动量系数 |
| `img_slot_max_text_tokens` | `512` | runtime 中保留的文本 token KV 上限 |
| `img_slot_tile_size` | `1024` | 原图切块边长，启用 ImgSlot 时必须为正整数 |

数据侧 placeholder 约定：

- `VisionPacker.pack()` 先对整图应用 `max_image_tokens` 预算，再用 `split_image_into_blocks()` 按 `ceil(width / tile_size)` 和 `ceil(height / tile_size)` 均匀切块。
- 每个 block 走 normal 本地 patch packing，产生一行 `image_grid_thw`；多 block 图像返回 2D grid。
- `encode_chatml_example()` 在含图样本前插入一个长度为 `m` 的 anchor span。
- 每个 block 的文本侧视觉 span 长度固定为 `k`，不是 `m + k`；整张图的文本侧开销是 `m + num_blocks * k`。
- `image_token_counts` 的形状语义是 `list[list[int]]`：外层按原图，内层按 block。

模型侧 runtime 约定：

- `_project_imgslot_kv()` 输入必须是 2D token 张量 `[seq, hidden]`，输出固定为 `[1, num_heads, seq, head_dim]`。
- `_build_imgslot_states_for_sample()` 要求 `text_tokens` 非空，并要求 `len(visual_pools) == len(visual_spans)`。
- 每个 visual pool 的 token 数必须至少为 `img_slot_k`；不足时显式报错，提示增大 tile size、减小 `img_slot_k` 或提高 block 分辨率。
- prefill 期间 Top-K 选择不 detach `visual_pool` / `text_tokens`，梯度可以回到 vision tower、text embedding 和 LoRA 路径。
- runtime 中缓存的 `A`、`V`、`V_k`、`V_v`、`V_topk`、`score_prev`、`topk_idx` 是 detach 后的状态，只用于 generation refresh。
- `use_cache=True` generation 首轮通过 `imgslot_first_prefill` 标记强制进入 embedding rewrite；后续 decode 每 `img_slot_delta` 步只刷新 full-attention layers 的 KV cache。

不要重新加入 legacy tuple/list cache 支持、Top-K padding、空文本 fallback 或宽松 `getattr(..., default)` 配置兜底，除非 THX 明确要求。

---

## 训练流程

`scripts/ft3.sh` 当前关键行为：

- 设置 `PYTHONPATH="$PROJECT_ROOT/src"`。
- 使用 `torchrun --nproc_per_node=1 -m qwen35_hf.train.sft`。
- 自动根据 CUDA 能力选择 bf16、fp16 或 fp32 参数。
- 从 `FT3_OUTPUT_DIR` / 默认 `checkpoints/fovea-ft3-imgslot` 查找最新 `checkpoint-*` 并传 `--resume_from_checkpoint`。
- 固定 `--max_image_tokens 8196`、`--lora_enable true`、`--unfreeze_vision true`、`--gradient_checkpointing true`、`--attn_implementation flash_attention_2`、`--model_max_length 32768`。
- 同时传 `--max_steps "$FT3_MAX_STEPS"`；分段停止由 `FT3_STOP_STEP` callback 负责。

`src/qwen35_hf/train/sft.py` 当前关键行为：

- 加载 `FoveaTokenizer` 和 `FoveaForConditionalGeneration`。
- 将 `img_slot_*` CLI/default 参数传给 `from_pretrained()`，并再次写回 `model.config`。
- 训练时设置 `model.config.use_cache = False`。
- 先冻结全模型；`--unfreeze_vision true` 时解冻 vision tower。
- LoRA target modules 覆盖 attention、MLP 和 linear-attention 相关投影。
- ImgSlot 子模块不作为 LoRA target，而是放入 `modules_to_save`，并在 PEFT 包装后显式保持 trainable。

LoRA / checkpoint 注意点：

- `adapter_model.safetensors` 保存 LoRA 权重和 `modules_to_save` 中的 ImgSlot 子模块。
- vision tower 可训练权重由 DeepSpeed checkpoint 的 `global_step*/mp_rank_00_model_states.pt` 保存；评测 adapter 会尝试从这里加载 `model.visual*` 权重。
- 修改可训练模块列表时，必须同步训练保存逻辑和评测加载逻辑。

---

## 数据与 ChatML

训练数据由 `LazySupervisedDataset` 懒加载：

- JSON 顶层必须是 list。
- 每条样本至少包含 `conversations`，可选 `image`。
- `image` 可以是字符串或字符串列表，路径相对 `image_folder`。
- `<image>` 占位会被移动到 user message 开头（当消息中恰好一个 `<image>` 且不在开头时）。
- 多图样本按 `<image>` 出现顺序依次消费；如果只有一个 `<image>` 但样本提供多张图，会把多张图的 block placeholders 连到同一个占位处。

assistant 监督模板固定为：

```text
<|im_start|>assistant
<think>

</think>

{answer}<|im_end|>
```

label 策略：

- system / user 全部 mask 为 `IGNORE_INDEX=-100`。
- assistant role prefix 和空 thinking scaffold mask。
- answer 内容和 `<|im_end|>` 参与监督。

`DataCollatorForQwen3_5SFT` 会右 padding 文本字段，截断到 `model_max_length`，并把不同样本的 `pixel_values` / `image_grid_thw` 沿视觉 patch / grid 维拼接，不做视觉 padding。

---

## 评测路径

`scripts/eval.sh` 默认运行：

```text
accelerate launch -m lmms_eval --model fovea --tasks xlrs-lite
```

`lmms_eval/models/simple/fovea.py` 当前行为：

- 加载 `FoveaForConditionalGeneration`，可选 `peft=...` 加载 LoRA adapter。
- `use_cache` 默认允许为 `True`，依赖 ImgSlot 首轮 prefill + 后续 refresh 逻辑。
- `LocalVisionImageProcessor` 复用训练侧 `VisionPacker`，并记录 `_last_image_block_counts`，用于 processor 把一张原图映射到多个 block spans。
- ImgSlot 启用时 processor 不返回 `mm_token_type_ids`；模型会走 embedding rewrite 路径。
- 当前 adapter 只支持 image inputs；video 输入会直接报错。
- `max_image_tokens` 默认 128，但 `eval.sh` 显式传 8196。

`xlrs-lite` 当前在 `lmms_eval/tasks/xlrs/mcq_utils.py` 中通过 `xlrs_process_docs()` 每个 `doc["category"]` 最多保留 60 条，任务 YAML 不使用 CLI `--limit`。答案解析会把 `(A)` 这类格式归一为 `A`。

---

## 修改原则

- 新模型入口、processor、tokenizer 代码使用 `Fovea*` 命名；仅为兼容保留 `Qwen3_5*` alias。
- `modeling_qwen3_5.py` 是基础 Qwen3.5 text/vision/backbone 实现；`modeling_fovea.py` 是 Fovea 条件生成和 ImgSlot runtime 实现。不要把 ImgSlot 主逻辑塞回基础 backbone。
- Qwen3.5/Fovea 专用评测逻辑放在 `lmms_eval/models/simple/fovea.py` 或对应 simple adapter；不要改通用 `qwen3_vl.py` 来服务 Fovea。
- 训练、评测、processor 的 ImgSlot placeholder 约定必须同步：anchor span、block span、`image_grid_thw` 行数和 `visual_pools` 数量必须一致。
- 遇到不确定的代码设计问题时，必须先询问 THX，不要自行引入兼容层或备用分支。
- 不能写兼容性代码，除非 THX 明确要求。
- 代码必须简洁，优先直接清晰的实现。避免“为了稳妥”添加未被当前输入结构使用的防御性路径。
- 涉及张量 reshape、permute、split、cat、scatter、mask、KV cache 写入时，关键变量附近必须标注形状。
- 注释只解释关键数据流、边界条件和设计意图；不要写无信息量注释。
- 修改 `img_slot_*` 默认值时，至少同步检查 `configuration_fovea.py`、`sft.py`、`VisionPacker`、`FoveaProcessor`、`lmms_eval/models/simple/fovea.py`、脚本文档和本文件。
- 修改 LoRA trainable / `modules_to_save` 时，同步检查 `sft.py` 的保存范围和 `fovea.py` 的 Deepspeed trainables 加载逻辑。
- 修改脚本可配置环境变量时，同步更新本文件的命令说明。

---

## 已知注意事项

- 根目录当前不是标准可编辑 Python 包；脚本依赖 `PYTHONPATH`。
- `configuration_fovea.py` 的类 docstring 若与类属性默认值不一致，应以类属性和实际入口为准，并在下次触碰该文件时同步修正。
- `warmup_ratio is deprecated` 是 Transformers 警告，不影响训练。
- `NCCL_DEBUG=INFO` 会产生大量日志；需要降噪时可改为 `WARN`。
- 参数打印中出现 `device=cpu` 可能发生在 DeepSpeed prepare 之前，不代表最终训练在 CPU 上。
- `PIL.Image.MAX_IMAGE_PIXELS = None` 已在训练数据路径全局关闭，用于避免超大图触发 PIL 限制。
