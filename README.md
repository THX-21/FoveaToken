# FoveaToken

FoveaToken 是一个基于 `llava-hf/llava-v1.6-vicuna-7b-hf` 的本地多模态实验仓库。当前机制是固定 fovea token 检索：64 个内置 fovea latent 会按 Qwen3.5 GatedDeltaNet 的 q/k/v/beta/g/z 线性 SSM 规则，随普通文本 token hidden states 做 chunk-parallel recurrent 更新；模型生成或读到 `<fovea>` 后，取当前位置对应的 64 个 latent，再用 q/k/v/o 多头注意力从 pooled high-resolution visual memory 聚合出 64 个连续视觉 embedding，并把它们作为虚拟 token 插到 `<fovea>` 后继续推理。

运行训练和评测时设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

## 代码结构

- `src/fovea_token/modeling_fovea.py`：`FoveaForConditionalGeneration` 和固定 64-token fovea retrieval。
- `src/fovea_token/modeling_llava_next.py`：从 transformers 4.53.0 vendored 的 HF LLaVA-NeXT 源码。
- `src/fovea_token/train/data.py`：训练 parquet 读取、HF LLaVA-NeXT chat template 编码、collator 和 LLaVA-NeXT 图像预处理适配；同时支持本地图片路径和官方 parquet 内嵌图片字节。
- `src/fovea_token/tokenizers/tokenization_fovea.py`：`<fovea>`、`<think>`、`</think>` token helper。
- `scripts/preprocess_vgr.py`：把原始 VGR region tag 离线转换成 `<fovea>` 训练 parquet。
- `scripts/preprocess_llava_next_data.py`：把官方 `lmms-lab/LLaVA-NeXT-Data` 的 `data/train-*.parquet` 转成当前仓库可训练的本地 parquet。
- `scripts/preprocess_pretrain_data.py`：把 Visual Genome regions 转成固定 fovea grounding parquet；旧 Stage A visual-code LM 已移除。
- `scripts/visualize_fovea.py`：把 64 个 fovea token 的检索注意力画到原图上；`--task train` 读取训练 parquet，其他任务名直接读取 `lmms-eval` 样本。
- `scripts/ft3.sh`：VGR 训练入口。
- `scripts/ft3_lora.sh`：单卡 LoRA 训练入口。
- `lmms-eval/lmms_eval/models/simple/fovea.py`：本地 lmms-eval adapter。

## 本地依赖

模型权重默认入口为 `llava-hf/llava-v1.6-vicuna-7b-hf`，如需完全离线请通过 `FT3_CKPT_PATH` 指到本地 checkpoint 目录。图像数据不会自动下载，`image_folder` 必须包含 VGR `image` 字段对应的相对路径，例如 `ai2d/abc_images/311.png`。

默认图像目录：

```bash
data/vgr/llava_next_raw_format
```

## 训练

默认 LLaVA-NeXT anyres 候选分辨率与原始 LLaVA-NeXT 保持一致，训练和评测会同步写入 model config 与 processor：

```python
[
    [336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008],
    [1008, 672], [672, 1008], [1344, 672], [672, 1344], [1008, 1008],
    [1344, 1008], [1008, 1344], [1344, 1344], [1680, 1344], [1344, 1680],
]
```

视觉 token 默认压缩策略：

- 初始 LLM 图像上下文：与原始 LLaVA-NeXT 一致，不做额外 pooling。
- Fovea 视觉池：默认也不做额外 pooling，保持原始 tile 分辨率。
- Retrieval 仍然只使用 high-res unpadded 特征，不使用 base image tokens 和 newline tokens。
- 推理时默认会在 assistant 开始回答处自动触发一次内部 `<fovea>` 检索；若开启 `enable_thinking=true`，触发位置是 `<think>` 之后而不是之前。可用 `fovea_auto_retrieve_on_answer_start=false` 关闭。

默认训练：

```bash
bash scripts/ft3.sh
```

单卡 LoRA：

```bash
bash scripts/ft3_lora.sh
```

训练默认读取离线预处理后的 `data/vgr/preprocessed`，目录模式会读取该目录下全部 parquet。也支持直接读取官方 `LLaVA-NeXT-Data` 的 `data/train-*.parquet`：如果 `image` 列里是内嵌图片对象就直接解码；如果是字符串路径就按 `image_folder` 读取。没有 `fovea_query_boxes` 的样本会按普通图文 SFT 处理。常用环境变量：

- `FT3_DATA_PATH`
- `FT3_IMAGE_FOLDER`
- `FT3_CKPT_PATH`
- `FT3_OUTPUT_DIR`
- `FT3_NUM_TRAIN_EPOCHS`
- `FT3_MAX_STEPS`
- `FT3_UNFREEZE_VISION`
- `FT3_FREEZE_EMBED_BASE`
- `FT3_DATALOADER_NUM_WORKERS`
- `FT3_REPORT_TO`

`scripts/ft3_lora.sh` 默认启用 LoRA；`FT3_UNFREEZE_VISION=true` 时只训练 vision tower LoRA，设为 `false` 时 vision tower 完全冻结。`FT3_FREEZE_EMBED_BASE=true` 时冻结 embedding/`lm_head` 的 base vocab rows，只训练 `<fovea>`、`<think>`、`</think>` 新增 token rows。

训练 prompt 使用 checkpoint 自带的 HF `processor.apply_chat_template()` 渲染，和原生 LLaVA-NeXT 推理入口保持一致。图像 `<image>` 占位符仍由本仓库按 pooled image token 数提前展开，避免和默认 processor 图像展开逻辑混用；assistant 内容中的 `<think>`/`</think>` 是本仓库新增的 atomic marker tokens。LLaVA-NeXT 原生没有 thinking chat-template 开关；评测时 `enable_thinking=true` 会在 assistant 生成前预填 `<think>`，否则预填 `<think>\n\n</think>`。
训练编码阶段如果 chat template 本身没有在 assistant 结尾给出 `eos`，会额外补一个 `eos_token_id`，并把它纳入 LM 监督。

## 离线预处理

```bash
PYTHONPATH="$PWD/src" python scripts/preprocess_vgr.py \
  --input data/vgr \
  --output data/vgr/preprocessed \
  --image_folder data/vgr/llava_next_raw_format
```

预处理会把 VGR assistant 文本里的：

```text
<SOT>[x1, y1, x2, y2]<EOT><image>
```

转换成：

```text
<fovea>
```

并把对应 box 写入 `fovea_query_boxes`。assistant 里不属于合法 region tag 的残留 `<image>`、孤立 `<SOT>`、孤立 `<EOT>` 会在这一步一起清掉。固定 fovea 机制不使用离散视觉码。

Visual Genome region grounding 样本：

```bash
PYTHONPATH="$PWD/src" python scripts/preprocess_pretrain_data.py \
  --output data/visual_genome/vg_regions_sample.parquet \
  --vg_region_descriptions /path/to/region_descriptions.json.zip \
  --vg_image_data /path/to/image_data.json.zip \
  --vg_image_root /path/to/visual_genome_images \
  --max_samples 8
```

把官方 `LLaVA-NeXT-Data` 约 `779k` 的 `data/train-*.parquet` 转成本仓库训练格式：

```bash
python scripts/preprocess_llava_next_data.py \
  --input lmms-lab/LLaVA-NeXT-Data \
  --output data/llava_next/preprocessed \
  --image_folder data/vgr/llava_next_raw_format \
  --processed_json data/vgr/llava_next_raw_format/llava_next_raw_format_processed.json
```

说明：

- 官方 `data/train-*.parquet` 里的 `image` 是内嵌图片对象，不是本仓库训练时直接读取的相对路径。
- 本脚本会优先用 `llava_next_raw_format_processed.json` 的 `id -> image` 映射恢复相对路径；缺失时再按图片 basename 在 `image_folder` 里回查。
- 转换后的样本会写成 `fovea_preprocessed=true`、`fovea_query_boxes=[]` 的普通图文 SFT 样本，不包含 `<fovea>` 框监督。
- 如果只想做冒烟测试，可以加 `--limit_shards 1`。

如果不想转换，也可以直接把官方 `data/train-*.parquet` 传给 `data_path`。当前训练读取逻辑已经支持 parquet 内嵌图片字节。

## 训练目标

训练 forward 是固定两次 decoder pass：

1. Pass A：正常多模态前向，得到所有 token 的最后 hidden states。
2. GatedDelta update：构造 `[B * 64, T, H]` latent/input 序列，投影出 `q/k/v/beta/g/z`，使用 Qwen3.5 的 chunk gated-delta-rule 更新 KV recurrent state 并得到每个位置的 64 个 fovea latent。
3. Retrieval：在 `<fovea>` 位置 gather 当前 64 个 latent，通过 q/k/v/o 多头检索 pooled high-res visual memory，得到 64 个虚拟视觉 embedding 与 `L_align`。`L_align` 用 GT box 的 patch overlap 分布监督 64 个 query 的区域覆盖，并加上 query 间去塌缩项。
4. Pass B：把 64 个虚拟 token 插到对应 `<fovea>` 后，扩展 attention mask 和 labels，再计算文本 LM loss。插入的虚拟 token labels 为 `IGNORE_INDEX`。

总损失：

```text
L = L_text + fovea_lambda_align * L_align
```

## 评测

```bash
bash scripts/eval.sh
```

默认使用 `--model fovea`。评测其他基线模型使用：

```bash
bash scripts/eval_others.sh
```

## 可视化

训练样本：

```bash
PYTHONPATH="$PWD/src:$PWD/lmms-eval" python scripts/visualize_fovea.py \
  --task train \
  --model_name_or_path checkpoints/fovea-vgr-ft/checkpoint-2400 \
  --index 0
```

`lmms-eval` 样本，例如 `mmstar`：

```bash
PYTHONPATH="$PWD/src:$PWD/lmms-eval" python scripts/visualize_fovea.py \
  --task mmstar \
  --model_name_or_path checkpoints/fovea-vgr-ft/checkpoint-2400 \
  --index 0 \
  --enable_thinking
```

输出默认写到：

```bash
outputs/fovea_visualize/
```
