# FoveaToken

FoveaToken 是一个本地 Qwen3.5 多模态实验仓库。当前只保留 visual-query replay 机制：模型生成 `<vq> <vis_i> ... </vq>` 离散视觉查询码，训练时用这些 code hidden states 对高分辨率 visual memory 做 multi-head retrieval，并把检索到的连续 replay vectors 写回 `</vq>` 后的 replay placeholder。

仓库不作为标准 editable package 使用，运行训练和评测时设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

## 代码结构

- `src/fovea_token/modeling_fovea.py`：`FoveaForConditionalGeneration` 和 packed visual-query retrieval/replay。
- `src/fovea_token/train/data.py`：离线预处理后的 VGR parquet 读取、ChatML 编码、collator。
- `src/fovea_token/train/image_packing.py`：Qwen vision patch packing 与 retrieval token box 生成。
- `src/fovea_token/tokenizers/tokenization_ibq.py`：离线 VGR 预处理使用的本地 IBQ 视觉码 tokenizer。
- `src/fovea_token/tokenizers/tokenization_visual_query.py`：`<vq>`、`</vq>`、`<vis_i>`、`<|replay_pad|>` token helpers。
- `scripts/preprocess_vgr.py`：把原始 VGR parquet 离线转换成训练 parquet。
- `scripts/preprocess_pretrain_data.py`：把 COCO captions / Visual Genome regions 转成 Stage A/B visual-token 预训练 parquet。
- `scripts/ft3.sh`：VGR 训练入口。
- `scripts/ft3_lora.sh`：单卡 LoRA 训练入口，不使用 DeepSpeed CPU optimizer offload。
- `lmms-eval/lmms_eval/models/simple/fovea.py`：本地 lmms-eval adapter。

## 本地依赖

训练不会下载图像、Qwen 权重或 IBQ 权重。离线预处理默认从项目内本地路径读取 IBQ，训练默认读取预处理后的 parquet 和本地图像：

```bash
src/fovea_token/models/Open-MAGVIT2
src/fovea_token/models/Open-MAGVIT2/configs/IBQ/gpu/pretrain_ibqgan_16384.yaml
src/fovea_token/models/Open-MAGVIT2/IBQ_pretrain_16384.ckpt
data/vgr/llava_next_raw_format
```

也可以在预处理命令中显式传入：

```bash
--ibq_repo /path/to/local/SEED-Voken \
--ibq_config /path/to/local/pretrain_ibqgan_16384.yaml \
--ibq_checkpoint /path/to/local/IBQ_pretrain_16384.ckpt \
--image_folder /path/to/llava_next_raw_format
```

图像数据不会自动下载。`image_folder` 必须包含 VGR `image` 字段对应的相对路径，例如 `ai2d/abc_images/311.png`。

## 训练

```bash
bash scripts/ft3.sh
```

如果只想跑单卡、避免 DeepSpeed `ZeRO-2 + CPU offload` 带来的慢速 optimizer step，可以使用：

```bash
bash scripts/ft3_lora.sh
```

训练默认读取离线预处理后的 `data/vgr/preprocessed`，目录模式会读取该目录下全部 parquet。VGR 默认包含：

```text
data/vgr/preprocessed/vgr_shortcot.parquet
data/vgr/preprocessed/vgr_longcot.parquet
```

常用环境变量：

- `FT3_DATA_PATH`
- `FT3_IMAGE_FOLDER`
- `FT3_CKPT_PATH`
- `FT3_OUTPUT_DIR`
- `FT3_NUM_TRAIN_EPOCHS`
- `FT3_MAX_STEPS`
- `FT3_GENERATED_REPLAY_PROB`
- `FT3_RETRIEVE_MAX_IMAGE_TOKENS`
- `FT3_UNFREEZE_VISION`
- `FT3_FREEZE_EMBED_BASE`
- `FT3_DATALOADER_NUM_WORKERS`
- `FT3_REPORT_TO`

注意：训练阶段不调用 IBQ codec。请先运行离线预处理，生成带 `fovea_query_boxes` 的 parquet。

`scripts/ft3_lora.sh` 默认：

- 启用 LoRA
- `FT3_UNFREEZE_VISION=true` 时只训练 vision tower LoRA；设为 `false` 时 vision tower 完全冻结
- `FT3_FREEZE_EMBED_BASE=true` 时冻结 embedding/`lm_head` 的 base vocab rows，只训练 visual-query token rows
- 不启用 DeepSpeed
- `FT3_DATALOADER_NUM_WORKERS=4`
- `FT3_REPORT_TO=none`

这样更接近单卡可持续训练的快速配置，显存和 optimizer state 压力会明显小于全参数版本。

LoRA 训练时不会全量更新 vision tower：`--unfreeze_vision true` 只会让 vision tower 内匹配的 attention/MLP LoRA 参数可训；`--unfreeze_vision false` 时 vision tower 没有可训参数。默认 `--freeze_embed_base true` 会冻结输入 embedding 和 `lm_head` 的 base vocab rows，只允许 visual-query 新增 token rows 更新；需要让整块 embedding/`lm_head` 正常更新时设为 `false`。

全参训练（`--lora_enable false`）同样遵守 `--unfreeze_vision`：设为 `true` 时 vision tower 参与全参训练并使用 `--vision_tower_lr`，设为 `false` 时只冻结 vision tower，其余模型参数仍按全参训练。

## 离线预处理

```bash
PYTHONPATH="$PWD/src" python scripts/preprocess_vgr.py \
  --input data/vgr \
  --output data/vgr/preprocessed \
  --image_folder data/vgr/llava_next_raw_format
```

预处理脚本会调用本地 IBQ checkpoint，把 VGR assistant 文本里的 region tag 替换成 visual-query tokens，并把对应 box 写入 `fovea_query_boxes`。训练时只读取这些结果，不再运行 codec。

### Stage A/B visual-token 预训练数据

`scripts/preprocess_pretrain_data.py` 支持两个额外离线预处理模式：

- `--stage coco_a`：读取 `Multimodal-Fatima/COCO_captions_train`，把整图 IBQ code 写成 image-conditioned code LM 样本。user 包含 `<image>` 和 caption 描述，assistant 只包含 `<vq> <vis_i> ... </vq>`，不包含 `<|replay_pad|>`，训练时使用图像 embedding 和普通 LM loss，不触发 retrieval。
- `--stage vg_b`：读取 Visual Genome `region_descriptions.json(.zip)` 和本地图像目录，把 region phrase + bbox crop 转成 VGR-style visual-query replay 定位样本。user 询问 caption 对应图中什么位置，assistant 以 `caption + <vq> <vis_i> ... </vq><|replay_pad|>... + bbox` 的形式回答，并写入 `fovea_query_boxes`。该阶段额外写入 `fovea_supervised_substrings`，只监督 `<vq> <vis_i> ... </vq>` 和 bbox 坐标；`<|replay_pad|>` 不写入监督子串，训练 labels 固定为 `IGNORE_INDEX`。

Stage A 默认 `--stage_a_max_visual_tokens 0`，表示 IBQ 编码不限制 visual code 数；Stage B 默认 `--stage_b_max_visual_tokens 256`。生成时 `visual_query_max_codes` 默认也是 256。

小样本 Stage A 测试：

```bash
PYTHONPATH="$PWD/src" python scripts/preprocess_pretrain_data.py \
  --stage coco_a \
  --output data/stage_a/coco_sample.parquet \
  --streaming \
  --max_samples 8
```

训练 Stage A 时，`FT3_IMAGE_FOLDER` 应指向 `data/stage_a`，因为预处理会把临时源图保存到输出目录旁的 `_stage_a_source_images/`。

Stage B 需要先在服务器准备 Visual Genome 图片和标注，例如 `images.zip`/`images2.zip` 解压后的图片目录，以及 `region_descriptions.json.zip`、`image_data.json.zip`：

```bash
PYTHONPATH="$PWD/src" python scripts/preprocess_pretrain_data.py \
  --stage vg_b \
  --output data/stage_b/vg_regions_sample.parquet \
  --vg_region_descriptions /path/to/region_descriptions.json.zip \
  --vg_image_data /path/to/image_data.json.zip \
  --vg_image_root /path/to/visual_genome_images \
  --max_samples 8
```

## 数据转换

VGR assistant 文本中的区域标注：

```text
<SOT>[x1, y1, x2, y2]<EOT><image>
```

会在离线预处理时转换为：

```text
<vq> <vis_i> ... </vq> <|replay_pad|> ...
```

每个 `<vq>` 绑定对应的归一化原图 box，用于 `L_align`。视觉码由本地 IBQ codec 在预处理阶段真实生成。assistant 里不属于合法 region tag 的残留 `<image>`、孤立 `<SOT>`、孤立 `<EOT>` 会在这一步一起清掉。

## 训练目标

训练 forward 是固定两次 decoder pass：

1. Pass A：masked teacher-forcing visual codes，计算 visual-code LM loss，并收集 `<vq>` code hidden states。
2. Packed retrieval：批量检索高分辨率 visual memory，得到 replay vectors 与 `L_align`。默认一半 batch 使用 Pass A 预测出的 generated visual codes 替换 Pass B 的 code 输入。
3. Pass B：scatter replay vectors，计算文本 loss。`<|replay_pad|>` 只承载 replay vectors，labels 固定为 `IGNORE_INDEX`，不参与 LM loss。visual-code loss 与文本 loss 在一次 mixed LM loss 中合并计算。

总损失：

```text
L = L_code + L_text + visual_query_lambda_align * L_align
```

## 评测

```bash
bash scripts/eval.sh
```

默认使用 `--model fovea`、`max_image_tokens=512`、任务 `xlrs-lite`。

评测其他基线模型使用：

```bash
bash scripts/eval_others.sh
```

默认使用 `--model qwen3_5`、任务 `xlrs-lite`。可通过 `EVAL_MODEL`、`EVAL_BASE_MODEL`、`EVAL_LIMIT`、`EVAL_FORCE_SIMPLE` 等环境变量覆盖。
