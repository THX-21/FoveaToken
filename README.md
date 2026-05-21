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
- `scripts/ft3.sh`：VGR 训练入口。
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

训练默认读取离线预处理后的 `data/vgr/preprocessed`，会合并：

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
- `FT3_DATALOADER_NUM_WORKERS`

注意：训练阶段不调用 IBQ codec。请先运行离线预处理，生成带 `fovea_query_boxes` 的 parquet。

## 离线预处理

```bash
PYTHONPATH="$PWD/src" python scripts/preprocess_vgr.py \
  --input data/vgr \
  --output data/vgr/preprocessed \
  --image_folder data/vgr/llava_next_raw_format
```

预处理脚本会调用本地 IBQ checkpoint，把 VGR assistant 文本里的 region tag 替换成 visual-query tokens，并把对应 box 写入 `fovea_query_boxes`。训练时只读取这些结果，不再运行 codec。

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
