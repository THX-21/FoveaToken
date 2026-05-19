# FoveaToken

FoveaToken 是一个本地 Qwen3.5 多模态实验仓库。当前只保留 visual-query replay 机制：模型生成 `<vq> <vis_i> ... </vq>` 离散视觉查询码，训练时用这些 code hidden states 对高分辨率 visual memory 做 multi-head retrieval，并把检索到的连续 replay vectors 写回 `</vq>` 后的 replay placeholder。

仓库不作为标准 editable package 使用，运行训练和评测时设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

## 代码结构

- `src/fovea_token/modeling_fovea.py`：`FoveaForConditionalGeneration` 和 packed visual-query retrieval/replay。
- `src/fovea_token/train/data.py`：VGR parquet 读取、ChatML 编码、IBQ code 替换、collator。
- `src/fovea_token/train/image_packing.py`：Qwen vision patch packing 与 retrieval token box 生成。
- `src/fovea_token/tokenizers/tokenization_ibq.py`：本地 IBQ 视觉码 tokenizer 与 cache。
- `src/fovea_token/tokenizers/tokenization_visual_query.py`：`<vq>`、`</vq>`、`<vis_i>`、`<|replay_pad|>` token helpers。
- `scripts/ft3.sh`：VGR 训练入口。
- `lmms-eval/lmms_eval/models/simple/fovea.py`：本地 lmms-eval adapter。

## 本地依赖

训练不会下载图像、Qwen 权重或 IBQ 权重。默认从项目内本地路径读取：

```bash
src/fovea_token/models/Open-MAGVIT2
src/fovea_token/models/Open-MAGVIT2/configs/IBQ/gpu/pretrain_ibqgan_16384.yaml
src/fovea_token/models/Open-MAGVIT2/IBQ_pretrain_16384.ckpt
data/vgr/llava_next_raw_format
```

也可以在训练命令中显式传入：

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

默认数据路径是 `data/vgr/data`，会合并：

```text
data/vgr/data/vgr_shortcot.parquet
data/vgr/data/vgr_longcot.parquet
```

常用环境变量：

- `FT3_DATA_PATH`
- `FT3_IMAGE_FOLDER`
- `FT3_CKPT_PATH`
- `FT3_OUTPUT_DIR`
- `FT3_NUM_TRAIN_EPOCHS`
- `FT3_MAX_STEPS`
- `FT3_IBQ_REPO`
- `FT3_IBQ_CONFIG`
- `FT3_IBQ_CHECKPOINT`
- `FT3_GENERATED_REPLAY_PROB`
- `FT3_RETRIEVE_MAX_IMAGE_TOKENS`
- `FT3_VISUAL_CODE_CACHE_DIR`
- `FT3_DATALOADER_NUM_WORKERS`

注意：当前 VGR 训练会在取样阶段调用本地 IBQ codec 编码 crop。默认使用 `FT3_DATALOADER_NUM_WORKERS=0`，避免 DataLoader worker 中初始化 CUDA 导致失败。

## 数据转换

VGR assistant 文本中的区域标注：

```text
<SOT>[x1, y1, x2, y2]<EOT><image>
```

会在懒加载时转换为：

```text
<vq> <vis_i> ... </vq> <|replay_pad|> ...
```

每个 `<vq>` 绑定对应的归一化原图 box，用于 `L_align`。视觉码由本地 IBQ codec 真实生成，并缓存到 `FT3_VISUAL_CODE_CACHE_DIR`。assistant 里不属于合法 region tag 的残留 `<image>`、孤立 `<SOT>`、孤立 `<EOT>` 会在这一步一起清掉。

## 训练目标

训练 forward 是固定两次 decoder pass：

1. Pass A：masked teacher-forcing visual codes，计算 visual-code LM loss，并收集 `<vq>` code hidden states。
2. Packed retrieval：批量检索高分辨率 visual memory，得到 replay vectors 与 `L_align`。默认一半 batch 使用 Pass A 预测出的 generated visual codes 替换 Pass B 的 code 输入。
3. Pass B：scatter replay vectors，计算文本 loss。visual-code loss 与文本 loss 在一次 mixed LM loss 中合并计算。

总损失：

```text
L = L_code + L_text + visual_query_lambda_align * L_align
```

## 评测

```bash
bash scripts/eval.sh
```

默认使用 `--model fovea`、`max_image_tokens=512`、任务 `xlrs-lite`。
