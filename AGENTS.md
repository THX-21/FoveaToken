# FoveaToken 当前约定

本仓库只保留固定 fovea token 检索机制，不保留历史 visual-query replay 机制、旧视觉 slot 路径、旧 JSON SFT 或旧 block span 兼容路径。

## 机制

FoveaToken 基于 `llava-hf/llava-v1.6-vicuna-7b-hf` / LLaVA-NeXT 多模态实现，新增逻辑集中在：

- `src/fovea_token/modeling_fovea.py`
- `src/fovea_token/modeling_llava_next.py`
- `src/fovea_token/train/data.py`
- `src/fovea_token/tokenizers/tokenization_fovea.py`
- `scripts/preprocess_vgr.py`
- `scripts/preprocess_pretrain_data.py`

核心数据流：

```text
<fovea>
-> first decoder pass hidden states for normal text tokens
-> Qwen3.5-style chunk gated-delta-rule updates 64 learned fovea latent streams
-> q/k/v/o multi-head retrieval over pooled high-resolution visual memory
-> insert 64 virtual visual embeddings after <fovea>
-> second decoder pass for observation / reasoning / final answer
```

64 个 fovea latent 是模型内置参数，不写入文本序列。普通文本 token 的 hidden states 会通过 Qwen3.5 GatedDeltaNet 风格的 `q/k/v/beta/g/z` 线性投影和 chunk gated-delta-rule 持续更新这 64 条 latent stream；只有检测到 `<fovea>` 时，才把当前位置对应的 64 个 latent 拿去视觉池检索并插入 64 个虚拟视觉 token。训练 labels 对这些插入的虚拟 token 固定为 `IGNORE_INDEX`。`L_align` 使用 `<fovea>` 绑定的原图归一化 box 监督 64 个 token 的检索注意力落到目标区域内。

视觉 token 默认压缩策略：初始 LLM 图像上下文中，base image tokens 经过 `2x2` pooling，high-res unpadded tokens 经过 `4x4` pooling，并保留 pooled high-res 每行 newline token；Fovea 视觉池不使用 base image tokens 和 newline tokens，只保留 high-res unpadded tokens，并统一经过 `2x2` pooling。

## 数据

原始数据只使用 VGR parquet：

```text
data/vgr/data/vgr_shortcot.parquet
data/vgr/data/vgr_longcot.parquet
```

VGR 中的：

```text
<SOT>[x1, y1, x2, y2]<EOT><image>
```

会转换为：

```text
<fovea>
```

每个 `<fovea>` 必须绑定一个原图归一化 box，用于 `L_align`。图像路径相对 `image_folder`。assistant 文本里不属于合法 region tag 的残留 `<image>`、孤立 `<SOT>`、孤立 `<EOT>` 会在离线预处理阶段直接清掉，不再参与 placeholder 展开。

训练 prompt 使用 checkpoint 自带的 HF `processor.apply_chat_template()` 渲染，保持和 LLaVA-NeXT 原生推理入口一致。由于本仓库使用自定义 pooled image token 数，`<image>` 占位符仍在模板渲染前由 `src/fovea_token/train/data.py` 按本地 token count 展开；assistant 文本中的 `<think>`/`</think>` 是本仓库新增的 atomic marker tokens。LLaVA-NeXT 原生没有 thinking chat-template 开关；评测时 `enable_thinking=true` 会在 assistant 生成前预填 `<think>`，否则预填 `<think>\n\n</think>`。

训练默认读取离线预处理结果：

```text
data/vgr/preprocessed/vgr_shortcot.parquet
data/vgr/preprocessed/vgr_longcot.parquet
```

`data_path` 指向目录时读取目录下全部 parquet；显式传入单个 parquet 文件时只读取该文件。固定 fovea 机制不使用离散视觉码。

LLaVA-NeXT anyres 候选分辨率使用扩展范围：

```python
[
    [336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008],
    [672, 1344], [1344, 672], [1344, 1344], [2016, 672], [672, 2016],
]
```

修改该列表时必须同步 model config 和 processor 的 `image_grid_pinpoints`，否则 `get_image_features()` 的 split/pack 与实际 tile preprocessing 会错位。

## 本地权重

不要自动下载模型或图像。脚本默认基模名是 `llava-hf/llava-v1.6-vicuna-7b-hf`；离线运行时必须通过 `FT3_CKPT_PATH` 或 `--model_name_or_path` 指向本地 checkpoint。训练默认读取预处理后的 parquet 和本地图像：

```bash
data/vgr/llava_next_raw_format
```

## 训练入口

默认入口：

```bash
bash scripts/ft3.sh
```

单卡快速入口：

```bash
bash scripts/ft3_lora.sh
```

双卡 DeepSpeed 冒烟测试：

```bash
bash scripts/test_deepspeed_2gpu.sh
```

常用环境变量：

- `FT3_DATA_PATH`
- `FT3_IMAGE_FOLDER`
- `FT3_CKPT_PATH`
- `FT3_OUTPUT_DIR`
- `FT3_UNFREEZE_VISION`
- `FT3_FREEZE_EMBED_BASE`
- `FT3_DATALOADER_NUM_WORKERS`
- `FT3_REPORT_TO`

运行时需要：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

LoRA 训练如果同时开启 `unfreeze_vision=true`，vision tower 只训练 LoRA 参数；`unfreeze_vision=false` 时 vision tower 完全冻结，不额外保存全量 `vision_tower.safetensors`。全参训练（`lora_enable=false`）也遵守 `unfreeze_vision`：`true` 时全量训练 vision tower，`false` 时冻结 vision tower、其余参数继续全参训练。`freeze_embed_base=true` 时 embedding/`lm_head` 的 base vocab rows 冻结，只训练 `<fovea>`、`<think>`、`</think>` 新增 token rows；设为 `false` 时不加 row-level gradient mask。

## 修改规则

- 不要恢复旧 visual-query replay 路径。
- 不要恢复旧视觉 slot 路径。
- 不要添加 JSON SFT 兼容读取。
- 禁止写兼容性代码和兜底代码。
- 不要在 `modeling_llava_next.py` 里加入 Fovea retrieval 逻辑；该文件应保持为 HF LLaVA-NeXT 源码搬运版本，Fovea retrieval 只放在 `modeling_fovea.py`。
- `FoveaForConditionalGeneration` 的 `fovea_*` 是 base checkpoint 中不存在的新参数；`__init__()` 里的显式初始化不够，`from_pretrained()` 返回前还需要再次检查这些新增权重是否变成全 0 或非 finite，并只修复异常权重。
- 修改 token、训练脚本、保存模块或数据字段时，同步更新 `README.md` 和本文件。
