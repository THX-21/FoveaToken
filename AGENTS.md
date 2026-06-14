# FoveaToken 当前约定

本仓库只保留固定 fovea token 检索机制，不保留历史 visual-query replay 机制、旧视觉 slot 路径、旧 JSON SFT 或旧 block span 兼容路径。

## 机制

FoveaToken 基于 transformers 官方 Qwen3.5 多模态实现，新增逻辑集中在：

- `src/fovea_token/modeling_fovea.py`
- `src/fovea_token/train/data.py`
- `src/fovea_token/tokenizers/tokenization_fovea.py`
- `scripts/preprocess_vgr.py`
- `scripts/preprocess_pretrain_data.py`

核心数据流：

```text
<fovea>
-> Qwen3.5-style text-conditioned 256-query retrieval over the original image token memory
-> training: use GT box to crop the original image and insert one crop image after each <fovea>
-> inference: use retrieved regions to crop the original image, re-encode each crop, then insert them after <fovea>
-> continue observation / reasoning / final answer
```

256 个 fovea latent 仍是模型内置参数，不写入文本序列。普通文本 token 的 hidden states 会通过 Qwen3.5 GatedDeltaNet 风格的 `q/k/v/beta/g/z` 线性投影和 chunk gated-delta-rule 持续更新这 256 条 latent/query；`<fovea>` 位置会用它们直接检索整图的 Qwen 图像 token memory。训练时不再把检索结果作为 256 个虚拟视觉 token 插入，而是直接用 `fovea_query_boxes` 从原图裁出 crop 图并插入到 `<fovea>` 后。推理时再根据检索注意力裁原图、重走视觉塔并插入真实 crop 图。`L_align` 继续用 `<fovea>` 绑定的原图归一化 box 生成 patch overlap 目标，约束 256 个 query 的注意力整体覆盖目标区域，并加上去塌缩项。

视觉 token 默认策略：初始 LLM 图像上下文与 transformers Qwen3.5 processor/model 保持一致。Fovea 检索直接复用整图的 Qwen 图像 token，并按 `image_grid_thw` 生成归一化 patch boxes 用于 `L_align`。训练和推理插入的 crop 图会再次经过官方 processor / vision tower 编码。

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

训练 prompt 使用 checkpoint 自带的 HF `processor.apply_chat_template()` 渲染，保持和 Qwen3.5 原生推理入口一致。`<image>` 占位符在模板渲染前由 `src/fovea_token/train/data.py` 按 `image_grid_thw` 对应 token count 展开；assistant 文本中的 `<think>`/`</think>` 是本仓库新增的 atomic marker tokens。训练前会把 assistant 开头的思维块规范成 Qwen 风格 `"<think>\n...\n</think>\n\n答案"`。
训练监督规则固定为：
- `"<|im_start|>assistant\n"` 不监督。
- assistant 开头的 `"<think>\n"` 不监督，只作为前缀上下文。
- `</think>`、其后的 `"\n\n"`、最终答案正文都继续监督。
- assistant 结尾只监督 `<|im_end|>`，不监督它后面的换行，也不额外补 `eos`。

训练默认读取离线预处理结果：

```text
data/vgr/preprocessed/vgr_shortcot.parquet
data/vgr/preprocessed/vgr_longcot.parquet
```

`data_path` 指向目录时读取目录下全部 parquet；显式传入单个 parquet 文件时只读取该文件。训练 parquet 的 `image` 字段既可以是相对 `image_folder` 的字符串路径，也可以是 parquet 里的内嵌图片对象；后者会直接按字节解码。没有 `fovea_query_boxes` 的样本按普通图文 SFT 处理。固定 fovea 机制不使用离散视觉码。

## 本地权重

运行环境需要 `transformers>=5.5.0`，以提供官方 Qwen3.5 模型、配置和 processor 类。

不要自动下载模型或图像。脚本默认基模名是 `Qwen/Qwen3.5-4B`；离线运行时必须通过 `FT3_CKPT_PATH` 或 `--model_name_or_path` 指向本地 checkpoint。训练默认读取预处理后的 parquet 和本地图像：

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
- `FT3_MAX_IMG_TOKENS`
- `FT3_UNFREEZE_VISION`
- `FT3_FREEZE_EMBED_BASE`
- `FT3_DATALOADER_NUM_WORKERS`
- `FT3_REPORT_TO`

运行时需要：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

LoRA 训练如果同时开启 `unfreeze_vision=true`，vision tower 只训练 LoRA 参数；`unfreeze_vision=false` 时 vision tower 完全冻结，不额外保存全量 `vision_tower.safetensors`。全参训练（`lora_enable=false`）也遵守 `unfreeze_vision`：`true` 时全量训练 vision tower，`false` 时冻结 vision tower、其余参数继续全参训练。`freeze_embed_base=true` 时 embedding/`lm_head` 的 base vocab rows 冻结，只训练 `<fovea>`、`<think>`、`</think>` 新增 token rows；设为 `false` 时不加 row-level gradient mask。
训练时初始整图上下文默认通过官方 `max_pixels` 路径限制到 `FT3_MAX_IMG_TOKENS=2048` 个 Qwen 图像 token；`<fovea>` 裁剪图另有独立的 `fovea_crop_max_image_tokens` 上限。

## 运行环境

所有 Python 命令默认使用项目根目录下的 `.venv`：

```bash
.venv/bin/python script.py
```

不要使用其他 conda 环境或系统 python。

## 修改规则

- 不要恢复旧 visual-query replay 路径。
- 不要恢复旧视觉 slot 路径。
- 不要添加 JSON SFT 兼容读取。
- 禁止写兼容性代码和兜底代码。
- 不要恢复本地 Qwen3.5 modeling/processor 实现；Qwen3.5 基模代码直接从 transformers 导入，Fovea retrieval 只放在 `modeling_fovea.py`。
- `FoveaForConditionalGeneration` 的 `fovea_*` 是 base checkpoint 中不存在的新参数；`__init__()` 里的显式初始化不够，`from_pretrained()` 返回前还需要再次检查这些新增权重是否变成全 0 或非 finite，并只修复异常权重。
- 修改 token、训练脚本、保存模块或数据字段时，同步更新 `README.md` 和本文件。
