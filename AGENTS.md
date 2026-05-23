# FoveaToken 当前约定

本仓库只保留 visual-query replay 机制，不保留历史机制、旧 JSON SFT 或旧 block span 兼容路径。

## 机制

FoveaToken 基于 Qwen3.5 多模态实现，新增逻辑集中在：

- `src/fovea_token/modeling_fovea.py`
- `src/fovea_token/train/data.py`
- `src/fovea_token/tokenizers/tokenization_ibq.py`
- `src/fovea_token/tokenizers/tokenization_visual_query.py`
- `scripts/preprocess_vgr.py`

核心数据流：

```text
<vq> <vis_i> ... </vq>
-> code hidden states
-> packed multi-head retrieval over high-resolution visual memory
-> replay vectors scattered into <|replay_pad|> positions
-> observation / reasoning / final answer
```

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
<vq> <vis_i> ... </vq> <|replay_pad|> ...
```

每个 `<vq>` 必须绑定一个原图归一化 box，用于 `L_align`。图像路径相对 `image_folder`。assistant 文本里不属于合法 region tag 的残留 `<image>`、孤立 `<SOT>`、孤立 `<EOT>` 会在离线预处理阶段直接清掉，不再参与 placeholder 展开。
`<|replay_pad|>` 只作为 replay vector 占位，训练 labels 必须设为 `IGNORE_INDEX`，不参与 LM loss。

训练默认读取离线预处理结果：

```text
data/vgr/preprocessed/vgr_shortcot.parquet
data/vgr/preprocessed/vgr_longcot.parquet
```

训练阶段不调用 IBQ codec；IBQ 只允许在 `scripts/preprocess_vgr.py` 离线预处理阶段使用。
离线预处理不使用视觉码缓存；每次运行都会重新调用 IBQ 对每个 crop 编码。

## 本地权重

不要自动下载模型或图像。离线预处理默认读取项目内本地 IBQ 路径，训练默认读取预处理后的 parquet 和本地图像：

```bash
src/fovea_token/models/Open-MAGVIT2
src/fovea_token/models/Open-MAGVIT2/configs/IBQ/gpu/pretrain_ibqgan_16384.yaml
src/fovea_token/models/Open-MAGVIT2/IBQ_pretrain_16384.ckpt
data/vgr/llava_next_raw_format
```

需要换路径时，通过预处理参数 `--ibq_repo`、`--ibq_config`、`--ibq_checkpoint`、`--image_folder` 显式传入。IBQ 必须是真实本地 16384 checkpoint，不使用伪码。

## 训练入口

默认入口：

```bash
bash scripts/ft3.sh
```

单卡快速入口：

```bash
bash scripts/ft3_lora.sh
```

常用环境变量：

- `FT3_DATA_PATH`
- `FT3_IMAGE_FOLDER`
- `FT3_CKPT_PATH`
- `FT3_OUTPUT_DIR`
- `FT3_GENERATED_REPLAY_PROB`
- `FT3_RETRIEVE_MAX_IMAGE_TOKENS`
- `FT3_DATALOADER_NUM_WORKERS`
- `FT3_REPORT_TO`

运行时需要：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

LoRA 训练如果同时开启 `unfreeze_vision=true`，checkpoint 目录需要连同额外的 `vision_tower.safetensors` 一起保留，恢复时会自动加载。

## 修改规则

- 不要恢复旧视觉 slot 路径。
- 不要添加 JSON SFT 兼容读取。
- 不要在 `modeling_qwen3_5.py` 里加入 Fovea retrieval 逻辑。
- `FoveaForConditionalGeneration` 的 `visual_query_*` 是 base checkpoint 中不存在的新参数；`__init__()` 里的显式初始化不够，`from_pretrained()` 返回前还需要再次检查这些新增 projection 权重是否变成全 0 或非 finite，并只修复异常权重。
- 修改 token、训练脚本、保存模块或数据字段时，同步更新 `README.md` 和本文件。
