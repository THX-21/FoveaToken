# FoveaToken 当前约定

本仓库只保留 visual-query replay 机制，不保留历史机制、旧 JSON SFT 或旧 block span 兼容路径。

## 机制

FoveaToken 基于 Qwen3.5 多模态实现，新增逻辑集中在：

- `src/qwen35_hf/modeling_fovea.py`
- `src/qwen35_hf/train/data.py`
- `src/qwen35_hf/visual_codec.py`
- `src/qwen35_hf/query_tokens.py`

核心数据流：

```text
<vq> <vis_i> ... </vq>
-> code hidden states
-> packed multi-head retrieval over high-resolution visual memory
-> replay vectors scattered into <|replay_pad|> positions
-> observation / reasoning / final answer
```

## 数据

训练数据只使用 VGR parquet：

```text
data/vgr/vgr_shortcot.parquet
data/vgr/vgr_longcot.parquet
```

VGR 中的：

```text
<SOT>[x1, y1, x2, y2]<EOT><image>
```

会转换为：

```text
<vq> <vis_i> ... </vq> <|replay_pad|> ...
```

每个 `<vq>` 必须绑定一个原图归一化 box，用于 `L_align`。图像路径相对 `image_folder`。

## 本地权重

不要自动下载模型或图像。训练默认读取项目内本地路径：

```bash
models/Open-MAGVIT2
models/Open-MAGVIT2/configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_16384.yaml
models/Open-MAGVIT2/tokenizer_16384.pt
data/llava_next_raw_format
```

需要换路径时，通过训练参数 `--magvit2_repo`、`--magvit2_config`、`--magvit2_checkpoint`、`--image_folder` 显式传入。Open-MAGVIT2 必须是真实本地 tokenizer 16384 checkpoint，不使用伪码。

## 训练入口

默认入口：

```bash
bash scripts/ft3.sh
```

常用环境变量：

- `FT3_DATA_PATH`
- `FT3_IMAGE_FOLDER`
- `FT3_CKPT_PATH`
- `FT3_OUTPUT_DIR`
- `FT3_MAGVIT2_REPO`
- `FT3_MAGVIT2_CONFIG`
- `FT3_MAGVIT2_CHECKPOINT`
- `FT3_GENERATED_REPLAY_PROB`
- `FT3_RETRIEVE_MAX_IMAGE_TOKENS`
- `FT3_VISUAL_CODE_CACHE_DIR`

运行时需要：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

## 修改规则

- 不要恢复旧视觉 slot 路径。
- 不要添加 JSON SFT 兼容读取。
- 不要在 `modeling_qwen3_5.py` 里加入 Fovea retrieval 逻辑。
- 修改 token、训练脚本、保存模块或数据字段时，同步更新 `README.md` 和本文件。
