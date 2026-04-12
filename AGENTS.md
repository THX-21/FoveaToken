# AGENTS.md

## 项目结构

- 当前工作目录是 `/root/wd/FoveaToken/qwen35_hf`，包含本地 Qwen3.5 HF 实现、训练脚本和 lmms-eval 适配。
- `src/qwen35_hf/` 存放模型、配置、tokenizer、processor 和训练代码。
- `scripts/` 存放训练/评测脚本。当前评测脚本在 `scripts/eval.sh`，不要在仓库根目录直接找 `eval.sh`。
- `lmms-eval/` 是本地 lmms-eval 副本，当前新增了 `qwen35_hf` 模型适配器。
- `checkpoints/` 和 `logs/` 是训练/评测产物目录，通常不要提交。

## 常用命令

从项目根目录执行：

```bash
cd /root/wd/FoveaToken/qwen35_hf
bash scripts/ft3.sh
```

运行评测：

```bash
cd /root/wd/FoveaToken/qwen35_hf
bash scripts/eval.sh
```

语法检查：

```bash
bash -n scripts/ft3.sh
bash -n scripts/eval.sh
/root/wd/FoveaToken/.venv/bin/python -m py_compile src/qwen35_hf/train/sft.py
/root/wd/FoveaToken/.venv/bin/python -m py_compile src/qwen35_hf/train/data.py
```

## 训练约定

- `scripts/ft3.sh` 使用 `torchrun` 启动，即使单卡也走 torch distributed，避免 Deepspeed 回退到 MPI。
- `scripts/ft3.sh` 会自动查找 `OUTPUT_DIR` 下最新的 `checkpoint-*` 并传入 `--resume_from_checkpoint` 继续训练。
- 默认训练基座是 `Qwen/Qwen3.5-9B`，默认数据路径是 `/mnt/data/GeoLLaVA-Data/ft3_whole_shuffle.json`。
- 当前训练是 LoRA + 解冻 vision tower：`--lora_enable true` 且 `--unfreeze_vision true`。
- `processor_backend` 当前脚本里是 `local`。
- `image_aspect_ratio` 当前脚本里是 `normal`；只有设置为 `anyres` 或包含 `anyres` 时，训练代码才会强制走本地 `pack_anyres_image` 路径。
- `scripts/ft3.sh` 当前传入 `--max_image_tokens 2048`。
- `max_image_tokens` 在本地 anyres 路径里表示最终图像 token 预算，不是旧的 tile 数。

## 数据和模板

- 训练编码入口是 `src/qwen35_hf/train/data.py` 的 `encode_chatml_example`。
- 当前训练模板手写 ChatML，并已对齐 Qwen3.5 官方 assistant 空 thinking scaffold：

```text
<|im_start|>assistant
<think>

</think>

answer<|im_end|>
```

- 图片占位会在数据编码阶段按 `image_token_counts` 展开为多个 `<|image_pad|>`，最终要和 `image_grid_thw.prod() // spatial_merge_size**2` 一致。
- 数据集中存在“多图但只有一个 `<image>` 文本占位”的样本，`replace_image_tokens_in_conversations` 已做兼容，会把单个占位展开成多个连续视觉占位。
- PIL 超大图检查已在 `data.py` 中关闭：`Image.MAX_IMAGE_PIXELS = None`。

## 图像处理

- 本地图像打包代码在 `src/qwen35_hf/train/image_packing.py`。
- `anyres` 路径使用 `pack_anyres_image`，会根据 `grid_pinpoints`、`patch_size`、`spatial_merge_size` 和 `max_image_tokens` 选择合法分辨率。
- 图像最终 token 数计算：

```text
image_tokens = image_grid_thw.prod() // spatial_merge_size**2
```

- `image_grid_thw = [1, 84, 84]` 对应 `1344x1344` canvas，最终图像 token 数是 `1764`。

## 评测约定

- `scripts/eval.sh` 使用 lmms-eval 的 `--model qwen35_hf`。
- `lmms-eval/lmms_eval/models/simple/qwen35_hf.py` 是本地适配器，直接调用：
  - `qwen35_hf.Qwen3_5ForConditionalGeneration`
  - `qwen35_hf.Qwen3_5Tokenizer`
  - `qwen35_hf.Qwen3VLProcessor`
- 评测适配器用 `VisionPacker` 和 `LocalVisionImageProcessor` 复用本地训练侧图像打包逻辑。
- `scripts/eval.sh` 当前实际 `--model_args` 使用 `processor_backend=local,image_aspect_ratio=normal,max_image_tokens=16384`。
- 评测 LoRA 的方式是先加载 `BASE_MODEL`，再通过 `PeftModel.from_pretrained` 加载 `LORA_CHECKPOINT`；当前 `scripts/eval.sh` 保留了 peft 示例注释，但实际命令没有传 `peft=${LORA_CHECKPOINT}`。
- `eval.sh` 是 bash 脚本，必须用 `bash scripts/eval.sh` 或 `./scripts/eval.sh` 运行，不要用 `python scripts/eval.sh`。
- `xlrs-lite` 的打分会把 `(A)` 标准化为 `A`，所以输出 `(A)` 可以算对。

## 已知注意事项

- `warmup_ratio is deprecated` 是 Transformers 警告，不是训练崩溃原因。
- 大量 `NCCL INFO` 通常只是日志噪音；可把 `NCCL_DEBUG` 降到 `WARN`。
- 如果出现 `processor is not defined`，检查 `sft.py` 中是否保留：

```python
processor = load_optional_processor(model_args.model_name_or_path, model_args.processor_backend)
```

- 如果看到参数打印里的 `device=cpu`，这是因为打印发生在 `Trainer/Deepspeed` prepare 之前，不代表训练在 CPU 上。

## 修改原则

- 修改训练模板、图像 token 展开或 lmms-eval 适配器时，要保证训练和评测的 prompt/process 逻辑一致。
- 不要把 `qwen35_hf` 的定制逻辑塞回通用 `qwen3_vl.py`；本地 Qwen3.5 评测逻辑应放在 `lmms_eval/models/simple/qwen35_hf.py`。
- 手动编辑文件时优先用 `apply_patch`，不要回滚用户已有改动。
