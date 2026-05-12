# AGENTS.md

本文件面向 Claude Code 及其他 AI 编程助手。**修改代码后如果影响项目结构、脚本入口或使用约定，需同步更新本文档。**

---

## 项目结构

```text
src/qwen35_hf/
  __init__.py             # Fovea 公共 API 与 Qwen3_5 兼容别名
  configuration_fovea.py  # Fovea / ImgSlot 配置
  modeling_qwen3_5.py     # Qwen3.5 text / vision / backbone 基础实现
  modeling_fovea.py       # FoveaForConditionalGeneration 与 ImgSlot 主逻辑
  processing_fovea.py     # FoveaProcessor 与视觉 placeholder 展开
  tokenization_fovea.py   # FoveaTokenizer
  train/
    sft.py                # 训练入口、LoRA、冻结策略、callback
    data.py               # SFT 数据集、ChatML 编码、collator
    image_packing.py      # 图像 resize / patch packing / block split

scripts/
  ft3.sh                  # LoRA + ImgSlot 训练脚本
  eval.sh                 # Fovea LoRA 评测脚本
  eval_qwen35.sh          # Qwen3.5 baseline 评测脚本
  train_eval_loop.sh      # 分段训练并评测最新 checkpoint
  zero2_tp2_gpu.json      # 当前训练默认使用的 DeepSpeed 配置

lmms-eval/
  lmms_eval/models/simple/fovea.py    # Fovea 自定义 lmms-eval adapter
  lmms_eval/tasks/xlrs/XLRS-lite.yaml # 当前默认评测任务
```

---

## 关键脚本摘要

### `scripts/ft3.sh`
- 训练主入口：`torchrun -m qwen35_hf.train.sft`
- 默认单机单卡，固定使用 `scripts/zero2_tp2_gpu.json`
- 自动从输出目录查找最新 `checkpoint-*` 续训
- 结构参数优先从 checkpoint / config 读取，脚本只透传训练期 loss 权重等调参项
- 常用环境变量：`FT3_JSON_PATH`、`FT3_IMAGE_FOLDER`、`FT3_CKPT_PATH`、`FT3_OUTPUT_DIR`、`FT3_MAX_STEPS`、`FT3_NUM_TRAIN_EPOCHS`

### `scripts/eval.sh`
- Fovea LoRA 评测入口，走本地 `lmms-eval` 的 `fovea` adapter
- 默认任务：`xlrs-lite`
- 默认离线模式：`HF_HUB_OFFLINE=1`
- 默认 `device_map=cuda:0`、`batch_size=1`
- 单进程下默认把 `device` 和 `device_map` 绑定到同一张卡
- 当前 `xlrs-lite` 任务按数据集全量样本评测，不再在 task 侧按 category 截断
- 常用环境变量：`EVAL_BASE_MODEL`、`EVAL_LORA_CHECKPOINT`、`EVAL_TASKS`、`EVAL_OUTPUT_PATH`、`EVAL_DEVICE`、`EVAL_DEVICE_MAP`、`EVAL_BATCH_SIZE`

### `scripts/eval_qwen35.sh`
- Qwen3.5 baseline 评测入口
- 不加载 Fovea LoRA，用于和 Fovea 路径对比

### `scripts/train_eval_loop.sh`
- 按 checkpoint step 分段训练
- 每段训练后自动评测最新 checkpoint
- 任一步失败即按原始退出码退出

---

## 必要注意事项

- 仓库根目录不是标准可编辑包；不要假设 `pip install -e .` 可用。
- 主训练与评测路径依赖：`PYTHONPATH="$PWD/src:$PWD/lmms-eval"`。
- 对外主入口优先使用 `FoveaForConditionalGeneration`、`FoveaConfig`、`FoveaProcessor`、`FoveaTokenizer`；`Qwen3_5*` 仅保留兼容别名。
- `modeling_qwen3_5.py` 是基础实现，`modeling_fovea.py` 是 Fovea / ImgSlot 主实现；不要把 ImgSlot 逻辑塞回基础 backbone。
- 训练、processor、评测三侧的 ImgSlot placeholder 约定必须同步，尤其是 block span、`image_grid_thw`、`image_block_counts`。
- 评测 canonical 路径由 `lmms-eval/lmms_eval/models/simple/fovea.py` 负责构造逻辑 image placeholder，并由 `FoveaProcessor` 展开成 block span；`modeling_fovea.py` 不再负责 prompt / placeholder 兼容修复。
- 修改 LoRA trainable / `modules_to_save` 时，必须同时检查训练保存逻辑和 `lmms-eval/lmms_eval/models/simple/fovea.py` 的评测加载逻辑。
- 修改脚本环境变量、默认入口或使用方式时，必须同步更新 `AGENTS.md` 和 `README.md`。
- 当前默认 attention 后端是 `sdpa`；如需切换 `flash_attention_2`，需按当前环境重新验证稳定性。
- 涉及 reshape、permute、split、cat、mask、KV cache 写入的改动，优先保证形状语义清晰、易核对。
