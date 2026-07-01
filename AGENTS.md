# FoveaToken 关键约定

## 当前机制

- 基座使用 transformers 官方 Qwen3.5 多模态实现；不要恢复本地 Qwen modeling / processor。
- Fovea 触发使用专用 `<fovea>` token。允许为它新增 tokenizer token，并同步 resize embedding、原始 `lm_head` 和 `fovea_aux_lm_head`。
- user 侧 `<|vision_start|>` 仍只表示原生图像 span 起点；assistant 侧 `<fovea>` 才是 Fovea retrieval trigger。数据侧必须保证 trigger 数量与 `fovea_query_boxes` 一致。
- 训练冻结 Qwen language model、vision tower、embedding、原始 `lm_head`；只训练 `fovea_*` 模块和 `fovea_aux_lm_head`。
- 训练默认语言监督走 `fovea_aux_lm_head`，总 loss 为 `lm_loss + fovea_lambda_align * align_loss`。其中默认 `lm_loss = aux_lm_loss`；若开启 `fovea_train_main_lm_head`，则改为 `lm_loss = 0.5 * (aux_lm_loss + main_lm_loss)`。正式推理 token 始终由原始 `lm_head` 生成，`<fovea>` 只作为隐藏检索信号。
- 推理时 aux head 若 argmax 为 `<fovea>`，只作为隐藏检索信号；该触发 token 不写入正式输出。单次触发只注入权重最高的 1 个 crop，格式仍为 `vision_start + image_features + vision_end`，并追加到 Fovea 条件 history。每条样本最多允许 4 次 hidden Fovea trigger：若当前用 `fovea_aux_lm_head` 生成正式文本，则 `<fovea>` token 必须直接屏蔽，不允许写入可见输出；若当前用原始 `lm_head` 生成正式文本，则超过 4 次后必须禁止 aux head 再触发检索。`max_new_tokens` 只统计正式文本 token，不统计这些 hidden trigger。

## 数据和入口

- 原始数据只使用 VGR parquet；预处理把 `<SOT>[box]<EOT><image>` 转成 assistant 侧 `<fovea>`，并保存 `fovea_query_boxes`。
- 默认训练读取 `data/vgr/preprocessed/*.parquet` 和本地图像 `data/vgr/llava_next_raw_format`。
- 默认训练入口是 `bash scripts/ft3.sh`；旧 `scripts/ft3_lora.sh` 已删除，不要恢复。
- 训练/评测默认基座切到 `Qwen/Qwen3.5-9B`；训练默认 `RUN_NAME` 为 `fovea-vgr-qwen-9b`，默认评测 checkpoint 也对应 `*-9b` 路径。
- 默认评测入口是 `bash scripts/eval.sh`；其中 `textvqa`、`vstar_bench`、`vsibench` 会按子任务展开，默认每个 benchmark 总样本预算为 `600`，可用 `EVAL_TASK_BUDGET` 覆盖。
- 运行 Python 默认用项目 `.venv`，并设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

## 禁止恢复

- 旧 visual-query replay、旧视觉 slot、旧 JSON SFT、旧 block span 兼容路径。
- LoRA、全参训练、vision unfreeze、embedding row-level 训练开关。
- 旧 `<|vision_start|>` 触发协议，以及和当前 `<fovea>` 协议并存的兼容分支。
- 新增额外的 `<think>`、`</think>` 等 special token rows。
- 兼容旧训练模式的分支或兜底逻辑。

## 保存和加载

- checkpoint 只保存 Fovea 新增模块和 `fovea_aux_lm_head`。
- `from_pretrained()` 返回前必须检查 Fovea 新增权重：全 0 或非 finite 时只修复异常权重。
- 若基座 checkpoint 不含解耦后的原始 `lm_head`，加载后必须用 input embedding 显式恢复它；新增 `<fovea>` token 对应的 input embedding、原始 `lm_head`、`fovea_aux_lm_head` row 在 resize / load 后也必须显式修复。
- `fovea_aux_lm_head` 缺失或异常时，从冻结的原始 `lm_head` 拷贝初始化。

## 同步修改

修改 token、数据字段、训练脚本、保存/加载模块或推理触发逻辑时，同步更新 `README.md` 和本文件。
