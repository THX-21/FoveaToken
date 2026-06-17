# FoveaToken 关键约定

## 当前机制

- 基座使用 transformers 官方 Qwen3.5 多模态实现；不要恢复本地 Qwen modeling / processor。
- 不新增 tokenizer token，不 resize embedding / `lm_head`。Fovea 触发复用 Qwen 内置 `<|vision_start|>`。
- `<|vision_start|>` 有双语义：user 侧是原生图像 span 起点；assistant 侧才是 Fovea retrieval trigger。数据侧必须用原始 labels 区分二者，并保证 trigger 数量与 `fovea_query_boxes` 一致。
- 训练冻结 Qwen language model、vision tower、embedding、原始 `lm_head`；只训练 `fovea_*` 模块和 `fovea_aux_lm_head`。
- 训练 loss 固定为 `aux_lm_loss + fovea_lambda_align * align_loss`。语言监督只走 `fovea_aux_lm_head`；正式推理 token 始终由冻结的原始 `lm_head` 生成。
- 推理时 aux head 若 argmax 为 `<|vision_start|>`，只作为隐藏检索信号；该触发 token 不写入正式输出。检索到的 crop 以 `vision_start + image_features + vision_end` 注入 decoder cache，并追加到 Fovea 条件 history。

## 数据和入口

- 原始数据只使用 VGR parquet；预处理把 `<SOT>[box]<EOT><image>` 转成 assistant 侧 `<|vision_start|>`，并保存 `fovea_query_boxes`。
- 默认训练读取 `data/vgr/preprocessed/*.parquet` 和本地图像 `data/vgr/llava_next_raw_format`。
- 默认训练入口是 `bash scripts/ft3.sh`；旧 `scripts/ft3_lora.sh` 已删除，不要恢复。
- 运行 Python 默认用项目 `.venv`，并设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/lmms-eval"
```

## 禁止恢复

- 旧 visual-query replay、旧视觉 slot、旧 JSON SFT、旧 block span 兼容路径。
- LoRA、全参训练、vision unfreeze、embedding row-level 训练开关。
- 新增 `<fovea>`、`<think>`、`</think>` 等 special token rows。
- 兼容旧训练模式的分支或兜底逻辑。

## 保存和加载

- checkpoint 只保存 Fovea 新增模块和 `fovea_aux_lm_head`。
- `from_pretrained()` 返回前必须检查 Fovea 新增权重：全 0 或非 finite 时只修复异常权重。
- `fovea_aux_lm_head` 缺失或异常时，从冻结的原始 `lm_head` 拷贝初始化。

## 同步修改

修改 token、数据字段、训练脚本、保存/加载模块或推理触发逻辑时，同步更新 `README.md` 和本文件。
