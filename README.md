# FoveaToken

当前默认基座是 `Qwen/Qwen3.5-9B`。

当前协议摘要：

- Fovea retrieval trigger 使用专用 `<fovea>` token。
- user 侧图像仍然走 Qwen 原生 `<|vision_start|><|image_pad|><|vision_end|>`。
- VGR 预处理会把 assistant 侧 `<SOT>[box]<EOT><image>` 改写成 `<fovea>`，并写入 `fovea_query_boxes`。
- 训练默认冻结 Qwen 基座，只训练 `fovea_*` 模块和 `fovea_aux_lm_head`。
- 训练总 loss 为 `lm_loss + fovea_lambda_align * align_loss`。默认 `lm_loss=aux_lm_loss`；如果打开 `fovea_train_main_lm_head`，则使用 `0.5 * (aux_lm_loss + main_lm_loss)`。
- 推理时 `<fovea>` 是隐藏检索信号，不写入最终文本输出；单次触发只注入权重最高的 1 个 crop，且每条样本最多只允许 4 次 hidden Fovea trigger。若当前用 `fovea_aux_lm_head` 生成正式文本，则 `<fovea>` token 会被直接屏蔽，不允许写入可见输出；若当前用原始 `lm_head` 生成正式文本，则超过 4 次后会禁止 aux head 再触发检索。`max_new_tokens` 只统计正式文本 token，不统计这些 hidden trigger。
- 加载基座 checkpoint 时，如果原始 `lm_head` 因解耦而缺失，会从 input embedding 显式恢复；新增 `<fovea>` token 的相关权重行也会在 resize / load 后显式修复。

- 训练入口：`bash scripts/ft3.sh`
- 默认训练输出：`checkpoints/fovea-vgr-qwen-9b`
- 默认原生 Qwen 评测：`bash scripts/eval_others.sh`
- 默认 Fovea checkpoint 评测：`bash scripts/eval.sh`
- `scripts/eval.sh` 对 `textvqa`、`vstar_bench`、`vsibench` 会按子任务展开评测；默认每个 benchmark 总样本预算为 `600`，可用 `EVAL_TASK_BUDGET` 覆盖。

常用默认值：

- `scripts/ft3.sh`：`FT3_CKPT_PATH=Qwen/Qwen3.5-9B`
- `scripts/eval_others.sh`：`EVAL_BASE_MODEL=Qwen/Qwen3.5-9B`
- `scripts/eval.sh`：`EVAL_FULL_CHECKPOINT=checkpoints/fovea-vgr-qwen-9b/checkpoint-666`

如果你已经有自己的 9B 本地 checkpoint，可以直接覆盖这些环境变量。
