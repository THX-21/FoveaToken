# FoveaToken

## 文件结构

- `src/fovea_token/`：Fovea 模型、配置、tokenizer 同步和训练代码。
- `src/fovea_token/modeling_fovea.py`：模型主体、Fovea retrieval、生成逻辑。
- `src/fovea_token/configuration_fovea.py`：Fovea 配置定义。
- `src/fovea_token/tokenizers/`：`<fovea>` token 同步逻辑。
- `src/fovea_token/train/`：SFT 数据读取、collator、训练入口和 checkpoint 保存加载。
- `scripts/`：训练、评测、抽样运行、可视化等入口脚本。
- `lmms-eval/`：本地 lmms-eval 代码和 Fovea 评测适配器。
- `data/`：训练和评测数据。
- `checkpoints/`：本地模型、训练输出和中间 checkpoint。

## 工程约束

- 使用 transformers 官方 Qwen3.5 多模态实现，不恢复本地 Qwen modeling / processor。
- Fovea 触发只使用专用 `<fovea>` token，不恢复旧 `<|vision_start|>` 触发协议。
- 不恢复旧 visual-query replay、旧视觉 slot、旧 JSON SFT、旧 block span 兼容路径。
- 不新增 `<think>`、`</think>` 等额外 special token rows。
- 不写兼容旧训练模式的分支，不写兜底逻辑。

## 修改原则

- 保持代码路径清晰、直接、最小化。
- 优先重构到正确结构，不要在错误结构上继续打补丁。
- 不为不再支持的模式保留兼容性代码。
- 修改 token、数据字段、训练脚本、保存加载或推理逻辑时，同步更新相关文档。
