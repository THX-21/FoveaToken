# qwen35-hf

`qwen35-hf` 是将 Hugging Face Transformers 中 `qwen3_5` 实现拆分为独立库的最小打包版本。

## 安装

先安装 PyTorch。对于 CUDA 12.8，使用官方 `cu128` 源：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

如果需要固定版本，可以显式指定，例如：

```bash
pip install "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
```

然后安装当前库：

```bash
pip install -e .
```

运行时直接依赖包括：

- `transformers>=4.57.1`
- `huggingface_hub>=0.30.0`
- `tokenizers>=0.21.0`
- `numpy`

其中 `torch` 需要用户按目标硬件环境单独安装。之所以不把它写死在 `pyproject.toml` 中，是因为标准 Python 包依赖无法可靠地为 `torch` 指定 CUDA 专用索引源；如果直接写成普通依赖，安装器往往会回落到默认源，结果未必是你需要的 `cu128` 版本。

可选加速依赖：

```bash
pip install -e ".[fast]"
```

其中 `fast` 会安装：

- `causal-conv1d`
- `flash-linear-attention`

这两者不是功能必需，但会显著影响线性注意力路径的性能；未安装时库会自动回退到纯 PyTorch 实现。

## 使用

```python
from qwen35_hf import Qwen3_5ForConditionalGeneration, Qwen3_5Tokenizer

model = Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-4B")
tokenizer = Qwen3_5Tokenizer.from_pretrained("Qwen/Qwen3.5-4B")
```
