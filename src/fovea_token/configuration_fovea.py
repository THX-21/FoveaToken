"""Fovea configuration on top of the transformers Qwen3.5 config."""

from transformers import Qwen3_5Config as _Qwen3_5Config
from transformers import Qwen3_5TextConfig, Qwen3_5VisionConfig


class FoveaConfig(_Qwen3_5Config):
    model_type = "fovea"

    def __init__(
        self,
        *args,
        fovea_num_tokens: int = 256,
        fovea_lambda_align: float = 0.2,
        fovea_align_eps: float = 1e-6,
        fovea_align_alpha: float = 1.0,
        fovea_align_beta: float = 0.1,
        fovea_crop_max_image_tokens: int = 1024,
        fovea_crop_threshold: float = 0.35,
        fovea_crop_margin: float = 1.0,
        fovea_crop_padding: float = 1.0,
        fovea_token_id: int | None = None,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        super().__init__(*args, **kwargs)
        self.fovea_num_tokens = int(fovea_num_tokens)
        self.fovea_lambda_align = float(fovea_lambda_align)
        self.fovea_align_eps = float(fovea_align_eps)
        self.fovea_align_alpha = float(fovea_align_alpha)
        self.fovea_align_beta = float(fovea_align_beta)
        self.fovea_crop_max_image_tokens = int(fovea_crop_max_image_tokens)
        self.fovea_crop_threshold = float(fovea_crop_threshold)
        self.fovea_crop_margin = float(fovea_crop_margin)
        self.fovea_crop_padding = float(fovea_crop_padding)
        self.fovea_token_id = fovea_token_id


FoveaTextConfig = Qwen3_5TextConfig
FoveaVisionConfig = Qwen3_5VisionConfig
Qwen3_5Config = FoveaConfig

__all__ = [
    "FoveaConfig",
    "FoveaTextConfig",
    "FoveaVisionConfig",
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
]
