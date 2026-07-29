"""Fovea configuration on top of the transformers Qwen2.5-VL config."""

from transformers import Qwen2_5_VLConfig as _Qwen2_5_VLConfig
from transformers import Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig


class FoveaConfig(_Qwen2_5_VLConfig):
    model_type = "fovea"

    def __init__(
        self,
        *args,
        fovea_num_tokens: int = 256,
        fovea_lambda_align: float = 0.2,
        fovea_align_eps: float = 1e-6,
        fovea_align_alpha: float = 1.0,
        fovea_align_beta: float = 0.1,
        fovea_crop_min_image_tokens: int = 64,
        fovea_crop_max_image_tokens: int = 1024,
        fovea_crop_threshold: float = 0.25,
        fovea_crop_region_scale: float = 1.2,
        fovea_crop_image_scale: float = 2.0,
        fovea_max_trigger_per_response: int = 4,
        fovea_trigger_token_ids: list[int] | None = None,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        for name, expected_model_type in (
            ("text_config", "qwen2_5_vl_text"),
            ("vision_config", "qwen2_5_vl_vision"),
        ):
            component = kwargs.get(name)
            if isinstance(component, dict):
                model_type = component.get("model_type")
                if model_type is not None and model_type != expected_model_type:
                    raise ValueError(
                        f"Fovea only supports Qwen2.5-VL; {name} has model_type={model_type!r}."
                    )
        kwargs["tie_word_embeddings"] = False
        super().__init__(*args, **kwargs)
        self.fovea_num_tokens = int(fovea_num_tokens)
        self.fovea_lambda_align = float(fovea_lambda_align)
        self.fovea_align_eps = float(fovea_align_eps)
        self.fovea_align_alpha = float(fovea_align_alpha)
        self.fovea_align_beta = float(fovea_align_beta)
        self.fovea_crop_min_image_tokens = int(fovea_crop_min_image_tokens)
        self.fovea_crop_max_image_tokens = int(fovea_crop_max_image_tokens)
        self.fovea_crop_threshold = float(fovea_crop_threshold)
        self.fovea_crop_region_scale = float(fovea_crop_region_scale)
        self.fovea_crop_image_scale = float(fovea_crop_image_scale)
        self.fovea_max_trigger_per_response = int(fovea_max_trigger_per_response)
        self.fovea_trigger_token_ids = fovea_trigger_token_ids


FoveaTextConfig = Qwen2_5_VLTextConfig
FoveaVisionConfig = Qwen2_5_VLVisionConfig

__all__ = [
    "FoveaConfig",
    "FoveaTextConfig",
    "FoveaVisionConfig",
    "Qwen2_5_VLTextConfig",
    "Qwen2_5_VLVisionConfig",
]
