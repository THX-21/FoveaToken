"""Fovea configuration on top of the HF LLaVA-NeXT config."""

from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig


DEFAULT_LLAVA_NEXT_IMAGE_GRID_PINPOINTS = [
    [336, 672],
    [672, 336],
    [672, 672],
    [1008, 336],
    [336, 1008],
    [1008, 672],
    [672, 1008],
    [1344, 672],
    [672, 1344],
    [1008, 1008],
    [1344, 1008],
    [1008, 1344],
    [1344, 1344],
    [1680, 1344],
    [1344, 1680],
]

def sync_expanded_image_grid_pinpoints(config, processor=None) -> list[list[int]]:
    """Use the original LLaVA-NeXT anyres candidate set everywhere."""

    pinpoints = [list(item) for item in DEFAULT_LLAVA_NEXT_IMAGE_GRID_PINPOINTS]
    config.image_grid_pinpoints = pinpoints
    if processor is not None:
        processor.image_grid_pinpoints = pinpoints
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            image_processor.image_grid_pinpoints = pinpoints
    return pinpoints


class FoveaConfig(LlavaNextConfig):
    model_type = "fovea"

    def __init__(
        self,
        *args,
        fovea_num_tokens: int = 64,
        fovea_lambda_align: float = 0.2,
        fovea_align_eps: float = 1e-6,
        fovea_align_alpha: float = 1.0,
        fovea_align_beta: float = 0.1,
        fovea_input_base_pool: int = 1,
        fovea_input_highres_pool: int = 2,
        fovea_retrieve_pool: int = 1,
        fovea_auto_retrieve_on_answer_start: bool = True,
        fovea_token_id: int | None = None,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        kwargs.setdefault("image_grid_pinpoints", DEFAULT_LLAVA_NEXT_IMAGE_GRID_PINPOINTS)
        super().__init__(*args, **kwargs)
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if not hasattr(self, attr):
                setattr(self, attr, getattr(self.text_config, attr, None))
        self.fovea_num_tokens = int(fovea_num_tokens)
        self.fovea_lambda_align = float(fovea_lambda_align)
        self.fovea_align_eps = float(fovea_align_eps)
        self.fovea_align_alpha = float(fovea_align_alpha)
        self.fovea_align_beta = float(fovea_align_beta)
        self.fovea_input_base_pool = int(fovea_input_base_pool)
        self.fovea_input_highres_pool = int(fovea_input_highres_pool)
        self.fovea_retrieve_pool = int(fovea_retrieve_pool)
        self.fovea_auto_retrieve_on_answer_start = bool(fovea_auto_retrieve_on_answer_start)
        self.fovea_token_id = fovea_token_id

    @property
    def image_token_id(self) -> int:
        return self.image_token_index

    @image_token_id.setter
    def image_token_id(self, value: int) -> None:
        self.image_token_index = value

    @property
    def pad_token_id(self) -> int | None:
        return getattr(self, "_pad_token_id", getattr(self.text_config, "pad_token_id", None))

    @pad_token_id.setter
    def pad_token_id(self, value: int | None) -> None:
        self._pad_token_id = value
        if hasattr(self, "text_config"):
            self.text_config.pad_token_id = value

    @property
    def bos_token_id(self) -> int | None:
        return getattr(self, "_bos_token_id", getattr(self.text_config, "bos_token_id", None))

    @bos_token_id.setter
    def bos_token_id(self, value: int | None) -> None:
        self._bos_token_id = value
        if hasattr(self, "text_config"):
            self.text_config.bos_token_id = value

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self, "_eos_token_id", getattr(self.text_config, "eos_token_id", None))

    @eos_token_id.setter
    def eos_token_id(self, value: int | None) -> None:
        self._eos_token_id = value
        if hasattr(self, "text_config"):
            self.text_config.eos_token_id = value


__all__ = [
    "DEFAULT_LLAVA_NEXT_IMAGE_GRID_PINPOINTS",
    "FoveaConfig",
    "sync_expanded_image_grid_pinpoints",
]
