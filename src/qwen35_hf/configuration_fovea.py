# Re-export and rename configuration classes to Fovea naming while keeping
# Qwen3.5 compatibility aliases for the underlying architecture-specific types.
from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import auto_docstring


@auto_docstring(checkpoint="Qwen/Qwen3.5-27B")
@strict
class FoveaTextConfig(PreTrainedConfig):
    r"""
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each key head in linear attention.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head in linear attention.
    linear_num_key_heads (`int`, *optional*, defaults to 16):
        Number of key heads used in linear attention layers.
    linear_num_value_heads (`int`, *optional*, defaults to 32):
        Number of value heads used in linear attention layers.
    """
    model_type = "fovea_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3.5-27B")
@strict
class FoveaVisionConfig(PreTrainedConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 3584):
        The output hidden size of the vision model.
    num_position_embeddings (`int`, *optional*, defaults to 2304):
        The maximum sequence length that this model might ever be used with.
    """
    model_type = "fovea"
    base_config_key = "vision_config"

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02


@auto_docstring(checkpoint="Qwen/Qwen3.5-27B")
@strict
class FoveaConfig(PreTrainedConfig):
    r"""
    img_slot_enable (`bool`, *optional*, defaults to `False`):
        Whether to enable block-based ImgSlot routing.
    img_slot_m (`int`, *optional*, defaults to 4):
        Number of dynamic anchor/query tokens per sample-level shared anchor span.
    img_slot_k (`int`, *optional*, defaults to 64):
        Number of selected visual tokens per image block span.
    img_slot_delta (`int`, *optional*, defaults to 8):
        Decode-step interval for refreshing ImgSlot KV entries.
    img_slot_beta (`float`, *optional*, defaults to 0.3):
        Update strength for the dynamic anchor tokens.
    img_slot_lambda (`float`, *optional*, defaults to 0.9):
        Momentum coefficient for visual Top-K scores.
    img_slot_max_text_tokens (`int`, *optional*, defaults to 512):
        Maximum number of text hidden states kept for ImgSlot decode refresh.
    img_slot_tile_size (`int`, *optional*):
        Required when ImgSlot is enabled. Original images are evenly split into
        blocks using `ceil(width / img_slot_tile_size)` and `ceil(height / img_slot_tile_size)`.
    """
    model_type = "fovea"
    sub_configs = {"vision_config": FoveaVisionConfig, "text_config": FoveaTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False
    img_slot_enable: bool = True
    img_slot_m: int = 8
    img_slot_k: int = 128
    img_slot_delta: int = 129
    img_slot_beta: float = 0.3
    img_slot_lambda: float = 0.9
    img_slot_max_text_tokens: int = 512
    img_slot_tile_size: int | None = 1024

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__post_init__(**kwargs)


Qwen3_5TextConfig = FoveaTextConfig
Qwen3_5VisionConfig = FoveaVisionConfig
Qwen3_5Config = FoveaConfig

__all__ = [
    "FoveaConfig",
    "FoveaTextConfig",
    "FoveaVisionConfig",
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
]
