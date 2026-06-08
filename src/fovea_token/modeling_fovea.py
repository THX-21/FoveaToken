"""Fixed-token Fovea model on top of HF LLaVA-NeXT."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from typing import TypedDict

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextModel,
    LlavaNextPreTrainedModel,
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
    unpad_image,
)
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple

from .configuration_fovea import FoveaConfig


class KwargsForCausalLM(TypedDict, total=False):
    pass


from .train.data import IGNORE_INDEX


class FoveaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (query.shape[-1] ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for idx in range(1, chunk_size):
        row = attn[..., idx, :idx].clone()
        sub = attn[..., :idx, :idx].clone()
        attn[..., idx, :idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for idx in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, idx], key[:, :, idx], value[:, :, idx]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, idx]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, idx]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, idx, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, idx] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, idx, -1, None, None].exp()
            + (k_i * (g[:, :, idx, -1, None] - g[:, :, idx]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class FoveaForConditionalGeneration(LlavaNextPreTrainedModel, GenerationMixin):
    """LLaVA-NeXT LM with 64 fixed fovea query tokens inserted after `<fovea>`."""

    config_class = FoveaConfig
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^image_newline": "model.image_newline",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    config: FoveaConfig

    def __init__(self, config):
        super().__init__(config)
        llava_config_dict = config.to_dict()
        for key in (
            "fovea_num_tokens",
            "fovea_lambda_align",
            "fovea_align_eps",
            "fovea_align_alpha",
            "fovea_align_beta",
            "fovea_input_base_pool",
            "fovea_input_highres_pool",
            "fovea_retrieve_pool",
            "fovea_auto_retrieve_on_answer_start",
            "fovea_token_id",
        ):
            llava_config_dict.pop(key, None)
        llava_config_dict["model_type"] = "llava_next"
        llava_config = LlavaNextConfig(**llava_config_dict)
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            setattr(llava_config, attr, getattr(config, attr, getattr(config.text_config, attr, None)))
        self.model = LlavaNextModel(llava_config)
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, config.text_config.vocab_size, bias=False)

        self.fovea_num_heads = int(config.text_config.num_attention_heads)
        self.fovea_head_dim = int(getattr(config.text_config, "head_dim", hidden_size // self.fovea_num_heads))
        if self.fovea_num_heads * self.fovea_head_dim != hidden_size:
            raise ValueError("Fovea retrieval requires num_attention_heads * head_dim to equal hidden_size.")

        eps = float(getattr(config.text_config, "rms_norm_eps", 1e-6))
        self.fovea_tokens = nn.Embedding(int(config.fovea_num_tokens), hidden_size)
        self.fovea_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_q_norm = LlamaRMSNorm(self.fovea_head_dim, eps=eps)
        self.fovea_k_norm = LlamaRMSNorm(self.fovea_head_dim, eps=eps)
        self.fovea_ssm_in_proj_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.fovea_ssm_in_proj_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_ssm_in_proj_b = nn.Linear(hidden_size, self.fovea_num_heads, bias=False)
        self.fovea_ssm_in_proj_a = nn.Linear(hidden_size, self.fovea_num_heads, bias=False)
        self.fovea_ssm_dt_bias = nn.Embedding(1, self.fovea_num_heads)
        self.fovea_ssm_A_log = nn.Embedding(1, self.fovea_num_heads)
        self.fovea_ssm_out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_ssm_norm = FoveaRMSNormGated(self.fovea_head_dim, eps=eps)
        self._fovea_aux: dict[str, torch.Tensor] = {}
        self._fovea_aux_history: list[dict[str, torch.Tensor]] = []
        self.post_init()
        self._init_fovea_modules()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        output_loading_info = bool(kwargs.get("output_loading_info", False))
        loaded = super().from_pretrained(*args, **kwargs)
        if output_loading_info:
            model, loading_info = loaded
            model._repair_fovea_init()
            return model, loading_info
        loaded._repair_fovea_init()
        return loaded

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, LlavaNextModel)):
            params = list(module.parameters(recurse=False))
            buffers = list(module.buffers(recurse=False))
            if params or buffers:
                if all(getattr(t, "_is_hf_initialized", False) for t in params + buffers):
                    return
        super()._init_weights(module)

    @torch.no_grad()
    def _init_fovea_modules(self) -> None:
        std = float(getattr(self.config.text_config, "initializer_range", 0.02))
        init.normal_(self.fovea_tokens.weight, mean=0.0, std=std)
        init.uniform_(self.fovea_ssm_A_log.weight, 1e-4, 16)
        self.fovea_ssm_A_log.weight.log_()
        init.ones_(self.fovea_ssm_dt_bias.weight)
        for module in (
            self.fovea_q_proj,
            self.fovea_k_proj,
            self.fovea_v_proj,
            self.fovea_o_proj,
            self.fovea_ssm_in_proj_qkv,
            self.fovea_ssm_in_proj_z,
            self.fovea_ssm_in_proj_b,
            self.fovea_ssm_in_proj_a,
            self.fovea_ssm_out_proj,
        ):
            init.normal_(module.weight, mean=0.0, std=std)
        init.ones_(self.fovea_q_norm.weight)
        init.ones_(self.fovea_k_norm.weight)
        init.ones_(self.fovea_ssm_norm.weight)

    @torch.no_grad()
    def _repair_fovea_init(self) -> None:
        std = float(getattr(self.config.text_config, "initializer_range", 0.02))
        tensors = [self.fovea_tokens.weight]
        tensors.extend(
            module.weight
            for module in (
                self.fovea_q_proj,
                self.fovea_k_proj,
                self.fovea_v_proj,
                self.fovea_o_proj,
                self.fovea_ssm_in_proj_qkv,
                self.fovea_ssm_in_proj_z,
                self.fovea_ssm_in_proj_b,
                self.fovea_ssm_in_proj_a,
                self.fovea_ssm_out_proj,
            )
        )
        for tensor in tensors:
            if torch.isfinite(tensor).all() and tensor.float().std() > 0:
                continue
            init.normal_(tensor, mean=0.0, std=std)
        a_log = self.fovea_ssm_A_log.weight
        a_log_float = a_log.float()
        if (
            not torch.isfinite(a_log).all()
            or a_log_float.exp().le(0).any()
            or a_log_float.abs().sum() == 0
            or (a_log.numel() > 1 and a_log_float.std() == 0)
        ):
            init.uniform_(a_log, 1e-4, 16)
            a_log.log_()
        dt_bias = self.fovea_ssm_dt_bias.weight
        if not torch.isfinite(dt_bias).all() or dt_bias.float().abs().sum() == 0:
            init.ones_(dt_bias)
        for module in (self.fovea_q_norm, self.fovea_k_norm, self.fovea_ssm_norm):
            if not torch.isfinite(module.weight).all() or module.weight.float().abs().sum() == 0:
                init.ones_(module.weight)

    def _unused_param_zero_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        """Attach zero-valued uses of conditional params for DDP consistency."""

        anchor = reference.new_zeros(())
        fovea_tensors = (
            self.fovea_tokens.weight,
            self.fovea_q_proj.weight,
            self.fovea_k_proj.weight,
            self.fovea_v_proj.weight,
            self.fovea_o_proj.weight,
            self.fovea_q_norm.weight,
            self.fovea_k_norm.weight,
            self.fovea_ssm_in_proj_qkv.weight,
            self.fovea_ssm_in_proj_z.weight,
            self.fovea_ssm_in_proj_b.weight,
            self.fovea_ssm_in_proj_a.weight,
            self.fovea_ssm_dt_bias.weight,
            self.fovea_ssm_A_log.weight,
            self.fovea_ssm_out_proj.weight,
            self.fovea_ssm_norm.weight,
        )
        vision_tensors = (
            self.model.multi_modal_projector.linear_1.weight,
            self.model.multi_modal_projector.linear_1.bias,
            self.model.multi_modal_projector.linear_2.weight,
            self.model.multi_modal_projector.linear_2.bias,
        )
        for tensor in fovea_tensors + vision_tensors:
            anchor = anchor + tensor.float().sum().to(device=reference.device) * 0.0
        vision_module = self.model.vision_tower
        for param in vision_module.parameters():
            anchor = anchor + param.float().sum().to(device=reference.device) * 0.0
        return anchor.to(dtype=reference.dtype)

    def _pool_spatial_feature_grid(self, feature_grid: torch.Tensor, pool_size: int) -> torch.Tensor:
        pool_size = int(pool_size)
        if pool_size <= 1:
            return feature_grid
        channels, height, width = feature_grid.shape
        pad_h = (pool_size - height % pool_size) % pool_size
        pad_w = (pool_size - width % pool_size) % pool_size
        feature = F.pad(feature_grid.unsqueeze(0), (0, pad_w, 0, pad_h))
        mask = feature_grid.new_ones((1, 1, height, width))
        mask = F.pad(mask, (0, pad_w, 0, pad_h))
        pooled = F.avg_pool2d(feature, kernel_size=pool_size, stride=pool_size)
        pooled_mask = F.avg_pool2d(mask, kernel_size=pool_size, stride=pool_size).clamp_min(1e-6)
        return (pooled / pooled_mask).squeeze(0)

    def _pool_flat_square_features(self, features: torch.Tensor, grid_size: int, pool_size: int) -> torch.Tensor:
        if int(pool_size) <= 1:
            return features
        grid = features.view(grid_size, grid_size, -1).permute(2, 0, 1).contiguous()
        pooled = self._pool_spatial_feature_grid(grid, pool_size)
        return pooled.flatten(1, 2).transpose(0, 1).contiguous()

    def _pack_image_features_pooled(
        self,
        image_features,
        image_sizes,
        *,
        base_pool: int,
        highres_pool: int,
        include_base: bool,
        include_newline: bool,
    ):
        new_image_features = []
        feature_lens = []
        grid_size = self.config.vision_config.image_size // self.config.vision_config.patch_size
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = self._pool_flat_square_features(image_feature[0], grid_size, base_pool)
                highres_feature = image_feature[1:]
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                highres_feature = highres_feature.view(num_patch_height, num_patch_width, grid_size, grid_size, -1)
                highres_feature = highres_feature.permute(4, 0, 2, 1, 3).contiguous()
                highres_feature = highres_feature.flatten(1, 2).flatten(2, 3)
                highres_feature = unpad_image(highres_feature, image_sizes[image_idx])
                highres_feature = self._pool_spatial_feature_grid(highres_feature, highres_pool)
                if include_newline:
                    highres_feature = torch.cat(
                        (
                            highres_feature,
                            self.model.image_newline[:, None, None]
                            .expand(*highres_feature.shape[:-1], 1)
                            .to(highres_feature.device, highres_feature.dtype),
                        ),
                        dim=-1,
                    )
                highres_feature = highres_feature.flatten(1, 2).transpose(0, 1).contiguous()
                if include_base:
                    image_feature = torch.cat((base_image_feature, highres_feature), dim=0)
                else:
                    image_feature = highres_feature
            else:
                image_feature = self._pool_flat_square_features(image_feature[0], grid_size, base_pool)
                if include_newline:
                    image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature)), dim=0)
                if not include_base:
                    image_feature = image_feature[:0]
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features[0].device)
        return new_image_features, feature_lens

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        base_pool: int | None = None,
        highres_pool: int | None = None,
        include_base: bool = True,
        include_newline: bool = True,
    ):
        return self._get_pooled_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            base_pool=base_pool,
            highres_pool=highres_pool,
            include_base=include_base,
            include_newline=include_newline,
        )

    def _get_main_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
    ):
        # Main-image features match original LLaVA-NeXT packing by default.
        return self._get_pooled_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            base_pool=self.config.fovea_input_base_pool,
            highres_pool=self.config.fovea_input_highres_pool,
            include_base=True,
            include_newline=True,
        )

    def _get_retrieve_image_features(
        self,
        retrieve_pixel_values: torch.FloatTensor,
        retrieve_image_sizes: torch.Tensor,
    ):
        # Retrieval memory keeps original tile resolution by default.
        return self._get_pooled_image_features(
            pixel_values=retrieve_pixel_values,
            image_sizes=retrieve_image_sizes,
            base_pool=self.config.fovea_retrieve_pool,
            highres_pool=self.config.fovea_retrieve_pool,
            include_base=False,
            include_newline=False,
        )

    def _get_pooled_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        base_pool: int | None = None,
        highres_pool: int | None = None,
        include_base: bool = True,
        include_newline: bool = True,
    ):
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if vision_feature_select_strategy != "default":
            raise ValueError("Fovea pooled image features require vision_feature_select_strategy='default'.")
        base_pool = int(base_pool if base_pool is not None else self.config.fovea_input_base_pool)
        highres_pool = int(highres_pool if highres_pool is not None else self.config.fovea_input_highres_pool)

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            pixel_values = torch.cat([pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)], dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.model.vision_tower(pixel_values, output_hidden_states=True)
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_features.hidden_states[vision_feature_layer]
        else:
            selected_image_feature = torch.cat([image_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer], dim=-1)
        selected_image_feature = selected_image_feature[:, 1:]
        image_features = self.model.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, _ = self._pack_image_features_pooled(
            image_features,
            image_sizes,
            base_pool=base_pool,
            highres_pool=highres_pool,
            include_base=include_base,
            include_newline=include_newline,
        )
        return image_features

    def _build_multimodal_embeddings(
        self,
        input_ids,
        pixel_values,
        image_sizes,
        input_embeds_override=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids) if input_embeds_override is None else input_embeds_override
        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self._get_main_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = input_ids.eq(int(self.config.image_token_index)).unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[image_mask].numel() != image_features.numel():
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens={int(input_ids.eq(int(self.config.image_token_index)).sum())}, features={image_features.shape[0]}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        return inputs_embeds, image_features

    def _language_forward_from_embeds(self, inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
        return self.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def _split_retrieve_features_by_sample(self, retrieve_pixel_values, retrieve_image_sizes, retrieve_image_counts):
        features = self._get_retrieve_image_features(retrieve_pixel_values, retrieve_image_sizes)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        counts = retrieve_image_counts.to(device="cpu", dtype=torch.long).tolist()
        per_sample, start = [], 0
        for count in counts:
            count = int(count)
            if count > 0:
                per_sample.append(torch.cat(features[start : start + count], dim=0))
            else:
                hidden_size = int(self.config.text_config.hidden_size)
                per_sample.append(retrieve_pixel_values.new_zeros((0, hidden_size)))
            start += count
        return per_sample

    def _retrieve_runtime_fovea_vectors(self, text_hidden_history, text_input_id_history, retrieve_memory_cache):
        last_pos = text_hidden_history.shape[1] - 1
        history_attention = text_input_id_history.new_ones(text_input_id_history.shape)
        history_position = torch.tensor([[0, last_pos]], device=text_hidden_history.device, dtype=torch.long)
        fovea_queries = self._condition_fovea_queries_from_text(
            text_hidden_history,
            history_attention,
            text_input_id_history,
            history_position,
        )
        trigger_batch = torch.zeros((1,), device=text_hidden_history.device, dtype=torch.long)
        return self._retrieve_fovea_from_memory(fovea_queries, trigger_batch, *retrieve_memory_cache)

    def _build_retrieve_memory(
        self,
        retrieve_pixel_values,
        retrieve_image_sizes,
        retrieve_patch_boxes,
        retrieve_image_counts,
        batch_size,
        device=None,
        dtype=None,
    ):
        if retrieve_pixel_values is None or retrieve_image_sizes is None or retrieve_patch_boxes is None or retrieve_image_counts is None:
            raise ValueError("Fovea retrieval requires retrieve_pixel_values, retrieve_image_sizes, retrieve_patch_boxes, and retrieve_image_counts.")
        memories = self._split_retrieve_features_by_sample(retrieve_pixel_values, retrieve_image_sizes, retrieve_image_counts)
        if len(memories) != batch_size:
            raise ValueError("retrieve_image_counts must contain one entry per batch sample.")
        return self._pad_sample_memory(
            memories,
            retrieve_patch_boxes,
            device or retrieve_pixel_values.device,
            dtype or retrieve_pixel_values.dtype,
        )

    def _pad_sample_memory(self, memories, patch_boxes, device, dtype):
        max_len = max(max(1, mem.shape[0]) for mem in memories)
        hidden = int(self.config.text_config.hidden_size)
        batch_size = len(memories)
        memory = torch.zeros((batch_size, max_len, hidden), device=device, dtype=dtype)
        mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.bool)
        boxes = torch.zeros((batch_size, max_len, 4), device=device, dtype=torch.float32)
        box_start = 0
        for batch_idx, mem in enumerate(memories):
            count = int(mem.shape[0])
            if count > 0:
                memory[batch_idx, :count] = mem.to(device=device, dtype=dtype)
                mask[batch_idx, :count] = True
            if box_start + count > patch_boxes.shape[0]:
                raise ValueError(
                    "retrieve_patch_boxes is shorter than the LLaVA-NeXT retrieval memory: "
                    f"need {box_start + count}, got {patch_boxes.shape[0]}."
                )
            if count > 0:
                boxes[batch_idx, :count] = patch_boxes[box_start : box_start + count].to(device=device, dtype=torch.float32)
            box_start += count
        if box_start != patch_boxes.shape[0]:
            raise ValueError(
                "retrieve_patch_boxes must match the LLaVA-NeXT retrieval memory length exactly: "
                f"used {box_start}, got {patch_boxes.shape[0]}."
            )
        return memory, mask, boxes

    def _condition_fovea_queries_from_text(self, hidden_states, attention_mask, input_ids, fovea_positions):
        device = hidden_states.device
        dtype = hidden_states.dtype
        num_fovea = int(self.config.fovea_num_tokens)
        text_mask = torch.ones(hidden_states.shape[:2], device=device, dtype=torch.bool)
        if attention_mask is not None:
            text_mask &= attention_mask.to(device=device).bool()
        if input_ids is not None:
            text_mask &= input_ids.to(device=device).ne(int(self.config.image_token_index))
        if fovea_positions.numel() == 0:
            return hidden_states.new_empty((0, num_fovea, hidden_states.shape[-1]))

        batch_size, seq_len, hidden_size = hidden_states.shape
        base = self.fovea_tokens.weight.to(device=device, dtype=dtype)
        x = hidden_states.unsqueeze(1) + base.view(1, num_fovea, 1, -1)
        x = x.reshape(batch_size * num_fovea, seq_len, hidden_size)
        flat_mask = text_mask[:, None, :].expand(batch_size, num_fovea, seq_len).reshape(batch_size * num_fovea, seq_len)
        x = x * flat_mask[..., None].to(dtype)

        mixed_qkv = self.fovea_ssm_in_proj_qkv(x)
        query, key, value = mixed_qkv.split(hidden_size, dim=-1)
        query = query.view(batch_size * num_fovea, seq_len, self.fovea_num_heads, self.fovea_head_dim)
        key = key.view(batch_size * num_fovea, seq_len, self.fovea_num_heads, self.fovea_head_dim)
        value = value.view(batch_size * num_fovea, seq_len, self.fovea_num_heads, self.fovea_head_dim)
        gate = self.fovea_ssm_in_proj_z(x).view(batch_size * num_fovea, seq_len, self.fovea_num_heads, self.fovea_head_dim)
        beta = torch.sigmoid(self.fovea_ssm_in_proj_b(x))
        a_log = self.fovea_ssm_A_log.weight[0].float()
        dt_bias = self.fovea_ssm_dt_bias.weight[0]
        g = -a_log.exp() * F.softplus(self.fovea_ssm_in_proj_a(x).float() + dt_bias)
        beta = beta * flat_mask[..., None].to(beta.dtype)
        g = torch.where(flat_mask[..., None], g, torch.zeros_like(g))

        core_attn_out, _ = _torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out = core_attn_out.reshape(-1, self.fovea_head_dim)
        gate = gate.reshape(-1, self.fovea_head_dim)
        states = self.fovea_ssm_norm(core_attn_out, gate)
        states = states.reshape(batch_size, num_fovea, seq_len, hidden_size)
        states = base.view(1, num_fovea, 1, -1) + self.fovea_ssm_out_proj(states)

        positions = fovea_positions.to(device=device, dtype=torch.long)
        return states[positions[:, 0], :, positions[:, 1], :]

    def _retrieve_fovea_from_memory(self, fovea_queries, trigger_batch, memory, memory_mask, patch_boxes, fovea_box_indices=None, fovea_boxes=None):
        device = fovea_queries.device
        dtype = fovea_queries.dtype
        num_triggers = int(fovea_queries.shape[0])
        num_fovea = int(self.config.fovea_num_tokens)
        heads = self.fovea_num_heads
        head_dim = self.fovea_head_dim
        memory_len = int(memory.shape[1])

        q_all = self.fovea_q_proj(fovea_queries).view(num_triggers, num_fovea, heads, head_dim)
        q_all = self.fovea_q_norm(q_all)

        vectors = fovea_queries.new_empty((num_triggers, num_fovea, heads * head_dim))
        attn_mean = torch.empty((num_triggers, num_fovea, memory_len), device=device, dtype=torch.float32)
        scale = head_dim**-0.5
        trigger_batch = trigger_batch.to(device=device, dtype=torch.long)

        for sample_idx in trigger_batch.unique(sorted=True):
            trigger_indices = torch.nonzero(trigger_batch == sample_idx, as_tuple=False).squeeze(-1)
            sample_id = int(sample_idx.item())
            sample_mask = memory_mask[sample_id]
            sample_memory = memory[sample_id : sample_id + 1]
            q = q_all.index_select(0, trigger_indices)
            k = self.fovea_k_proj(sample_memory).view(1, memory_len, heads, head_dim)
            v = self.fovea_v_proj(sample_memory).view(1, memory_len, heads, head_dim)
            k = self.fovea_k_norm(k).squeeze(0).permute(1, 0, 2)
            v = v.squeeze(0).permute(1, 0, 2)

            scores = torch.einsum("tnhd,hmd->tnhm", q, k) * scale
            scores = scores.masked_fill((~sample_mask).view(1, 1, 1, -1), torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            context = torch.einsum("tnhm,hmd->tnhd", attn, v).reshape(trigger_indices.shape[0], num_fovea, -1)
            retrieved = self.fovea_o_proj(context).to(dtype)
            vectors.index_copy_(0, trigger_indices, retrieved)
            attn_mean.index_copy_(0, trigger_indices, attn.mean(dim=2).to(torch.float32))

        align_loss = vectors.new_zeros(())
        if fovea_boxes is not None and fovea_boxes.numel() > 0 and fovea_box_indices is not None and fovea_box_indices.numel() > 0:
            boxes_for_trigger = fovea_boxes.to(device=device, dtype=torch.float32).index_select(0, fovea_box_indices.to(device=device, dtype=torch.long))
            boxes_for_memory = patch_boxes.index_select(0, trigger_batch).to(device=device, dtype=torch.float32)
            mask_for_trigger = memory_mask.index_select(0, trigger_batch)
            valid_boxes = boxes_for_memory.ge(0).all(dim=-1)
            eps = float(self.config.fovea_align_eps)
            alpha = float(self.config.fovea_align_alpha)
            beta = float(self.config.fovea_align_beta)

            target_boxes = boxes_for_trigger.unsqueeze(1)
            inter_x1 = torch.max(target_boxes[..., 0], boxes_for_memory[..., 0])
            inter_y1 = torch.max(target_boxes[..., 1], boxes_for_memory[..., 1])
            inter_x2 = torch.min(target_boxes[..., 2], boxes_for_memory[..., 2])
            inter_y2 = torch.min(target_boxes[..., 3], boxes_for_memory[..., 3])
            inter_area = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)

            box_area = (
                (boxes_for_trigger[:, 2] - boxes_for_trigger[:, 0]).clamp_min(0)
                * (boxes_for_trigger[:, 3] - boxes_for_trigger[:, 1]).clamp_min(0)
            )
            q = inter_area / box_area[:, None].clamp_min(eps)
            q = q * mask_for_trigger.to(q.dtype) * valid_boxes.to(q.dtype)
            q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)

            a_mean = attn_mean.mean(dim=1)
            a_max = attn_mean.max(dim=1).values
            l_main = -(q * a_mean.clamp_min(eps).log()).sum(dim=-1)
            l_cover = -(q * a_max.clamp_min(eps).log()).sum(dim=-1)

            attn_flat = F.normalize(attn_mean, p=2, dim=-1, eps=eps)
            cosine = torch.matmul(attn_flat, attn_flat.transpose(-1, -2))
            div_mask = 1.0 - torch.eye(num_fovea, device=device, dtype=cosine.dtype)
            l_div = (cosine * div_mask.unsqueeze(0)).sum(dim=(-1, -2)) / max(num_fovea * (num_fovea - 1), 1)

            align_loss = (l_main + alpha * l_cover + beta * l_div).mean()
        entry = {
            "fovea_attn_mean": attn_mean.detach(),
            "trigger_batch": trigger_batch.detach(),
        }
        self._fovea_aux = entry
        self._fovea_aux_history.append(entry)
        return vectors, align_loss

    def _expand_with_fovea_tokens(self, base_embeds, attention_mask, labels, fovea_positions, fovea_vectors):
        batch_size, seq_len, hidden_size = base_embeds.shape
        num_fovea = int(self.config.fovea_num_tokens)
        by_batch: list[list[tuple[int, int]]] = [[] for _ in range(batch_size)]
        for query_idx, (batch_idx, token_pos) in enumerate(fovea_positions.to(device="cpu", dtype=torch.long).tolist()):
            if 0 <= batch_idx < batch_size and 0 <= token_pos < seq_len:
                by_batch[batch_idx].append((token_pos, query_idx))
        for items in by_batch:
            items.sort(key=lambda item: item[0])

        lengths = [seq_len + len(items) * num_fovea for items in by_batch]
        max_len = max(lengths)
        new_embeds = base_embeds.new_zeros((batch_size, max_len, hidden_size))
        new_attention = attention_mask.new_zeros((batch_size, max_len)) if attention_mask is not None else None
        new_labels = labels.new_full((batch_size, max_len), IGNORE_INDEX) if labels is not None else None

        for batch_idx, items in enumerate(by_batch):
            src_cursor = 0
            dst_cursor = 0
            for token_pos, query_idx in items:
                copy_len = token_pos - src_cursor + 1
                if copy_len > 0:
                    new_embeds[batch_idx, dst_cursor : dst_cursor + copy_len] = base_embeds[batch_idx, src_cursor : token_pos + 1]
                    if new_attention is not None:
                        new_attention[batch_idx, dst_cursor : dst_cursor + copy_len] = attention_mask[batch_idx, src_cursor : token_pos + 1]
                    if new_labels is not None:
                        new_labels[batch_idx, dst_cursor : dst_cursor + copy_len] = labels[batch_idx, src_cursor : token_pos + 1]
                    dst_cursor += copy_len
                new_embeds[batch_idx, dst_cursor : dst_cursor + num_fovea] = fovea_vectors[query_idx].to(base_embeds.dtype)
                if new_attention is not None:
                    new_attention[batch_idx, dst_cursor : dst_cursor + num_fovea] = 1
                dst_cursor += num_fovea
                src_cursor = token_pos + 1
            tail_len = seq_len - src_cursor
            if tail_len > 0:
                new_embeds[batch_idx, dst_cursor : dst_cursor + tail_len] = base_embeds[batch_idx, src_cursor:]
                if new_attention is not None:
                    new_attention[batch_idx, dst_cursor : dst_cursor + tail_len] = attention_mask[batch_idx, src_cursor:]
                if new_labels is not None:
                    new_labels[batch_idx, dst_cursor : dst_cursor + tail_len] = labels[batch_idx, src_cursor:]
        return new_embeds, new_attention, new_labels

    def _cached_decode_position_ids(self, attention_mask, num_new_tokens: int = 1):
        return attention_mask.long().cumsum(-1)[:, -int(num_new_tokens) :] - 1

    def _sample_next_token(self, logits, *, do_sample: bool, temperature=None, top_p=None, top_k=None):
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        scaled_logits = logits
        temp = None if temperature is None else float(temperature)
        if temp is not None and temp > 0:
            scaled_logits = scaled_logits / temp

        if top_k is not None:
            k = min(int(top_k), int(scaled_logits.shape[-1]))
            if k > 0:
                threshold = torch.topk(scaled_logits, k=k, dim=-1).values[..., -1, None]
                scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, torch.finfo(scaled_logits.dtype).min)

        if top_p is not None:
            p = float(top_p)
            if 0.0 < p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, dim=-1, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                sorted_remove = cumulative_probs > p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool).scatter(-1, sorted_indices, sorted_remove)
                scaled_logits = scaled_logits.masked_fill(remove_mask, torch.finfo(scaled_logits.dtype).min)

        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def fovea_generate(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        image_sizes=None,
        retrieve_pixel_values=None,
        retrieve_image_sizes=None,
        retrieve_patch_boxes=None,
        retrieve_image_counts=None,
        max_new_tokens=128,
        eos_token_id=None,
        pad_token_id=None,
        use_cache=True,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        fovea_auto_retrieve_on_answer_start=None,
        **kwargs,
    ):
        self._fovea_aux = {}
        self._fovea_aux_history = []
        if input_ids.shape[0] != 1:
            raise ValueError("Fovea generation currently expects batch_size=1.")
        if int(num_beams) != 1:
            raise ValueError("Fovea generation currently supports num_beams=1 only.")
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        model_kwargs = {"use_cache": use_cache}
        for key in ("output_attentions", "output_hidden_states", "return_dict"):
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        output_input_ids = input_ids
        model_input_ids = input_ids
        inputs_embeds, _ = self._build_multimodal_embeddings(model_input_ids, pixel_values, image_sizes)
        retrieve_memory_cache = None
        if retrieve_pixel_values is not None:
            retrieve_memory_cache = self._build_retrieve_memory(
                retrieve_pixel_values,
                retrieve_image_sizes,
                retrieve_patch_boxes,
                retrieve_image_counts,
                input_ids.shape[0],
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
        fovea_id = getattr(self.config, "fovea_token_id", None)
        if fovea_auto_retrieve_on_answer_start is None:
            fovea_auto_retrieve_on_answer_start = getattr(self.config, "fovea_auto_retrieve_on_answer_start", True)
        fovea_auto_retrieve_on_answer_start = bool(fovea_auto_retrieve_on_answer_start)
        text_hidden_history = None
        text_input_id_history = model_input_ids
        if fovea_id is not None and retrieve_memory_cache is not None:
            prompt_positions = torch.nonzero(model_input_ids.eq(int(fovea_id)), as_tuple=False)
            if prompt_positions.numel() > 0:
                prompt_outputs = self._language_forward_from_embeds(inputs_embeds, attention_mask=attention_mask, **model_kwargs)
                prompt_hidden = prompt_outputs[0]
                text_hidden_history = prompt_hidden
                trigger_batch = prompt_positions[:, 0].to(device=prompt_hidden.device, dtype=torch.long)
                fovea_queries = self._condition_fovea_queries_from_text(
                    prompt_hidden,
                    attention_mask,
                    input_ids,
                    prompt_positions,
                )
                fovea_vectors, _ = self._retrieve_fovea_from_memory(
                    fovea_queries,
                    trigger_batch,
                    *retrieve_memory_cache,
                )
                inputs_embeds, attention_mask, _ = self._expand_with_fovea_tokens(
                    inputs_embeds,
                    attention_mask,
                    None,
                    prompt_positions.to(device=prompt_hidden.device, dtype=torch.long),
                    fovea_vectors,
                )
        outputs = self._language_forward_from_embeds(inputs_embeds, attention_mask=attention_mask, **model_kwargs)
        if text_hidden_history is None:
            text_hidden_history = outputs[0]
        logits = self.lm_head(outputs[0][:, -1:, :]).squeeze(1)
        past_key_values = outputs.past_key_values
        if (
            fovea_id is not None
            and retrieve_memory_cache is not None
            and fovea_auto_retrieve_on_answer_start
            and (model_input_ids.shape[1] == 0 or int(model_input_ids[0, -1].item()) != int(fovea_id))
        ):
            auto_fovea = model_input_ids.new_full((model_input_ids.shape[0], 1), int(fovea_id))
            auto_fovea_embed = self.get_input_embeddings()(auto_fovea).to(inputs_embeds.dtype)
            model_input_ids = torch.cat([model_input_ids, auto_fovea], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
            outputs = self._language_forward_from_embeds(
                auto_fovea_embed,
                attention_mask=attention_mask,
                position_ids=self._cached_decode_position_ids(attention_mask),
                past_key_values=past_key_values,
                **model_kwargs,
            )
            auto_hidden = outputs[0][:, -1, :]
            past_key_values = outputs.past_key_values
            text_hidden_history = torch.cat([text_hidden_history, auto_hidden.unsqueeze(1)], dim=1)
            text_input_id_history = torch.cat([text_input_id_history, auto_fovea], dim=1)
            fovea_vectors, _ = self._retrieve_runtime_fovea_vectors(
                text_hidden_history,
                text_input_id_history,
                retrieve_memory_cache,
            )
            fovea_embeds = fovea_vectors.squeeze(0).unsqueeze(0).to(device=auto_hidden.device, dtype=inputs_embeds.dtype)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, fovea_embeds.shape[1]))], dim=1)
            outputs = self._language_forward_from_embeds(
                fovea_embeds,
                attention_mask=attention_mask,
                position_ids=self._cached_decode_position_ids(attention_mask, fovea_embeds.shape[1]),
                past_key_values=past_key_values,
                **model_kwargs,
            )
            logits = self.lm_head(outputs[0][:, -1, :])
            past_key_values = outputs.past_key_values
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_ids = {int(eos_token_id)} if isinstance(eos_token_id, int) else {int(item) for item in (eos_token_id or [])}

        for _ in range(int(max_new_tokens)):
            next_token = self._sample_next_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            next_embed = self.get_input_embeddings()(next_token).to(inputs_embeds.dtype)
            model_input_ids = torch.cat([model_input_ids, next_token], dim=1)
            output_input_ids = torch.cat([output_input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
            if int(next_token.item()) in eos_ids:
                break

            outputs = self._language_forward_from_embeds(
                next_embed,
                attention_mask=attention_mask,
                position_ids=self._cached_decode_position_ids(attention_mask),
                past_key_values=past_key_values,
                **model_kwargs,
            )
            token_hidden = outputs[0][:, -1, :]
            logits = self.lm_head(token_hidden)
            past_key_values = outputs.past_key_values
            text_hidden_history = torch.cat([text_hidden_history, token_hidden.unsqueeze(1)], dim=1)
            text_input_id_history = torch.cat([text_input_id_history, next_token], dim=1)

            if fovea_id is not None and int(next_token.item()) == int(fovea_id) and retrieve_memory_cache is not None:
                fovea_vectors, _ = self._retrieve_runtime_fovea_vectors(
                    text_hidden_history,
                    text_input_id_history,
                    retrieve_memory_cache=retrieve_memory_cache,
                )
                fovea_embeds = fovea_vectors.squeeze(0).unsqueeze(0).to(device=token_hidden.device, dtype=inputs_embeds.dtype)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, fovea_embeds.shape[1]))], dim=1)
                outputs = self._language_forward_from_embeds(
                    fovea_embeds,
                    attention_mask=attention_mask,
                    position_ids=self._cached_decode_position_ids(attention_mask, fovea_embeds.shape[1]),
                    past_key_values=past_key_values,
                    **model_kwargs,
                )
                logits = self.lm_head(outputs[0][:, -1, :])
                past_key_values = outputs.past_key_values
        return output_input_ids

    def generate(self, *args, **kwargs):
        if kwargs.get("retrieve_pixel_values") is not None:
            return self.fovea_generate(*args, **kwargs)
        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        if cache_position is not None and cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes
        return model_inputs

    def _fovea_forward(
        self,
        input_ids,
        attention_mask,
        labels,
        pixel_values,
        image_sizes,
        retrieve_pixel_values,
        retrieve_image_sizes,
        retrieve_patch_boxes,
        retrieve_image_counts,
        fovea_positions,
        fovea_box_indices,
        fovea_boxes,
        logits_to_keep=0,
        **kwargs,
    ):
        self._fovea_aux = {}
        self._fovea_aux_history = []
        vision_feature_layer = kwargs.pop("vision_feature_layer", None)
        vision_feature_select_strategy = kwargs.pop("vision_feature_select_strategy", None)
        embeds_a, _ = self._build_multimodal_embeddings(
            input_ids,
            pixel_values,
            image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        outputs_a = self._language_forward_from_embeds(embeds_a, attention_mask=attention_mask, **kwargs)
        hidden_a = outputs_a[0]
        fovea_positions = fovea_positions.to(device=hidden_a.device, dtype=torch.long)
        trigger_batch = fovea_positions[:, 0]
        fovea_queries = self._condition_fovea_queries_from_text(
            hidden_a,
            attention_mask,
            input_ids,
            fovea_positions,
        )

        memory = self._build_retrieve_memory(
            retrieve_pixel_values,
            retrieve_image_sizes,
            retrieve_patch_boxes,
            retrieve_image_counts,
            hidden_a.shape[0],
            hidden_a.device,
            hidden_a.dtype,
        )
        fovea_vectors, align_loss = self._retrieve_fovea_from_memory(
            fovea_queries,
            trigger_batch,
            *memory,
            fovea_box_indices=fovea_box_indices,
            fovea_boxes=fovea_boxes,
        )
        embeds_b, attention_b, labels_b = self._expand_with_fovea_tokens(embeds_a, attention_mask, labels, fovea_positions, fovea_vectors)
        outputs_b = self._language_forward_from_embeds(embeds_b, attention_mask=attention_b, **kwargs)
        hidden_b = outputs_b[0]
        full_logits_b = self.lm_head(hidden_b)
        lm_loss = self.loss_function(logits=full_logits_b, labels=labels_b, vocab_size=full_logits_b.shape[-1])
        loss = lm_loss + float(self.config.fovea_lambda_align) * align_loss
        loss = loss + self._unused_param_zero_anchor(loss)
        self._fovea_aux = {
            "lm_loss": lm_loss.detach(),
            "align_loss": align_loss.detach(),
            "num_queries": hidden_b.new_tensor(float(fovea_positions.shape[0])),
            "tokens_per_query": hidden_b.new_tensor(float(self.config.fovea_num_tokens)),
            "fovea_attn_mean": self._fovea_aux.get("fovea_attn_mean"),
        }
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=full_logits_b[:, slice_indices, :],
            past_key_values=outputs_b.past_key_values,
            hidden_states=outputs_b.hidden_states,
            attentions=outputs_b.attentions,
            image_hidden_states=None,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        retrieve_pixel_values: torch.Tensor | None = None,
        retrieve_image_sizes: torch.LongTensor | None = None,
        retrieve_patch_boxes: torch.Tensor | None = None,
        retrieve_image_counts: torch.Tensor | None = None,
        fovea_positions: torch.LongTensor | None = None,
        fovea_box_indices: torch.LongTensor | None = None,
        fovea_boxes: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> tuple | LlavaNextCausalLMOutputWithPast:
        if (
            retrieve_pixel_values is None
            and retrieve_image_sizes is None
            and retrieve_patch_boxes is None
            and retrieve_image_counts is None
            and fovea_positions is None
            and fovea_box_indices is None
            and fovea_boxes is None
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            vision_feature_layer = kwargs.pop("vision_feature_layer", None)
            vision_feature_select_strategy = kwargs.pop("vision_feature_select_strategy", None)
            vision_feature_layer = (
                vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
            )
            vision_feature_select_strategy = (
                vision_feature_select_strategy
                if vision_feature_select_strategy is not None
                else self.config.vision_feature_select_strategy
            )

            outputs = self.model(
                input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = outputs[0]
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            loss = None
            if labels is not None:
                loss = self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
                )
                loss = loss + self._unused_param_zero_anchor(loss)

            return LlavaNextCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=outputs.image_hidden_states,
            )

        model_kwargs = {
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
            "cache_position": cache_position,
            **kwargs,
        }
        if labels is not None and fovea_positions is not None and fovea_positions.numel() > 0:
            return self._fovea_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                retrieve_pixel_values=retrieve_pixel_values,
                retrieve_image_sizes=retrieve_image_sizes,
                retrieve_patch_boxes=retrieve_patch_boxes,
                retrieve_image_counts=retrieve_image_counts,
                fovea_positions=fovea_positions,
                fovea_box_indices=fovea_box_indices,
                fovea_boxes=fovea_boxes,
                logits_to_keep=logits_to_keep,
                **model_kwargs,
            )

        vision_feature_layer = model_kwargs.pop("vision_feature_layer", None)
        vision_feature_select_strategy = model_kwargs.pop("vision_feature_select_strategy", None)
        if inputs_embeds is None:
            inputs_embeds, image_hidden_states = self._build_multimodal_embeddings(
                input_ids,
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
        else:
            image_hidden_states = None
        outputs = self._language_forward_from_embeds(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **model_kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
            loss = loss + self._unused_param_zero_anchor(loss)
        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


__all__ = ["FoveaForConditionalGeneration"]
