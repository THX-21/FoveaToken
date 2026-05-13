import copy
from typing import Any

import torch
from torch import nn
from torch.nn import init

from .modeling_qwen3_5 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    apply_rotary_pos_emb,
    auto_docstring,
    can_return_tuple,
    Cache,
    eager_attention_forward,
    GenerationMixin,
    rotate_half,
    TransformersKwargs,
    Unpack,
)
from .configuration_fovea import FoveaConfig
from .train.data import IGNORE_INDEX


class FoveaForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    """Multimodal causal LM head with ImgSlot prefill rewrite and decode-time KV refresh."""
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    config: FoveaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        hidden_size = config.text_config.hidden_size
        num_slot_tokens = max(1, int(config.img_slot_k))
        self.imgslot_num_heads = int(config.text_config.num_attention_heads)
        self.imgslot_head_dim = int(getattr(config.text_config, "head_dim", hidden_size // self.imgslot_num_heads))
        if self.imgslot_num_heads * self.imgslot_head_dim != hidden_size:
            raise ValueError("ImgSlot attention requires num_attention_heads * head_dim to equal hidden_size.")
        # Shared internal A/query token table: [k, hidden].
        self.imgslot_a_tokens = nn.Embedding(num_slot_tokens, hidden_size)
        # Shared ImgSlot attention blocks. These do not replace decoder self-attn;
        # they only synthesize/refresh slot tokens before writing them back into
        # decoder KV cache.
        for prefix in ("text", "img"):
            for name, out_dim in (("q", hidden_size * 2), ("k", hidden_size), ("v", hidden_size), ("o", hidden_size)):
                setattr(self, f"imgslot_{prefix}_{name}_proj", nn.Linear(hidden_size, out_dim, bias=config.text_config.attention_bias))
            for name in ("q", "k"):
                setattr(self, f"imgslot_{prefix}_{name}_norm", Qwen3_5RMSNorm(self.imgslot_head_dim, eps=config.text_config.rms_norm_eps))
        self.imgslot_attention_dropout = config.text_config.attention_dropout
        self.imgslot_scaling = self.imgslot_head_dim**-0.5
        self.num_key_value_groups = 1
        self.imgslot_attn_norm = nn.LayerNorm(hidden_size)
        self.imgslot_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.imgslot_ffn_norm = nn.LayerNorm(hidden_size)
        self.imgslot_visual_norm = nn.LayerNorm(hidden_size)
        # Runtime state used only during generation refresh.
        # states: list[batch] -> list[visual_block_state]
        # text_keys/text_values: per-sample cached text KV, each shaped
        #   [1, num_attention_heads, text_seq, head_dim]
        self._imgslot_runtime = {"enabled": False, "states": [], "delta": 1, "step": 0}
        self._imgslot_aux = torch.zeros((), device=self.lm_head.weight.device)
        self.post_init()
        self._init_imgslot_modules()

    def _imgslot_init_groups(self):
        linears = [getattr(self, f"imgslot_{p}_{n}_proj") for p in ("text", "img") for n in ("q", "k", "v", "o")]
        return linears + [self.imgslot_ffn[0], self.imgslot_ffn[2]], [getattr(self, f"imgslot_{p}_{n}_norm") for p in ("text", "img") for n in ("q", "k")], [self.imgslot_attn_norm, self.imgslot_ffn_norm, self.imgslot_visual_norm]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        output_loading_info = bool(kwargs.get("output_loading_info", False))
        loaded = super().from_pretrained(*args, **kwargs)
        if output_loading_info:
            model, loading_info = loaded
            model._repair_imgslot_init()
            return model, loading_info
        loaded._repair_imgslot_init()
        return loaded

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _imgslot_is_enabled(self) -> bool:
        return bool(self.config.img_slot_enable)

    def _imgslot_config(self) -> dict[str, float | int]:
        cfg = self.config
        if self._imgslot_is_enabled() and cfg.img_slot_tile_size is None:
            raise ValueError("img_slot_tile_size is required when img_slot_enable=true.")
        return {
            "k": int(self.config.img_slot_k),
            "delta": int(self.config.img_slot_delta),
            "beta": float(self.config.img_slot_beta),
            "lam": float(self.config.img_slot_lambda),
            "max_text_tokens": int(self.config.img_slot_max_text_tokens),
        }

    @torch.no_grad()
    def _init_imgslot_modules(self) -> None:
        """Explicitly initialize newly-added ImgSlot weights.

        Qwen/Qwen3.5 checkpoints do not contain ImgSlot parameters. Reinitialize
        them here so missing-key load paths start from the same small-random
        distribution as the rest of the model instead of silently landing on
        zero weights.
        """

        std = float(getattr(self.config.text_config, "initializer_range", 0.02))
        linear_modules, norm_modules, layer_norm_modules = self._imgslot_init_groups()

        init.normal_(self.imgslot_a_tokens.weight, mean=0.0, std=std)
        for module in linear_modules:
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        for module in norm_modules:
            init.zeros_(module.weight)
        for module in layer_norm_modules:
            init.ones_(module.weight)
            init.zeros_(module.bias)

    @torch.no_grad()
    def _repair_imgslot_init(self) -> None:
        """Fix ImgSlot missing-key tensors left uninitialized by some loaders."""

        std = float(getattr(self.config.text_config, "initializer_range", 0.02))
        linear_modules, norm_modules, layer_norm_modules = self._imgslot_init_groups()
        for module in (self.imgslot_a_tokens, *linear_modules):
            weight = module.weight
            if torch.isfinite(weight).all() and weight.float().std() > 0:
                continue
            init.normal_(weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)

        for module in norm_modules:
            if not torch.isfinite(module.weight).all():
                init.zeros_(module.weight)

        for module in layer_norm_modules:
            if not torch.isfinite(module.weight).all():
                init.ones_(module.weight)
            if not torch.isfinite(module.bias).all():
                init.zeros_(module.bias)

    def _reset_imgslot_runtime(self) -> None:
        self._imgslot_runtime = {"enabled": False, "states": [], "delta": 1, "step": 0}

    def _clone_imgslot_runtime_item(self, item, *, deep: bool):
        if deep:
            return copy.deepcopy(item)
        if isinstance(item, torch.Tensor):
            return item.clone()
        return copy.deepcopy(item)

    def _reorder_imgslot_runtime(self, beam_order: list[int]) -> None:
        runtime = getattr(self, "_imgslot_runtime", None)
        if not runtime or not runtime.get("enabled"):
            return
        for key in ("states", "text_keys", "text_values"):
            items = runtime.get(key)
            if not items:
                continue
            if len(items) != len(beam_order):
                raise ValueError(f"ImgSlot runtime[{key}] batch {len(items)} does not match beam order length {len(beam_order)}.")
            runtime[key] = [self._clone_imgslot_runtime_item(items[idx], deep=(key == "states")) for idx in beam_order]

    def _expand_imgslot_runtime(self, expand_size: int) -> None:
        runtime = getattr(self, "_imgslot_runtime", None)
        if expand_size == 1 or not runtime or not runtime.get("enabled"):
            return

        for key in ("states", "text_keys", "text_values"):
            items = runtime.get(key)
            if not items:
                continue
            runtime[key] = [self._clone_imgslot_runtime_item(item, deep=(key == "states")) for item in items for _ in range(expand_size)]

    def _image_block_counts_tensor(self, image_block_counts, device) -> torch.Tensor | None:
        if image_block_counts is None:
            return None
        if isinstance(image_block_counts, torch.Tensor):
            image_block_counts = image_block_counts.to(device=device, dtype=torch.long)
        elif image_block_counts and isinstance(image_block_counts[0], (list, tuple)):
            max_images = max(len(sample_counts) for sample_counts in image_block_counts)
            padded_counts = [list(sample_counts) + [0] * (max_images - len(sample_counts)) for sample_counts in image_block_counts]
            image_block_counts = torch.tensor(padded_counts, dtype=torch.long, device=device)
        else:
            image_block_counts = torch.tensor(image_block_counts, dtype=torch.long, device=device)
        if image_block_counts.dim() == 1:
            image_block_counts = image_block_counts.unsqueeze(0)
        return image_block_counts

    def _split_imgslot_patches(self, pixel_values, image_grid_thw, sample_block_counts):
        merge_size = int(getattr(self.model.visual, "spatial_merge_size", 1))
        if merge_size <= 0:
            raise ValueError(f"Invalid visual spatial_merge_size: {merge_size}.")
        patch_sizes = (image_grid_thw.prod(-1) // (merge_size**2)).to(device=pixel_values.device, dtype=torch.long).tolist()
        block_patches = list(torch.split(pixel_values, patch_sizes, dim=0))
        sample_patches, start = [], 0
        for count in sample_block_counts:
            sample_patches.append(torch.cat(block_patches[start : start + count], dim=0))
            start += count
        if start != len(block_patches):
            raise ValueError("Packed ImgSlot patch splits contain extra blocks after sample grouping.")
        return sample_patches

    def _expand_imgslot_packed_images(self, pixel_values, image_grid_thw, image_block_offsets, image_block_counts, expand_size: int):
        if pixel_values is None or image_grid_thw is None or image_block_offsets is None or image_block_counts is None or expand_size == 1:
            return pixel_values, image_grid_thw, image_block_offsets, image_block_counts

        image_block_counts = self._image_block_counts_tensor(image_block_counts, device=image_grid_thw.device)
        sample_block_counts = image_block_counts.sum(dim=1).tolist()
        if sum(sample_block_counts) != image_grid_thw.shape[0]:
            raise ValueError(
                f"Expanded ImgSlot image metadata mismatch: summed block counts {sum(sample_block_counts)} != grid rows {image_grid_thw.shape[0]}."
            )
        if image_block_offsets.shape[0] != image_grid_thw.shape[0]:
            raise ValueError("Expanded ImgSlot image_block_offsets row count must match image_grid_thw rows.")

        grid_splits = torch.split(image_grid_thw, sample_block_counts, dim=0)
        offset_splits = torch.split(image_block_offsets, sample_block_counts, dim=0)
        patch_splits = self._split_imgslot_patches(pixel_values, image_grid_thw, sample_block_counts)
        if len(grid_splits) != len(sample_block_counts) or len(offset_splits) != len(sample_block_counts) or len(patch_splits) != len(sample_block_counts):
            raise ValueError("Packed ImgSlot image splits do not align with per-sample block counts.")

        expanded_patches, expanded_grids, expanded_offsets, expanded_counts = [], [], [], []
        for patches, grids, offsets, counts in zip(patch_splits, grid_splits, offset_splits, image_block_counts):
            expanded_patches.extend(patches.clone() for _ in range(expand_size))
            expanded_grids.extend(grids.clone() for _ in range(expand_size))
            expanded_offsets.extend(offsets.clone() for _ in range(expand_size))
            expanded_counts.extend(counts.unsqueeze(0).clone() for _ in range(expand_size))
        return torch.cat(expanded_patches, dim=0), torch.cat(expanded_grids, dim=0), torch.cat(expanded_offsets, dim=0), torch.cat(expanded_counts, dim=0)

    def _project_imgslot_kv(self, tokens, k_proj, v_proj, k_norm):
        # tokens: [seq, hidden]
        # -> key/value: [1, num_attention_heads, seq, head_dim]
        proj_k = k_proj(tokens)
        proj_v = v_proj(tokens)
        num_heads = self.imgslot_num_heads
        head_dim = self.imgslot_head_dim
        hidden_shape = (1, tokens.shape[0], -1, head_dim)
        key = k_norm(proj_k.view(hidden_shape)).transpose(1, 2)
        value = proj_v.view(hidden_shape).transpose(1, 2)
        return key, value

    def _project_imgslot_text_tokens(self, text_tokens):
        return self._project_imgslot_kv(
            text_tokens,
            self.imgslot_text_k_proj,
            self.imgslot_text_v_proj,
            self.imgslot_text_k_norm,
        )

    def _project_imgslot_visual_pool(self, visual_pool):
        return self._project_imgslot_kv(
            visual_pool,
            self.imgslot_img_k_proj,
            self.imgslot_img_v_proj,
            self.imgslot_img_k_norm,
        )

    def _apply_imgslot_rope(self, states: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        if positions is None:
            return states
        # states: [batch, heads, seq, head_dim]
        # positions: [3, batch, seq] or [batch, seq]
        batch_size, _num_heads, seq_len, _head_dim = states.shape
        if positions.ndim == 2:
            positions = positions[None, ...].expand(3, batch_size, seq_len)
        dummy = states.new_zeros((batch_size, seq_len, self.config.text_config.hidden_size))
        cos, sin = self.model.language_model.rotary_emb(dummy, positions.to(device=states.device))
        cos = cos.unsqueeze(1).to(device=states.device, dtype=states.dtype)
        sin = sin.unsqueeze(1).to(device=states.device, dtype=states.dtype)
        rotary_dim = cos.shape[-1]
        states_rot, states_pass = states[..., :rotary_dim], states[..., rotary_dim:]
        states_embed = (states_rot * cos) + (rotate_half(states_rot) * sin)
        return torch.cat([states_embed, states_pass], dim=-1)

    def _run_imgslot_attention(
        self,
        anchor_tokens,
        key_states,
        value_states,
        q_proj,
        q_norm,
        o_proj,
        mask=None,
        query_positions=None,
        key_positions=None,
    ):
        # anchor_tokens: [block_count, slot_seq, hidden]
        # key_states/value_states: [block_count, num_attention_heads, kv_seq, head_dim]
        # mask: [block_count, kv_seq] where True means valid token.
        hidden_dim = anchor_tokens.shape[-1]
        head_dim = self.imgslot_head_dim
        input_shape = anchor_tokens.shape[:-1]
        block_count, slot_seq = input_shape
        hidden_shape = (*input_shape, -1, head_dim)
        query_states, gate = torch.chunk(
            q_proj(anchor_tokens).view(*input_shape, -1, head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        query_states = self._apply_imgslot_rope(query_states, query_positions)
        key_states = self._apply_imgslot_rope(key_states, key_positions)
        key_states = key_states.to(device=query_states.device, dtype=query_states.dtype)
        value_states = value_states.to(device=query_states.device, dtype=query_states.dtype)

        attention_mask = None
        if mask is not None:
            mask = mask.to(device=query_states.device)
            attention_mask = query_states.new_full(
                (block_count, 1, 1, key_states.shape[-2]),
                torch.finfo(query_states.dtype).min,
            )
            attention_mask = attention_mask.masked_fill(mask[:, None, None, :], 0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config.text_config._attn_implementation,
            eager_attention_forward,
        )
        use_eager_attention = attention_interface is eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.imgslot_attention_dropout,
            scaling=self.imgslot_scaling,
            is_causal=False,
        )
        if attn_weights is None:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.imgslot_attention_dropout,
                scaling=self.imgslot_scaling,
                is_causal=False,
            )
        elif not use_eager_attention:
            attn_output = attn_output.to(query_states.dtype)
        context = attn_output.reshape(block_count, slot_seq, hidden_dim).contiguous()
        context = (context * torch.sigmoid(gate)).to(anchor_tokens.dtype)
        updated_anchor_tokens = self.imgslot_attn_norm(anchor_tokens + o_proj(context))
        updated_anchor_tokens = self.imgslot_ffn_norm(updated_anchor_tokens + self.imgslot_ffn(updated_anchor_tokens))
        return updated_anchor_tokens, attn_weights

    def _run_imgslot_text_blocks(self, anchor_tokens, text_key, text_value, text_mask=None):
        return self._run_imgslot_attention(
            anchor_tokens,
            text_key,
            text_value,
            self.imgslot_text_q_proj,
            self.imgslot_text_q_norm,
            self.imgslot_text_o_proj,
            mask=text_mask,
        )[0]

    def _run_imgslot_global_visual(self, shared_a_tokens, visual_pools, visual_positions, device, dtype):
        visual_tokens = torch.cat([pool.to(device=device, dtype=dtype) for pool in visual_pools], dim=0).unsqueeze(0)
        visual_pos = torch.cat(visual_positions, dim=0).unsqueeze(0)
        visual_key, visual_value = self._project_imgslot_visual_pool(visual_tokens[0])
        return self._run_imgslot_attention(
            shared_a_tokens,
            visual_key.to(device=device, dtype=dtype),
            visual_value.to(device=device, dtype=dtype),
            self.imgslot_img_q_proj,
            self.imgslot_img_q_norm,
            self.imgslot_img_o_proj,
            mask=torch.ones((1, visual_tokens.shape[1]), dtype=torch.bool, device=device),
            key_positions=visual_pos.permute(2, 0, 1),
        )[0][0]

    def _build_shared_imgslot_a(self, visual_pools, visual_positions, text_key, text_value, device, dtype, prev_a=None):
        base_a = (prev_a if prev_a is not None else self.imgslot_a_tokens.weight[: int(self._imgslot_config()["k"])]).to(device=device, dtype=dtype).unsqueeze(0)
        if text_key is not None and text_value is not None and text_key.shape[-2] > 0:
            text_a = self._run_imgslot_text_blocks(base_a, text_key.to(device=device, dtype=dtype), text_value.to(device=device, dtype=dtype))
            if prev_a is not None:
                beta = float(self._imgslot_config()["beta"])
                text_a = (1.0 - beta) * base_a + beta * text_a
            base_a = text_a
        return self._run_imgslot_global_visual(base_a, visual_pools, visual_positions, device, dtype)

    def _compress_imgslot_visual_tokens(self, shared_a_tokens, visual_pool, visual_pos):
        # shared_a_tokens: [k, hidden], visual_pool: [visual_seq, hidden],
        # visual_pos: [visual_seq, 3] in global image coordinates.
        key_states, value_states = self._project_imgslot_visual_pool(visual_pool)
        updated_a_tokens, attention_weights = self._run_imgslot_attention(
            shared_a_tokens.unsqueeze(0),
            key_states,
            value_states,
            self.imgslot_img_q_proj,
            self.imgslot_img_q_norm,
            self.imgslot_img_o_proj,
            key_positions=visual_pos.unsqueeze(0).permute(2, 0, 1),
        )
        attention_scores = attention_weights.float().mean(dim=1)[0]
        slot_tokens = updated_a_tokens[0]
        slot_pos = torch.matmul(attention_scores, visual_pos.float()).to(visual_pos.dtype)
        attention_prob = attention_scores.clamp_min(1e-6)
        attention_entropy = -(attention_prob * attention_prob.log()).sum(dim=-1).mean()
        aux_stats = {
            "attention_entropy": attention_entropy.to(slot_tokens.dtype),
            "num_blocks": torch.ones((), device=visual_pool.device),
        }
        return slot_tokens, slot_pos, aux_stats

    def _compress_imgslot_blocks(self, shared_a_tokens, visual_pools, visual_positions):
        replacements, slot_positions, aux_stats = [], [], []
        for visual_pool, visual_pos in zip(visual_pools, visual_positions):
            compressed_slots, slot_pos, block_aux = self._compress_imgslot_visual_tokens(shared_a_tokens, visual_pool, visual_pos)
            replacements.append(compressed_slots)
            slot_positions.append(slot_pos)
            aux_stats.append(block_aux)
        return replacements, slot_positions, aux_stats

    def _merge_imgslot_aux_stats(self, aux_stats: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
        if not aux_stats:
            zero = torch.zeros((), device=device)
            return {"attention_entropy": zero, "num_blocks": zero}
        return {
            "attention_entropy": torch.stack([item["attention_entropy"] for item in aux_stats]).mean(),
            "num_blocks": torch.stack([item["num_blocks"].to(device) for item in aux_stats]).sum(),
        }

    def _build_imgslot_visual_positions(self, visual_grids, image_block_offsets, device, dtype):
        spatial_merge_size = int(self.config.vision_config.spatial_merge_size)
        visual_positions = []
        for grid, block_offset in zip(visual_grids, image_block_offsets):
            grid = grid.detach().reshape(-1).to(device="cpu", dtype=torch.long)
            if grid.numel() != 3 or torch.any(grid <= 0):
                raise ValueError(f"Invalid ImgSlot visual grid: {grid.tolist()}.")
            block_offset = block_offset.to(device=device, dtype=torch.long)
            positions = self.model.get_vision_position_ids(
                0,
                grid.to(device=device),
                1,
                spatial_merge_size,
                device=device,
            ).transpose(0, 1)
            # block_offset: [t_global, h_offset, w_offset] in merged-grid units.
            positions = positions + block_offset.to(device=device, dtype=positions.dtype)
            visual_positions.append(positions.to(device=device, dtype=dtype))
        return visual_positions

    def _build_imgslot_prefill_position_ids(self, input_ids, attention_mask, device, dtype):
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            text_pos = attention_mask.to(device=device, dtype=dtype).cumsum(-1) - 1
            text_pos = text_pos.masked_fill(attention_mask.to(device=device) == 0, 0)
        else:
            text_pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        return text_pos.unsqueeze(0).expand(4, -1, -1).clone()

    def _build_imgslot_states_for_sample(
        self,
        visual_pools,
        visual_grids,
        image_block_offsets,
        text_tokens,
        visual_spans,
        text_key=None,
        text_value=None,
    ):
        if len(visual_pools) != len(visual_spans):
            raise ValueError("ImgSlot sample visual block count must match visual span count.")
        if len(visual_grids) != len(visual_spans):
            raise ValueError("ImgSlot sample visual grid count must match visual span count.")
        if len(image_block_offsets) != len(visual_spans):
            raise ValueError("ImgSlot sample block offset count must match visual span count.")
        if not visual_pools:
            raise ValueError("ImgSlot sample must contain at least one visual pool.")

        visual_device = visual_pools[0].device
        visual_dtype = visual_pools[0].dtype
        visual_positions = self._build_imgslot_visual_positions(
            visual_grids,
            image_block_offsets,
            visual_device,
            torch.float32,
        )
        if text_tokens.numel() > 0 and (text_key is None or text_value is None):
            text_key, text_value = self._project_imgslot_text_tokens(text_tokens.to(visual_dtype))
        shared_a_tokens_single = self._build_shared_imgslot_a(visual_pools, visual_positions, text_key, text_value, visual_device, visual_dtype)

        states = []
        visual_replacements, visual_slot_positions, aux_stats = self._compress_imgslot_blocks(shared_a_tokens_single, visual_pools, visual_positions)
        for (span_start, span_length), visual_pool, visual_pos, compressed_slots, slot_pos in zip(
            visual_spans,
            visual_pools,
            visual_positions,
            visual_replacements,
            visual_slot_positions,
        ):
            states.append(
                {
                    "span": (int(span_start), int(span_length)),
                    "A": shared_a_tokens_single.detach(),
                    "V": visual_pool.detach(),
                    "visual_pos": visual_pos.detach(),
                    "compressed_slots": compressed_slots.detach(),
                    "slot_pos": slot_pos.detach(),
                }
            )
        return visual_replacements, visual_slot_positions, states, self._merge_imgslot_aux_stats(aux_stats, visual_device)

    def _refresh_imgslot_states_for_sample(self, states: list[dict[str, Any]], text_key, text_value):
        if not states:
            return [], None

        visual_dtype = states[0]["V"].dtype
        visual_device = states[0]["V"].device
        imgslot_config = self._imgslot_config()
        slot_target = int(imgslot_config["k"])

        shared_a_prev = states[0]["A"].to(device=visual_device, dtype=visual_dtype)
        visual_pools = [state["V"].to(device=visual_device, dtype=visual_dtype) for state in states]
        visual_positions = [state["visual_pos"].to(device=visual_device, dtype=torch.float32) for state in states]
        shared_a_tokens = self._build_shared_imgslot_a(visual_pools, visual_positions, text_key, text_value, visual_device, visual_dtype, shared_a_prev)

        visual_records = []
        compressed_blocks, slot_positions, aux_stats = self._compress_imgslot_blocks(shared_a_tokens, visual_pools, visual_positions)
        for state, compressed_slots, slot_pos in zip(states, compressed_blocks, slot_positions):
            prev_slots = state["compressed_slots"].to(device=visual_device, dtype=visual_dtype)
            compressed_slots = imgslot_config["lam"] * prev_slots + (1.0 - imgslot_config["lam"]) * compressed_slots
            state["A"] = shared_a_tokens.detach()
            state["compressed_slots"] = compressed_slots.detach()
            state["slot_pos"] = slot_pos.detach()
            span_start, span_length = state["span"]
            if span_length != slot_target:
                raise ValueError(f"ImgSlot visual span length {span_length} must equal k ({slot_target}).")
            visual_records.append(
                {
                    "span_start": span_start,
                    "span_length": span_length,
                    "slot_tokens": state["compressed_slots"][:span_length],
                    "slot_positions": state["slot_pos"][:span_length],
                }
            )
        return visual_records, self._merge_imgslot_aux_stats(aux_stats, visual_device)

    def _image_placeholder_spans(self, input_ids: torch.Tensor) -> list[list[tuple[int, int]]]:
        image_token_id = self.config.image_token_id
        all_spans: list[list[tuple[int, int]]] = []
        for sample_ids in input_ids:
            positions = torch.where(sample_ids == image_token_id)[0].tolist()
            sample_spans: list[tuple[int, int]] = []
            if positions:
                start = previous = positions[0]
                for pos in positions[1:]:
                    if pos != previous + 1:
                        sample_spans.append((start, previous - start + 1))
                        start = pos
                    previous = pos
                sample_spans.append((start, previous - start + 1))
            all_spans.append(sample_spans)
        return all_spans

    def _prebuild_imgslot_inputs(self, input_ids, attention_mask, inputs_embeds, visual_pools, image_grid_thw, image_block_offsets, labels=None):
        embed_device = inputs_embeds.device
        # With device_map="auto", token tensors can stay on cuda:0 while the
        # embedding shard that produced inputs_embeds lives on another GPU.
        # All masks used to index inputs_embeds must follow inputs_embeds.
        input_ids = input_ids.to(device=embed_device)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        else:
            attention_mask = attention_mask.to(device=embed_device)
        if labels is not None:
            labels = labels.to(device=embed_device)
        spans_by_sample = self._image_placeholder_spans(input_ids)
        if sum(len(spans) for spans in spans_by_sample) != len(visual_pools):
            span_counts = [len(spans) for spans in spans_by_sample]
            raise ValueError(
                f"ImgSlot visual block count must match placeholder visual span count: "
                f"visual_blocks={len(visual_pools)}, span_counts={span_counts}."
            )
        if image_grid_thw is None or image_grid_thw.shape[0] != len(visual_pools):
            raise ValueError("ImgSlot image_grid_thw row count must match visual pool count.")
        if image_block_offsets is None or image_block_offsets.shape[0] != len(visual_pools):
            raise ValueError("ImgSlot image_block_offsets row count must match visual pool count.")
        batch_states: list[list[dict[str, Any]]] = []
        batch_aux_stats: list[dict[str, torch.Tensor]] = []
        pool_index = 0
        # new_inputs_embeds: [batch, seq, hidden]
        new_inputs_embeds = inputs_embeds.clone()
        position_ids = self._build_imgslot_prefill_position_ids(
            input_ids,
            attention_mask,
            embed_device,
            torch.float32,
        )
        imgslot_config = self._imgslot_config()
        slot_target = int(imgslot_config["k"])
        max_text_tokens = int(imgslot_config["max_text_tokens"])
        runtime_text_keys: list[torch.Tensor] = []
        runtime_text_values: list[torch.Tensor] = []
        for batch_idx, spans in enumerate(spans_by_sample):
            if not spans:
                batch_states.append([])
                runtime_text_keys.append(inputs_embeds.new_zeros((1, self.imgslot_num_heads, 0, self.imgslot_head_dim)).detach())
                runtime_text_values.append(inputs_embeds.new_zeros((1, self.imgslot_num_heads, 0, self.imgslot_head_dim)).detach())
                continue
            if any(span_length != slot_target for _, span_length in spans):
                raise ValueError(f"ImgSlot visual span length must equal k ({slot_target}).")
            visual_spans = spans
            # valid_text_mask: [seq]
            # text_tokens: [text_seq, hidden], with image placeholders removed and
            # supervision targets excluded so ImgSlot does not condition on future
            # answer tokens during training.
            valid_text_mask = (input_ids[batch_idx] != self.config.image_token_id) & attention_mask[batch_idx].bool()
            if labels is not None:
                valid_text_mask = valid_text_mask & labels[batch_idx].eq(IGNORE_INDEX)
            text_tokens = inputs_embeds[batch_idx][valid_text_mask][-max_text_tokens:]
            text_key = text_value = None
            if text_tokens.numel() > 0:
                text_key, text_value = self._project_imgslot_text_tokens(text_tokens)
            sample_visual_pools = []
            sample_visual_grids = []
            sample_block_offsets = []
            for _ in range(len(visual_spans)):
                sample_visual_pools.append(visual_pools[pool_index].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
                sample_visual_grids.append(image_grid_thw[pool_index].to(device=inputs_embeds.device))
                sample_block_offsets.append(image_block_offsets[pool_index].to(device=inputs_embeds.device))
                pool_index += 1
            visual_replacements, visual_slot_positions, sample_states, sample_aux = self._build_imgslot_states_for_sample(
                sample_visual_pools,
                sample_visual_grids,
                sample_block_offsets,
                text_tokens,
                visual_spans,
                text_key=text_key,
                text_value=text_value,
            )
            for replacement, slot_positions, (span_start, span_length) in zip(visual_replacements, visual_slot_positions, visual_spans):
                new_inputs_embeds[batch_idx, span_start : span_start + span_length] = replacement.to(inputs_embeds.dtype)
                position_ids[1:, batch_idx, span_start : span_start + span_length] = slot_positions[:span_length].transpose(0, 1).to(position_ids.dtype)
            batch_states.append(sample_states)
            batch_aux_stats.append(sample_aux)
            if text_key is None or text_value is None:
                runtime_text_keys.append(inputs_embeds.new_zeros((1, self.imgslot_num_heads, 0, self.imgslot_head_dim)).detach())
                runtime_text_values.append(inputs_embeds.new_zeros((1, self.imgslot_num_heads, 0, self.imgslot_head_dim)).detach())
            else:
                runtime_text_keys.append(text_key.detach())
                runtime_text_values.append(text_value.detach())
        self._imgslot_runtime = {
            "enabled": True,
            "states": batch_states,
            "delta": imgslot_config["delta"],
            "step": 0,
            "text_keys": runtime_text_keys,
            "text_values": runtime_text_values,
        }
        self._imgslot_aux = self._merge_imgslot_aux_stats(batch_aux_stats, inputs_embeds.device)
        return new_inputs_embeds, attention_mask, position_ids

    def _write_imgslot_records_for_layer(self, past_key_values, layer_idx, layer, records):
        if not records:
            return
        attention = layer.self_attn
        layer_device = next(layer.parameters()).device
        slot_tokens = torch.stack([record["slot_tokens"].to(device=layer_device) for record in records], dim=0)
        slot_positions = torch.stack([record["slot_positions"].to(device=layer_device) for record in records], dim=0)
        hidden_shape = (*slot_tokens.shape[:-1], -1, attention.head_dim)
        # slot_tokens: [num_records, slot_seq, hidden]
        # slot_keys/slot_values: [num_records, num_kv_heads/num_heads, slot_seq, head_dim]
        slot_keys = attention.k_norm(attention.k_proj(slot_tokens).view(hidden_shape)).transpose(1, 2)
        slot_values = attention.v_proj(slot_tokens).view(hidden_shape).transpose(1, 2)
        dummy_query = slot_keys.new_zeros((slot_tokens.shape[0], attention.config.num_attention_heads, slot_tokens.shape[1], attention.head_dim))
        position_embeddings = self.model.language_model.rotary_emb(slot_tokens, slot_positions.permute(2, 0, 1))
        _, slot_keys = apply_rotary_pos_emb(dummy_query, slot_keys, *position_embeddings)
        cache_layer = past_key_values.layers[layer_idx]
        keys, values = cache_layer.keys, cache_layer.values
        if keys.numel() == 0:
            return
        for record_idx, record in enumerate(records):
            span_start = record["span_start"]
            span_end = span_start + record["span_length"]
            if span_end <= keys.shape[-2]:
                keys[record["batch_idx"], :, span_start:span_end] = slot_keys[record_idx].to(keys.dtype)
                values[record["batch_idx"], :, span_start:span_end] = slot_values[record_idx].to(values.dtype)

    def _collect_imgslot_refresh_records(self, runtime):
        text_keys_by_batch = runtime["text_keys"]
        text_values_by_batch = runtime["text_values"]
        visual_records = []
        aux_stats = []
        aux_device = None
        for batch_idx, states in enumerate(runtime["states"]):
            if batch_idx >= len(text_keys_by_batch) or batch_idx >= len(text_values_by_batch):
                continue
            sample_visual_records, sample_aux = self._refresh_imgslot_states_for_sample(
                states,
                text_keys_by_batch[batch_idx],
                text_values_by_batch[batch_idx],
            )
            for record in sample_visual_records:
                record["batch_idx"] = batch_idx
                visual_records.append(record)
                aux_device = record["slot_tokens"].device
            if sample_aux is not None:
                aux_stats.append(sample_aux)
        if aux_device is not None:
            self._imgslot_aux = self._merge_imgslot_aux_stats(aux_stats, aux_device)
        return visual_records

    def _maybe_refresh_imgslot_cache(self, past_key_values):
        runtime = self._imgslot_runtime
        if not runtime["enabled"] or past_key_values is None:
            return past_key_values
        runtime["step"] += 1
        if runtime["step"] % runtime["delta"]:
            return past_key_values
        layers = self.model.language_model.layers
        layer_types = self.config.text_config.layer_types
        visual_records = self._collect_imgslot_refresh_records(runtime)
        if not visual_records:
            return past_key_values

        # Only overwrite layers that use native full attention. Non-full-attention
        # layers keep their own state transition logic untouched.
        for layer_idx, layer in enumerate(layers):
            if layer_idx < len(layer_types) and layer_types[layer_idx] != "full_attention":
                continue
            self._write_imgslot_records_for_layer(past_key_values, layer_idx, layer, visual_records)
        return past_key_values

    def _append_imgslot_text_hidden(self, hidden_states, past_key_values):
        runtime = getattr(self, "_imgslot_runtime", None)
        if not runtime or not runtime.get("enabled") or past_key_values is None or hidden_states is None:
            return
        if hidden_states.shape[1] == 0:
            return
        # last_hidden: [batch, 1, hidden]
        last_hidden = hidden_states[:, -1:, :].detach()
        text_keys = runtime.setdefault("text_keys", [])
        text_values = runtime.setdefault("text_values", [])
        if len(text_keys) != last_hidden.shape[0] or len(text_values) != last_hidden.shape[0]:
            return
        max_text_tokens = int(self._imgslot_config()["max_text_tokens"])
        for batch_idx in range(last_hidden.shape[0]):
            # last_hidden[batch_idx]: [1, hidden] -> new_key/new_value: [1, num_attention_heads, 1, head_dim]
            new_key, new_value = self._project_imgslot_text_tokens(last_hidden[batch_idx])
            # Append one new text token into the per-sample runtime KV memory, then
            # keep only the latest max_text_tokens entries along seq dim (-2).
            text_keys[batch_idx] = torch.cat([text_keys[batch_idx].to(new_key.device, new_key.dtype), new_key], dim=-2)[:, :, -max_text_tokens:, :].detach()
            text_values[batch_idx] = torch.cat([text_values[batch_idx].to(new_value.device, new_value.dtype), new_value], dim=-2)[:, :, -max_text_tokens:, :].detach()

    @auto_docstring
    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[TransformersKwargs]):
        """
        video_grid_thw (`torch.LongTensor`, *optional*):
            Temporal, height, width grid metadata for each video in `pixel_values_videos`.
        """
        return self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    @auto_docstring
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[TransformersKwargs]):
        """
        image_grid_thw (`torch.LongTensor`, *optional*):
            Temporal, height, width grid metadata for each image in `pixel_values`.
        """
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_block_offsets: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        image_block_counts: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5CausalLMOutputWithPast:
        pre_refresh_past_key_values = past_key_values
        imgslot_first_prefill = bool(kwargs.pop("imgslot_first_prefill", False))
        has_imgslot_prefill = self._imgslot_is_enabled() and (imgslot_first_prefill or past_key_values is None) and input_ids is not None and pixel_values is not None
        if self._imgslot_is_enabled() and not has_imgslot_prefill and past_key_values is None:
            self._reset_imgslot_runtime()
        # Decode path: refresh cached slot tokens before entering the decoder so the
        # next token attends to the latest ImgSlot visual state.
        if self._imgslot_is_enabled() and past_key_values is not None:
            past_key_values = self._maybe_refresh_imgslot_cache(past_key_values)

        # Prefill path with images:
        # - input_ids: [batch, seq]
        # - inputs_embeds: [batch, seq, hidden]
        # - visual_pools: flat list over all image blocks in the batch, each item is
        #   [visual_seq_i, hidden]
        # This rewrites placeholder spans in inputs_embeds, then skips the normal
        # Qwen multimodal scatter path by clearing pixel/grid/mm-token inputs.
        if has_imgslot_prefill:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            visual_pools = list(self.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, return_dict=True).pooler_output)
            if not visual_pools:
                raise RuntimeError("ImgSlot: empty visual pools.")
            inputs_embeds, attention_mask, position_ids = self._prebuild_imgslot_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                visual_pools=visual_pools,
                image_grid_thw=image_grid_thw,
                image_block_offsets=image_block_offsets,
                labels=labels,
            )
            input_ids = None
            pixel_values = None
            image_grid_thw = None
            image_block_offsets = None
            mm_token_type_ids = None

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs[0]
        # hidden_states: [batch, step_seq, hidden]
        self._append_imgslot_text_hidden(hidden_states, pre_refresh_past_key_values)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits: [batch, kept_seq, vocab]
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        image_block_offsets=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        image_block_counts=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Packed ImgSlot image tensors are not batch-major, so avoid the generic
        # tensor repeat_interleave path during beam expansion. Keep them only for
        # the first multimodal prefill and let forward() consume them once.
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=None if is_first_iteration and pixel_values is not None else pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=None if is_first_iteration and image_grid_thw is not None else image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if self._imgslot_is_enabled() and is_first_iteration and pixel_values is not None:
            # ImgSlot prefill must see the full prompt because image placeholders
            # are rewritten in-place before normal decoding starts.
            model_inputs["input_ids"] = input_ids
            model_inputs["attention_mask"] = attention_mask
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw
            model_inputs["image_block_offsets"] = image_block_offsets
            model_inputs["image_block_counts"] = image_block_counts
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
            model_inputs["imgslot_first_prefill"] = True
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_block_offsets"] = None
        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            batch_size = text_positions.shape[0]
            delta = self.model.rope_deltas
            if delta.shape[0] != batch_size:
                if batch_size % delta.shape[0] != 0:
                    raise ValueError(f"rope_deltas batch {delta.shape[0]} cannot expand to generation batch {batch_size}.")
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = text_positions[None, ...] + delta.to(device=text_positions.device)
            return position_ids

        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if is_input_ids and model_kwargs.get("mm_token_type_ids") is not None and (model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None):
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device)

        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)
        return position_ids

    def _reorder_cache(self, past_key_values, beam_idx):
        if not hasattr(past_key_values, "reorder_cache"):
            raise TypeError("past_key_values does not support reorder_cache required for beam search.")
        past_key_values.reorder_cache(beam_idx)

        beam_order = beam_idx.tolist()
        if self.model.rope_deltas is not None:
            if self.model.rope_deltas.shape[0] != len(beam_order):
                raise ValueError(
                    f"rope_deltas batch {self.model.rope_deltas.shape[0]} does not match beam order length {len(beam_order)}."
                )
            self.model.rope_deltas = self.model.rope_deltas.index_select(0, beam_idx.to(self.model.rope_deltas.device)).clone()

        self._reorder_imgslot_runtime(beam_order)
        return past_key_values

    def _expand_inputs_for_generation(self, expand_size: int = 1, is_encoder_decoder: bool = False, input_ids: torch.LongTensor | None = None, **model_kwargs):
        pixel_values = model_kwargs.pop("pixel_values", None)
        image_grid_thw = model_kwargs.pop("image_grid_thw", None)
        image_block_offsets = model_kwargs.pop("image_block_offsets", None)
        image_block_counts = model_kwargs.pop("image_block_counts", None)
        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )
        pixel_values, image_grid_thw, image_block_offsets, image_block_counts = self._expand_imgslot_packed_images(
            pixel_values,
            image_grid_thw,
            image_block_offsets,
            image_block_counts,
            expand_size,
        )
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw
        if image_block_offsets is not None:
            model_kwargs["image_block_offsets"] = image_block_offsets
        if image_block_counts is not None:
            model_kwargs["image_block_counts"] = image_block_counts
        self._expand_imgslot_runtime(expand_size)
        return input_ids, model_kwargs


__all__ = ["FoveaForConditionalGeneration"]
