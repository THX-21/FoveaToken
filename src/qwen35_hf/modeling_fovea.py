import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    apply_rotary_pos_emb,
    auto_docstring,
    can_return_tuple,
    Cache,
    GenerationMixin,
    rotate_half,
    TransformersKwargs,
    Unpack,
)
from .configuration_fovea import FoveaConfig


class FoveaForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    """Multimodal causal LM head for text, image, and video generation.

    ImgSlot overview:
    - Prefill: replace text-side image placeholder spans with one shared anchor span
      plus one compressed visual span per image block.
    - Decode: keep a detached runtime state per sample, and every `delta` steps
      refresh the slot tokens then overwrite their KV cache entries on
      full-attention layers only.
    """
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    config: FoveaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        hidden_size = config.text_config.hidden_size
        num_anchor_tokens = max(1, int(config.img_slot_m))
        imgslot_num_experts = int(config.img_slot_num_experts)
        imgslot_slots_per_expert = int(config.img_slot_slots_per_expert)
        imgslot_total_slots = imgslot_num_experts * imgslot_slots_per_expert
        if imgslot_total_slots != int(config.img_slot_k):
            raise ValueError("img_slot_num_experts * img_slot_slots_per_expert must equal img_slot_k.")
        self.imgslot_num_heads = int(config.text_config.num_attention_heads)
        self.imgslot_head_dim = int(getattr(config.text_config, "head_dim", hidden_size // self.imgslot_num_heads))
        if self.imgslot_num_heads * self.imgslot_head_dim != hidden_size:
            raise ValueError("ImgSlot attention requires num_attention_heads * head_dim to equal hidden_size.")
        # Anchor token table: [m, hidden].
        self.imgslot_a_tokens = nn.Embedding(num_anchor_tokens, hidden_size)
        # Shared ImgSlot attention blocks. These do not replace decoder self-attn;
        # they only synthesize/refresh slot tokens before writing them back into
        # decoder KV cache.
        for stream in ("text", "img"):
            for proj in ("q", "k", "v", "o"):
                setattr(self, f"imgslot_{stream}_{proj}_proj", nn.Linear(hidden_size, hidden_size, bias=False))
        self.imgslot_attn_norm = nn.LayerNorm(hidden_size)
        self.imgslot_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.imgslot_ffn_norm = nn.LayerNorm(hidden_size)
        self.imgslot_visual_norm = nn.LayerNorm(hidden_size)
        self.imgslot_gate_proj = nn.Linear(hidden_size, 1, bias=False)
        self.imgslot_expert_proj = nn.Linear(hidden_size, imgslot_num_experts, bias=False)
        self.imgslot_subslot_proj = nn.Linear(hidden_size, imgslot_total_slots, bias=False)
        self.imgslot_value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_slot_out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Runtime state used only during generation refresh.
        # states: list[batch] -> list[visual_block_state]
        # text_keys/text_values: per-sample cached text KV, each shaped
        #   [1, num_attention_heads, text_seq, head_dim]
        self._imgslot_runtime = {"enabled": False, "states": [], "delta": 1, "step": 0}
        self._imgslot_aux = torch.zeros((), device=self.lm_head.weight.device)
        self.post_init()

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
        total_slots = int(cfg.img_slot_num_experts) * int(cfg.img_slot_slots_per_expert)
        if total_slots != int(cfg.img_slot_k):
            raise ValueError("img_slot_num_experts * img_slot_slots_per_expert must equal img_slot_k.")
        return {
            "m": int(cfg.img_slot_m),
            "k": int(cfg.img_slot_k),
            "delta": int(cfg.img_slot_delta),
            "beta": float(cfg.img_slot_beta),
            "lam": float(cfg.img_slot_lambda),
            "max_text_tokens": int(cfg.img_slot_max_text_tokens),
            "num_experts": int(cfg.img_slot_num_experts),
            "slots_per_expert": int(cfg.img_slot_slots_per_expert),
            "gate_temperature": float(cfg.img_slot_gate_temperature),
            "route_temperature": float(cfg.img_slot_route_temperature),
            "gate_sparsity_coef": float(cfg.img_slot_gate_sparsity_coef),
            "expert_balance_coef": float(cfg.img_slot_expert_balance_coef),
            "slot_balance_coef": float(cfg.img_slot_slot_balance_coef),
            "route_entropy_coef": float(cfg.img_slot_route_entropy_coef),
        }

    def _project_imgslot_kv(self, tokens, k_proj, v_proj):
        # tokens: [seq, hidden]
        # -> key/value: [1, num_attention_heads, seq, head_dim]
        num_heads = self.imgslot_num_heads
        head_dim = self.imgslot_head_dim
        key = k_proj(tokens).view(1, -1, num_heads, head_dim).transpose(1, 2)
        value = v_proj(tokens).view(1, -1, num_heads, head_dim).transpose(1, 2)
        return key, value

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
        cos = cos.unsqueeze(1).to(states.dtype)
        sin = sin.unsqueeze(1).to(states.dtype)
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
        o_proj,
        mask=None,
        query_positions=None,
        key_positions=None,
    ):
        # anchor_tokens: [block_count, slot_seq, hidden]
        # key_states/value_states: [block_count, num_attention_heads, kv_seq, head_dim]
        # mask: [block_count, kv_seq] where True means valid token.
        # query: [block_count, num_heads, slot_seq, head_dim]
        # attention_logits: [block_count, num_heads, slot_seq, kv_seq]
        # context: [block_count, slot_seq, hidden]
        num_heads = self.imgslot_num_heads
        hidden_dim = anchor_tokens.shape[-1]
        head_dim = self.imgslot_head_dim
        block_count = anchor_tokens.shape[0]
        query = q_proj(anchor_tokens).view(block_count, -1, num_heads, head_dim).transpose(1, 2)
        query = self._apply_imgslot_rope(query, query_positions)
        key_states = self._apply_imgslot_rope(key_states, key_positions)
        attention_logits = torch.matmul(query.float(), key_states.float().transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            attention_logits = attention_logits.masked_fill(~mask[:, None, None, :], torch.finfo(attention_logits.dtype).min)
        # Stabilize softmax over the kv dimension.
        attention_logits = attention_logits - attention_logits.amax(dim=-1, keepdim=True)
        attention_weights = torch.softmax(attention_logits, dim=-1).to(value_states.dtype)
        context = torch.matmul(attention_weights.float(), value_states.float()).transpose(1, 2).reshape(block_count, anchor_tokens.shape[1], hidden_dim)
        context = context.to(anchor_tokens.dtype)
        updated_anchor_tokens = self.imgslot_attn_norm(anchor_tokens + o_proj(context))
        updated_anchor_tokens = self.imgslot_ffn_norm(updated_anchor_tokens + self.imgslot_ffn(updated_anchor_tokens))
        return updated_anchor_tokens, attention_weights

    def _run_imgslot_text_blocks(self, anchor_tokens, text_key, text_value, text_mask=None):
        return self._run_imgslot_attention(
            anchor_tokens,
            text_key,
            text_value,
            self.imgslot_text_q_proj,
            self.imgslot_text_o_proj,
            mask=text_mask,
        )[0]

    def _run_imgslot_visual_blocks(
        self,
        anchor_tokens,
        visual_key,
        visual_value,
        visual_tokens,
        visual_lengths,
        visual_mask=None,
        anchor_positions=None,
        visual_positions=None,
    ):
        updated_anchor_tokens, attention_weights = self._run_imgslot_attention(
            anchor_tokens,
            visual_key,
            visual_value,
            self.imgslot_img_q_proj,
            self.imgslot_img_o_proj,
            mask=visual_mask,
            query_positions=anchor_positions,
            key_positions=visual_positions,
        )
        # attention_weights: [1, num_heads, m, total_visual_seq]
        # token_anchor_weight: [1, total_visual_seq, m]
        anchor_attn = attention_weights.float().mean(dim=1)
        token_anchor_weight = torch.softmax(anchor_attn.transpose(1, 2), dim=-1).to(updated_anchor_tokens.dtype)
        token_anchor_context = torch.matmul(token_anchor_weight, updated_anchor_tokens)
        visual_context_global = self.imgslot_visual_norm(visual_tokens + token_anchor_context)
        visual_contexts = list(visual_context_global.split(visual_lengths, dim=1))
        token_scores = anchor_attn.mean(dim=1).to(updated_anchor_tokens.dtype)
        visual_token_scores = list(token_scores.split(visual_lengths, dim=1))
        return updated_anchor_tokens, visual_contexts, visual_token_scores, attention_weights

    def _compress_imgslot_visual_tokens(self, visual_context, visual_pool, visual_pos, visual_mask, visual_gate_score):
        # visual_context/visual_pool: [block_count, visual_seq, hidden]
        # visual_pos: [block_count, visual_seq, 3]
        # visual_mask: [block_count, visual_seq]
        # visual_gate_score: [block_count, visual_seq]
        imgslot_config = self._imgslot_config()
        gate_temperature = max(float(imgslot_config["gate_temperature"]), float(self.config.img_slot_min_temperature))
        route_temperature = max(float(imgslot_config["route_temperature"]), float(self.config.img_slot_min_temperature))
        num_experts = int(imgslot_config["num_experts"])
        slots_per_expert = int(imgslot_config["slots_per_expert"])
        total_slots = int(imgslot_config["k"])

        gate_input = visual_context * visual_gate_score.unsqueeze(-1).to(visual_context.dtype)
        gate_logits = self.imgslot_gate_proj(gate_input).squeeze(-1)
        expert_logits = self.imgslot_expert_proj(visual_context)
        subslot_logits = self.imgslot_subslot_proj(visual_context).view(*visual_context.shape[:2], num_experts, slots_per_expert)
        slot_logits = (
            gate_logits.unsqueeze(-1).unsqueeze(-1) / gate_temperature
            + expert_logits.unsqueeze(-1) / route_temperature
            + subslot_logits / route_temperature
        )
        if visual_mask is not None:
            slot_logits = slot_logits.masked_fill(~visual_mask.unsqueeze(-1).unsqueeze(-1), torch.finfo(slot_logits.dtype).min)
        dispatch = F.softmax(slot_logits, dim=1)
        if visual_mask is not None:
            dispatch = dispatch * visual_mask.unsqueeze(-1).unsqueeze(-1).to(dispatch.dtype)
            dispatch = dispatch / dispatch.sum(dim=1, keepdim=True).clamp_min(1e-6)
        dispatch_flat = dispatch.reshape(dispatch.shape[0], dispatch.shape[1], total_slots)

        value_states = self.imgslot_value_proj(visual_pool)
        slot_tokens = torch.einsum("bnes,bnh->besh", dispatch.float(), value_states.float())
        slot_tokens = slot_tokens.reshape(visual_context.shape[0], total_slots, value_states.shape[-1])
        slot_tokens = self.imgslot_slot_out_proj(slot_tokens.to(visual_pool.dtype))
        slot_pos = torch.einsum("bnes,bnc->besc", dispatch.float(), visual_pos.float())
        slot_pos = slot_pos.reshape(visual_context.shape[0], total_slots, visual_pos.shape[-1])

        expert_usage = dispatch.sum(dim=(1, 3))
        expert_usage_norm = expert_usage / expert_usage.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expert_uniform = torch.full_like(expert_usage_norm, 1.0 / expert_usage_norm.shape[-1])
        expert_balance_loss = ((expert_usage_norm - expert_uniform) ** 2).mean()

        slot_usage = dispatch_flat.sum(dim=1)
        slot_usage_norm = slot_usage / slot_usage.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_uniform = torch.full_like(slot_usage_norm, 1.0 / slot_usage_norm.shape[-1])
        slot_balance_loss = ((slot_usage_norm - slot_uniform) ** 2).mean()

        dispatch_entropy = -(dispatch.clamp_min(1e-6) * dispatch.clamp_min(1e-6).log()).sum(dim=1).mean()

        aux_loss = (
            float(imgslot_config["gate_sparsity_coef"]) * gate_logits.sigmoid().mean()
            + float(imgslot_config["expert_balance_coef"]) * expert_balance_loss
            + float(imgslot_config["slot_balance_coef"]) * slot_balance_loss
            + float(imgslot_config["route_entropy_coef"]) * dispatch_entropy
        )
        return slot_tokens, slot_pos.to(visual_pos.dtype), aux_loss

    def _merge_imgslot_aux_stats(self, aux_stats: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        return torch.stack(aux_stats).mean() if aux_stats else torch.zeros((), device=device)

    def _build_imgslot_visual_positions(self, visual_grids, image_block_offsets, device, dtype):
        spatial_merge_size = int(self.config.vision_config.spatial_merge_size)
        visual_positions = []
        for grid, block_offset in zip(visual_grids, image_block_offsets):
            # positions: [3, visual_seq] -> [visual_seq, 3]
            positions = self.model.get_vision_position_ids(
                0,
                grid,
                1,
                spatial_merge_size,
                device=device,
            ).transpose(0, 1)
            # block_offset: [t_global, h_offset, w_offset] in merged-grid units.
            positions = positions + block_offset.to(device=device, dtype=positions.dtype)
            visual_positions.append(positions.to(device=device, dtype=dtype))
        return visual_positions

    def _run_imgslot_global_visual_context(self, anchor_tokens, anchor_positions, visual_pools, visual_positions):
        visual_device = visual_pools[0].device
        visual_dtype = visual_pools[0].dtype
        visual_lengths = [visual_pool.shape[0] for visual_pool in visual_pools]
        # visual_tokens_global: [1, total_visual_seq, hidden]
        # visual_positions_global: [1, total_visual_seq, 3]
        visual_tokens_global = torch.cat([visual_pool.to(device=visual_device, dtype=visual_dtype) for visual_pool in visual_pools], dim=0).unsqueeze(0)
        visual_positions_global = torch.cat([position.to(device=visual_device, dtype=torch.float32) for position in visual_positions], dim=0).unsqueeze(0)
        visual_key_global, visual_value_global = self._project_imgslot_kv(visual_tokens_global[0], self.imgslot_img_k_proj, self.imgslot_img_v_proj)
        visual_mask_global = torch.ones((1, visual_tokens_global.shape[1]), dtype=torch.bool, device=visual_device)
        return self._run_imgslot_visual_blocks(
            anchor_tokens,
            visual_key_global.to(device=visual_device, dtype=visual_dtype),
            visual_value_global.to(device=visual_device, dtype=visual_dtype),
            visual_tokens_global,
            visual_lengths,
            visual_mask=visual_mask_global,
            anchor_positions=anchor_positions.transpose(0, 1).unsqueeze(1),
            visual_positions=visual_positions_global.permute(2, 0, 1),
        )

    def _compress_imgslot_blocks(self, visual_pools, visual_positions, visual_contexts, visual_gate_scores, visual_device):
        slot_tokens, slot_positions, aux_stats = [], [], []
        for block_idx, (visual_pool, visual_pos) in enumerate(zip(visual_pools, visual_positions)):
            block_mask = torch.ones((1, visual_pool.shape[0]), dtype=torch.bool, device=visual_device)
            compressed_slots, slot_pos, block_aux = self._compress_imgslot_visual_tokens(
                visual_contexts[block_idx],
                visual_pool.unsqueeze(0),
                visual_pos.unsqueeze(0),
                block_mask,
                visual_gate_scores[block_idx],
            )
            slot_tokens.append(compressed_slots[0])
            slot_positions.append(slot_pos[0])
            aux_stats.append(block_aux)
        return slot_tokens, slot_positions, aux_stats

    def _build_imgslot_anchor_positions(self, num_anchors: int, current_last_t: int | float, device, dtype):
        anchor_t = torch.arange(num_anchors, device=device, dtype=dtype) + float(current_last_t) + 1.0
        # Text-like anchor coordinates: temporal/height/width share the same deterministic sequence.
        return torch.stack([anchor_t, anchor_t, anchor_t], dim=-1)

    def _split_imgslot_sample_spans(self, spans: list[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        imgslot_config = self._imgslot_config()
        num_anchors = int(imgslot_config["m"])
        topk_target = int(imgslot_config["k"])
        if not spans:
            raise ValueError("ImgSlot sample must contain one anchor span and at least one visual span.")
        anchor_span = spans[0]
        if anchor_span[1] != num_anchors:
            raise ValueError(f"ImgSlot anchor span length {anchor_span[1]} must equal m ({num_anchors}).")
        visual_spans = spans[1:]
        if not visual_spans:
            raise ValueError("ImgSlot sample must contain at least one visual span.")
        for _, span_length in visual_spans:
            if span_length != topk_target:
                raise ValueError(f"ImgSlot visual span length {span_length} must equal k ({topk_target}).")
        return anchor_span, visual_spans

    def _build_imgslot_states_for_sample(
        self,
        visual_pools,
        visual_grids,
        image_block_offsets,
        text_tokens,
        anchor_span,
        visual_spans,
        text_key=None,
        text_value=None,
        current_last_t: int | float = 0,
    ):
        if len(visual_pools) != len(visual_spans):
            raise ValueError("ImgSlot sample visual block count must match visual span count.")
        if len(visual_grids) != len(visual_spans):
            raise ValueError("ImgSlot sample visual grid count must match visual span count.")
        if len(image_block_offsets) != len(visual_spans):
            raise ValueError("ImgSlot sample block offset count must match visual span count.")
        if not visual_pools:
            raise ValueError("ImgSlot sample must contain at least one visual pool.")

        imgslot_config = self._imgslot_config()
        num_anchors = int(imgslot_config["m"])
        slot_target = int(imgslot_config["k"])
        anchor_span_start, anchor_span_length = anchor_span
        if anchor_span_length != num_anchors:
            raise ValueError(f"ImgSlot anchor span length {anchor_span_length} must equal m ({num_anchors}).")
        visual_device = visual_pools[0].device
        visual_dtype = visual_pools[0].dtype
        # text_tokens: [text_seq, hidden]
        if text_tokens.numel() == 0:
            raise ValueError("ImgSlot requires non-empty text_tokens.")
        if text_key is None or text_value is None:
            text_key, text_value = self._project_imgslot_kv(text_tokens.to(visual_dtype), self.imgslot_text_k_proj, self.imgslot_text_v_proj)

        visual_positions = self._build_imgslot_visual_positions(visual_grids, image_block_offsets, visual_device, torch.float32)

        # anchor_seed: [m, hidden] -> unsqueeze -> [1, m, hidden]
        anchor_seed = self.imgslot_a_tokens.weight[:num_anchors].to(device=visual_device, dtype=visual_dtype)
        text_key = text_key.to(device=visual_device, dtype=visual_dtype)
        text_value = text_value.to(device=visual_device, dtype=visual_dtype)
        # anchor_text: [1, m, hidden]
        anchor_text = self._run_imgslot_text_blocks(anchor_seed.unsqueeze(0), text_key, text_value)
        anchor_positions = self._build_imgslot_anchor_positions(num_anchors, current_last_t, visual_device, torch.float32)
        # shared_anchor_tokens: [1, m, hidden]
        # visual_contexts: list[[1, visual_seq_i, hidden]]
        shared_anchor_tokens, visual_contexts, visual_gate_scores, _anchor_attn = self._run_imgslot_global_visual_context(
            anchor_text,
            anchor_positions,
            visual_pools,
            visual_positions,
        )
        shared_anchor_tokens_single = shared_anchor_tokens[0]

        visual_replacements, visual_slot_positions, aux_stats = self._compress_imgslot_blocks(
            visual_pools,
            visual_positions,
            visual_contexts,
            visual_gate_scores,
            visual_device,
        )
        states = []
        for (span_start, span_length), visual_pool, visual_pos, compressed_slots, slot_pos in zip(
            visual_spans,
            visual_pools,
            visual_positions,
            visual_replacements,
            visual_slot_positions,
        ):
            if span_length != slot_target:
                raise ValueError(f"ImgSlot visual span length {span_length} must equal k ({slot_target}).")
            states.append(
                {
                    "anchor_span": (int(anchor_span_start), int(anchor_span_length)),
                    "span": (int(span_start), int(span_length)),
                    "A": shared_anchor_tokens_single.detach(),  # [m, hidden]
                    "V": visual_pool.detach(),  # [visual_seq_i, hidden]
                    "visual_pos": visual_pos.detach(),  # [visual_seq_i, 3]
                    "compressed_slots": compressed_slots.detach(),  # [k, hidden]
                    "slot_pos": slot_pos.detach(),  # [k, 3]
                }
            )
        return shared_anchor_tokens_single, anchor_positions, visual_replacements, visual_slot_positions, states, self._merge_imgslot_aux_stats(aux_stats, visual_device)

    def _refresh_imgslot_states_for_sample(self, states: list[dict[str, Any]], text_key, text_value, current_last_t: int | float):
        if not states:
            return None, [], None

        imgslot_config = self._imgslot_config()
        visual_dtype = states[0]["V"].dtype
        visual_device = states[0]["V"].device
        slot_target = int(imgslot_config["k"])
        num_anchors = int(imgslot_config["m"])

        text_key = text_key.to(device=visual_device, dtype=visual_dtype)
        text_value = text_value.to(device=visual_device, dtype=visual_dtype)
        # anchors_prev: [1, m, hidden]
        anchors_prev = states[0]["A"].to(device=visual_device, dtype=visual_dtype).unsqueeze(0)
        # anchors_text: [1, m, hidden]
        anchors_text = self._run_imgslot_text_blocks(anchors_prev, text_key, text_value)
        # Exponential-style smoothing between previous anchor state and the
        # latest text-conditioned anchor update.
        anchors_text = (1.0 - imgslot_config["beta"]) * anchors_prev + imgslot_config["beta"] * anchors_text

        visual_pools = [state["V"].to(device=visual_device, dtype=visual_dtype) for state in states]
        visual_positions = [state["visual_pos"].to(device=visual_device, dtype=torch.float32) for state in states]
        anchor_positions = self._build_imgslot_anchor_positions(num_anchors, current_last_t, visual_device, torch.float32)
        # anchors_new: [1, m, hidden]
        # visual_contexts: list[[1, visual_seq_i, hidden]]
        anchors_new, visual_contexts, visual_gate_scores, _anchor_attn = self._run_imgslot_global_visual_context(
            anchors_text,
            anchor_positions,
            visual_pools,
            visual_positions,
        )
        shared_anchor_tokens = anchors_new[0]

        anchor_start, anchor_length = states[0]["anchor_span"]
        anchor_record = {
            "span_start": anchor_start,
            "span_length": anchor_length,
            "slot_tokens": shared_anchor_tokens[:anchor_length],
            "slot_positions": anchor_positions[:anchor_length],
        }
        compressed_slot_list, slot_pos_list, aux_stats = self._compress_imgslot_blocks(
            visual_pools,
            visual_positions,
            visual_contexts,
            visual_gate_scores,
            visual_device,
        )
        visual_records = []
        for state, compressed_slots, slot_pos in zip(states, compressed_slot_list, slot_pos_list):
            prev_slots = state["compressed_slots"].to(device=visual_device, dtype=visual_dtype)
            compressed_slots = imgslot_config["lam"] * prev_slots + (1.0 - imgslot_config["lam"]) * compressed_slots
            state["A"] = shared_anchor_tokens.detach()
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
        return anchor_record, visual_records, self._merge_imgslot_aux_stats(aux_stats, visual_device)

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

    def _prebuild_imgslot_inputs(self, input_ids, attention_mask, inputs_embeds, visual_pools, image_grid_thw, image_block_offsets):
        spans_by_sample = self._image_placeholder_spans(input_ids)
        if sum(max(0, len(spans) - 1) for spans in spans_by_sample) != len(visual_pools):
            raise ValueError("ImgSlot visual block count must match placeholder visual span count.")
        if image_grid_thw is None or image_grid_thw.shape[0] != len(visual_pools):
            raise ValueError("ImgSlot image_grid_thw row count must match visual pool count.")
        if image_block_offsets is None or image_block_offsets.shape[0] != len(visual_pools):
            raise ValueError("ImgSlot image_block_offsets row count must match visual pool count.")
        batch_states: list[list[dict[str, Any]]] = []
        batch_aux_stats: list[dict[str, torch.Tensor]] = []
        pool_index = 0
        # new_inputs_embeds: [batch, seq, hidden]
        new_inputs_embeds = inputs_embeds.clone()
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            text_pos = attention_mask.to(device=inputs_embeds.device, dtype=torch.float32).cumsum(-1) - 1
            text_pos = text_pos.masked_fill(attention_mask.to(device=inputs_embeds.device) == 0, 0)
        else:
            text_pos = torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
        position_ids = text_pos.unsqueeze(0).expand(4, -1, -1).clone()
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        imgslot_config = self._imgslot_config()
        max_text_tokens = int(imgslot_config["max_text_tokens"])
        runtime_text_keys: list[torch.Tensor] = []
        runtime_text_values: list[torch.Tensor] = []
        for batch_idx, spans in enumerate(spans_by_sample):
            if not spans:
                batch_states.append([])
                runtime_text_keys.append(inputs_embeds.new_zeros((1, self.imgslot_num_heads, 0, self.imgslot_head_dim)).detach())
                runtime_text_values.append(inputs_embeds.new_zeros((1, self.imgslot_num_heads, 0, self.imgslot_head_dim)).detach())
                continue
            anchor_span, visual_spans = self._split_imgslot_sample_spans(spans)
            # valid_text_mask: [seq]
            # text_tokens: [text_seq, hidden], with image placeholders removed.
            valid_text_mask = (input_ids[batch_idx] != self.config.image_token_id) & attention_mask[batch_idx].bool()
            text_tokens = inputs_embeds[batch_idx][valid_text_mask][-max_text_tokens:]
            text_key = text_value = None
            if text_tokens.numel() > 0:
                text_key, text_value = self._project_imgslot_kv(text_tokens, self.imgslot_text_k_proj, self.imgslot_text_v_proj)
            sample_visual_pools = []
            sample_visual_grids = []
            sample_block_offsets = []
            for _ in range(len(visual_spans)):
                sample_visual_pools.append(visual_pools[pool_index].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
                sample_visual_grids.append(image_grid_thw[pool_index].to(device=inputs_embeds.device))
                sample_block_offsets.append(image_block_offsets[pool_index].to(device=inputs_embeds.device))
                pool_index += 1
            current_last_t = float(position_ids[1:, batch_idx, attention_mask[batch_idx].bool()].max().item())
            anchor_replacement, anchor_positions, visual_replacements, visual_slot_positions, sample_states, sample_aux = self._build_imgslot_states_for_sample(
                sample_visual_pools,
                sample_visual_grids,
                sample_block_offsets,
                text_tokens,
                anchor_span,
                visual_spans,
                text_key=text_key,
                text_value=text_value,
                current_last_t=current_last_t,
            )
            # Write one shared anchor block back into the first placeholder span.
            anchor_start, anchor_length = anchor_span
            new_inputs_embeds[batch_idx, anchor_start : anchor_start + anchor_length] = anchor_replacement.to(inputs_embeds.dtype)
            position_ids[1:, batch_idx, anchor_start : anchor_start + anchor_length] = anchor_positions[:anchor_length].transpose(0, 1).to(position_ids.dtype)
            # Write one compressed visual block back into each remaining placeholder span.
            for replacement, slot_positions, (span_start, span_length) in zip(visual_replacements, visual_slot_positions, visual_spans):
                new_inputs_embeds[batch_idx, span_start : span_start + span_length] = replacement.to(inputs_embeds.dtype)
                position_ids[1:, batch_idx, span_start : span_start + span_length] = slot_positions[:span_length].transpose(0, 1).to(position_ids.dtype)
            batch_states.append(sample_states)
            batch_aux_stats.append(sample_aux)
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

    def _collect_imgslot_refresh_records(self, runtime, current_last_t: int | float):
        text_keys_by_batch = runtime["text_keys"]
        text_values_by_batch = runtime["text_values"]
        anchor_records = []
        visual_records = []
        aux_stats = []
        aux_device = None
        for batch_idx, states in enumerate(runtime["states"]):
            if batch_idx >= len(text_keys_by_batch) or batch_idx >= len(text_values_by_batch):
                continue
            anchor_record, sample_visual_records, sample_aux = self._refresh_imgslot_states_for_sample(
                states,
                text_keys_by_batch[batch_idx],
                text_values_by_batch[batch_idx],
                current_last_t,
            )
            if anchor_record is not None:
                anchor_record["batch_idx"] = batch_idx
                anchor_records.append(anchor_record)
                aux_device = anchor_record["slot_tokens"].device
            for record in sample_visual_records:
                record["batch_idx"] = batch_idx
                visual_records.append(record)
                aux_device = record["slot_tokens"].device
            if sample_aux is not None:
                aux_stats.append(sample_aux)
        if aux_device is not None:
            self._imgslot_aux = self._merge_imgslot_aux_stats(aux_stats, aux_device)
        return anchor_records, visual_records

    def _maybe_refresh_imgslot_cache(self, past_key_values):
        runtime = self._imgslot_runtime
        if not runtime["enabled"] or past_key_values is None:
            return past_key_values
        runtime["step"] += 1
        if runtime["step"] % runtime["delta"]:
            return past_key_values
        layers = self.model.language_model.layers
        layer_types = self.config.text_config.layer_types
        current_last_t = max(0, past_key_values.get_seq_length() - 1)
        anchor_records, visual_records = self._collect_imgslot_refresh_records(runtime, current_last_t)
        if not anchor_records and not visual_records:
            return past_key_values

        # Only overwrite layers that use native full attention. Non-full-attention
        # layers keep their own state transition logic untouched.
        for layer_idx, layer in enumerate(layers):
            if layer_idx < len(layer_types) and layer_types[layer_idx] != "full_attention":
                continue
            self._write_imgslot_records_for_layer(past_key_values, layer_idx, layer, anchor_records)
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
            new_key, new_value = self._project_imgslot_kv(last_hidden[batch_idx], self.imgslot_text_k_proj, self.imgslot_text_v_proj)
            # Append one new text token into the per-sample runtime KV memory, then
            # keep only the latest max_text_tokens entries along seq dim (-2).
            text_keys[batch_idx] = torch.cat([text_keys[batch_idx].to(new_key.device, new_key.dtype), new_key], dim=-2)[:, :, -max_text_tokens:, :].detach()
            text_values[batch_idx] = torch.cat([text_values[batch_idx].to(new_value.device, new_value.dtype), new_value], dim=-2)[:, :, -max_text_tokens:, :].detach()

    @auto_docstring
    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[TransformersKwargs]):
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(sum(raw_video_patches), patch_volume)`):
            Packed flattened raw video patches across the whole batch.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The per-video `(T, H, W)` grid in LLM token space.
        """
        return self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    @auto_docstring
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[TransformersKwargs]):
        r"""
        pixel_values (`torch.FloatTensor` of shape `(sum(raw_image_patches), patch_volume)`):
            Packed flattened raw image patches across the whole batch.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The per-image `(T, H, W)` grid in LLM token space.
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
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5CausalLMOutputWithPast:
        pre_refresh_past_key_values = past_key_values
        imgslot_first_prefill = bool(kwargs.pop("imgslot_first_prefill", False))
        # Decode path: refresh cached slot tokens before entering the decoder so the
        # next token attends to the latest anchor/Top-K visual state.
        if self._imgslot_is_enabled() and past_key_values is not None:
            past_key_values = self._maybe_refresh_imgslot_cache(past_key_values)

        # Prefill path with images:
        # - input_ids: [batch, seq]
        # - inputs_embeds: [batch, seq, hidden]
        # - visual_pools: flat list over all image blocks in the batch, each item is
        #   [visual_seq_i, hidden]
        # This rewrites placeholder spans in inputs_embeds, then skips the normal
        # Qwen multimodal scatter path by clearing pixel/grid/mm-token inputs.
        if self._imgslot_is_enabled() and (imgslot_first_prefill or past_key_values is None) and input_ids is not None and pixel_values is not None:
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
            lm_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
            aux_loss = self._imgslot_aux.to(lm_loss.device) * float(self.config.img_slot_aux_loss_coef)
            loss = lm_loss + aux_loss

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
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            image_block_offsets=image_block_offsets,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if self._imgslot_is_enabled() and is_first_iteration and pixel_values is not None:
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
            position_ids = text_positions[None, ...] + self.model.rope_deltas
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


__all__ = ["FoveaForConditionalGeneration"]
