import math
from typing import Any

import torch
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
    TransformersKwargs,
    Unpack,
)
from .configuration_fovea import FoveaConfig


class FoveaForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    """Multimodal causal LM head for text, image, and video generation.

    ImgSlot overview:
    - Prefill: replace text-side image placeholder spans with one shared anchor span
      plus one Top-K visual span per image block.
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
        num_heads = next((h for h in (16, 12, 8, 6, 4, 3, 2, 1) if hidden_size % h == 0), 1)
        self.imgslot_num_heads = num_heads
        # Anchor token table: [m, hidden].
        self.imgslot_a_tokens = nn.Embedding(num_anchor_tokens, hidden_size)
        # Shared ImgSlot attention blocks. These do not replace decoder self-attn;
        # they only synthesize/refresh slot tokens before writing them back into
        # decoder KV cache.
        self.imgslot_text_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_text_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_text_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_text_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_img_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_img_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_img_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_img_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.imgslot_attn_norm = nn.LayerNorm(hidden_size)
        self.imgslot_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.imgslot_ffn_norm = nn.LayerNorm(hidden_size)
        # Runtime state used only during generation refresh.
        # states: list[batch] -> list[visual_block_state]
        # text_keys/text_values: per-sample cached text KV, each shaped
        #   [1, num_heads, text_seq, head_dim]
        self._imgslot_runtime = {"enabled": False, "states": [], "delta": 1, "step": 0}
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _imgslot_is_enabled(self) -> bool:
        return bool(self.config.img_slot_enable)

    def _imgslot_config(self) -> dict[str, float | int]:
        if self._imgslot_is_enabled() and self.config.img_slot_tile_size is None:
            raise ValueError("img_slot_tile_size is required when img_slot_enable=true.")
        return {
            "m": int(self.config.img_slot_m),
            "k": int(self.config.img_slot_k),
            "delta": int(self.config.img_slot_delta),
            "beta": float(self.config.img_slot_beta),
            "lam": float(self.config.img_slot_lambda),
            "max_text_tokens": int(self.config.img_slot_max_text_tokens),
        }

    def _project_imgslot_kv(self, tokens, k_proj, v_proj):
        # tokens: [seq, hidden]
        # -> key/value: [1, num_heads, seq, head_dim]
        num_heads = self.imgslot_num_heads
        hidden_dim = tokens.shape[-1]
        head_dim = hidden_dim // num_heads
        key = k_proj(tokens).view(1, -1, num_heads, head_dim).transpose(1, 2)
        value = v_proj(tokens).view(1, -1, num_heads, head_dim).transpose(1, 2)
        return key, value

    def _project_imgslot_text_tokens(self, text_tokens):
        return self._project_imgslot_kv(
            text_tokens,
            self.imgslot_text_k_proj,
            self.imgslot_text_v_proj,
        )

    def _project_imgslot_visual_pool(self, visual_pool):
        return self._project_imgslot_kv(
            visual_pool,
            self.imgslot_img_k_proj,
            self.imgslot_img_v_proj,
        )

    def _run_imgslot_attention(self, anchor_tokens, key_states, value_states, q_proj, o_proj, mask=None):
        # anchor_tokens: [block_count, slot_seq, hidden]
        # key_states/value_states: [block_count, num_heads, kv_seq, head_dim]
        # mask: [block_count, kv_seq] where True means valid token.
        # query: [block_count, num_heads, slot_seq, head_dim]
        # attention_logits: [block_count, num_heads, slot_seq, kv_seq]
        # context: [block_count, slot_seq, hidden]
        num_heads = self.imgslot_num_heads
        hidden_dim = anchor_tokens.shape[-1]
        head_dim = hidden_dim // num_heads
        block_count = anchor_tokens.shape[0]
        query = q_proj(anchor_tokens).view(block_count, -1, num_heads, head_dim).transpose(1, 2)
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
        updated_anchor_tokens, _attention_weights = self._run_imgslot_attention(
            anchor_tokens,
            text_key,
            text_value,
            self.imgslot_text_q_proj,
            self.imgslot_text_o_proj,
            mask=text_mask,
        )
        return updated_anchor_tokens

    def _run_imgslot_visual_blocks(self, anchor_tokens, visual_key, visual_value, visual_mask=None):
        updated_anchor_tokens, attention_weights = self._run_imgslot_attention(
            anchor_tokens,
            visual_key,
            visual_value,
            self.imgslot_img_q_proj,
            self.imgslot_img_o_proj,
            mask=visual_mask,
        )
        # attention_weights: [block_count, num_heads, m, visual_seq]
        # mean(1): average over heads -> [block_count, m, visual_seq]
        # max(1): any anchor can nominate a visual token -> [block_count, visual_seq]
        scores = attention_weights.mean(1).max(1).values
        if visual_mask is not None:
            scores = scores.masked_fill(~visual_mask, torch.finfo(scores.dtype).min)
        return updated_anchor_tokens, scores

    def _pad_imgslot_visual_kv(self, visual_kv, device, dtype):
        # visual_kv: list[(key, value)] where each key/value is
        #   [1, num_heads, visual_seq_i, head_dim]
        # After padding:
        #   visual_keys_padded/visual_values_padded: [block_count, num_heads, max_visual_tokens, head_dim]
        #   visual_mask: [block_count, max_visual_tokens]
        block_count = len(visual_kv)
        visual_lengths = [visual_key.shape[-2] for visual_key, _visual_value in visual_kv]
        max_visual_tokens = max(visual_lengths)
        num_heads = self.imgslot_num_heads
        head_dim = visual_kv[0][0].shape[-1]
        visual_keys_padded = visual_kv[0][0].new_zeros((block_count, num_heads, max_visual_tokens, head_dim), device=device, dtype=dtype)
        visual_values_padded = visual_kv[0][1].new_zeros((block_count, num_heads, max_visual_tokens, head_dim), device=device, dtype=dtype)
        visual_mask = torch.zeros((block_count, max_visual_tokens), dtype=torch.bool, device=device)
        for block_idx, (visual_key, visual_value) in enumerate(visual_kv):
            visual_len = visual_key.shape[-2]
            visual_keys_padded[block_idx, :, :visual_len, :] = visual_key[0].to(device=device, dtype=dtype)
            visual_values_padded[block_idx, :, :visual_len, :] = visual_value[0].to(device=device, dtype=dtype)
            visual_mask[block_idx, :visual_len] = True
        return visual_keys_padded, visual_values_padded, visual_mask

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

    def _build_imgslot_states_for_sample(self, visual_pools, text_tokens, anchor_span, visual_spans, text_key=None, text_value=None):
        if len(visual_pools) != len(visual_spans):
            raise ValueError("ImgSlot sample visual block count must match visual span count.")
        if not visual_pools:
            raise ValueError("ImgSlot sample must contain at least one visual pool.")

        imgslot_config = self._imgslot_config()
        num_anchors = int(imgslot_config["m"])
        topk_target = int(imgslot_config["k"])
        anchor_span_start, anchor_span_length = anchor_span
        if anchor_span_length != num_anchors:
            raise ValueError(f"ImgSlot anchor span length {anchor_span_length} must equal m ({num_anchors}).")
        visual_device = visual_pools[0].device
        visual_dtype = visual_pools[0].dtype
        # text_tokens: [text_seq, hidden]
        if text_tokens.numel() == 0:
            raise ValueError("ImgSlot requires non-empty text_tokens.")
        if text_key is None or text_value is None:
            text_key, text_value = self._project_imgslot_text_tokens(text_tokens.to(visual_dtype))

        block_count = len(visual_pools)
        # Each visual_pool: [visual_seq_i, hidden]
        # visual_kv entries: ([1, num_heads, visual_seq_i, head_dim], same for value)
        visual_kv = [self._project_imgslot_visual_pool(visual_pool) for visual_pool in visual_pools]
        visual_keys_padded, visual_values_padded, visual_mask = self._pad_imgslot_visual_kv(
            visual_kv,
            device=visual_device,
            dtype=visual_dtype,
        )

        # anchor_seed: [m, hidden] -> unsqueeze -> [1, m, hidden]
        anchor_seed = self.imgslot_a_tokens.weight[:num_anchors].to(device=visual_device, dtype=visual_dtype)
        text_key = text_key.to(device=visual_device, dtype=visual_dtype)
        text_value = text_value.to(device=visual_device, dtype=visual_dtype)
        # anchor_text: [1, m, hidden]
        anchor_text = self._run_imgslot_text_blocks(anchor_seed.unsqueeze(0), text_key, text_value)
        # Expand one shared anchor set to all visual blocks, then let each block attend
        # its own visual pool.
        # shared_anchor_tokens: [block_count, m, hidden]
        # scores_padded: [block_count, max_visual_tokens]
        shared_anchor_tokens, scores_padded = self._run_imgslot_visual_blocks(
            anchor_text.expand(block_count, -1, -1),
            visual_keys_padded,
            visual_values_padded,
            visual_mask=visual_mask,
        )
        shared_anchor_tokens_single = shared_anchor_tokens[0]

        visual_replacements = []
        states = []
        for block_idx, ((span_start, span_length), visual_pool, visual_pair) in enumerate(zip(visual_spans, visual_pools, visual_kv)):
            if span_length != topk_target:
                raise ValueError(f"ImgSlot visual span length {span_length} must equal k ({topk_target}).")
            if visual_pool.shape[0] < topk_target:
                raise ValueError(
                    f"ImgSlot visual pool has {visual_pool.shape[0]} tokens, fewer than k ({topk_target}). "
                    "Increase tile size, reduce img_slot_k, or increase block resolution."
                )
            # scores: [visual_seq_i] -> topk_idx: [k] -> visual_topk: [k, hidden]
            scores = scores_padded[block_idx, : visual_pool.shape[0]]
            topk_idx = torch.topk(scores, k=topk_target, dim=0).indices
            visual_topk = visual_pool[topk_idx]
            visual_replacements.append(visual_topk)
            states.append(
                {
                    # Shared anchor span for the whole sample.
                    "anchor_span": (int(anchor_span_start), int(anchor_span_length)),
                    # This visual block's text-side replacement span.
                    "span": (int(span_start), int(span_length)),
                    # Detached runtime state used only for generation refresh.
                    "A": shared_anchor_tokens_single.detach(),  # [m, hidden]
                    "V": visual_pool.detach(),  # [visual_seq_i, hidden]
                    "V_k": visual_pair[0].detach(),  # [1, num_heads, visual_seq_i, head_dim]
                    "V_v": visual_pair[1].detach(),  # [1, num_heads, visual_seq_i, head_dim]
                    "V_topk": visual_topk.detach(),  # [k, hidden]
                    "score_prev": scores.detach(),  # [visual_seq_i]
                    "topk_idx": topk_idx.detach(),  # [k]
                }
            )
        return shared_anchor_tokens_single, visual_replacements, states

    def _refresh_imgslot_states_for_sample(self, states: list[dict[str, Any]], text_key, text_value):
        if not states:
            return None, []

        imgslot_config = self._imgslot_config()
        visual_dtype = states[0]["V"].dtype
        visual_device = states[0]["V"].device
        block_count = len(states)
        topk_target = int(imgslot_config["k"])

        text_key = text_key.to(device=visual_device, dtype=visual_dtype)
        text_value = text_value.to(device=visual_device, dtype=visual_dtype)
        # anchors_prev: [1, m, hidden]
        anchors_prev = states[0]["A"].to(device=visual_device, dtype=visual_dtype).unsqueeze(0)
        # anchors_text: [1, m, hidden]
        anchors_text = self._run_imgslot_text_blocks(anchors_prev, text_key, text_value)
        # Exponential-style smoothing between previous anchor state and the
        # latest text-conditioned anchor update.
        anchors_text = (1.0 - imgslot_config["beta"]) * anchors_prev + imgslot_config["beta"] * anchors_text

        visual_kv = [(state["V_k"], state["V_v"]) for state in states]
        visual_keys_padded, visual_values_padded, visual_mask = self._pad_imgslot_visual_kv(
            visual_kv,
            device=visual_device,
            dtype=visual_dtype,
        )
        # anchors_new: [block_count, m, hidden]
        # scores_current_padded: [block_count, max_visual_tokens]
        anchors_new, scores_current_padded = self._run_imgslot_visual_blocks(
            anchors_text.expand(block_count, -1, -1),
            visual_keys_padded,
            visual_values_padded,
            visual_mask=visual_mask,
        )
        shared_anchor_tokens = anchors_new[0]

        anchor_start, anchor_length = states[0]["anchor_span"]
        anchor_record = {
            "span_start": anchor_start,
            "span_length": anchor_length,
            "slot_tokens": shared_anchor_tokens[:anchor_length],
        }
        visual_records = []
        for state_idx, state in enumerate(states):
            visual_pool = state["V"]
            if visual_pool.shape[0] < topk_target:
                raise ValueError(
                    f"ImgSlot visual pool has {visual_pool.shape[0]} tokens, fewer than k ({topk_target}). "
                    "Increase tile size, reduce img_slot_k, or increase block resolution."
                )
            # scores_current / score_prev: [visual_seq_i]
            # topk_idx: [k], visual_topk: [k, hidden]
            scores_current = scores_current_padded[state_idx, : visual_pool.shape[0]]
            scores = imgslot_config["lam"] * state["score_prev"] + (1.0 - imgslot_config["lam"]) * scores_current
            topk_idx = torch.topk(scores, k=topk_target, dim=0).indices
            visual_topk = visual_pool[topk_idx]
            state["A"] = shared_anchor_tokens.detach()
            state["V_topk"] = visual_topk.detach()
            state["score_prev"] = scores.detach()
            state["topk_idx"] = topk_idx.detach()
            states[state_idx] = state
            span_start, span_length = state["span"]
            visual_records.append(
                {
                    "span_start": span_start,
                    "span_length": span_length,
                    "slot_tokens": state["V_topk"][:span_length],
                }
            )
        return anchor_record, visual_records

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

    def _build_imgslot_visual_pools(self, pixel_values, image_grid_thw):
        if pixel_values is None or image_grid_thw is None:
            return None
        image_outputs = self.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, return_dict=True)
        return list(image_outputs.pooler_output)

    def _prebuild_imgslot_inputs(self, input_ids, attention_mask, inputs_embeds, visual_pools):
        spans_by_sample = self._image_placeholder_spans(input_ids)
        if sum(max(0, len(spans) - 1) for spans in spans_by_sample) != len(visual_pools):
            raise ValueError("ImgSlot visual block count must match placeholder visual span count.")
        batch_states: list[list[dict[str, Any]]] = []
        pool_index = 0
        # new_inputs_embeds: [batch, seq, hidden]
        new_inputs_embeds = inputs_embeds.clone()
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        imgslot_config = self._imgslot_config()
        max_text_tokens = int(imgslot_config["max_text_tokens"])
        runtime_text_keys: list[torch.Tensor] = []
        runtime_text_values: list[torch.Tensor] = []
        for batch_idx, spans in enumerate(spans_by_sample):
            if not spans:
                batch_states.append([])
                runtime_text_keys.append(inputs_embeds.new_zeros((1, self.imgslot_num_heads, 0, inputs_embeds.shape[-1] // self.imgslot_num_heads)).detach())
                runtime_text_values.append(inputs_embeds.new_zeros((1, self.imgslot_num_heads, 0, inputs_embeds.shape[-1] // self.imgslot_num_heads)).detach())
                continue
            anchor_span, visual_spans = self._split_imgslot_sample_spans(spans)
            # valid_text_mask: [seq]
            # text_tokens: [text_seq, hidden], with image placeholders removed.
            valid_text_mask = (input_ids[batch_idx] != self.config.image_token_id) & attention_mask[batch_idx].bool()
            text_tokens = inputs_embeds[batch_idx][valid_text_mask][-max_text_tokens:]
            text_key = text_value = None
            if text_tokens.numel() > 0:
                text_key, text_value = self._project_imgslot_text_tokens(text_tokens)
            sample_visual_pools = []
            for _ in range(len(visual_spans)):
                sample_visual_pools.append(visual_pools[pool_index].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
                pool_index += 1
            anchor_replacement, visual_replacements, sample_states = self._build_imgslot_states_for_sample(
                sample_visual_pools,
                text_tokens,
                anchor_span,
                visual_spans,
                text_key=text_key,
                text_value=text_value,
            )
            # Write one shared anchor block back into the first placeholder span.
            anchor_start, anchor_length = anchor_span
            new_inputs_embeds[batch_idx, anchor_start : anchor_start + anchor_length] = anchor_replacement.to(inputs_embeds.dtype)
            # Write one Top-K visual block back into each remaining placeholder span.
            for replacement, (span_start, span_length) in zip(visual_replacements, visual_spans):
                new_inputs_embeds[batch_idx, span_start : span_start + span_length] = replacement.to(inputs_embeds.dtype)
            batch_states.append(sample_states)
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
        return new_inputs_embeds, attention_mask

    def _slot_kv_for_layer(self, layer, slot_tokens, slot_positions):
        attention = layer.self_attn
        layer_device = next(layer.parameters()).device
        # slot_tokens: [num_records, slot_seq, hidden]
        # slot_positions: [num_records, slot_seq]
        slot_tokens = slot_tokens.to(device=layer_device)
        slot_positions = slot_positions.to(device=layer_device)
        input_shape = slot_tokens.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        # key_states/value_states before RoPE: [num_records, num_kv_heads/num_heads, slot_seq, head_dim]
        key_states = attention.k_norm(attention.k_proj(slot_tokens).view(hidden_shape)).transpose(1, 2)
        value_states = attention.v_proj(slot_tokens).view(hidden_shape).transpose(1, 2)
        # Rebuild per-layer KV exactly in the decoder layer's parameter space, then
        # apply RoPE to key so the overwritten cache matches native full-attn cache layout.
        dummy_query = key_states.new_zeros((slot_tokens.shape[0], attention.config.num_attention_heads, slot_tokens.shape[1], attention.head_dim))
        position_embeddings = self.model.language_model.rotary_emb(slot_tokens, slot_positions)
        _, key_states = apply_rotary_pos_emb(dummy_query, key_states, *position_embeddings)
        return key_states, value_states

    def _overwrite_cache_layer(self, past_key_values, layer_idx: int, batch_idx: int, span_start: int, span_end: int, key_states, value_states) -> bool:
        cache_layer = past_key_values.layers[layer_idx]
        keys = cache_layer.keys
        values = cache_layer.values
        if keys.numel() == 0 or span_end > keys.shape[-2]:
            return False
        # keys/values: [batch, num_kv_heads/num_heads, cache_seq, head_dim]
        # key_states/value_states: [1, num_kv_heads/num_heads, slot_seq, head_dim]
        keys[batch_idx, :, span_start:span_end] = key_states[0].to(keys.dtype)
        values[batch_idx, :, span_start:span_end] = value_states[0].to(values.dtype)
        return True

    def _build_imgslot_positions(self, records, layer_device):
        return torch.stack([
            torch.arange(record["span_start"], record["span_start"] + record["span_length"], device=layer_device, dtype=torch.long)
            for record in records
        ], dim=0)

    def _write_imgslot_records_for_layer(self, past_key_values, layer_idx, layer, records):
        if not records:
            return
        layer_device = next(layer.parameters()).device
        slot_tokens = torch.stack([record["slot_tokens"].to(device=layer_device) for record in records], dim=0)
        slot_positions = self._build_imgslot_positions(records, layer_device)
        slot_kv = self._slot_kv_for_layer(layer, slot_tokens, slot_positions)
        if slot_kv is None:
            return
        slot_keys, slot_values = slot_kv
        for record_idx, record in enumerate(records):
            span_start = record["span_start"]
            span_end = span_start + record["span_length"]
            self._overwrite_cache_layer(
                past_key_values,
                layer_idx,
                record["batch_idx"],
                span_start,
                span_end,
                slot_keys[record_idx : record_idx + 1],
                slot_values[record_idx : record_idx + 1],
            )

    def _collect_imgslot_refresh_records(self, runtime):
        text_keys_by_batch = runtime["text_keys"]
        text_values_by_batch = runtime["text_values"]
        anchor_records = []
        visual_records = []
        for batch_idx, states in enumerate(runtime["states"]):
            if batch_idx >= len(text_keys_by_batch) or batch_idx >= len(text_values_by_batch):
                continue
            anchor_record, sample_visual_records = self._refresh_imgslot_states_for_sample(
                states,
                text_keys_by_batch[batch_idx],
                text_values_by_batch[batch_idx],
            )
            if anchor_record is not None:
                anchor_record["batch_idx"] = batch_idx
                anchor_records.append(anchor_record)
            for record in sample_visual_records:
                record["batch_idx"] = batch_idx
                visual_records.append(record)
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
        anchor_records, visual_records = self._collect_imgslot_refresh_records(runtime)
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
            # last_hidden[batch_idx]: [1, hidden] -> new_key/new_value: [1, num_heads, 1, head_dim]
            new_key, new_value = self._project_imgslot_text_tokens(last_hidden[batch_idx])
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
            visual_pools = self._build_imgslot_visual_pools(pixel_values, image_grid_thw)
            if not visual_pools:
                raise RuntimeError("ImgSlot: empty visual pools.")
            inputs_embeds, attention_mask = self._prebuild_imgslot_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                visual_pools=visual_pools,
            )
            input_ids = None
            pixel_values = None
            image_grid_thw = None
            mm_token_type_ids = None
            if position_ids is None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

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

    def _get_image_nums_and_video_nums(self, input_ids: torch.LongTensor | None, inputs_embeds: torch.Tensor | None = None):
        return super()._get_image_nums_and_video_nums(input_ids=input_ids, inputs_embeds=inputs_embeds)

    def _expand_inputs_for_generation(self, expand_size: int = 1, is_encoder_decoder: bool = False, input_ids: torch.LongTensor | None = None, **model_kwargs):
        return super()._expand_inputs_for_generation(expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **model_kwargs)


__all__ = ["FoveaForConditionalGeneration"]
