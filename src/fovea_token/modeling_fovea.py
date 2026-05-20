"""Fovea visual-query replay model.

This file intentionally contains only the current mechanism:
`<vq> <vis_i> ... </vq>` hidden states retrieve high-resolution visual memory,
then replay vectors are scattered into the replay placeholder positions for a
second decoder pass.
"""

import torch
from torch import nn
from torch.nn import init

from .configuration_fovea import FoveaConfig
from .modeling_qwen3_5 import (
    Cache,
    GenerationMixin,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    TransformersKwargs,
    Unpack,
    auto_docstring,
    can_return_tuple,
)
from .train.data import IGNORE_INDEX


class FoveaForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    """Qwen3.5 multimodal LM with packed visual-query retrieval."""

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    config: FoveaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, config.text_config.vocab_size, bias=False)

        self.visual_query_num_heads = int(config.text_config.num_attention_heads)
        self.visual_query_head_dim = int(getattr(config.text_config, "head_dim", hidden_size // self.visual_query_num_heads))
        if self.visual_query_num_heads * self.visual_query_head_dim != hidden_size:
            raise ValueError("Visual-query retrieval requires num_attention_heads * head_dim to equal hidden_size.")

        self.visual_query_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.visual_query_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.visual_query_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.visual_query_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.visual_query_q_norm = Qwen3_5RMSNorm(self.visual_query_head_dim, eps=config.text_config.rms_norm_eps)
        self.visual_query_k_norm = Qwen3_5RMSNorm(self.visual_query_head_dim, eps=config.text_config.rms_norm_eps)
        self._visual_query_aux: dict[str, torch.Tensor] = {}
        self.post_init()
        self._init_visual_query_modules()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        output_loading_info = bool(kwargs.get("output_loading_info", False))
        loaded = super().from_pretrained(*args, **kwargs)
        if output_loading_info:
            model, loading_info = loaded
            model._repair_visual_query_init()
            return model, loading_info
        loaded._repair_visual_query_init()
        return loaded

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @torch.no_grad()
    def _init_visual_query_modules(self) -> None:
        std = float(getattr(self.config.text_config, "initializer_range", 0.02))
        for module in (
            self.visual_query_q_proj,
            self.visual_query_k_proj,
            self.visual_query_v_proj,
            self.visual_query_o_proj,
        ):
            init.normal_(module.weight, mean=0.0, std=std)
        init.zeros_(self.visual_query_q_norm.weight)
        init.zeros_(self.visual_query_k_norm.weight)

    @torch.no_grad()
    def _repair_visual_query_init(self) -> None:
        """Repair missing visual-query params after checkpoint loading."""

        std = float(getattr(self.config.text_config, "initializer_range", 0.02))
        for module in (
            self.visual_query_q_proj,
            self.visual_query_k_proj,
            self.visual_query_v_proj,
            self.visual_query_o_proj,
        ):
            weight = module.weight
            if torch.isfinite(weight).all() and weight.float().std() > 0:
                continue
            init.normal_(weight, mean=0.0, std=std)
        for module in (self.visual_query_q_norm, self.visual_query_k_norm):
            if not torch.isfinite(module.weight).all():
                init.zeros_(module.weight)

    @auto_docstring
    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[TransformersKwargs]):
        r"""
        pixel_values_videos (`torch.FloatTensor`):
            Packed video pixel tensor.
        video_grid_thw (`torch.LongTensor`, *optional*):
            Temporal-height-width grid metadata for each packed video.
        """
        return self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    @auto_docstring
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[TransformersKwargs]):
        r"""
        pixel_values (`torch.FloatTensor`):
            Packed image pixel tensor.
        image_grid_thw (`torch.LongTensor`, *optional*):
            Temporal-height-width grid metadata for each packed image.
        """
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    def _build_multimodal_embeddings_and_positions(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        pixel_values_videos=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        input_embeds_override=None,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids) if input_embeds_override is None else input_embeds_override
        if pixel_values is not None:
            image_outputs = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_outputs = self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, return_dict=True)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        self.model.rope_deltas = None
        position_ids = self.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        return inputs_embeds, position_ids

    def _position_ids_from_embeds(
        self,
        input_ids,
        attention_mask,
        inputs_embeds,
        image_grid_thw,
        video_grid_thw=None,
        mm_token_type_ids=None,
    ):
        self.model.rope_deltas = None
        return self.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )

    def _language_forward_from_embeds(self, inputs_embeds, attention_mask, position_ids, **kwargs):
        return self.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def _masked_visual_code_embeddings(self, input_ids, labels, vq_code_label_mask):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        mask_id = self.config.mask_vis_token_id
        if mask_id is None or vq_code_label_mask is None or vq_code_label_mask.numel() == 0:
            return inputs_embeds
        if not self.training:
            return inputs_embeds
        mask = vq_code_label_mask.to(device=input_ids.device).bool()
        mask_ratio = torch.rand((), device=input_ids.device)
        mask = mask & (torch.rand(mask.shape, device=input_ids.device) < mask_ratio) & labels.ne(IGNORE_INDEX)
        if not mask.any():
            return inputs_embeds
        mask_embeds = self.get_input_embeddings()(torch.full_like(input_ids, int(mask_id)))
        return torch.where(mask.unsqueeze(-1), mask_embeds, inputs_embeds)

    def _use_generated_replay(self) -> bool:
        if not self.training:
            return False
        prob = float(getattr(self.config, "visual_query_generated_replay_prob", 0.0))
        if prob <= 0:
            return False
        if prob >= 1:
            return True
        return bool(torch.rand((), device=next(self.parameters()).device) < prob)

    def _generated_code_ids(self, input_ids, logits, vq_code_positions):
        vis_start = self.config.vis_token_start_id
        vis_end = self.config.vis_token_end_id
        if vis_start is None or vis_end is None or vq_code_positions.numel() == 0:
            return input_ids
        code_pos = vq_code_positions.to(device=logits.device, dtype=torch.long)
        code_batch = code_pos[:, 0]
        code_token = code_pos[:, 1]
        predictor_token = (code_token - 1).clamp_min(0)
        code_logits = logits[code_batch, predictor_token, int(vis_start) : int(vis_end) + 1]
        generated = code_logits.argmax(dim=-1).to(device=input_ids.device) + int(vis_start)
        generated_ids = input_ids.clone()
        generated_ids[code_batch.to(input_ids.device), code_token.to(input_ids.device)] = generated
        return generated_ids

    def _retrieve_image_position_bases(self, position_ids, mm_token_type_ids, retrieve_image_counts, device):
        counts = retrieve_image_counts.to(device="cpu", dtype=torch.long).tolist()
        if position_ids is None or mm_token_type_ids is None:
            return [0] * sum(int(count) for count in counts)
        bases = []
        for batch_idx, count in enumerate(counts):
            image_mask = mm_token_type_ids[batch_idx].to(device=position_ids.device).eq(1)
            starts = torch.where(image_mask & torch.cat([image_mask.new_tensor([True]), ~image_mask[:-1]]))[0]
            for item in starts[: int(count)]:
                bases.append(int(position_ids[:, batch_idx, item].min().item()))
            bases.extend([0] * (int(count) - min(len(starts), int(count))))
        return bases

    def _split_retrieve_features_by_sample(
        self,
        retrieve_pixel_values,
        retrieve_grid_thw,
        retrieve_image_counts,
        batch_size,
        position_ids=None,
        mm_token_type_ids=None,
    ):
        outputs = self.model.get_image_features(pixel_values=retrieve_pixel_values, image_grid_thw=retrieve_grid_thw, return_dict=True)
        flat_features = list(outputs.pooler_output)
        position_bases = self._retrieve_image_position_bases(
            position_ids,
            mm_token_type_ids,
            retrieve_image_counts,
            retrieve_pixel_values.device,
        )
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        flat_positions = [
            self.model.get_vision_position_ids(
                position_bases[idx],
                grid,
                1,
                spatial_merge_size,
                device=retrieve_pixel_values.device,
            ).transpose(0, 1)
            for idx, grid in enumerate(retrieve_grid_thw)
        ]
        counts = retrieve_image_counts.to(device="cpu", dtype=torch.long).tolist()
        per_sample, per_sample_positions, start = [], [], 0
        for count in counts:
            count = int(count)
            sample_features = flat_features[start : start + count]
            sample_positions = flat_positions[start : start + count]
            per_sample.append(torch.cat(sample_features, dim=0))
            per_sample_positions.append(torch.cat(sample_positions, dim=0))
            start += count
        if len(per_sample) != batch_size:
            raise ValueError("retrieve_image_counts must contain one entry per batch sample.")
        return per_sample, per_sample_positions

    def _pad_sample_memory(self, memories, memory_positions, patch_boxes, device, dtype):
        max_len = max(mem.shape[0] for mem in memories)
        hidden = memories[0].shape[-1]
        batch_size = len(memories)
        memory = torch.zeros((batch_size, max_len, hidden), device=device, dtype=dtype)
        positions = torch.zeros((batch_size, max_len, 3), device=device, dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.bool)
        boxes = torch.zeros((batch_size, max_len, 4), device=device, dtype=torch.float32)
        box_start = 0
        for batch_idx, (mem, pos) in enumerate(zip(memories, memory_positions)):
            count = mem.shape[0]
            memory[batch_idx, :count] = mem.to(device=device, dtype=dtype)
            positions[batch_idx, :count] = pos.to(device=device, dtype=torch.long)
            mask[batch_idx, :count] = True
            boxes[batch_idx, :count] = patch_boxes[box_start : box_start + count].to(device=device, dtype=torch.float32)
            box_start += count
        return memory, positions, mask, boxes

    def _retrieve_visual_queries(
        self,
        hidden_states,
        retrieve_pixel_values,
        retrieve_grid_thw,
        retrieve_patch_boxes,
        retrieve_image_counts,
        vq_code_positions,
        vq_replay_positions,
        vq_code_query_indices,
        vq_boxes,
        position_ids=None,
        mm_token_type_ids=None,
    ):
        if retrieve_pixel_values is None or retrieve_grid_thw is None or retrieve_patch_boxes is None or retrieve_image_counts is None:
            raise ValueError("Visual-query training requires retrieve_pixel_values, retrieve_grid_thw, retrieve_patch_boxes, and retrieve_image_counts.")

        memories, memory_positions = self._split_retrieve_features_by_sample(
            retrieve_pixel_values,
            retrieve_grid_thw,
            retrieve_image_counts,
            hidden_states.shape[0],
            position_ids,
            mm_token_type_ids,
        )
        memory, memory_positions, memory_mask, patch_boxes = self._pad_sample_memory(
            memories,
            memory_positions,
            retrieve_patch_boxes,
            hidden_states.device,
            hidden_states.dtype,
        )

        code_pos = vq_code_positions.to(device=hidden_states.device, dtype=torch.long)
        code_batch = code_pos[:, 0]
        code_token = code_pos[:, 1]
        query_hidden = hidden_states[code_batch, code_token]
        memory_for_code = memory.index_select(0, code_batch)
        memory_pos_for_code = memory_positions.index_select(0, code_batch)
        mask_for_code = memory_mask.index_select(0, code_batch)
        boxes_for_code = patch_boxes.index_select(0, code_batch)

        heads = self.visual_query_num_heads
        head_dim = self.visual_query_head_dim
        q = self.visual_query_q_proj(query_hidden).view(-1, heads, head_dim)
        k = self.visual_query_k_proj(memory_for_code).view(query_hidden.shape[0], memory.shape[1], heads, head_dim)
        v = self.visual_query_v_proj(memory_for_code).view(query_hidden.shape[0], memory.shape[1], heads, head_dim)
        q = self.visual_query_q_norm(q).unsqueeze(2)
        k = self.visual_query_k_norm(k).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)).squeeze(2) * (head_dim**-0.5)
        scores = scores.masked_fill((~mask_for_code).unsqueeze(1), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn.unsqueeze(2), v).squeeze(2).reshape(query_hidden.shape[0], -1)
        replay_vectors = self.visual_query_o_proj(context).to(hidden_states.dtype)

        attn_mean = attn.mean(dim=1)
        centers = (boxes_for_code[..., :2] + boxes_for_code[..., 2:]) * 0.5
        if vq_boxes is None or vq_boxes.numel() == 0:
            align_loss = replay_vectors.new_zeros(())
        else:
            query_boxes = vq_boxes.to(device=hidden_states.device, dtype=torch.float32).index_select(
                0,
                vq_code_query_indices.to(device=hidden_states.device, dtype=torch.long),
            )
            inside = (
                (centers[..., 0] >= query_boxes[:, 0:1])
                & (centers[..., 0] <= query_boxes[:, 2:3])
                & (centers[..., 1] >= query_boxes[:, 1:2])
                & (centers[..., 1] <= query_boxes[:, 3:4])
                & mask_for_code
            )
            align_loss = -((attn_mean * inside.to(attn_mean.dtype)).sum(dim=-1).clamp_min(float(self.config.visual_query_align_eps)).log()).mean()
        replay_positions = (attn_mean.float().unsqueeze(-1) * memory_pos_for_code.float()).sum(dim=1)
        replay_spatial_offsets = replay_positions[:, 1:] - replay_positions[:, 0:1]
        return {
            "vectors": replay_vectors,
            "spatial_offsets": replay_spatial_offsets,
            "align_loss": align_loss,
            "replay_positions": vq_replay_positions.to(device=hidden_states.device, dtype=torch.long),
            "replay_query_indices": vq_code_query_indices.to(device=hidden_states.device, dtype=torch.long),
        }

    def _scatter_replay(self, base_embeds, position_ids, retrieval):
        replay_pos = retrieval["replay_positions"]
        vectors = retrieval["vectors"].to(device=base_embeds.device, dtype=base_embeds.dtype)
        if replay_pos.shape[0] != vectors.shape[0]:
            raise ValueError("Replay position count must equal retrieved vector count.")
        out = base_embeds.clone()
        out[replay_pos[:, 0], replay_pos[:, 1]] = vectors
        if position_ids is None:
            return out, None
        pos = position_ids.clone()
        replay_query = retrieval["replay_query_indices"].to(device=pos.device, dtype=torch.long)
        batch_ids = replay_pos[:, 0].to(device=pos.device, dtype=torch.long)
        token_ids = replay_pos[:, 1].to(device=pos.device, dtype=torch.long)

        for batch_idx in batch_ids.unique(sorted=True):
            batch_mask = batch_ids == batch_idx
            batch_queries = replay_query[batch_mask]
            batch_tokens = token_ids[batch_mask]
            batch_idx = int(batch_idx.item())
            for query_idx in batch_queries.unique(sorted=True):
                query_tokens = batch_tokens[batch_queries == query_idx]
                first_token = int(query_tokens.min().item())
                extra_tokens = int(query_tokens.numel() - 1)
                if extra_tokens > 0 and first_token + 1 < pos.shape[-1]:
                    pos[:, batch_idx, first_token + 1 :] -= extra_tokens

        base_t = pos[0, batch_ids, token_ids]
        spatial_offsets = retrieval["spatial_offsets"].round().to(device=pos.device, dtype=pos.dtype)
        replay_visual_pos = torch.stack(
            [
                base_t,
                base_t + spatial_offsets[:, 0],
                base_t + spatial_offsets[:, 1],
            ],
            dim=0,
        )
        pos[:, batch_ids, token_ids] = replay_visual_pos
        return out, pos

    def _cached_decode_position_ids(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
        position_ids = position_ids.clamp_min(0).view(1, attention_mask.shape[0], 1).repeat(3, 1, 1)
        if self.model.rope_deltas is not None:
            position_ids = position_ids + self.model.rope_deltas.to(position_ids.device).view(1, -1, 1)
        return position_ids

    def _visual_query_state(self, input_ids, prompt_len, max_codes=64):
        ids = input_ids[0, prompt_len:].tolist()
        start_id = self.config.vq_start_token_id
        end_id = self.config.vq_end_token_id
        vis_start = self.config.vis_token_start_id
        vis_end = self.config.vis_token_end_id
        if None in (start_id, end_id, vis_start, vis_end):
            return False, 0, False
        start = None
        for idx, token_id in enumerate(ids):
            if token_id == start_id:
                start = idx
            elif token_id == end_id and start is not None:
                start = None
        if start is None:
            return False, 0, False
        code_count = sum(1 for token_id in ids[start + 1 :] if int(vis_start) <= token_id <= int(vis_end))
        return True, code_count, code_count < int(max_codes)

    def _constrained_next_token(self, logits, input_ids, prompt_len):
        vis_start = int(self.config.vis_token_start_id)
        vis_end = int(self.config.vis_token_end_id)
        end_id = int(self.config.vq_end_token_id)
        replay_id = int(self.config.replay_token_id)
        in_query, code_count, can_add_code = self._visual_query_state(input_ids, prompt_len)
        masked = logits.new_full(logits.shape, torch.finfo(logits.dtype).min)
        if in_query:
            if can_add_code:
                masked[:, vis_start : vis_end + 1] = logits[:, vis_start : vis_end + 1]
            if code_count > 0:
                masked[:, end_id] = logits[:, end_id]
        else:
            masked.copy_(logits)
            masked[:, vis_start : vis_end + 1] = torch.finfo(logits.dtype).min
            masked[:, end_id] = torch.finfo(logits.dtype).min
            masked[:, replay_id] = torch.finfo(logits.dtype).min
        return masked.argmax(dim=-1, keepdim=True)

    def _query_spans_with_replay(self, input_ids):
        ids = input_ids[0].tolist()
        start_id = self.config.vq_start_token_id
        end_id = self.config.vq_end_token_id
        replay_id = self.config.replay_token_id
        vis_start = self.config.vis_token_start_id
        vis_end = self.config.vis_token_end_id
        spans = []
        pos = 0
        while pos < len(ids):
            if ids[pos] != start_id:
                pos += 1
                continue
            end = pos + 1
            while end < len(ids) and ids[end] != end_id:
                end += 1
            if end >= len(ids):
                break
            codes = [idx for idx in range(pos + 1, end) if int(vis_start) <= ids[idx] <= int(vis_end)]
            replay_start = end + 1
            replay_end = replay_start
            while replay_end < len(ids) and ids[replay_end] == replay_id:
                replay_end += 1
            spans.append((codes, replay_start, replay_end))
            pos = replay_end
        return spans

    def _insert_missing_replay_pads(self, input_ids, attention_mask, mm_token_type_ids, inputs_embeds):
        replay_id = int(self.config.replay_token_id)
        spans = self._query_spans_with_replay(input_ids)
        offset = 0
        for codes, replay_start, replay_end in spans:
            missing = len(codes) - (replay_end - replay_start)
            if missing <= 0:
                continue
            insert_at = replay_start + offset
            pad_ids = input_ids.new_full((1, missing), replay_id)
            pad_mask = attention_mask.new_ones((1, missing))
            pad_types = mm_token_type_ids.new_zeros((1, missing)) if mm_token_type_ids is not None else None
            pad_embeds = self.get_input_embeddings()(pad_ids).to(inputs_embeds.dtype)
            input_ids = torch.cat([input_ids[:, :insert_at], pad_ids, input_ids[:, insert_at:]], dim=1)
            attention_mask = torch.cat([attention_mask[:, :insert_at], pad_mask, attention_mask[:, insert_at:]], dim=1)
            inputs_embeds = torch.cat([inputs_embeds[:, :insert_at], pad_embeds, inputs_embeds[:, insert_at:]], dim=1)
            if mm_token_type_ids is not None:
                mm_token_type_ids = torch.cat([mm_token_type_ids[:, :insert_at], pad_types, mm_token_type_ids[:, insert_at:]], dim=1)
            offset += missing
        return input_ids, attention_mask, mm_token_type_ids, inputs_embeds

    def _inference_replay_metadata(self, input_ids):
        code_positions, replay_positions, query_indices = [], [], []
        for query_idx, (codes, replay_start, replay_end) in enumerate(self._query_spans_with_replay(input_ids)):
            if not codes or replay_end - replay_start != len(codes):
                continue
            code_positions.extend(codes)
            replay_positions.extend(range(replay_start, replay_end))
            query_indices.extend([query_idx] * len(codes))
        device = input_ids.device
        return (
            torch.tensor([[0, pos] for pos in code_positions], device=device, dtype=torch.long),
            torch.tensor([[0, pos] for pos in replay_positions], device=device, dtype=torch.long),
            torch.tensor(query_indices, device=device, dtype=torch.long),
        )

    def _refresh_replay_cache(
        self,
        input_ids,
        attention_mask,
        mm_token_type_ids,
        inputs_embeds,
        image_grid_thw,
        video_grid_thw,
        retrieve_pixel_values,
        retrieve_grid_thw,
        retrieve_patch_boxes,
        retrieve_image_counts,
        model_kwargs,
    ):
        input_ids, attention_mask, mm_token_type_ids, inputs_embeds = self._insert_missing_replay_pads(
            input_ids,
            attention_mask,
            mm_token_type_ids,
            inputs_embeds,
        )
        code_pos, replay_pos, query_indices = self._inference_replay_metadata(input_ids)
        if code_pos.numel() == 0:
            return input_ids, attention_mask, mm_token_type_ids, inputs_embeds, None
        position_ids = self._position_ids_from_embeds(
            input_ids,
            attention_mask,
            inputs_embeds,
            image_grid_thw,
            video_grid_thw,
            mm_token_type_ids,
        )
        outputs_a = self._language_forward_from_embeds(inputs_embeds, attention_mask, position_ids, **model_kwargs)
        retrieval = self._retrieve_visual_queries(
            outputs_a[0],
            retrieve_pixel_values,
            retrieve_grid_thw,
            retrieve_patch_boxes,
            retrieve_image_counts,
            code_pos,
            replay_pos,
            query_indices,
            None,
            position_ids,
            mm_token_type_ids,
        )
        inputs_embeds, position_ids = self._scatter_replay(inputs_embeds, position_ids, retrieval)
        outputs_b = self._language_forward_from_embeds(inputs_embeds, attention_mask, position_ids, **model_kwargs)
        return input_ids, attention_mask, mm_token_type_ids, inputs_embeds, outputs_b

    @torch.no_grad()
    def visual_query_generate(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        retrieve_pixel_values=None,
        retrieve_grid_thw=None,
        retrieve_patch_boxes=None,
        retrieve_image_counts=None,
        max_new_tokens=128,
        eos_token_id=None,
        pad_token_id=None,
        use_cache=True,
        do_sample=False,
        num_beams=1,
        **kwargs,
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("visual-query replay generation currently expects batch_size=1.")
        if do_sample or int(num_beams) != 1:
            raise ValueError("visual-query replay generation supports greedy decoding only.")
        if not use_cache:
            raise ValueError("visual-query replay generation requires use_cache=True.")
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        if mm_token_type_ids is None:
            mm_token_type_ids = input_ids.new_zeros(input_ids.shape)

        model_kwargs = {"use_cache": use_cache}
        for key in ("output_attentions", "output_hidden_states", "return_dict"):
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        prompt_len = input_ids.shape[1]
        inputs_embeds, position_ids = self._build_multimodal_embeddings_and_positions(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            mm_token_type_ids,
        )
        outputs = self._language_forward_from_embeds(inputs_embeds, attention_mask, position_ids, **model_kwargs)
        logits = self.lm_head(outputs[0][:, -1:, :]).squeeze(1)
        past_key_values = outputs.past_key_values
        eos_ids = {int(eos_token_id)} if isinstance(eos_token_id, int) else {int(item) for item in (eos_token_id or [])}

        for _ in range(int(max_new_tokens)):
            next_token = self._constrained_next_token(logits, input_ids, prompt_len)
            next_embed = self.get_input_embeddings()(next_token).to(inputs_embeds.dtype)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
            mm_token_type_ids = torch.cat([mm_token_type_ids, mm_token_type_ids.new_zeros((1, 1))], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

            if int(next_token.item()) in eos_ids:
                break

            if int(next_token.item()) == int(self.config.vq_end_token_id) and retrieve_pixel_values is not None:
                input_ids, attention_mask, mm_token_type_ids, inputs_embeds, outputs = self._refresh_replay_cache(
                    input_ids,
                    attention_mask,
                    mm_token_type_ids,
                    inputs_embeds,
                    image_grid_thw,
                    video_grid_thw,
                    retrieve_pixel_values,
                    retrieve_grid_thw,
                    retrieve_patch_boxes,
                    retrieve_image_counts,
                    model_kwargs,
                )
                if outputs is not None:
                    logits = self.lm_head(outputs[0][:, -1:, :]).squeeze(1)
                    past_key_values = outputs.past_key_values
                    continue

            position_ids = self._cached_decode_position_ids(attention_mask)
            outputs = self(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
        return input_ids

    def generate(self, *args, **kwargs):
        if kwargs.get("retrieve_pixel_values") is not None:
            return self.visual_query_generate(*args, **kwargs)
        return super().generate(*args, **kwargs)

    def _visual_query_forward(
        self,
        input_ids,
        attention_mask,
        labels,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        mm_token_type_ids,
        retrieve_pixel_values,
        retrieve_grid_thw,
        retrieve_patch_boxes,
        retrieve_image_counts,
        vq_code_positions,
        vq_replay_positions,
        vq_code_query_indices,
        vq_boxes,
        vq_code_label_mask,
        logits_to_keep=0,
        **kwargs,
    ):
        masked_embeds = self._masked_visual_code_embeddings(input_ids, labels, vq_code_label_mask)
        embeds_a, position_ids_a = self._build_multimodal_embeddings_and_positions(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            mm_token_type_ids,
            input_embeds_override=masked_embeds,
        )
        outputs_a = self._language_forward_from_embeds(embeds_a, attention_mask, position_ids_a, **kwargs)
        hidden_a = outputs_a[0]
        logits_a = self.lm_head(hidden_a)
        code_mask = vq_code_label_mask.to(device=labels.device).bool()
        use_generated = self._use_generated_replay()
        pass_b_input_ids = self._generated_code_ids(input_ids, logits_a, vq_code_positions) if use_generated else input_ids

        retrieval = self._retrieve_visual_queries(
            hidden_a,
            retrieve_pixel_values,
            retrieve_grid_thw,
            retrieve_patch_boxes,
            retrieve_image_counts,
            vq_code_positions,
            vq_replay_positions,
            vq_code_query_indices,
            vq_boxes,
            position_ids_a,
            mm_token_type_ids,
        )
        if use_generated:
            base_embeds = embeds_a.clone()
            code_pos = vq_code_positions.to(device=base_embeds.device, dtype=torch.long)
            generated_embeds = self.get_input_embeddings()(pass_b_input_ids)
            base_embeds[code_pos[:, 0], code_pos[:, 1]] = generated_embeds[code_pos[:, 0], code_pos[:, 1]]
            position_ids_b = self._position_ids_from_embeds(
                pass_b_input_ids,
                attention_mask,
                base_embeds,
                image_grid_thw,
                video_grid_thw,
                mm_token_type_ids,
            )
        else:
            base_embeds = embeds_a
            position_ids_b = self._position_ids_from_embeds(
                input_ids,
                attention_mask,
                base_embeds,
                image_grid_thw,
                video_grid_thw,
                mm_token_type_ids,
            )
        embeds_b, position_ids_b = self._scatter_replay(base_embeds, position_ids_b, retrieval)
        outputs_b = self._language_forward_from_embeds(embeds_b, attention_mask, position_ids_b, **kwargs)
        hidden_b = outputs_b[0]
        full_logits_b = self.lm_head(hidden_b)
        code_predictor_mask = torch.zeros_like(code_mask)
        code_predictor_mask[:, :-1] = code_mask[:, 1:]
        mixed_logits = torch.where(code_predictor_mask.unsqueeze(-1).to(full_logits_b.device), logits_a, full_logits_b)
        lm_loss = self.loss_function(
            logits=mixed_logits,
            labels=labels,
            vocab_size=mixed_logits.shape[-1],
        )
        align_loss = retrieval["align_loss"]
        loss = lm_loss + float(self.config.visual_query_lambda_align) * align_loss
        self._visual_query_aux = {
            "lm_loss": lm_loss.detach(),
            "align_loss": align_loss.detach(),
            "generated_replay": hidden_b.new_tensor(float(use_generated)),
            "num_queries": hidden_b.new_tensor(float(vq_boxes.shape[0])),
        }
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=full_logits_b[:, slice_indices, :],
            past_key_values=outputs_b.past_key_values,
            hidden_states=outputs_b.hidden_states,
            attentions=outputs_b.attentions,
            rope_deltas=self.model.rope_deltas,
        )

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
        retrieve_pixel_values: torch.Tensor | None = None,
        retrieve_grid_thw: torch.LongTensor | None = None,
        retrieve_patch_boxes: torch.Tensor | None = None,
        retrieve_image_counts: torch.Tensor | None = None,
        vq_code_positions: torch.LongTensor | None = None,
        vq_replay_positions: torch.LongTensor | None = None,
        vq_code_query_indices: torch.LongTensor | None = None,
        vq_boxes: torch.Tensor | None = None,
        vq_code_label_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5CausalLMOutputWithPast:
        if labels is not None and vq_code_positions is not None and vq_code_positions.numel() > 0:
            return self._visual_query_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                retrieve_pixel_values=retrieve_pixel_values,
                retrieve_grid_thw=retrieve_grid_thw,
                retrieve_patch_boxes=retrieve_patch_boxes,
                retrieve_image_counts=retrieve_image_counts,
                vq_code_positions=vq_code_positions,
                vq_replay_positions=vq_replay_positions,
                vq_code_query_indices=vq_code_query_indices,
                vq_boxes=vq_boxes,
                vq_code_label_mask=vq_code_label_mask,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
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


__all__ = ["FoveaForConditionalGeneration"]
