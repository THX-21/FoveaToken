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
        """Return video features for packed video inputs.

        Args:
            pixel_values_videos: Packed video pixel tensor.
            video_grid_thw: Temporal-height-width grid metadata for each packed video.
        """
        return self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    @auto_docstring
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[TransformersKwargs]):
        """Return image features for packed image inputs.

        Args:
            pixel_values: Packed image pixel tensor.
            image_grid_thw: Temporal-height-width grid metadata for each packed image.
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

    def _split_retrieve_features_by_sample(self, retrieve_pixel_values, retrieve_grid_thw, retrieve_image_counts, batch_size):
        outputs = self.model.get_image_features(pixel_values=retrieve_pixel_values, image_grid_thw=retrieve_grid_thw, return_dict=True)
        flat_features = list(outputs.pooler_output)
        counts = retrieve_image_counts.to(device="cpu", dtype=torch.long).tolist()
        per_sample, start = [], 0
        for count in counts:
            count = int(count)
            sample_features = flat_features[start : start + count]
            per_sample.append(torch.cat(sample_features, dim=0))
            start += count
        if len(per_sample) != batch_size:
            raise ValueError("retrieve_image_counts must contain one entry per batch sample.")
        return per_sample

    def _pad_sample_memory(self, memories, patch_boxes, device, dtype):
        max_len = max(mem.shape[0] for mem in memories)
        hidden = memories[0].shape[-1]
        batch_size = len(memories)
        memory = torch.zeros((batch_size, max_len, hidden), device=device, dtype=dtype)
        mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.bool)
        boxes = torch.zeros((batch_size, max_len, 4), device=device, dtype=torch.float32)
        box_start = 0
        for batch_idx, mem in enumerate(memories):
            count = mem.shape[0]
            memory[batch_idx, :count] = mem.to(device=device, dtype=dtype)
            mask[batch_idx, :count] = True
            boxes[batch_idx, :count] = patch_boxes[box_start : box_start + count].to(device=device, dtype=torch.float32)
            box_start += count
        return memory, mask, boxes

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
    ):
        if retrieve_pixel_values is None or retrieve_grid_thw is None or retrieve_patch_boxes is None or retrieve_image_counts is None:
            raise ValueError("Visual-query training requires retrieve_pixel_values, retrieve_grid_thw, retrieve_patch_boxes, and retrieve_image_counts.")

        memories = self._split_retrieve_features_by_sample(
            retrieve_pixel_values,
            retrieve_grid_thw,
            retrieve_image_counts,
            hidden_states.shape[0],
        )
        memory, memory_mask, patch_boxes = self._pad_sample_memory(memories, retrieve_patch_boxes, hidden_states.device, hidden_states.dtype)

        code_pos = vq_code_positions.to(device=hidden_states.device, dtype=torch.long)
        code_batch = code_pos[:, 0]
        code_token = code_pos[:, 1]
        query_hidden = hidden_states[code_batch, code_token]
        memory_for_code = memory.index_select(0, code_batch)
        mask_for_code = memory_mask.index_select(0, code_batch)
        boxes_for_code = patch_boxes.index_select(0, code_batch)

        heads = self.visual_query_num_heads
        head_dim = self.visual_query_head_dim
        q = self.visual_query_q_proj(query_hidden).view(-1, heads, head_dim)
        k = self.visual_query_k_proj(memory_for_code).view(query_hidden.shape[0], memory.shape[1], heads, head_dim)
        v = self.visual_query_v_proj(memory_for_code).view(query_hidden.shape[0], memory.shape[1], heads, head_dim)
        q = self.visual_query_q_norm(q).transpose(0, 1).unsqueeze(2)
        k = self.visual_query_k_norm(k).permute(0, 2, 1, 3).transpose(0, 1)
        v = v.permute(0, 2, 1, 3).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-1, -2)).squeeze(2) * (head_dim**-0.5)
        scores = scores.masked_fill((~mask_for_code).unsqueeze(0), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn.unsqueeze(2), v).squeeze(2).transpose(0, 1).reshape(query_hidden.shape[0], -1)
        replay_vectors = self.visual_query_o_proj(context).to(hidden_states.dtype)

        attn_mean = attn.mean(dim=0)
        centers = (boxes_for_code[..., :2] + boxes_for_code[..., 2:]) * 0.5
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
        soft_xy = (attn_mean.float().unsqueeze(-1) * centers.float()).sum(dim=1)
        scale = max(memory.shape[1] ** 0.5, 1.0)
        replay_offsets = torch.stack([soft_xy[:, 1] * scale, soft_xy[:, 0] * scale], dim=-1)
        return {
            "vectors": replay_vectors,
            "offsets": replay_offsets,
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

        replay_offsets = retrieval["offsets"].round().to(device=pos.device, dtype=pos.dtype)
        base_t = pos[0, batch_ids, token_ids]
        base_h = pos[1, batch_ids, token_ids]
        base_w = pos[2, batch_ids, token_ids]
        replay_visual_pos = torch.stack(
            [
                base_t,
                base_h + replay_offsets[:, 0],
                base_w + replay_offsets[:, 1],
            ],
            dim=0,
        )
        pos[:, batch_ids, token_ids] = replay_visual_pos
        return out, pos

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
