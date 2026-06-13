"""Fixed-token Fovea model on top of the transformers Qwen3.5 multimodal model."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers import Qwen3_5Model, Qwen3_5PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5RMSNorm,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from .configuration_fovea import FoveaConfig


class KwargsForCausalLM(TransformersKwargs, total=False):
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
    mask_diag = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn_base = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)

    # --- functional prefix scan (no in-place ops on non-leaf tensors) ---
    # Compute (I - A_base)^{-1} - I row-by-row for strictly lower triangular A_base.
    # Avoiding in-place attn[..., idx, :idx] = ... which breaks autograd through
    # k_beta, key, and decay_mask (all depend on the SSM projections).
    accum_rows: list[torch.Tensor] = [attn_base[..., 0:1, :]]
    for idx in range(1, chunk_size):
        orig_prefix = attn_base[..., idx:idx + 1, :idx]                  # (..., 1, idx)
        prev_mat = torch.cat(accum_rows, dim=-2)[..., :idx]              # (..., idx, idx)
        prev_mat = prev_mat.unsqueeze(-3)                                # (..., 1, idx, idx)
        new_prefix = orig_prefix + (orig_prefix.unsqueeze(-1) * prev_mat).sum(-2)
        tail = attn_base[..., idx:idx + 1, idx:]                         # keep tail zero
        accum_rows.append(torch.cat([new_prefix, tail], dim=-1))
    attn = torch.cat(accum_rows, dim=-2) + torch.eye(chunk_size, dtype=attn_base.dtype, device=attn_base.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    core_attn_chunks: list[torch.Tensor] = []
    for idx in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, idx], key[:, :, idx], value[:, :, idx]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, idx]).masked_fill(mask, 0)
        v_prime = (k_cumdecay[:, :, idx]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, idx, :, None].exp()) @ last_recurrent_state
        core_attn_chunks.append((attn_inter + attn @ v_new).unsqueeze(2))
        last_recurrent_state = (
            last_recurrent_state * g[:, :, idx, -1, None, None].exp()
            + (k_i * (g[:, :, idx, -1, None] - g[:, :, idx]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = torch.cat(core_attn_chunks, dim=2)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class FoveaForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    """Qwen3.5 multimodal LM with 64 fixed fovea vectors inserted after `<fovea>`."""

    config_class = FoveaConfig
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    config: FoveaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, config.text_config.vocab_size, bias=False)

        self.fovea_num_heads = int(config.text_config.num_attention_heads)
        # Qwen3.5 configs may expose a `head_dim` that does not satisfy
        # `num_attention_heads * head_dim == hidden_size`. Fovea blocks reshape
        # full hidden states across heads, so they must derive their own
        # per-head width from `hidden_size`.
        if hidden_size % self.fovea_num_heads != 0:
            raise ValueError("Fovea retrieval requires hidden_size to be divisible by num_attention_heads.")
        self.fovea_head_dim = hidden_size // self.fovea_num_heads

        eps = float(getattr(config.text_config, "rms_norm_eps", 1e-6))
        self.fovea_tokens = nn.Embedding(int(config.fovea_num_tokens), hidden_size)
        self.fovea_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fovea_q_norm = Qwen3_5RMSNorm(self.fovea_head_dim, eps=eps)
        self.fovea_k_norm = Qwen3_5RMSNorm(self.fovea_head_dim, eps=eps)
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
        self._fovea_grad_diag: dict[str, float] = {}
        self._fovea_grad_trace: dict[str, float] = {}  # survives aux overwrites
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

    def _trace_grad(self, tensor: torch.Tensor, name: str) -> None:
        """Register a backward hook that records the gradient norm of *tensor*."""
        if not tensor.requires_grad:
            return

        def hook(grad):
            if grad is not None:
                self._fovea_grad_trace[name] = grad.detach().float().norm().item()
            else:
                self._fovea_grad_trace[name] = -1.0  # signal: hook fired but grad is None

        tensor.register_hook(hook)

    def _register_fovea_backward_hooks(self) -> None:
        """Register full backward hooks on fovea modules for gradient diagnostic."""

        modules_to_watch = {
            "fovea_tokens": self.fovea_tokens,
            "fovea_q_proj": self.fovea_q_proj,
            "fovea_k_proj": self.fovea_k_proj,
            "fovea_v_proj": self.fovea_v_proj,
            "fovea_o_proj": self.fovea_o_proj,
            "fovea_q_norm": self.fovea_q_norm,
            "fovea_k_norm": self.fovea_k_norm,
            "fovea_ssm_in_qkv": self.fovea_ssm_in_proj_qkv,
            "fovea_ssm_in_z": self.fovea_ssm_in_proj_z,
            "fovea_ssm_in_beta": self.fovea_ssm_in_proj_b,
            "fovea_ssm_in_a": self.fovea_ssm_in_proj_a,
            "fovea_ssm_dt_bias": self.fovea_ssm_dt_bias,
            "fovea_ssm_A_log": self.fovea_ssm_A_log,
            "fovea_ssm_out": self.fovea_ssm_out_proj,
            "fovea_ssm_norm": self.fovea_ssm_norm,
        }

        def _make_bw_hook(module_name):
            def hook(_module, _grad_input, _grad_output):
                grads = {}
                for pname, param in _module.named_parameters():
                    if param.grad is not None:
                        grads[pname] = param.grad.detach().float().norm().item()
                    else:
                        grads[pname] = 0.0
                self._fovea_grad_diag[module_name] = grads
                # Also inject into aux so the callback finds it
                aux = self._fovea_aux
                if isinstance(aux, dict):
                    aux.setdefault("_bw_grads", {})
                    aux["_bw_grads"].setdefault(module_name, {})
                    aux["_bw_grads"][module_name] = grads
            return hook

        for name, module in modules_to_watch.items():
            module.register_full_backward_hook(_make_bw_hook(name))

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
            # HF from_pretrained sometimes zeroes custom module weights; only reinit
            # if the tensor is essentially dead (abs max < 1e-20 catches subnormals).
            if tensor.float().abs().max() >= 1e-20:
                continue
            init.normal_(tensor, mean=0.0, std=std)
        if self.fovea_ssm_A_log.weight.float().abs().max() < 1e-20:
            init.uniform_(self.fovea_ssm_A_log.weight, 1e-4, 16)
            self.fovea_ssm_A_log.weight.log_()
        if self.fovea_ssm_dt_bias.weight.float().abs().max() < 1e-20:
            init.ones_(self.fovea_ssm_dt_bias.weight)
        for module in (self.fovea_q_norm, self.fovea_k_norm, self.fovea_ssm_norm):
            if module.weight.float().abs().max() < 1e-20:
                init.ones_(module.weight)

    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[KwargsForCausalLM]):
        return self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None, **kwargs: Unpack[KwargsForCausalLM]):
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
        image_hidden_states = None
        if pixel_values is not None:
            image_outputs = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, return_dict=True)
            image_hidden_states = image_outputs.pooler_output
            image_embeds = torch.cat(image_hidden_states, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
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
        return inputs_embeds, position_ids, image_hidden_states

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

    def _language_forward_from_embeds(self, inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
        return self.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def _split_retrieve_features_by_sample(self, retrieve_pixel_values, retrieve_grid_thw, retrieve_image_counts):
        outputs = self.model.get_image_features(pixel_values=retrieve_pixel_values, image_grid_thw=retrieve_grid_thw, return_dict=True)
        features = list(outputs.pooler_output)
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
        retrieve_grid_thw,
        retrieve_patch_boxes,
        retrieve_image_counts,
        batch_size,
        device=None,
        dtype=None,
    ):
        if retrieve_pixel_values is None or retrieve_grid_thw is None or retrieve_patch_boxes is None or retrieve_image_counts is None:
            raise ValueError("Fovea retrieval requires retrieve_pixel_values, retrieve_grid_thw, retrieve_patch_boxes, and retrieve_image_counts.")
        memories = self._split_retrieve_features_by_sample(retrieve_pixel_values, retrieve_grid_thw, retrieve_image_counts)
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
                    "retrieve_patch_boxes is shorter than the Qwen3.5 retrieval memory: "
                    f"need {box_start + count}, got {patch_boxes.shape[0]}."
                )
            if count > 0:
                boxes[batch_idx, :count] = patch_boxes[box_start : box_start + count].to(device=device, dtype=torch.float32)
            box_start += count
        if box_start != patch_boxes.shape[0]:
            raise ValueError(
                "retrieve_patch_boxes must match the Qwen3.5 retrieval memory length exactly: "
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
            text_mask &= input_ids.to(device=device).ne(int(self.config.image_token_id))
        if fovea_positions.numel() == 0:
            return hidden_states.new_empty((0, num_fovea, hidden_states.shape[-1]))

        batch_size, seq_len, hidden_size = hidden_states.shape
        positions = fovea_positions.to(device=device, dtype=torch.long)
        a_log = self.fovea_ssm_A_log.weight[0].float()
        dt_bias = self.fovea_ssm_dt_bias.weight[0]

        # Single SSM pass over the text (no per-fovea-token broadcast).
        # The SSM compresses the sequence into a context representation;
        # fovea tokens interact with it only at trigger positions.
        x = hidden_states * text_mask[..., None].to(dtype)

        mixed_qkv = self.fovea_ssm_in_proj_qkv(x)
        query, key, value = mixed_qkv.split(hidden_size, dim=-1)
        query = query.view(batch_size, seq_len, self.fovea_num_heads, self.fovea_head_dim)
        key = key.view(batch_size, seq_len, self.fovea_num_heads, self.fovea_head_dim)
        value = value.view(batch_size, seq_len, self.fovea_num_heads, self.fovea_head_dim)
        gate = self.fovea_ssm_in_proj_z(x).view(batch_size, seq_len, self.fovea_num_heads, self.fovea_head_dim)
        beta = torch.sigmoid(self.fovea_ssm_in_proj_b(x))
        g = -a_log.exp() * F.softplus(self.fovea_ssm_in_proj_a(x).float() + dt_bias)
        beta = beta * text_mask[..., None].to(beta.dtype)
        g = torch.where(text_mask[..., None], g, torch.zeros_like(g))

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
        gate_flat = gate.reshape(-1, self.fovea_head_dim)
        states = self.fovea_ssm_norm(core_attn_out, gate_flat)
        states = states.reshape(batch_size, seq_len, hidden_size)
        states = self.fovea_ssm_out_proj(states)

        # At trigger positions, fovea token embeddings interact with the SSM context.
        trigger_states = states[positions[:, 0], positions[:, 1], :]            # (num_triggers, hidden_size)
        base = self.fovea_tokens.weight.to(device=device, dtype=dtype)          # (num_fovea, hidden_size)
        fovea_queries = trigger_states.unsqueeze(1) + base.unsqueeze(0)          # (num_triggers, num_fovea, hidden_size)

        self._trace_grad(fovea_queries, "ssm_fovea_queries")
        return fovea_queries

    def _retrieve_fovea_from_memory(self, fovea_queries, trigger_batch, memory, memory_mask, patch_boxes, fovea_box_indices=None, fovea_boxes=None):
        device = fovea_queries.device
        dtype = fovea_queries.dtype
        num_triggers = int(fovea_queries.shape[0])
        num_fovea = int(self.config.fovea_num_tokens)
        if num_triggers == 0:
            return (fovea_queries.new_empty((0, num_fovea, fovea_queries.shape[-1])),
                    fovea_queries.new_zeros(()))
        heads = self.fovea_num_heads
        head_dim = self.fovea_head_dim
        memory_len = int(memory.shape[1])

        q_all = self.fovea_q_proj(fovea_queries).view(num_triggers, num_fovea, heads, head_dim)
        q_all = self.fovea_q_norm(q_all)
        self._trace_grad(q_all, "retrieval_q_all")     # after q_proj + q_norm

        scale = head_dim**-0.5
        trigger_batch = trigger_batch.to(device=device, dtype=torch.long)

        # Build output tensors functionally (no in-place into detached leaf)
        # to preserve autograd connectivity through fovea retrieval.
        vector_parts: list[torch.Tensor] = [None] * num_triggers  # type: ignore[list-item]
        attn_mean_parts: list[torch.Tensor] = [None] * num_triggers  # type: ignore[list-item]

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
            # num diag: v, memory, q
            mem_f = sample_memory.detach().float()
            v_f = v.detach().float()
            q_f = q.detach().float()
            self._fovea_aux.setdefault("_num_diag", {})
            self._fovea_aux["_num_diag"][f"mem_s{sample_id}"] = f"norm={mem_f.norm():.3e}"
            self._fovea_aux["_num_diag"][f"v_s{sample_id}"] = f"mean={v_f.mean():.3e} norm={v_f.norm():.3e}"
            self._fovea_aux["_num_diag"][f"q_s{sample_id}"] = f"mean={q_f.mean():.3e} norm={q_f.norm():.3e}"

            scores = torch.einsum("tnhd,hmd->tnhm", q, k) * scale
            scores = scores.masked_fill((~sample_mask).view(1, 1, 1, -1), torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            context = torch.einsum("tnhm,hmd->tnhd", attn, v).reshape(trigger_indices.shape[0], num_fovea, -1)
            # numerical diagnostics
            ctx_f = context.detach().float()
            self._fovea_aux.setdefault("_num_diag", {})
            self._fovea_aux["_num_diag"][f"ctx_s{sample_id}"] = f"mean={ctx_f.mean():.3e} std={ctx_f.std():.3e} min={ctx_f.min():.3e} max={ctx_f.max():.3e} norm={ctx_f.norm():.3e}"
            retrieved_raw = self.fovea_o_proj(context)
            # numerical diagnostics: context (input), weight, output
            ctx_f = context.detach().float()
            w_f = self.fovea_o_proj.weight.detach().float()
            out_f = retrieved_raw.detach().float()
            self._fovea_aux.setdefault("_num_diag", {})
            self._fovea_aux["_num_diag"][f"ctx_s{sample_id}"] = (
                f"mean={ctx_f.mean():.3e} std={ctx_f.std():.3e} min={ctx_f.min():.3e} max={ctx_f.max():.3e} norm={ctx_f.norm():.3e}"
            )
            self._fovea_aux["_num_diag"][f"w_s{sample_id}"] = (
                f"mean={w_f.mean():.3e} std={w_f.std():.3e} min={w_f.min():.3e} max={w_f.max():.3e} norm={w_f.norm():.3e}"
            )
            self._fovea_aux["_num_diag"][f"out_s{sample_id}"] = (
                f"mean={out_f.mean():.3e} std={out_f.std():.3e} min={out_f.min():.3e} max={out_f.max():.3e} norm={out_f.norm():.3e}"
            )
            # end numerical diagnostics
            self._trace_grad(context, f"retrieval_context_s{sample_id}")  # before o_proj
            self._trace_grad(retrieved_raw, f"retrieval_oproj_out_s{sample_id}")  # o_proj raw output
            retrieved = retrieved_raw.to(dtype)
            self._trace_grad(retrieved, f"retrieval_retrieved_s{sample_id}")  # after .to(dtype)
            attn_flat = attn.mean(dim=2).to(torch.float32)
            for i_local, i_global in enumerate(trigger_indices.tolist()):
                vector_parts[i_global] = retrieved[i_local : i_local + 1]
                attn_mean_parts[i_global] = attn_flat[i_local : i_local + 1]

        vectors = torch.cat(vector_parts, dim=0)
        attn_mean = torch.cat(attn_mean_parts, dim=0)
        self._trace_grad(vectors, "retrieval_vectors")
        self._trace_grad(attn_mean, "retrieval_attn_mean")

        align_loss = vectors.new_zeros(())
        if fovea_boxes is not None and fovea_boxes.numel() > 0 and fovea_box_indices is not None and fovea_box_indices.numel() > 0:
            boxes_for_trigger = fovea_boxes.to(device=device, dtype=torch.float32).index_select(0, fovea_box_indices.to(device=device, dtype=torch.long))
            boxes_for_memory = patch_boxes.index_select(0, trigger_batch).to(device=device, dtype=torch.float32)
            mask_for_trigger = memory_mask.index_select(0, trigger_batch)
            valid_boxes = boxes_for_memory.ge(0).all(dim=-1)
            eps = float(self.config.fovea_align_eps)  # config default 1e-6
            alpha = float(self.config.fovea_align_alpha)
            beta = float(self.config.fovea_align_beta)
            # Use a larger floor for log-stability to prevent FP16 gradient explosion.
            # log_clamp = 1e-4 keeps max gradient ~ 1e4 instead of 1e6.
            log_clamp = max(float(eps), 1e-4)

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
            # Use a_mean + log_clamp (not clamp_min) for smooth, bounded gradient 1/(x+log_clamp)
            l_main = -(q * (a_mean + log_clamp).log()).sum(dim=-1)
            l_cover = -(q * (a_max + log_clamp).log()).sum(dim=-1)

            attn_flat = F.normalize(attn_mean, p=2, dim=-1, eps=1e-4)
            cosine = torch.matmul(attn_flat, attn_flat.transpose(-1, -2))
            div_mask = 1.0 - torch.eye(num_fovea, device=device, dtype=cosine.dtype)
            l_div = (cosine * div_mask.unsqueeze(0)).sum(dim=(-1, -2)) / max(num_fovea * (num_fovea - 1), 1)

            align_loss = (l_main + alpha * l_cover + beta * l_div).mean()
        entry = {
            "fovea_attn_mean": attn_mean.detach(),
            "trigger_batch": trigger_batch.detach(),
            "retrieve_patch_boxes": patch_boxes.detach(),
            "_num_diag": self._fovea_aux.get("_num_diag", {}),
        }
        self._fovea_aux = entry
        self._fovea_aux_history.append(entry)
        return vectors, align_loss

    def _expand_with_fovea_tokens(self, base_embeds, attention_mask, labels, fovea_positions, fovea_vectors):
        batch_size, seq_len, hidden_size = base_embeds.shape
        num_fovea = int(self.config.fovea_num_tokens)
        fovea_vectors = fovea_vectors.to(base_embeds.dtype)

        by_batch: list[list[tuple[int, int]]] = [[] for _ in range(batch_size)]
        for query_idx, (batch_idx, token_pos) in enumerate(fovea_positions.to(device="cpu", dtype=torch.long).tolist()):
            if 0 <= batch_idx < batch_size and 0 <= token_pos < seq_len:
                by_batch[batch_idx].append((token_pos, query_idx))
        for items in by_batch:
            items.sort(key=lambda item: item[0])

        # Build per-sample embedding sequences functionally via concatenation
        # to preserve autograd connectivity through fovea_vectors.
        new_embed_parts: list[torch.Tensor] = []
        new_attn_parts: list[torch.Tensor] = []
        new_label_parts: list[torch.Tensor] = []
        for batch_idx, items in enumerate(by_batch):
            src_cursor = 0
            embed_chunks: list[torch.Tensor] = []
            attn_chunks: list[torch.Tensor] = []
            label_chunks: list[torch.Tensor] = []
            for token_pos, query_idx in items:
                if token_pos >= src_cursor:
                    embed_chunks.append(base_embeds[batch_idx, src_cursor : token_pos + 1])
                    if attention_mask is not None:
                        attn_chunks.append(attention_mask[batch_idx, src_cursor : token_pos + 1])
                    if labels is not None:
                        label_chunks.append(labels[batch_idx, src_cursor : token_pos + 1])
                embed_chunks.append(fovea_vectors[query_idx])
                if attention_mask is not None:
                    attn_chunks.append(attention_mask.new_ones((num_fovea,)))
                if labels is not None:
                    label_chunks.append(labels.new_full((num_fovea,), IGNORE_INDEX))
                src_cursor = token_pos + 1
            if src_cursor < seq_len:
                embed_chunks.append(base_embeds[batch_idx, src_cursor:])
                if attention_mask is not None:
                    attn_chunks.append(attention_mask[batch_idx, src_cursor:])
                if labels is not None:
                    label_chunks.append(labels[batch_idx, src_cursor:])
            new_embed_parts.append(torch.cat(embed_chunks, dim=0))
            if attention_mask is not None:
                new_attn_parts.append(torch.cat(attn_chunks, dim=0))
            if labels is not None:
                new_label_parts.append(torch.cat(label_chunks, dim=0))

        # Pad all samples to the same length (functional, preserves autograd)
        max_len = max(p.shape[0] for p in new_embed_parts)
        padded_embeds: list[torch.Tensor] = []
        padded_attns: list[torch.Tensor] = []
        padded_labels: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            pad_len = max_len - new_embed_parts[batch_idx].shape[0]
            padded_embeds.append(F.pad(new_embed_parts[batch_idx], (0, 0, 0, pad_len)))
            if attention_mask is not None:
                padded_attns.append(F.pad(new_attn_parts[batch_idx], (0, pad_len)))
            if labels is not None:
                padded_labels.append(F.pad(new_label_parts[batch_idx], (0, pad_len), value=IGNORE_INDEX))
        new_embeds = torch.stack(padded_embeds, dim=0)
        new_attention = torch.stack(padded_attns, dim=0) if attention_mask is not None else None
        new_labels = torch.stack(padded_labels, dim=0) if labels is not None else None

        return new_embeds, new_attention, new_labels

    def _expand_position_ids_with_fovea(self, position_ids, attention_mask, fovea_positions):
        if position_ids is None:
            return None
        streams, batch_size, seq_len = position_ids.shape
        num_fovea = int(self.config.fovea_num_tokens)
        by_batch: list[list[int]] = [[] for _ in range(batch_size)]
        for batch_idx, token_pos in fovea_positions.to(device="cpu", dtype=torch.long).tolist():
            if 0 <= batch_idx < batch_size and 0 <= token_pos < seq_len:
                by_batch[batch_idx].append(token_pos)
        for items in by_batch:
            items.sort()
        lengths = [seq_len + len(items) * num_fovea for items in by_batch]
        max_len = max(lengths)
        expanded = position_ids.new_zeros((streams, batch_size, max_len))
        for batch_idx, items in enumerate(by_batch):
            src_cursor = 0
            dst_cursor = 0
            offset = 0
            for token_pos in items:
                copy_len = token_pos - src_cursor + 1
                if copy_len > 0:
                    expanded[:, batch_idx, dst_cursor : dst_cursor + copy_len] = position_ids[
                        :, batch_idx, src_cursor : token_pos + 1
                    ] + offset
                    dst_cursor += copy_len
                base = expanded[:, batch_idx, dst_cursor - 1 : dst_cursor]
                delta = torch.arange(1, num_fovea + 1, device=position_ids.device, dtype=position_ids.dtype).view(1, -1)
                expanded[:, batch_idx, dst_cursor : dst_cursor + num_fovea] = base + delta
                dst_cursor += num_fovea
                offset += num_fovea
                src_cursor = token_pos + 1
            tail_len = seq_len - src_cursor
            if tail_len > 0:
                expanded[:, batch_idx, dst_cursor : dst_cursor + tail_len] = position_ids[:, batch_idx, src_cursor:] + offset
        return expanded

    def _cached_decode_position_ids(self, attention_mask, num_new_tokens: int = 1):
        num_new_tokens = int(num_new_tokens)
        position_ids = attention_mask.long().cumsum(-1)[:, -num_new_tokens:] - 1
        position_ids = position_ids.clamp_min(0).view(1, attention_mask.shape[0], num_new_tokens).repeat(3, 1, 1)
        if self.model.rope_deltas is not None:
            position_ids = position_ids + self.model.rope_deltas.to(position_ids.device).view(1, -1, 1)
        return position_ids

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
        image_grid_thw=None,
        pixel_values_videos=None,
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
        inputs_embeds, position_ids, _ = self._build_multimodal_embeddings_and_positions(
            model_input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        retrieve_memory_cache = None
        if retrieve_pixel_values is not None:
            retrieve_memory_cache = self._build_retrieve_memory(
                retrieve_pixel_values,
                retrieve_grid_thw,
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
                prompt_outputs = self._language_forward_from_embeds(inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs)
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
                position_ids = self._expand_position_ids_with_fovea(
                    position_ids,
                    attention_mask,
                    prompt_positions.to(device=prompt_hidden.device, dtype=torch.long),
                )
        outputs = self._language_forward_from_embeds(inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs)
        if text_hidden_history is None:
            text_hidden_history = outputs[0]
        logits = self.lm_head(outputs[0][:, -1:, :]).squeeze(1)
        past_key_values = outputs.past_key_values

        force_fovea_first = (
            fovea_auto_retrieve_on_answer_start
            and fovea_id is not None
            and retrieve_memory_cache is not None
            and (model_input_ids.shape[1] == 0 or int(model_input_ids[0, -1].item()) != int(fovea_id))
        )

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_ids = {int(eos_token_id)} if isinstance(eos_token_id, int) else {int(item) for item in (eos_token_id or [])}

        for _ in range(int(max_new_tokens)):
            if force_fovea_first:
                next_token = model_input_ids.new_full((model_input_ids.shape[0], 1), int(fovea_id))
                force_fovea_first = False
            else:
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
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
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
            model_inputs["image_grid_thw"] = image_grid_thw
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_grid_thw"] = video_grid_thw
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
        return model_inputs

    def _fovea_forward(
        self,
        input_ids,
        attention_mask,
        labels,
        pixel_values,
        image_grid_thw,
        pixel_values_videos,
        video_grid_thw,
        mm_token_type_ids,
        retrieve_pixel_values,
        retrieve_grid_thw,
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
        if not getattr(self, "_fovea_bw_hooks_done", False):
            self._register_fovea_backward_hooks()
            self._fovea_bw_hooks_done = True
        embeds_a, position_ids_a, image_hidden_states = self._build_multimodal_embeddings_and_positions(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        outputs_a = self._language_forward_from_embeds(embeds_a, attention_mask=attention_mask, position_ids=position_ids_a, **kwargs)
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
            retrieve_grid_thw,
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
        position_ids_b = self._expand_position_ids_with_fovea(position_ids_a, attention_b, fovea_positions)
        outputs_b = self._language_forward_from_embeds(embeds_b, attention_mask=attention_b, position_ids=position_ids_b, **kwargs)
        hidden_b = outputs_b[0]
        full_logits_b = self.lm_head(hidden_b)
        lm_loss = self.loss_function(logits=full_logits_b, labels=labels_b, vocab_size=full_logits_b.shape[-1])
        loss = lm_loss + float(self.config.fovea_lambda_align) * align_loss
        self._fovea_aux = {
            "lm_loss": lm_loss.detach(),
            "align_loss": align_loss.detach(),
            "num_queries": hidden_b.new_tensor(float(fovea_positions.shape[0])),
            "tokens_per_query": hidden_b.new_tensor(float(self.config.fovea_num_tokens)),
            "fovea_attn_mean": self._fovea_aux.get("fovea_attn_mean"),
            "_num_diag": self._fovea_aux.get("_num_diag", {}),
        }
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=full_logits_b[:, slice_indices, :],
            past_key_values=outputs_b.past_key_values,
            hidden_states=outputs_b.hidden_states,
            attentions=outputs_b.attentions,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
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
        retrieve_grid_thw: torch.LongTensor | None = None,
        retrieve_patch_boxes: torch.Tensor | None = None,
        retrieve_image_counts: torch.Tensor | None = None,
        fovea_positions: torch.LongTensor | None = None,
        fovea_box_indices: torch.LongTensor | None = None,
        fovea_boxes: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> tuple | Qwen3_5CausalLMOutputWithPast:
        if (
            retrieve_pixel_values is None
            and retrieve_grid_thw is None
            and retrieve_patch_boxes is None
            and retrieve_image_counts is None
            and fovea_positions is None
            and fovea_box_indices is None
            and fovea_boxes is None
        ):
            outputs = self.model(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
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

            return Qwen3_5CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
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
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                retrieve_pixel_values=retrieve_pixel_values,
                retrieve_grid_thw=retrieve_grid_thw,
                retrieve_patch_boxes=retrieve_patch_boxes,
                retrieve_image_counts=retrieve_image_counts,
                fovea_positions=fovea_positions,
                fovea_box_indices=fovea_box_indices,
                fovea_boxes=fovea_boxes,
                logits_to_keep=logits_to_keep,
                **model_kwargs,
            )

        if inputs_embeds is None:
            inputs_embeds, position_ids, image_hidden_states = self._build_multimodal_embeddings_and_positions(
                input_ids,
                attention_mask,
                pixel_values,
                image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
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
        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["FoveaForConditionalGeneration"]
