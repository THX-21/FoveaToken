import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoProcessor, AutoTokenizer, HfArgumentParser, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from fovea_token import FoveaForConditionalGeneration
from fovea_token.tokenizers.tokenization_fovea import FOVEA_SPECIAL_TOKENS, add_fovea_tokens, sync_fovea_token_ids

from .data import DataCollatorForQwen3_5SFT, LazySupervisedDataset, VisionPacker


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-4B")
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    unfreeze_vision: bool = field(default=True)
    freeze_embed_base: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    system_message: str = field(default="You are a helpful assistant.")
    max_img_tokens: int = field(default=2048)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=32768)
    report_to: str | None = field(default="tensorboard")
    attn_implementation: str = field(default="sdpa")
    vision_tower_lr: Optional[float] = field(default=None)


def freeze_for_vision_plus_lora(model: FoveaForConditionalGeneration, unfreeze_vision: bool, lora_enable: bool) -> None:
    if not lora_enable:
        model.requires_grad_(True)
        if not unfreeze_vision:
            get_visual_module(model).requires_grad_(False)
        return
    model.requires_grad_(False)


def unfreeze_fovea_parameters(model, lora_enable: bool) -> None:
    """Keep fovea retrieval modules and the new trigger token row trainable."""

    for name, param in model.named_parameters():
        is_fovea_module = "fovea_" in name or "embed_tokens" in name or "lm_head" in name
        if not is_fovea_module:
            continue
        if lora_enable and "modules_to_save" not in name:
            continue
        if not lora_enable and ".original_module." in name:
            continue
        param.requires_grad_(True)


def get_trainable_special_token_ids(config, tokenizer=None, vocab_size: int | None = None) -> list[int]:
    token_id = getattr(config, "fovea_token_id", None)
    token_ids = [] if token_id is None else [int(token_id)]
    if tokenizer is not None:
        token_ids.extend(int(tokenizer.convert_tokens_to_ids(token)) for token in FOVEA_SPECIAL_TOKENS)
    if vocab_size is None:
        vocab_size = int(getattr(getattr(config, "text_config", config), "vocab_size", 0))
    return sorted({token_id for token_id in token_ids if 0 <= token_id < int(vocab_size)})


def freeze_base_embedding_rows(model, config, tokenizer=None) -> None:
    """Freeze base vocab rows while allowing newly added text token rows to learn."""

    registered_params = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embed_tokens" not in name and "lm_head" not in name:
            continue
        if param.ndim != 2:
            continue
        trainable_token_ids = get_trainable_special_token_ids(config, tokenizer=tokenizer, vocab_size=param.shape[0])
        if not trainable_token_ids:
            continue
        if id(param) in registered_params:
            continue
        registered_params.add(id(param))
        row_mask = torch.zeros((param.shape[0], 1), device=param.device, dtype=param.dtype)
        row_mask[torch.tensor(trainable_token_ids, device=param.device, dtype=torch.long)] = 1
        param.register_hook(lambda grad, mask=row_mask: grad * mask.to(device=grad.device, dtype=grad.dtype))


def configure_vision_trainability_for_lora(model, unfreeze_vision: bool) -> None:
    """For LoRA runs, train only vision LoRA weights when requested."""

    visual_module = get_visual_module(model)
    for name, param in visual_module.named_parameters():
        param.requires_grad_("lora_" in name if unfreeze_vision else False)


def get_visual_module(model):
    """Return the vision tower for both raw and PEFT-wrapped models."""

    candidates = [model]
    for attr in ("model", "base_model"):
        child = getattr(model, attr, None)
        if child is not None:
            candidates.append(child)

    for candidate in candidates:
        visual = getattr(candidate, "vision_tower", None)
        if visual is not None:
            return visual
        visual = getattr(candidate, "visual", None)
        if visual is not None:
            return visual
        inner_model = getattr(candidate, "model", None)
        visual = getattr(inner_model, "vision_tower", None) if inner_model is not None else None
        if visual is not None:
            return visual
        visual = getattr(inner_model, "visual", None) if inner_model is not None else None
        if visual is not None:
            return visual

    raise AttributeError("Could not locate the vision tower on the current model wrapper stack.")


def maybe_enable_lora(model, model_args: ModelArguments):
    if not model_args.lora_enable:
        return model

    from peft import LoraConfig, get_peft_model

    fovea_modules = [
        "fovea_tokens",
        "fovea_q_proj",
        "fovea_k_proj",
        "fovea_v_proj",
        "fovea_o_proj",
        "fovea_q_norm",
        "fovea_k_norm",
        "fovea_ssm_in_proj_qkv",
        "fovea_ssm_in_proj_z",
        "fovea_ssm_in_proj_b",
        "fovea_ssm_in_proj_a",
        "fovea_ssm_dt_bias",
        "fovea_ssm_A_log",
        "fovea_ssm_out_proj",
        "fovea_ssm_norm",
        "embed_tokens",
        "lm_head",
    ]
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if model_args.unfreeze_vision:
        target_modules.extend(
            [
                "out_proj",
                "fc1",
                "fc2",
                "linear_1",
                "linear_2",
            ]
        )
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        exclude_modules=fovea_modules,
        modules_to_save=fovea_modules,
        ensure_weight_tying=False,
    )
    model = get_peft_model(model, lora_config)
    return model


def print_loading_summary(model, loading_info: dict, source_label: str) -> None:
    """Print a compact summary of checkpoint loading results."""

    missing_keys = loading_info.get("missing_keys", [])
    unexpected_keys = loading_info.get("unexpected_keys", [])
    mismatched_keys = loading_info.get("mismatched_keys", [])
    error_msgs = loading_info.get("error_msgs", [])

    total_model_keys = len(model.state_dict())
    loaded_key_count = total_model_keys - len(missing_keys) - len(mismatched_keys)

    print(f"=== Checkpoint Loading Summary ({source_label}) ===")
    print(f"total_model_keys: {total_model_keys}")
    print(f"loaded_keys: {loaded_key_count}")
    print(f"missing_keys: {len(missing_keys)}")
    print(f"unexpected_keys: {len(unexpected_keys)}")
    print(f"mismatched_keys: {len(mismatched_keys)}")
    print(f"error_msgs: {len(error_msgs)}")

    if missing_keys:
        print(f"missing_key_names_from_{source_label}:")
        for name in missing_keys:
            print(f"  {name}")

    if unexpected_keys:
        print(f"unexpected_key_names_from_{source_label}:")
        for name in unexpected_keys:
            print(f"  {name}")

    if mismatched_keys:
        print(f"mismatched_key_names_from_{source_label}:")
        for item in mismatched_keys:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                print(f"  {item[0]} checkpoint_shape={item[1]} model_shape={item[2]}")
            else:
                print(f"  {item}")

    if error_msgs:
        print(f"loading_errors_from_{source_label}:")
        for msg in error_msgs:
            print(f"  {msg}")


def should_load_model_from_checkpoint(model_name_or_path: str, resume_from_checkpoint: str | None, lora_enable: bool) -> bool:
    if lora_enable or not resume_from_checkpoint:
        return False
    if os.path.abspath(model_name_or_path) == os.path.abspath(resume_from_checkpoint):
        return True
    return os.path.isfile(os.path.join(resume_from_checkpoint, "model.safetensors"))


def print_parameter_summary(model) -> None:
    total_numel = 0
    trainable_numel = 0
    frozen_numel = 0
    lora_numel = 0
    trainable_names: list[str] = []

    for name, param in model.named_parameters():
        numel = param.numel()
        total_numel += numel
        if param.requires_grad:
            trainable_numel += numel
            trainable_names.append(name)
        else:
            frozen_numel += numel
        if "lora_" in name.lower():
            lora_numel += numel

    trainable_ratio = (trainable_numel / total_numel) if total_numel else 0.0
    print("=== Parameter Summary ===")
    print(f"total_parameters: {total_numel}")
    print(f"trainable_parameters: {trainable_numel}")
    print(f"frozen_parameters: {frozen_numel}")
    print(f"trainable_ratio: {trainable_ratio:.6f}")
    print(f"lora_parameters: {lora_numel}")
    print("=== Trainable Parameter Names ===")
    for name in trainable_names:
        print(name)


def sync_tokenizer_special_tokens_with_model(tokenizer, model: FoveaForConditionalGeneration) -> None:
    """Keep tokenizer special-token defaults aligned with the checkpoint config."""

    text_config = getattr(model.config, "text_config", model.config)
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(text_config, attr, None)
        if token_id is None:
            token_id = getattr(model.config, attr, None)
        if token_id is None and getattr(model, "generation_config", None) is not None:
            token_id = getattr(model.generation_config, attr, None)
        if token_id is None:
            token_id = getattr(tokenizer, attr, None)
        if token_id is None:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is None:
            continue
        setattr(tokenizer, attr, token_id)
        setattr(tokenizer, attr.replace("_id", ""), token)
        setattr(text_config, attr, token_id)
        # Mirror text-config special tokens onto the top-level config so
        # downstream Transformers utilities no longer see a mismatch.
        setattr(model.config, attr, token_id)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = getattr(model.config, "bos_token_id", None)
        model.generation_config.eos_token_id = getattr(model.config, "eos_token_id", None)
        model.generation_config.pad_token_id = getattr(model.config, "pad_token_id", None)


class FT3Trainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        vision_tower_lr = self.args.vision_tower_lr

        # embed_tokens / lm_head always get weight_decay=0, even when
        # freeze_embed_base=true (grad hooks zero non-special rows but
        # AdamW weight decay still acts directly on param.data).
        no_wd_prefixes = ("embed_tokens", "lm_head")

        if vision_tower_lr is None and self.args.weight_decay <= 0:
            return super().create_optimizer()

        if vision_tower_lr is None:
            # build custom groups just for the no-wd prefixes
            decay_parameter_names = set(get_parameter_names(self.model, ALL_LAYERNORM_LAYERS))
            decay_parameter_names = {name for name in decay_parameter_names if not name.endswith("bias")}
            optimizer_grouped_parameters = [
                {"params": [], "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
                {"params": [], "weight_decay": 0.0, "lr": self.args.learning_rate},
            ]
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(name.startswith(p) for p in no_wd_prefixes):
                    optimizer_grouped_parameters[1]["params"].append(param)
                elif name in decay_parameter_names:
                    optimizer_grouped_parameters[0]["params"].append(param)
                else:
                    optimizer_grouped_parameters[1]["params"].append(param)
            optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            return self.optimizer

        decay_parameter_names = set(get_parameter_names(self.model, ALL_LAYERNORM_LAYERS))
        decay_parameter_names = {name for name in decay_parameter_names if not name.endswith("bias")}
        vision_module = get_visual_module(self.model)
        vision_param_ids = {id(param) for param in vision_module.parameters() if param.requires_grad}

        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
            {"params": [], "weight_decay": 0.0, "lr": self.args.learning_rate},
            {"params": [], "weight_decay": self.args.weight_decay, "lr": vision_tower_lr},
            {"params": [], "weight_decay": 0.0, "lr": vision_tower_lr},
        ]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_vision = id(param) in vision_param_ids
            uses_decay = name in decay_parameter_names
            no_wd = any(name.startswith(p) for p in no_wd_prefixes)
            if is_vision and (uses_decay and not no_wd):
                optimizer_grouped_parameters[2]["params"].append(param)
            elif is_vision:
                optimizer_grouped_parameters[3]["params"].append(param)
            elif uses_decay and not no_wd:
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)

        optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if group["params"]]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

class StopAtStepCallback(transformers.TrainerCallback):
    """Stop training cleanly at the requested global step.

    Reads `FT3_STOP_STEP` from the environment so the shell loop can request
    segmented training without changing the global max_steps schedule.
    """

    def __init__(self) -> None:
        stop_step = os.environ.get("FT3_STOP_STEP")
        self.stop_step = int(stop_step) if stop_step else None

    def on_step_end(self, _args, state, control, **_kwargs):
        if self.stop_step is None:
            return control
        if state.global_step >= self.stop_step:
            control.should_save = True
            control.should_training_stop = True
        return control


class FoveaMetricsCallback(transformers.TrainerCallback):
    """Publish fovea auxiliary metrics into Trainer logs."""

    @staticmethod
    def _find_fovea_aux(model):
        seen = set()
        stack = [model]
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            aux = getattr(current, "_fovea_aux", None)
            if isinstance(aux, dict):
                return aux
            get_base_model = getattr(current, "get_base_model", None)
            if callable(get_base_model):
                stack.append(get_base_model())
            for attr in ("base_model", "model"):
                stack.append(getattr(current, attr, None))
        return None

    def on_log(self, _args, state, control, model=None, logs=None, **_kwargs):
        if model is None or logs is None:
            return control
        aux = self._find_fovea_aux(model)
        if aux is None:
            return control
        # Numerical diagnostics from fovea retrieval
        num_diag = aux.get("_num_diag")
        if isinstance(num_diag, dict):
            for k, v in num_diag.items():
                logs[f"num/{k}"] = v

        for key in ("lm_loss", "align_loss", "num_queries", "tokens_per_query"):
            value = aux.get(key)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach()
            if isinstance(value, torch.Tensor) and value.device.type == "meta":
                continue
            if hasattr(value, "item"):
                value = value.item()
            logs[f"fovea/{key}"] = value
        # --- tensor-level gradient trace (survives aux overwrites) ---
        grad_trace = getattr(model, "_fovea_grad_trace", None)
        # Drill through PEFT / DeepSpeed wrappers to find the raw model
        if grad_trace is None:
            for attr in ("module", "base_model", "model"):
                inner = getattr(model, attr, None)
                if inner is not None:
                    grad_trace = getattr(inner, "_fovea_grad_trace", None)
                    if grad_trace is not None:
                        break
        if isinstance(grad_trace, dict):
            for key, value in grad_trace.items():
                logs[f"trace/{key}"] = value
        # Log backward gradient diagnostic
        bw_grads = aux.get("_bw_grads")
        if isinstance(bw_grads, dict):
            for module_name, pgrads in bw_grads.items():
                if not isinstance(pgrads, dict):
                    continue
                for pname, gnorm in pgrads.items():
                    logs[f"bw/{module_name}/{pname}"] = gnorm
        # Log tensor-level gradient diagnostic
        tensor_grads = aux.get("_tensor_grads")
        if isinstance(tensor_grads, dict):
            for tname, grad_list in tensor_grads.items():
                if isinstance(grad_list, list) and grad_list:
                    logs[f"tg/{tname}/norm"] = grad_list[-1]
        return control


class PerModuleGradNormCallback(transformers.TrainerCallback):
    """Log per-module gradient norms after each optimizer step.

    Registers backward hooks on all trainable parameters during on_train_begin
    and accumulates per-module squared gradient norms. Per-step norms are
    logged at the same cadence as the trainer's default logging.
    """

    def __init__(self, log_prefix: str = "grad"):
        self.log_prefix = log_prefix
        self._handles: list = []
        self._accumulated: dict[str, float] = {}

    def on_train_begin(self, _args, state, control, model=None, **kwargs):
        if model is None:
            return control
        self._register_hooks(model)
        return control

    @staticmethod
    def _module_group(name: str) -> str:
        """Collapse parameter names into top-level groups."""
        full = name

        # Fovea SSM params
        if "fovea_ssm_in_proj_qkv" in full:
            return "fovea_ssm_in_qkv"
        if "fovea_ssm_in_proj_z" in full:
            return "fovea_ssm_in_z"
        if "fovea_ssm_in_proj_b" in full:
            return "fovea_ssm_in_beta"
        if "fovea_ssm_in_proj_a" in full:
            return "fovea_ssm_in_a"
        if "fovea_ssm_out_proj" in full:
            return "fovea_ssm_out"
        if "fovea_ssm_dt_bias" in full:
            return "fovea_ssm_dt_bias"
        if "fovea_ssm_A_log" in full:
            return "fovea_ssm_A_log"
        if "fovea_ssm_norm" in full:
            return "fovea_ssm_norm"

        # Fovea retrieval attn
        if "fovea_q_proj" in full or "fovea_q_norm" in full:
            return "fovea_attn_q"
        if "fovea_k_proj" in full or "fovea_k_norm" in full:
            return "fovea_attn_k"
        if "fovea_v_proj" in full:
            return "fovea_attn_v"
        if "fovea_o_proj" in full:
            return "fovea_attn_o"

        if "fovea_tokens" in full:
            return "fovea_tokens"

        # LoRA weights
        if "lora_A" in full:
            if "visual" in full or "vision" in full:
                return "lora_A_vision"
            if "self_attn" in full:
                return "lora_A_attn"
            return "lora_A_ffn"
        if "lora_B" in full:
            if "visual" in full or "vision" in full:
                return "lora_B_vision"
            if "self_attn" in full:
                return "lora_B_attn"
            return "lora_B_ffn"

        if "embed_tokens" in full and "modules_to_save" in full:
            return "embed_tokens"
        if "lm_head" in full and "modules_to_save" in full:
            return "lm_head"

        return "other"

    def _make_hook(self, group_name: str):
        def hook(grad, _group=group_name):
            if grad is None:
                return
            sq_norm = grad.detach().float().norm().pow(2).item()
            self._accumulated[_group] = self._accumulated.get(_group, 0.0) + sq_norm
        return hook

    def _register_hooks(self, model) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            group = self._module_group(name)
            handle = param.register_hook(self._make_hook(group))
            self._handles.append(handle)

    def on_step_end(self, args, state, control, **kwargs):
        if not self._accumulated:
            return control
        # Snapshot and reset for the next optimizer step
        state.__dict__["_per_module_grads"] = {
            f"{self.log_prefix}/{group}": sq_sum ** 0.5
            for group, sq_sum in self._accumulated.items()
        }
        self._accumulated = {}
        return control

    def on_log(self, _args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        per_module = state.__dict__.get("_per_module_grads", {})
        logs.update(per_module)
        return control

    def __del__(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()



def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    add_fovea_tokens(tokenizer)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    processor.tokenizer = tokenizer

    init_model_path = model_args.model_name_or_path
    loading_source_label = "base_model"
    if should_load_model_from_checkpoint(
        model_name_or_path=model_args.model_name_or_path,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        lora_enable=model_args.lora_enable,
    ):
        init_model_path = training_args.resume_from_checkpoint
        loading_source_label = "resume_checkpoint"
        print(f"Initializing model weights from resume checkpoint: {init_model_path}")

    model, loading_info = FoveaForConditionalGeneration.from_pretrained(
        init_model_path,
        torch_dtype="auto",
        attn_implementation=training_args.attn_implementation,
        output_loading_info=True,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    sync_fovea_token_ids(model.config, tokenizer)
    model.config.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    model.config.video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    model.config.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    model.config.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    # print_loading_summary(model, loading_info, source_label=loading_source_label)
    sync_tokenizer_special_tokens_with_model(tokenizer, model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    model.config.use_cache = False
    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    freeze_for_vision_plus_lora(model, unfreeze_vision=model_args.unfreeze_vision, lora_enable=model_args.lora_enable)
    model = maybe_enable_lora(model, model_args)
    sync_tokenizer_special_tokens_with_model(tokenizer, model)
    unfreeze_fovea_parameters(model, lora_enable=model_args.lora_enable)
    if model_args.freeze_embed_base:
        freeze_base_embedding_rows(model, model.config, tokenizer=tokenizer)
    if model_args.lora_enable:
        configure_vision_trainability_for_lora(model, unfreeze_vision=model_args.unfreeze_vision)
    # print_parameter_summary(model)

    vision_packer = VisionPacker(
        processor=processor,
        vision_config=model.config.vision_config,
        max_image_tokens=data_args.max_img_tokens,
    )
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        processor=processor,
        tokenizer=tokenizer,
        vision_packer=vision_packer,
        image_token_id=model.config.image_token_id,
        system_message=data_args.system_message,
        model_max_length=training_args.model_max_length,
    )
    data_collator = DataCollatorForQwen3_5SFT(
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    trainer = FT3Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=[StopAtStepCallback, FoveaMetricsCallback, PerModuleGradNormCallback],
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
