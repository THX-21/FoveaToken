import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import HfArgumentParser, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from fovea_token import FoveaForConditionalGeneration, FoveaTokenizer
from fovea_token.tokenizers.tokenization_ibq import (
    DEFAULT_IBQ_CHECKPOINT,
    DEFAULT_IBQ_CONFIG,
    DEFAULT_IBQ_REPO,
    IBQCodec,
)
from fovea_token.tokenizers.tokenization_visual_query import add_visual_query_tokens, sync_visual_query_token_ids

from .data import DataCollatorForQwen3_5SFT, LazySupervisedDataset, VisionPacker


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-9B")
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    unfreeze_vision: bool = field(default=True)
    visual_query_generated_replay_prob: float = field(default=0.5)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_image_tokens: Optional[int] = field(default=None)
    retrieve_max_image_tokens: Optional[int] = field(default=4096)
    ibq_repo: str = field(default=DEFAULT_IBQ_REPO)
    ibq_checkpoint: str = field(default=DEFAULT_IBQ_CHECKPOINT)
    ibq_config: str = field(default=DEFAULT_IBQ_CONFIG)
    visual_code_cache_dir: Optional[str] = field(default=None)
    system_message: str = field(default="You are a helpful assistant.")


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
        return
    model.requires_grad_(False)
    if unfreeze_vision:
        get_visual_module(model).requires_grad_(True)


def unfreeze_visual_query_parameters(model) -> None:
    """Keep retrieval modules and new token embeddings trainable."""

    for name, param in model.named_parameters():
        if "visual_query_" in name or "embed_tokens" in name or "lm_head" in name:
            param.requires_grad_(True)


def get_visual_module(model):
    """Return the vision tower for both raw and PEFT-wrapped models."""

    candidates = [model]
    for attr in ("model", "base_model"):
        child = getattr(model, attr, None)
        if child is not None:
            candidates.append(child)

    for candidate in candidates:
        visual = getattr(candidate, "visual", None)
        if visual is not None:
            return visual
        inner_model = getattr(candidate, "model", None)
        visual = getattr(inner_model, "visual", None) if inner_model is not None else None
        if visual is not None:
            return visual

    raise AttributeError("Could not locate the vision tower on the current model wrapper stack.")


def maybe_enable_lora(model, model_args: ModelArguments):
    if not model_args.lora_enable:
        return model

    from peft import LoraConfig, get_peft_model

    visual_query_modules = [
        "visual_query_q_proj",
        "visual_query_k_proj",
        "visual_query_v_proj",
        "visual_query_o_proj",
        "visual_query_q_norm",
        "visual_query_k_norm",
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
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ]
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        exclude_modules=visual_query_modules,
        modules_to_save=visual_query_modules,
    )
    model = get_peft_model(model, lora_config)
    return model


def print_loading_summary(model, loading_info: dict) -> None:
    """Print a compact summary of checkpoint loading results."""

    missing_keys = loading_info.get("missing_keys", [])
    unexpected_keys = loading_info.get("unexpected_keys", [])
    mismatched_keys = loading_info.get("mismatched_keys", [])
    error_msgs = loading_info.get("error_msgs", [])

    total_model_keys = len(model.state_dict())
    loaded_key_count = total_model_keys - len(missing_keys) - len(mismatched_keys)

    print("=== Checkpoint Loading Summary ===")
    print(f"total_model_keys: {total_model_keys}")
    print(f"loaded_keys: {loaded_key_count}")
    print(f"missing_keys: {len(missing_keys)}")
    print(f"unexpected_keys: {len(unexpected_keys)}")
    print(f"mismatched_keys: {len(mismatched_keys)}")
    print(f"error_msgs: {len(error_msgs)}")

    if missing_keys:
        print("missing_key_names:")
        for name in missing_keys:
            print(f"  {name}")

    if unexpected_keys:
        print("unexpected_key_names:")
        for name in unexpected_keys:
            print(f"  {name}")

    if mismatched_keys:
        print("mismatched_key_names:")
        for item in mismatched_keys:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                print(f"  {item[0]} checkpoint_shape={item[1]} model_shape={item[2]}")
            else:
                print(f"  {item}")

    if error_msgs:
        print("loading_errors:")
        for msg in error_msgs:
            print(f"  {msg}")


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


def sync_tokenizer_special_tokens_with_model(tokenizer: FoveaTokenizer, model: FoveaForConditionalGeneration) -> None:
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
        if vision_tower_lr is None:
            return super().create_optimizer()

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
            if is_vision and uses_decay:
                optimizer_grouped_parameters[2]["params"].append(param)
            elif is_vision:
                optimizer_grouped_parameters[3]["params"].append(param)
            elif uses_decay:
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


class VisualQueryMetricsCallback(transformers.TrainerCallback):
    """Publish visual-query auxiliary metrics into Trainer logs."""

    def on_log(self, _args, state, control, model=None, logs=None, **_kwargs):
        if model is None or logs is None or not hasattr(model, "_visual_query_aux"):
            return control
        aux = getattr(model, "_visual_query_aux", None)
        if not isinstance(aux, dict):
            return control
        for key in ("lm_loss", "align_loss", "generated_replay", "num_queries"):
            value = aux.get(key)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach()
            if isinstance(value, torch.Tensor) and value.device.type == "meta":
                continue
            if hasattr(value, "item"):
                value = value.item()
            logs[f"visual_query/{key}"] = value
        return control



def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = FoveaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    add_visual_query_tokens(tokenizer)
    model, loading_info = FoveaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation=training_args.attn_implementation,
        output_loading_info=True,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.config.visual_query_generated_replay_prob = float(model_args.visual_query_generated_replay_prob)
    sync_visual_query_token_ids(model.config, tokenizer)
    print_loading_summary(model, loading_info)
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
    if model_args.unfreeze_vision:
        get_visual_module(model).requires_grad_(True)
    unfreeze_visual_query_parameters(model)
    print_parameter_summary(model)

    vision_packer = VisionPacker(
        vision_config=model.config.vision_config,
        max_image_tokens=data_args.max_image_tokens,
    )
    visual_codec = IBQCodec.from_paths(
        repo=data_args.ibq_repo,
        checkpoint=data_args.ibq_checkpoint,
        config=data_args.ibq_config,
        cache_dir=data_args.visual_code_cache_dir,
    )

    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        tokenizer=tokenizer,
        vision_packer=vision_packer,
        image_token_id=model.config.image_token_id,
        system_message=data_args.system_message,
        visual_codec=visual_codec,
        retrieve_max_image_tokens=data_args.retrieve_max_image_tokens,
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
        processing_class=tokenizer,
        callbacks=[StopAtStepCallback, VisualQueryMetricsCallback],
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
