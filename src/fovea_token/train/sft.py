import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

from fovea_token import FoveaForConditionalGeneration
from fovea_token.tokenizers.tokenization_fovea import sync_fovea_token_ids

from .data import DataCollatorForQwen3_5SFT, LazySupervisedDataset, VisionPacker


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-9B")


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    system_message: str = field(default="You are a helpful assistant.")
    max_img_tokens: int = field(default=2048)
    fovea_crop_max_img_tokens: int = field(default=1024)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=32768)
    report_to: str | None = field(default="tensorboard")
    attn_implementation: str = field(default="sdpa")
    freeze_base_model: bool = field(default=True)
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)


FOVEA_EXTRA_WEIGHTS_NAME = "fovea_extra.safetensors"
FOVEA_EXTRA_CONFIG_NAME = "fovea_extra_config.json"


def get_fovea_base_model(model):
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base = get_base_model()
        if isinstance(base, FoveaForConditionalGeneration):
            return base
    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, FoveaForConditionalGeneration):
            return current
        current = getattr(current, "model", None)
    return model


def is_fovea_extra_key(name: str) -> bool:
    return name.startswith("fovea_") or ".fovea_" in name


def is_lora_excluded_module(name: str) -> bool:
    return is_fovea_extra_key(name) or name == "lm_head" or name.endswith("embed_tokens")


def clear_fovea_token_row_hooks(model: FoveaForConditionalGeneration) -> None:
    if hasattr(model, "_fovea_input_embedding_hook") and model._fovea_input_embedding_hook is not None:
        model._fovea_input_embedding_hook.remove()
        model._fovea_input_embedding_hook = None
    if hasattr(model, "_fovea_lm_head_hook") and model._fovea_lm_head_hook is not None:
        model._fovea_lm_head_hook.remove()
        model._fovea_lm_head_hook = None


def configure_low_contamination_training(model: FoveaForConditionalGeneration) -> None:
    """Freeze Qwen and train only Fovea retrieval plus the added token rows."""

    model.requires_grad_(False)
    configure_fovea_extra_training(model, token_rows_only=True)


def configure_fovea_extra_training(
    model: FoveaForConditionalGeneration,
    token_rows_only: bool,
    train_token_rows: bool = True,
) -> None:
    """Train Fovea modules plus either full token matrices or only the added token rows."""

    for name, param in model.named_parameters():
        if name.startswith("fovea_"):
            param.requires_grad_(True)

    clear_fovea_token_row_hooks(model)
    model._fovea_save_full_token_matrices = not token_rows_only

    if not train_token_rows:
        return

    fovea_token_id = getattr(model.config, "fovea_token_id", None)
    if fovea_token_id is None or int(fovea_token_id) < 0:
        raise ValueError("Frozen-base Fovea training requires a valid fovea_token_id.")

    input_embedding_weight = model.get_input_embeddings().weight
    input_embedding_weight.requires_grad_(True)
    model.lm_head.weight.requires_grad_(True)

    if not token_rows_only:
        return

    allowed_row = int(fovea_token_id)

    def keep_only_fovea_row(grad):
        if grad is None:
            return grad
        masked = grad.new_zeros(grad.shape)
        masked[allowed_row].copy_(grad[allowed_row])
        return masked

    model._fovea_input_embedding_hook = input_embedding_weight.register_hook(keep_only_fovea_row)

    model._fovea_lm_head_hook = model.lm_head.weight.register_hook(keep_only_fovea_row)


def configure_training_mode(model: FoveaForConditionalGeneration, freeze_base_model: bool) -> None:
    clear_fovea_token_row_hooks(model)
    if freeze_base_model:
        configure_low_contamination_training(model)
        return
    model._fovea_save_full_token_matrices = True
    model.requires_grad_(True)


def resolve_lora_target_modules(model: FoveaForConditionalGeneration) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        if is_lora_excluded_module(name):
            continue
        if isinstance(module, torch.nn.Linear):
            targets.append(name)
    if not targets:
        raise ValueError("No LoRA target modules were found.")
    return targets


def configure_lora_training(model: FoveaForConditionalGeneration, training_args: TrainingArguments):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires peft. Install project train dependencies or `pip install peft`.") from exc

    model.requires_grad_(False)
    clear_fovea_token_row_hooks(model)

    target_modules = resolve_lora_target_modules(model)
    print("=== LoRA Target Modules ===")
    for name in target_modules:
        print(name)

    fovea_token_id = getattr(model.config, "fovea_token_id", None)
    if fovea_token_id is None or int(fovea_token_id) < 0:
        raise ValueError("LoRA Fovea training requires a valid fovea_token_id.")
    fovea_token_id = int(fovea_token_id)

    peft_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        trainable_token_indices={
            "model.language_model.embed_tokens": [fovea_token_id],
            "lm_head": [fovea_token_id],
        },
    )
    peft_model = get_peft_model(model, peft_config)
    configure_fovea_extra_training(model, token_rows_only=True, train_token_rows=False)
    return peft_model


def collect_fovea_extra_state(model: FoveaForConditionalGeneration) -> dict[str, torch.Tensor]:
    state = {}
    for name, tensor in model.state_dict().items():
        if is_fovea_extra_key(name):
            state[name] = tensor.detach().cpu()

    token_id = getattr(model.config, "fovea_token_id", None)
    if getattr(model, "_fovea_save_full_token_matrices", False):
        state["model.embed_tokens.weight"] = model.get_input_embeddings().weight.detach().cpu()
        state["lm_head.weight"] = model.lm_head.weight.detach().cpu()
    elif token_id is not None and int(token_id) >= 0:
        token_id = int(token_id)
        state["model.embed_tokens.weight.fovea_row"] = model.get_input_embeddings().weight[token_id].detach().cpu()
        state["lm_head.weight.fovea_row"] = model.lm_head.weight[token_id].detach().cpu()
    return state


def save_fovea_extra(model: FoveaForConditionalGeneration, output_dir: str) -> None:
    from safetensors.torch import save_file

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_file(collect_fovea_extra_state(model), str(root / FOVEA_EXTRA_WEIGHTS_NAME))
    extra_config = {
        "fovea_token_id": getattr(model.config, "fovea_token_id", None),
        "base_model_name_or_path": getattr(model, "name_or_path", None),
    }
    (root / FOVEA_EXTRA_CONFIG_NAME).write_text(json.dumps(extra_config, indent=2, sort_keys=True) + "\n")


def load_fovea_extra(model: FoveaForConditionalGeneration, checkpoint_dir: str) -> None:
    weights_path = Path(checkpoint_dir) / FOVEA_EXTRA_WEIGHTS_NAME
    if not weights_path.exists():
        return
    from safetensors.torch import load_file

    state = load_file(str(weights_path), device="cpu")
    row_keys = {
        "model.embed_tokens.weight.fovea_row",
        "lm_head.weight.fovea_row",
    }
    module_state = {key: value for key, value in state.items() if key not in row_keys}
    missing, unexpected = model.load_state_dict(module_state, strict=False)
    relevant_missing = [key for key in missing if is_fovea_extra_key(key)]
    if relevant_missing:
        print(f"Warning: missing Fovea extra keys while loading {weights_path}: {relevant_missing}")
    if unexpected:
        print(f"Warning: unexpected Fovea extra keys while loading {weights_path}: {unexpected}")

    token_id = getattr(model.config, "fovea_token_id", None)
    if token_id is not None and int(token_id) >= 0:
        token_id = int(token_id)
        with torch.no_grad():
            if "model.embed_tokens.weight.fovea_row" in state:
                row = state["model.embed_tokens.weight.fovea_row"].to(model.get_input_embeddings().weight)
                model.get_input_embeddings().weight[token_id].copy_(row)
            if "lm_head.weight.fovea_row" in state:
                row = state["lm_head.weight.fovea_row"].to(model.lm_head.weight)
                model.lm_head.weight[token_id].copy_(row)


def print_parameter_summary(model) -> None:
    total = trainable = 0
    trainable_names = []
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            trainable_names.append(name)
    print("=== Parameter Summary ===")
    print(f"total_parameters: {total}")
    print(f"trainable_parameters: {trainable}")
    print(f"trainable_ratio: {(trainable / total) if total else 0.0:.6f}")
    print("=== Trainable Parameter Names ===")
    for name in trainable_names:
        print(name)
    fovea_token_id = getattr(model.config, "fovea_token_id", None)
    if getattr(model, "_fovea_input_embedding_hook", None) is not None and fovea_token_id is not None:
        print("=== Partially Trainable Input Embedding Weight ===")
        print(f"model.language_model.embed_tokens.weight[row {int(fovea_token_id)} only]")


def sync_tokenizer_special_tokens_with_model(tokenizer, model: FoveaForConditionalGeneration) -> None:
    """Keep tokenizer, text config, top-level config, and generation config aligned."""

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
        setattr(model.config, attr, token_id)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = getattr(model.config, "bos_token_id", None)
        model.generation_config.eos_token_id = getattr(model.config, "eos_token_id", None)
        model.generation_config.pad_token_id = getattr(model.config, "pad_token_id", None)


def set_multimodal_token_ids(model: FoveaForConditionalGeneration, tokenizer) -> None:
    sync_fovea_token_ids(model.config, tokenizer, model)
    model.config.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    model.config.video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    model.config.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    model.config.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")


FULL_MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def has_full_model_weights(path: str | None) -> bool:
    if not path:
        return False
    return any(os.path.isfile(os.path.join(path, name)) for name in FULL_MODEL_WEIGHT_FILENAMES)


def should_initialize_from_resume(model_name_or_path: str, resume_from_checkpoint: str | None) -> bool:
    if not resume_from_checkpoint:
        return False
    if os.path.abspath(model_name_or_path) == os.path.abspath(resume_from_checkpoint):
        return True
    return has_full_model_weights(resume_from_checkpoint)


def resolve_initialization_paths(model_name_or_path: str, resume_from_checkpoint: str | None) -> tuple[str, str]:
    init_model_path = model_name_or_path
    fovea_extra_path = model_name_or_path
    if not resume_from_checkpoint:
        return init_model_path, fovea_extra_path

    if should_initialize_from_resume(model_name_or_path, resume_from_checkpoint):
        init_model_path = resume_from_checkpoint
        fovea_extra_path = resume_from_checkpoint
        print(f"Initializing model weights from resume checkpoint: {init_model_path}")
        return init_model_path, fovea_extra_path

    if os.path.isfile(os.path.join(resume_from_checkpoint, FOVEA_EXTRA_WEIGHTS_NAME)):
        fovea_extra_path = resume_from_checkpoint
        print(f"Initializing Fovea extra weights from resume checkpoint: {fovea_extra_path}")
    return init_model_path, fovea_extra_path


def find_fovea_metrics(model):
    seen = set()
    stack = [model]
    empty_metrics = None
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        metrics = getattr(current, "_fovea_metrics", None)
        if isinstance(metrics, dict):
            if metrics:
                return metrics
            if empty_metrics is None:
                empty_metrics = metrics
        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            stack.append(get_base_model())
        for attr in (
            "module",
            "base_model",
            "model",
            "model_wrapped",
            "_orig_mod",
            "deepspeed",
            "engine",
            "wrapped_module",
        ):
            stack.append(getattr(current, attr, None))
    return empty_metrics


class FoveaTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latest_fovea_logs: dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        metrics = getattr(outputs, "fovea_metrics", None)
        if not isinstance(metrics, dict):
            unwrapped_model = self.accelerator.unwrap_model(model)
            metrics = (
                find_fovea_metrics(model)
                or find_fovea_metrics(unwrapped_model)
                or find_fovea_metrics(getattr(self, "model_wrapped", None))
                or find_fovea_metrics(getattr(self, "model", None))
            )
        cached_logs: dict[str, float] = {}
        if metrics is not None:
            for key in ("lm_loss", "align_loss", "num_queries", "tokens_per_query"):
                value = metrics.get(key)
                if value is None:
                    continue
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "item"):
                    value = value.item()
                cached_logs[f"fovea/{key}"] = value
        self._latest_fovea_logs = cached_logs
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs.update(self._latest_fovea_logs)
        super().log(logs, start_time=start_time)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        output_dir = output_dir or self.args.output_dir
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        base_model = get_fovea_base_model(unwrapped_model)

        save_pretrained = getattr(unwrapped_model, "save_pretrained", None)
        if callable(save_pretrained) and unwrapped_model is not base_model:
            if self.args.should_save:
                save_pretrained(output_dir, safe_serialization=True)
                save_fovea_extra(base_model, output_dir)
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(output_dir)
            self.accelerator.wait_for_everyone()
            return

        if self.args.should_save:
            save_fovea_extra(base_model, output_dir)
        super().save_model(output_dir, _internal_call=_internal_call)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    processor = transformers.AutoProcessor.from_pretrained(model_args.model_name_or_path)
    processor.tokenizer = tokenizer

    init_model_path, fovea_extra_path = resolve_initialization_paths(
        model_args.model_name_or_path,
        training_args.resume_from_checkpoint,
    )

    model = FoveaForConditionalGeneration.from_pretrained(
        init_model_path,
        torch_dtype="auto",
        attn_implementation=training_args.attn_implementation,
    )
    load_fovea_extra(model, fovea_extra_path)
    set_multimodal_token_ids(model, tokenizer)
    model.config.fovea_crop_max_image_tokens = data_args.fovea_crop_max_img_tokens
    sync_tokenizer_special_tokens_with_model(tokenizer, model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    model.config.use_cache = False
    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if training_args.freeze_base_model:
        configure_training_mode(model, freeze_base_model=training_args.freeze_base_model)
    elif training_args.use_lora:
        model = configure_lora_training(model, training_args)
    else:
        configure_training_mode(model, freeze_base_model=False)
    print_parameter_summary(model)

    fovea_model = get_fovea_base_model(model)
    vision_packer = VisionPacker(
        processor=processor,
        vision_config=fovea_model.config.vision_config,
        max_image_tokens=data_args.max_img_tokens,
    )
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        processor=processor,
        tokenizer=tokenizer,
        vision_packer=vision_packer,
        image_token_id=fovea_model.config.image_token_id,
        system_message=data_args.system_message,
        fovea_crop_max_image_tokens=data_args.fovea_crop_max_img_tokens,
        model_max_length=training_args.model_max_length,
    )
    data_collator = DataCollatorForQwen3_5SFT(
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    trainer = FoveaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
