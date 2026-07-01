import os
from dataclasses import dataclass, field

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
    train_main_lm_head_loss: bool = field(default=False)


def configure_low_contamination_training(model: FoveaForConditionalGeneration) -> None:
    """Freeze Qwen and train only Fovea retrieval plus the auxiliary head."""

    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith("fovea_"):
            param.requires_grad_(True)

    fovea_token_id = getattr(model.config, "fovea_token_id", None)
    if fovea_token_id is None or int(fovea_token_id) < 0:
        raise ValueError("Frozen-base Fovea training requires a valid fovea_token_id.")

    input_embedding_weight = model.get_input_embeddings().weight
    input_embedding_weight.requires_grad_(True)

    allowed_row = int(fovea_token_id)

    def keep_only_fovea_row(grad):
        if grad is None:
            return grad
        masked = grad.new_zeros(grad.shape)
        masked[allowed_row].copy_(grad[allowed_row])
        return masked

    if hasattr(model, "_fovea_input_embedding_hook") and model._fovea_input_embedding_hook is not None:
        model._fovea_input_embedding_hook.remove()
    model._fovea_input_embedding_hook = input_embedding_weight.register_hook(keep_only_fovea_row)


def configure_training_mode(model: FoveaForConditionalGeneration, freeze_base_model: bool) -> None:
    if hasattr(model, "_fovea_input_embedding_hook") and model._fovea_input_embedding_hook is not None:
        model._fovea_input_embedding_hook.remove()
        model._fovea_input_embedding_hook = None
    if freeze_base_model:
        configure_low_contamination_training(model)
        return
    model.requires_grad_(True)


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


def should_initialize_from_resume(model_name_or_path: str, resume_from_checkpoint: str | None) -> bool:
    if not resume_from_checkpoint:
        return False
    if os.path.abspath(model_name_or_path) == os.path.abspath(resume_from_checkpoint):
        return True
    return os.path.isfile(os.path.join(resume_from_checkpoint, "model.safetensors"))


def find_fovea_aux(model):
    seen = set()
    stack = [model]
    empty_aux = None
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        aux = getattr(current, "_fovea_aux", None)
        if isinstance(aux, dict):
            if aux:
                return aux
            if empty_aux is None:
                empty_aux = aux
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
    return empty_aux


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
        aux = getattr(outputs, "fovea_metrics", None)
        if not isinstance(aux, dict):
            unwrapped_model = self.accelerator.unwrap_model(model)
            aux = (
                find_fovea_aux(model)
                or find_fovea_aux(unwrapped_model)
                or find_fovea_aux(getattr(self, "model_wrapped", None))
                or find_fovea_aux(getattr(self, "model", None))
            )
        cached_logs: dict[str, float] = {}
        if aux is not None:
            for key in ("lm_loss", "aux_lm_loss", "main_lm_loss", "align_loss", "num_queries", "tokens_per_query"):
                value = aux.get(key)
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

    init_model_path = model_args.model_name_or_path
    if should_initialize_from_resume(model_args.model_name_or_path, training_args.resume_from_checkpoint):
        init_model_path = training_args.resume_from_checkpoint
        print(f"Initializing model weights from resume checkpoint: {init_model_path}")

    model = FoveaForConditionalGeneration.from_pretrained(
        init_model_path,
        torch_dtype="auto",
        attn_implementation=training_args.attn_implementation,
    )
    set_multimodal_token_ids(model, tokenizer)
    model.config.fovea_crop_max_image_tokens = data_args.fovea_crop_max_img_tokens
    model.config.fovea_train_main_lm_head = training_args.train_main_lm_head_loss
    sync_tokenizer_special_tokens_with_model(tokenizer, model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    model.config.use_cache = False
    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    configure_training_mode(model, freeze_base_model=training_args.freeze_base_model)
    print_parameter_summary(model)

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
