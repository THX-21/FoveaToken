import os
from dataclasses import dataclass, field

import torch
import transformers

from fovea_token import FoveaForConditionalGeneration
from fovea_token.tokenizers.tokenization_fovea import sync_fovea_token_ids

from .data import DataCollatorForQwen3_5SFT, LazySupervisedDataset, VisionPacker


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-4B")


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


def configure_low_contamination_training(model: FoveaForConditionalGeneration) -> None:
    """Freeze Qwen and train only Fovea retrieval plus the auxiliary head."""

    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith("fovea_"):
            param.requires_grad_(True)


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
    sync_fovea_token_ids(model.config, tokenizer)
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


class FoveaMetricsCallback(transformers.TrainerCallback):
    @staticmethod
    def _find_aux(model):
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
            for attr in ("module", "base_model", "model"):
                stack.append(getattr(current, attr, None))
        return None

    def on_log(self, _args, state, control, model=None, logs=None, **_kwargs):
        if model is None or logs is None:
            return control
        aux = self._find_aux(model)
        if aux is None:
            return control
        for key in ("lm_loss", "aux_lm_loss", "align_loss", "num_queries", "tokens_per_query"):
            value = aux.get(key)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "item"):
                value = value.item()
            logs[f"fovea/{key}"] = value
        return control


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
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer/model vocab mismatch: Fovea must not add or resize token rows.")
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

    configure_low_contamination_training(model)
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

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=[FoveaMetricsCallback],
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
