#!/usr/bin/env python
"""Visualize Fovea attention for training samples or lmms-eval samples."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "lmms-eval"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from fovea_token.fovea_crop import crop_attended_regions, normalize_patch_boxes
from fovea_token.tokenizers.tokenization_fovea import FOVEA_TOOL_CALL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Fovea attention.")
    parser.add_argument("--task", default="chartqa", help="`train` or an lmms-eval task like `mmstar`.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--index", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9,10])
    parser.add_argument("--output_dir", default="outputs/fovea_visualize")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--force_simple", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--data_path", default="data/VLM-R3-data/preprocessed/vlir_sft_12k.parquet")
    parser.add_argument("--image_folder", default="data/VLM-R3-data/preprocessed")
    parser.add_argument("--disable_fovea_retrieval", type=lambda x: x.lower() == "true", default=False,
                        help="Disable fovea retrieval pipeline.")
    parser.add_argument("--crop", default="true", help="Crop the most-attended regions and save them.")
    parser.add_argument("--crop_threshold", type=float, default=0.2,
                        help="Threshold for cropping: boxes with weight > threshold * max_weight are cropped.")
    parser.add_argument("--crop_margin", type=float, default=1.0,
                        help="Patch count tolerance for connectivity. 0 = strictly adjacent, 1 = one-patch gap allowed.")
    parser.add_argument("--crop_padding", type=float, default=1.0,
                        help="Expand each crop region by this many patches outward.")
    return parser.parse_args()


def to_numpy_image(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def render_heatmap(image: np.ndarray, boxes: torch.Tensor, weights: torch.Tensor, alpha: float, cmap: str) -> np.ndarray:
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    boxes = normalize_patch_boxes(boxes).detach().cpu().numpy()
    weights = weights.detach().cpu().numpy()
    for box, weight in zip(boxes, weights):
        x1 = max(0, min(w, int(round(box[0] * w))))
        y1 = max(0, min(h, int(round(box[1] * h))))
        x2 = max(0, min(w, int(round(box[2] * w))))
        y2 = max(0, min(h, int(round(box[3] * h))))
        if x2 > x1 and y2 > y1:
            heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], float(weight))
    if heat.max() > 0:
        heat = heat / heat.max()
    colored = plt.get_cmap(cmap)(heat)[..., :3]
    return (image * (1.0 - alpha) + colored * 255.0 * alpha).clip(0, 255).astype(np.uint8)


def get_sample_qa(record: dict) -> tuple[str, str]:
    question_parts = []
    answer_parts = []
    for turn in record.get("conversations", []):
        role = turn.get("from")
        value = str(turn.get("value", "")).strip()
        if not value:
            continue
        if role in {"human", "user"}:
            question_parts.append(value)
        elif role in {"gpt", "assistant"}:
            answer_parts.append(value)
    return "\n".join(question_parts), "\n".join(answer_parts)


def wrap_for_title(text: str, width: int = 90) -> str:
    lines = []
    for block in text.splitlines() or [""]:
        block = " ".join(block.split())
        if not block:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(block, width=width) or [""])
    return "\n".join(lines)


def format_box_text(box: torch.Tensor | None) -> str:
    if box is None or box.numel() != 4:
        return "n/a"
    values = [float(v) for v in box.detach().cpu().tolist()]
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _parse_scalar(text: str):
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _init_task(task_name: str, model_name: str, task_type: str):
    from lmms_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager(model_name=model_name)
    task_dict = get_task_dict(task_name, task_manager=task_manager, task_type=task_type)
    if task_name not in task_dict:
        raise ValueError(f"Task '{task_name}' not found in resolved task dict: {sorted(task_dict)}")
    return task_dict[task_name]


def _init_model(force_simple: bool, model_kwargs: dict):
    from lmms_eval.models import MODEL_REGISTRY_V2, get_model

    resolved = MODEL_REGISTRY_V2.resolve("fovea", force_simple=force_simple)
    model_cls = get_model("fovea", force_simple=force_simple)
    return resolved, model_cls(**model_kwargs)


def _task_eval_split(task) -> str:
    if task.has_test_docs():
        return task.config.test_split
    if task.has_validation_docs():
        return task.config.validation_split
    raise ValueError(f"Task '{task.config.task}' has no test or validation split.")


def _run_simple_sample(model, task, doc_idx: int, context: str, gen_kwargs: dict):
    from lmms_eval.api.instance import Instance, unwrap_generation_output

    split = _task_eval_split(task)
    task_name = task.config.task
    model.task_dict = {task_name: {split: task.task_docs}}
    request = Instance(
        request_type="generate_until",
        arguments=(context, gen_kwargs, task.doc_to_visual, doc_idx, task_name, split),
        idx=0,
        metadata={"task": task_name, "doc_id": doc_idx, "repeats": 1},
    )
    output = model.generate_until([request])[0]
    text, token_counts = unwrap_generation_output(output)
    return text, token_counts


def _fovea_metrics_history_from_lmms(lmms_model) -> list:
    model = getattr(lmms_model, "model", None)
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        model = get_base_model()
    return getattr(model, "_fovea_metrics_history", None) or []


def _fovea_metrics_history_from_model(model) -> list:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        model = get_base_model()
    return getattr(model, "_fovea_metrics_history", None) or []


_TRAIN_STATE = None  # singleton: (model, tokenizer, processor, dataset, collator)

MODEL_WEIGHT_FILENAMES = {
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


def _as_local_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.exists() else None


def _is_lora_checkpoint_dir(path: str | None) -> bool:
    candidate = _as_local_path(path)
    return bool(candidate and candidate.is_dir() and (candidate / "adapter_config.json").exists())


def _is_full_checkpoint_dir(path: str | None) -> bool:
    candidate = _as_local_path(path)
    if not candidate or not candidate.is_dir() or not (candidate / "config.json").exists():
        return False
    return any((candidate / name).exists() for name in MODEL_WEIGHT_FILENAMES)


def _lora_base_model_name(adapter_path: str) -> str:
    adapter_config = Path(adapter_path).expanduser() / "adapter_config.json"
    data = json.loads(adapter_config.read_text())
    base_model = data.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"LoRA checkpoint {adapter_path} does not define base_model_name_or_path.")
    return base_model


def _load_auto_resource(factory, preferred_source: str, fallback_source: str | None = None, **kwargs):
    try:
        return factory.from_pretrained(preferred_source, **kwargs)
    except Exception:
        if fallback_source is None or fallback_source == preferred_source:
            raise
        return factory.from_pretrained(fallback_source, **kwargs)


def _init_train_state(args):
    global _TRAIN_STATE
    if _TRAIN_STATE is not None:
        return _TRAIN_STATE

    from fovea_token import FoveaForConditionalGeneration
    from fovea_token.train.sft import FOVEA_EXTRA_WEIGHTS_NAME, load_fovea_extra
    from fovea_token.tokenizers.tokenization_fovea import sync_fovea_trigger_ids
    from fovea_token.train.data import DataCollatorForFoveaSFT, LazySupervisedDataset, VisionPacker
    from transformers import AutoProcessor, AutoTokenizer

    print("Loading model (once)...")
    checkpoint = args.model_name_or_path
    is_lora = _is_lora_checkpoint_dir(checkpoint)
    model_source = _lora_base_model_name(checkpoint) if is_lora else checkpoint
    tokenizer = _load_auto_resource(AutoTokenizer, checkpoint, model_source, use_fast=True)
    processor = _load_auto_resource(AutoProcessor, checkpoint, model_source)
    processor.tokenizer = tokenizer

    model = FoveaForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    if is_lora:
        from peft import PeftModel

        if (Path(checkpoint).expanduser() / FOVEA_EXTRA_WEIGHTS_NAME).exists():
            load_fovea_extra(model, checkpoint)
        model = PeftModel.from_pretrained(model, checkpoint)
        fovea_model = model.get_base_model()
    else:
        if _is_full_checkpoint_dir(checkpoint):
            load_fovea_extra(model, checkpoint)
        fovea_model = model
    if len(tokenizer) != fovea_model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer/model vocab mismatch: Fovea must not add or resize token rows.")
    sync_fovea_trigger_ids(fovea_model.config, tokenizer)
    fovea_model.config.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    fovea_model.config.video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    fovea_model.config.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    fovea_model.config.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    model.eval()

    packer = VisionPacker(processor=processor, vision_config=fovea_model.config.vision_config)
    dataset = LazySupervisedDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        processor=processor,
        tokenizer=tokenizer,
        vision_packer=packer,
        image_token_id=fovea_model.config.image_token_id,
    )
    collator = DataCollatorForFoveaSFT(tokenizer=tokenizer, model_max_length=32768)
    _TRAIN_STATE = (model, tokenizer, processor, dataset, collator)
    return _TRAIN_STATE


def load_train_sample(args, index: int):
    model, tokenizer, processor, dataset, collator = _init_train_state(args)

    item = dataset[index]
    batch = collator([item])
    batch = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in batch.items()}
    with torch.no_grad():
        model(**{k: v for k, v in batch.items() if v is not None})

    record = dataset.records[index]
    image_name = record["image"]
    image = Image.open(Path(args.image_folder) / image_name).convert("RGB")
    question, answer = get_sample_qa(record)

    # Insert Fovea tool calls into answer at trigger positions if not already present.
    if FOVEA_TOOL_CALL not in answer:
        labels = batch["labels"][0]  # [seq_len]
        answer_mask = labels != -100
        answer_start = int(answer_mask.nonzero(as_tuple=True)[0][0].item())
        fovea_positions = batch["fovea_positions"]  # [N, 2]: [batch_idx, position]
        fovea_offsets = sorted(
            (int(p[1].item()) - answer_start for p in fovea_positions if int(p[0].item()) == 0),
            reverse=True,
        )
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        for offset in fovea_offsets:
            offset = max(0, min(offset, len(answer_tokens)))
            prefix = tokenizer.decode(answer_tokens[:offset], skip_special_tokens=False)
            suffix = tokenizer.decode(answer_tokens[offset:], skip_special_tokens=False)
            answer = prefix + FOVEA_TOOL_CALL + suffix
            answer_tokens = tokenizer.encode(answer, add_special_tokens=False)

    history = []
    for call_idx, entry in enumerate(_fovea_metrics_history_from_model(model)):
        attn_mean = entry["fovea_attn_mean"]
        if attn_mean.ndim == 2:
            attn_mean = attn_mean.unsqueeze(0)
        for trigger_idx, attn in enumerate(attn_mean):
            query_box = batch["fovea_boxes"][trigger_idx] if trigger_idx < int(batch["fovea_boxes"].shape[0]) else None
            history.append(
                {
                    "call_idx": call_idx,
                    "trigger_idx": trigger_idx,
                    "attn": attn,
                    "boxes": entry["retrieve_patch_boxes"],
                    "query_box": query_box,
                }
            )
    return {
        "image": image,
        "question": question,
        "answer": answer,
        "prediction": answer,
        "history": history,
        "sample_name": f"train_{index:05d}",
    }


def load_lmms_eval_sample(args, index: int):
    resolved_model, lmms_model = _init_model(
        force_simple=args.force_simple,
        model_kwargs={
            "pretrained": args.model_name_or_path,
            "device": args.device,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "disable_fovea_retrieval": args.disable_fovea_retrieval,
        },
    )
    task = _init_task(args.task, resolved_model.model_id, resolved_model.model_type)
    doc = task.task_docs[index]
    doc_index = doc.get("index", index)
    question = task.doc_to_text(doc) if hasattr(task, "doc_to_text") else ""
    answer = doc.get("answer", task.doc_to_target(doc) if hasattr(task, "doc_to_target") else "")
    visuals = task.doc_to_visual(doc) or []
    if len(visuals) != 1:
        raise ValueError(f"Current lmms-eval visualization expects one image, got {len(visuals)}.")
    visual = visuals[0]
    if not isinstance(visual, Image.Image):
        raise ValueError(f"Current lmms-eval visualization expects PIL image input, got {type(visual)}.")

    gen_kwargs = dict(getattr(task.config, "generation_kwargs", {}) or {})
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["do_sample"] = args.temperature > 0
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k

    with torch.no_grad():
        prediction, _token_counts = _run_simple_sample(lmms_model, task, index, question, gen_kwargs)

    fovea_history = _fovea_metrics_history_from_lmms(lmms_model)
    history = []
    for call_idx, entry in enumerate(fovea_history):
        attn_mean = entry["fovea_attn_mean"]
        if attn_mean.ndim == 2:
            attn_mean = attn_mean.unsqueeze(0)
        trigger_offset = entry.get("trigger_offset", None)
        for trigger_idx, attn in enumerate(attn_mean):
            history.append(
                {
                    "call_idx": call_idx,
                    "trigger_idx": trigger_idx,
                    "attn": attn,
                    "boxes": entry["retrieve_patch_boxes"],
                    "query_box": None,
                    "trigger_offset": trigger_offset,
                }
            )
    return {
        "image": visual,
        "question": question,
        "answer": str(answer),
        "prediction": prediction,
        "history": history,
        "sample_name": f"{args.task}_{int(doc_index):05d}",
    }


def save_visualizations(sample: dict, args) -> None:
    image_np = to_numpy_image(sample["image"])
    out_dir = Path(args.output_dir) / sample["sample_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sample["history"]:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_np)
        ax.axis("off")
        title = (
            f"Task: {args.task}\n"
            f"Q: {wrap_for_title(sample['question'])}\n\n"
            f"GT: {wrap_for_title(sample['answer'])}\n\n"
            f"Pred: {wrap_for_title(sample['prediction'])}\n\n"
            "No Fovea trigger was recorded for this sample."
        )
        ax.set_title(title, fontsize=10, loc="left", pad=16)
        fig.tight_layout()
        fig.savefig(out_dir / "no_fovea_trigger.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    for entry in sample["history"]:
        stem = f"call_{entry['call_idx']:02d}_trigger_{entry['trigger_idx']:02d}"
        summary = entry["attn"].mean(dim=0)
        summary_panel = render_heatmap(image_np, entry["boxes"], summary, args.alpha, args.cmap)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(summary_panel)
        ax.axis("off")
        title = (
            f"Task: {args.task}  Call: {entry['call_idx']:02d}  Trigger: {entry['trigger_idx']:02d}  "
            f"Box: {format_box_text(entry['query_box'])}\n"
            f"Q: {wrap_for_title(sample['question'])}\n\n"
            f"GT: {wrap_for_title(sample['answer'])}\n\n"
            f"Pred: {wrap_for_title(sample['prediction'])}"
        )
        ax.set_title(title, fontsize=10, loc="left", pad=16)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Crop most-attended regions
        if getattr(args, "crop", False):
            crops = crop_attended_regions(
                sample["image"], entry["boxes"], summary,
                threshold=args.crop_threshold, margin=args.crop_margin, padding=args.crop_padding,
            )
            for crop_idx, c in enumerate(crops[:1]):
                crop_path = out_dir / f"{stem}_crop_{crop_idx:02d}.png"
                c.crop.save(crop_path)
                print(f"Saved crop: {crop_path}  box={c.box}  weight={c.weight:.4f}")

        num_tokens = int(entry["attn"].shape[0])
        grid_size = int(np.ceil(num_tokens**0.5))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
        for token_idx, ax in enumerate(axes.flat):
            if token_idx >= num_tokens:
                ax.axis("off")
                continue
            panel = render_heatmap(image_np, entry["boxes"], entry["attn"][token_idx], args.alpha, args.cmap)
            ax.imshow(panel)
            ax.set_title(f"{token_idx:02d}", fontsize=8)
            ax.axis("off")
        fig.suptitle(
            f"Task: {args.task}  Call: {entry['call_idx']:02d}  Trigger: {entry['trigger_idx']:02d}",
            fontsize=16,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_overview.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "lmms-eval"))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
        args.device_map = "cpu"

    for idx in args.index:
        print(f"\n=== Processing index {idx} ===")
        if args.task == "train":
            sample = load_train_sample(args, idx)
        else:
            sample = load_lmms_eval_sample(args, idx)
        save_visualizations(sample, args)


if __name__ == "__main__":
    main()
