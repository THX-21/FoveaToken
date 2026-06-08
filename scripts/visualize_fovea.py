#!/usr/bin/env python
"""Visualize Fovea attention for training samples or lmms-eval samples."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Fovea attention.")
    parser.add_argument("--task", default="mmstar", help="`train` or an lmms-eval task like `mmstar`.")
    parser.add_argument("--model_name_or_path", default="checkpoints/fovea-vgr-ft/checkpoint-2800")
    parser.add_argument("--index", type=int, default=50)
    parser.add_argument("--output_dir", default="outputs/fovea_visualize")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--device_map", default="cuda:3")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--force_simple", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--data_path", default="data/vgr/preprocessed")
    parser.add_argument("--image_folder", default="data/vgr/llava_next_raw_format")
    return parser.parse_args()


def to_numpy_image(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def render_heatmap(image: np.ndarray, boxes: torch.Tensor, weights: torch.Tensor, alpha: float, cmap: str) -> np.ndarray:
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    boxes = boxes.detach().cpu().numpy()
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


def _build_simple_messages(model, visuals, question: str):
    content = []
    for visual in visuals:
        content.append({"type": "image", "image": visual})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _prepare_simple_inputs(model, visuals, question: str):
    texts = model._apply_chat_template([_build_simple_messages(model, visuals, question)])
    inputs = model.processor(
        text=texts,
        images=list(visuals),
        return_mm_token_type_ids=True,
        return_tensors="pt",
    )
    if getattr(model, "vision_packer", None) is not None and visuals and not getattr(model, "disable_fovea_retrieval", False):
        retrieve_pixels = []
        retrieve_grids = []
        retrieve_boxes = []
        for visual in visuals:
            pixels, grid, boxes = model.vision_packer.pack_retrieve(visual, getattr(model, "retrieve_max_image_tokens", None))
            retrieve_pixels.append(pixels)
            retrieve_grids.append(grid)
            retrieve_boxes.append(boxes)
        inputs["retrieve_pixel_values"] = torch.cat(retrieve_pixels, dim=0)
        inputs["retrieve_grid_thw"] = torch.stack(retrieve_grids, dim=0)
        inputs["retrieve_patch_boxes"] = torch.cat(retrieve_boxes, dim=0)
        inputs["retrieve_image_counts"] = torch.tensor([len(visuals)], dtype=torch.long)
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }, texts[0]


def _decode_generated_text(model, full_ids: torch.Tensor, prompt_len: int) -> str:
    generated = full_ids[prompt_len:]
    return model.processor.batch_decode(
        [generated],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]


def load_train_sample(args):
    from fovea_token import FoveaForConditionalGeneration
    from fovea_token.tokenizers.tokenization_fovea import add_fovea_tokens, sync_fovea_token_ids
    from fovea_token.train.data import DataCollatorForQwen3_5SFT, LazySupervisedDataset, VisionPacker
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    add_fovea_tokens(tokenizer)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    processor.tokenizer = tokenizer

    model = FoveaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    sync_fovea_token_ids(model.config, tokenizer)
    model.config.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    model.config.video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    model.config.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    model.config.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    model.eval()

    packer = VisionPacker(processor=processor, vision_config=model.config.vision_config)
    dataset = LazySupervisedDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        processor=processor,
        tokenizer=tokenizer,
        vision_packer=packer,
        image_token_id=model.config.image_token_id,
    )
    item = dataset[args.index]
    batch = DataCollatorForQwen3_5SFT(tokenizer=tokenizer, model_max_length=32768)([item])
    batch = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in batch.items()}
    with torch.no_grad():
        model(**{k: v for k, v in batch.items() if v is not None})

    record = dataset.records[args.index]
    image_name = record["image"]
    image = Image.open(Path(args.image_folder) / image_name).convert("RGB")
    question, answer = get_sample_qa(record)
    history = []
    for call_idx, entry in enumerate(model._fovea_aux_history):
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
                    "boxes": batch["retrieve_patch_boxes"],
                    "query_box": query_box,
                }
            )
    return {
        "image": image,
        "question": question,
        "answer": answer,
        "prediction": answer,
        "history": history,
        "sample_name": f"train_{args.index:05d}",
    }


def load_lmms_eval_sample(args):
    resolved_model, lmms_model = _init_model(
        force_simple=args.force_simple,
        model_kwargs={
            "pretrained": args.model_name_or_path,
            "device": args.device,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "enable_thinking": args.enable_thinking,
        },
    )
    task = _init_task(args.task, resolved_model.model_id, resolved_model.model_type)
    doc = task.task_docs[args.index]
    doc_index = doc.get("index", args.index)
    question = task.doc_to_text(doc) if hasattr(task, "doc_to_text") else ""
    answer = doc.get("answer", task.doc_to_target(doc) if hasattr(task, "doc_to_target") else "")
    visuals = task.doc_to_visual(doc) or []
    if len(visuals) != 1:
        raise ValueError(f"Current lmms-eval visualization expects one image, got {len(visuals)}.")
    visual = visuals[0]
    if not isinstance(visual, Image.Image):
        raise ValueError(f"Current lmms-eval visualization expects PIL image input, got {type(visual)}.")

    inputs, _prompt = _prepare_simple_inputs(lmms_model, visuals, question)
    gen_kwargs = lmms_model._build_generate_kwargs(
        {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
    )
    with torch.no_grad():
        cont = lmms_model.model.generate(**inputs, **gen_kwargs)
    prediction = _decode_generated_text(lmms_model, cont[0], inputs["input_ids"].shape[1])

    history = []
    for call_idx, entry in enumerate(lmms_model.model._fovea_aux_history):
        attn_mean = entry["fovea_attn_mean"]
        if attn_mean.ndim == 2:
            attn_mean = attn_mean.unsqueeze(0)
        for trigger_idx, attn in enumerate(attn_mean):
            history.append(
                {
                    "call_idx": call_idx,
                    "trigger_idx": trigger_idx,
                    "attn": attn,
                    "boxes": inputs["retrieve_patch_boxes"],
                    "query_box": None,
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
            "No <fovea> trigger was recorded for this sample."
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

        fig, axes = plt.subplots(8, 8, figsize=(24, 24))
        for token_idx, ax in enumerate(axes.flat):
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

    if args.task == "train":
        sample = load_train_sample(args)
    else:
        sample = load_lmms_eval_sample(args)
    save_visualizations(sample, args)


if __name__ == "__main__":
    main()
