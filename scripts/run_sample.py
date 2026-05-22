#!/usr/bin/env python
"""Run one MMStar sample through the real Fovea visual-query inference path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Fovea sample with visual-query replay enabled.")
    parser.add_argument("--checkpoint", default="checkpoints/fovea-visual-query-replay/checkpoint-2400")
    parser.add_argument("--dataset", default="Lin-Chen/MMStar")
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--max_new_tokens", type=int, default=1280)
    parser.add_argument("--max_image_tokens", type=int, default=512)
    parser.add_argument("--retrieve_max_image_tokens", type=int, default=4096)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face network access instead of offline cache only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "lmms-eval"))

    if not args.online:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    from datasets import load_dataset
    from lmms_eval.models.simple.fovea import Fovea
    from lmms_eval.tasks._task_utils.reasoning_utils import DEFAULT_REASONING_SYSTEM_PROMPT
    from lmms_eval.tasks.mmstar.utils import mmstar_doc_to_text, mmstar_doc_to_visual

    checkpoint = str((root / args.checkpoint).resolve()) if not Path(args.checkpoint).is_absolute() else args.checkpoint

    print(f"[sample] load dataset={args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    doc = ds[args.index]

    print(f"[sample] load model={checkpoint}")
    model = Fovea(
        pretrained=checkpoint,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        enable_thinking=args.enable_thinking,
        max_image_tokens=args.max_image_tokens,
        retrieve_max_image_tokens=args.retrieve_max_image_tokens,
        batch_size=1,
    )

    context = mmstar_doc_to_text(doc, {})
    visual = mmstar_doc_to_visual(doc)[0]
    message = [
        {"role": "system", "content": DEFAULT_REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]},
    ]
    text = model._apply_chat_template([message])[0]

    inputs = model.processor(
        text=[text],
        images=[visual],
        image_counts_per_sample=[1],
        return_mm_token_type_ids=True,
        return_tensors="pt",
    )

    retrieve_pixels, retrieve_grid, retrieve_boxes = model.vision_packer.pack_retrieve(
        visual,
        model.retrieve_max_image_tokens,
    )
    inputs["retrieve_pixel_values"] = retrieve_pixels
    inputs["retrieve_grid_thw"] = retrieve_grid.unsqueeze(0)
    inputs["retrieve_patch_boxes"] = retrieve_boxes
    inputs["retrieve_image_counts"] = torch.tensor([1], dtype=torch.long)
    inputs = inputs.to(model.device)

    print("[sample] generate with visual-query replay")
    outputs = model.model.generate(
        **inputs,
        **model._build_generate_kwargs({"max_new_tokens": args.max_new_tokens}),
    )
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    raw = model.processor.batch_decode(
        [generated_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    clean = model.processor.batch_decode(
        [generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("---RAW-ANSWER---")
    print(raw)
    print("---CLEAN-ANSWER---")
    print(model._strip_thinking(clean))
    print("---END---")
    print(f"generated_tokens={generated_ids.shape[0]}")


if __name__ == "__main__":
    main()
