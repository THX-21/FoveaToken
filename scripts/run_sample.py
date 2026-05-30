#!/usr/bin/env python
"""Run a few lmms-eval task samples through a chosen lmms-eval model."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lmms-eval samples through a chosen lmms-eval model.")
    parser.add_argument("--model", default="fovea", help="lmms-eval model name")
    parser.add_argument(
        "--model_args",
        default="pretrained=checkpoints/fovea-vgr-ft/checkpoint-100,device=cuda:0,device_map=cuda:0,attn_implementation=sdpa",
        help="Comma-separated lmms-eval model args, e.g. pretrained=...,device=cuda:0",
    )
    parser.add_argument("--force_simple", action="store_true", help="Force the simple model/task path")
    parser.add_argument("--task", default="mmstar", help="lmms-eval task name")
    parser.add_argument("--index", type=int, default=0, help="Start index in the task docs")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of consecutive samples to run")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


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


def _parse_model_args(text: str) -> dict:
    if not text.strip():
        return {}
    parsed = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid model arg '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        parsed[key.strip()] = _parse_scalar(value.strip())
    return parsed


def _init_task(task_name: str, model_name: str, task_type: str):
    from lmms_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager(model_name=model_name)
    task_dict = get_task_dict(task_name, task_manager=task_manager, task_type=task_type)
    if task_name not in task_dict:
        raise ValueError(f"Task '{task_name}' not found in resolved task dict: {sorted(task_dict)}")
    return task_dict[task_name]


def _init_model(model_name: str, model_args: dict, force_simple: bool):
    from lmms_eval.models import MODEL_REGISTRY_V2, get_model

    resolved = MODEL_REGISTRY_V2.resolve(model_name, force_simple=force_simple)
    model_cls = get_model(model_name, force_simple=force_simple)
    return resolved, model_cls(**model_args)


def _build_simple_messages(model, visuals, question: str):
    content = []
    for visual in visuals:
        content.append({"type": "image", "image": visual})
    content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": model.system_prompt},
        {"role": "user", "content": content},
    ]


def _prepare_simple_inputs(model, visuals, question: str):
    texts = model._apply_chat_template([_build_simple_messages(model, visuals, question)])
    processor_kwargs = {
        "text": texts,
        "images": list(visuals),
        "image_counts_per_sample": [len(visuals)],
        "return_tensors": "pt",
    }
    inputs = model.processor(**processor_kwargs)

    if hasattr(model, "vision_packer") and visuals:
        retrieve_pixels, retrieve_sizes, retrieve_boxes = model.vision_packer.pack_retrieve(visuals[0])
        import torch

        inputs["retrieve_pixel_values"] = retrieve_pixels
        inputs["retrieve_image_sizes"] = retrieve_sizes
        inputs["retrieve_patch_boxes"] = retrieve_boxes
        inputs["retrieve_image_counts"] = torch.tensor([1], dtype=torch.long)

    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }, texts[0]


def _prepare_chat_inputs(model, task, doc):
    raise RuntimeError("scripts/run_sample.py currently supports the simple lmms-eval path for Fovea/LLaVA-NeXT.")


def _decode_raw(model, gen_ids):
    decoder = getattr(model, "processor", None) or getattr(model, "tokenizer", None)
    if decoder is None or not hasattr(decoder, "batch_decode"):
        raise RuntimeError(f"Model '{type(model).__name__}' does not expose a batch_decode-capable processor/tokenizer.")
    raw = decoder.batch_decode(
        [gen_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    clean = decoder.batch_decode(
        [gen_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return raw, clean


def _count_fovea_tokens(model, ids_list: list[int]) -> str:
    cfg = getattr(model.model, "config", None)
    if cfg is None or getattr(cfg, "fovea_token_id", None) is None:
        return "n/a"
    return str(ids_list.count(int(cfg.fovea_token_id)))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "lmms-eval"))

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch

    model_args = _parse_model_args(args.model_args)
    resolved_model, model = _init_model(args.model, model_args, args.force_simple)
    task = _init_task(args.task, resolved_model.model_id, resolved_model.model_type)
    docs = task.task_docs

    print(f"[sample] model={args.model} resolved_type={resolved_model.model_type}")
    print(f"[sample] model_args={model_args}")
    print(f"[sample] task={args.task} docs={len(docs)}")

    total_time = 0.0
    for i in range(args.num_samples):
        doc_idx = args.index + i
        doc = docs[doc_idx]
        question = task.doc_to_text(doc) if hasattr(task, "doc_to_text") else ""
        answer = doc.get("answer", task.doc_to_target(doc) if hasattr(task, "doc_to_target") else "N/A")

        if resolved_model.model_type == "chat":
            inputs, context = _prepare_chat_inputs(model, task, doc)
        else:
            visuals = task.doc_to_visual(doc) or []
            inputs, context = _prepare_simple_inputs(model, visuals, question)

        gen_kwargs = model._build_generate_kwargs(
            {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.temperature > 0,
            }
        )

        t0 = time.time()
        with torch.no_grad():
            cont = model.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0
        total_time += elapsed

        gen_ids = cont[0, inputs["input_ids"].shape[1] :]
        n_tok = int(gen_ids.shape[0])
        hit_limit = n_tok >= args.max_new_tokens
        raw, clean = _decode_raw(model, gen_ids)
        has_think_end = "</think>" in clean
        fovea_count = _count_fovea_tokens(model, gen_ids.tolist())

        print(f"\n{'=' * 80}")
        print(
            f"Sample {doc_idx} | time={elapsed:.1f}s | tokens={n_tok} | "
            f"</think>={has_think_end} | fovea={fovea_count} | hit_limit={hit_limit}"
        )
        print(f"GT Answer: {answer}")
        if question:
            print(f"Question: {question}")
        else:
            print(f"Context: {context}")
        print(f"{'─' * 80}")
        print(raw)
        print(f"{'=' * 80}")

    print(f"\nTotal time: {total_time:.1f}s | Avg: {total_time / args.num_samples:.1f}s/sample")


if __name__ == "__main__":
    main()
