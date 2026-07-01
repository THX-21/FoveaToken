#!/usr/bin/env python
"""Run a few lmms-eval task samples through a chosen lmms-eval model."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lmms-eval samples through a chosen lmms-eval model.")
    parser.add_argument("--model", default="fovea", help="lmms-eval model name")
    parser.add_argument(
        "--model_args",
        default="pretrained=checkpoints/fovea-vgr-qwen-9b/checkpoint-250,device=cuda:4,device_map=cuda:4,attn_implementation=sdpa,enable_thinking=True,max_image_tokens=2048,disable_fovea_retrieval=False,fovea_use_aux_head=True",
        help="Comma-separated lmms-eval model args, e.g. pretrained=...,device=cuda:0",
    )
    parser.add_argument("--force_simple", action="store_true", help="Force the simple model/task path")
    parser.add_argument("--task", default="mmstar", help="lmms-eval task name")
    parser.add_argument("--index", type=int, default=0, help="Start index in the task docs")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of consecutive samples to run")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--show_prompt", action="store_true", help="Print the full rendered prompt")
    parser.add_argument("--show_full", action="store_true", help="Print the full decoded prompt+generation")
    parser.add_argument("--preview_chars", type=int, default=3000, help="Max chars to show for preview blocks")
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


def _task_eval_split(task) -> str:
    if task.has_test_docs():
        return task.config.test_split
    if task.has_validation_docs():
        return task.config.validation_split
    raise ValueError(f"Task '{task.config.task}' has no test or validation split.")


def _run_chat_sample(model, task, doc_idx: int, context: str, gen_kwargs: dict):
    from lmms_eval.api.instance import Instance, unwrap_generation_output

    split = _task_eval_split(task)
    task_name = task.config.task
    model.task_dict = {task_name: {split: task.task_docs}}
    request = Instance(
        request_type="generate_until",
        arguments=(context, task.doc_to_messages, gen_kwargs, doc_idx, task_name, split),
        idx=0,
        metadata={"task": task_name, "doc_id": doc_idx, "repeats": 1},
    )
    output = model.generate_until([request])[0]
    text, token_counts = unwrap_generation_output(output)
    return text, token_counts


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


def _restore_logged_output(model, text: str) -> str:
    from lmms_eval.api.reasoning import restore_prefilled_reasoning_prefix

    return restore_prefilled_reasoning_prefix(
        text,
        [["<think>", "</think>"]],
        getattr(model, "enable_thinking", None),
    )


def _compact_text(text: str, limit: int) -> str:
    text = re.sub(r"(?:<image>){4,}", lambda m: f"<image>x{len(m.group(0)) // 7}", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _annotate_fovea_triggers(text: str, tokenizer, trigger_offsets: list[int]) -> str:
    """Insert [FOVEA] markers at token offsets (relative to generation start)."""
    if not trigger_offsets or tokenizer is None:
        return text
    offsets = sorted(set(trigger_offsets))
    ids = tokenizer(text, add_special_tokens=False).input_ids
    char_pos = [0]
    for tid in ids:
        char_pos.append(char_pos[-1] + len(tokenizer.decode([tid])))
    # Skip prefilled <think> token(s) at the start so that trigger_offset=0
    # maps to the first *generated* token, not the prefilled one.
    gen_start = 0
    if text.startswith("<think>"):
        gen_start = len(tokenizer("<think>", add_special_tokens=False).input_ids)
    result = []
    oi = 0
    for i, (s, e) in enumerate(zip(char_pos[:-1], char_pos[1:])):
        if i >= gen_start:
            while oi < len(offsets) and offsets[oi] == i - gen_start:
                result.append("[FOVEA]")
                oi += 1
        result.append(text[s:e])
    while oi < len(offsets):
        result.append("[FOVEA]")
        oi += 1
    return "".join(result)


def _print_block(title: str, text: str, limit: int) -> None:
    print(f"[{title}]")
    print(_compact_text(text, limit))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "lmms-eval"))

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
        context = question
        answer = doc.get("answer", task.doc_to_target(doc) if hasattr(task, "doc_to_target") else "N/A")

        sample_gen_kwargs = dict(getattr(task.config, "generation_kwargs", {}) or {})
        if args.max_new_tokens is not None:
            sample_gen_kwargs["max_new_tokens"] = args.max_new_tokens
        if args.temperature is not None:
            sample_gen_kwargs["temperature"] = args.temperature
            sample_gen_kwargs["do_sample"] = args.temperature > 0
        if args.top_p is not None:
            sample_gen_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            sample_gen_kwargs["top_k"] = args.top_k

        if resolved_model.model_type == "chat":
            t0 = time.time()
            raw, token_counts = _run_chat_sample(model, task, doc_idx, question, sample_gen_kwargs)
            elapsed = time.time() - t0
            total_time += elapsed
            display_raw = _restore_logged_output(model, raw)

            n_tok = token_counts.output_tokens if token_counts is not None else "n/a"
            max_new_tokens = sample_gen_kwargs.get("max_new_tokens")
            hit_limit = n_tok != "n/a" and max_new_tokens is not None and int(n_tok) >= int(max_new_tokens)
            has_think_end = "</think>" in display_raw
            fovea_hist = getattr(getattr(model, "model", None), "_fovea_aux_history", None) or []
            fovea_count = len(fovea_hist)
        else:
            t0 = time.time()
            raw, token_counts = _run_simple_sample(model, task, doc_idx, question, sample_gen_kwargs)
            elapsed = time.time() - t0
            total_time += elapsed
            display_raw = _restore_logged_output(model, raw)

            n_tok = token_counts.output_tokens if token_counts is not None else "n/a"
            max_new_tokens = sample_gen_kwargs.get("max_new_tokens")
            hit_limit = n_tok != "n/a" and max_new_tokens is not None and int(n_tok) >= int(max_new_tokens)
            has_think_end = "</think>" in display_raw
            fovea_hist = getattr(getattr(model, "model", None), "_fovea_aux_history", None) or []
            fovea_count = len(fovea_hist)

        fovea_offsets = [e["trigger_offset"] for e in fovea_hist if "trigger_offset" in e]
        display_annotated = _annotate_fovea_triggers(display_raw, getattr(model, "tokenizer", None), fovea_offsets) if fovea_offsets else display_raw

        print(f"\n{'=' * 80}")
        print(f"Sample {doc_idx}")
        print(f"Time: {elapsed:.1f}s | Tokens: {n_tok} | </think>: {has_think_end} | Fovea: {fovea_count} | Limit: {hit_limit}")
        print(f"GT: {answer}")
        print(f"Q: {question or context}")
        print(f"{'─' * 80}")
        _print_block("OUTPUT", display_annotated, args.preview_chars)
        print(f"{'=' * 80}")

    print(f"\nTotal time: {total_time:.1f}s | Avg: {total_time / args.num_samples:.1f}s/sample")


if __name__ == "__main__":
    main()
