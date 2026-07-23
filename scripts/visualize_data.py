#!/usr/bin/env python
"""Visualize training or lmms-eval benchmark samples."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "lmms-eval"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize training or lmms-eval benchmark samples.")
    parser.add_argument(
        "--task",
        nargs="+",
        default=["train"],
        help="`train` and/or lmms-eval task names, such as `chartqa textvqa_val`.",
    )
    parser.add_argument("--index", type=int, nargs="*", default=None, help="Sample indices for every task.")
    parser.add_argument("--num_samples", type=int, default=10, help="Random samples per task; overrides --index.")
    parser.add_argument("--output_dir", default="outputs/data_visualize")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_path", default="data/VLM-R3-data/preprocessed/vlir_sft_12k.parquet")
    parser.add_argument("--image_folder", default="data/VLM-R3-data/preprocessed")
    parser.add_argument("--model_type", choices=("simple", "chat"), default="simple",
                        help="lmms-eval model implementation type used when resolving tasks.")
    parser.add_argument("--model_name", default="fovea", help="Model name used when resolving lmms-eval tasks.")
    parser.add_argument("--box_color", default="#FF4444")
    parser.add_argument("--box_alpha", type=float, default=0.3)
    parser.add_argument("--box_linewidth", type=float, default=2.0)
    parser.add_argument("--title_width", type=int, default=120)
    return parser.parse_args()


def wrap_text(text: str, width: int) -> str:
    lines = []
    for block in text.splitlines() or [""]:
        block = " ".join(block.split())
        lines.extend(textwrap.wrap(block, width=width) or [""])
    return "\n".join(lines)


def select_indices(total: int, args: argparse.Namespace, seed_offset: int) -> list[int]:
    if args.num_samples > 0:
        rng = np.random.default_rng(args.random_seed + seed_offset)
        return sorted(rng.choice(total, size=min(args.num_samples, total), replace=False).tolist())
    if args.index:
        return [index for index in args.index if 0 <= index < total]
    return list(range(min(10, total)))


def _as_images(visuals) -> list[Image.Image]:
    if isinstance(visuals, Image.Image):
        return [visuals]
    return [image for image in visuals if isinstance(image, Image.Image)]


def _draw_boxes(ax, boxes, image_size: tuple[int, int], args: argparse.Namespace) -> None:
    image_width, image_height = image_size
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = np.asarray(box, dtype=float).flatten()[:4]
        rect = mpatches.Rectangle(
            (x1 * image_width, y1 * image_height),
            (x2 - x1) * image_width,
            (y2 - y1) * image_height,
            linewidth=args.box_linewidth,
            edgecolor=args.box_color,
            facecolor=args.box_color,
            alpha=args.box_alpha,
        )
        ax.add_patch(rect)
        ax.text(x1 * image_width, y1 * image_height - 4, f"box {index}", fontsize=8, color=args.box_color)


def render_sample(
    images: list[Image.Image],
    question: str,
    answer: str,
    title: str,
    args: argparse.Namespace,
    boxes: list | None = None,
):
    details = [title, "", "[QUESTION]", wrap_text(question, args.title_width)]
    if answer:
        details.extend(["", "[ANSWER]", wrap_text(answer, args.title_width)])
    detail_text = "\n".join(details)
    text_height = max(3.0, 0.16 * (detail_text.count("\n") + 1))

    figure = plt.figure(figsize=(12 * len(images), 12 + text_height))
    grid = figure.add_gridspec(2, len(images), height_ratios=(12, text_height), hspace=0.04)
    for index, image in enumerate(images):
        axis = figure.add_subplot(grid[0, index])
        axis.imshow(np.asarray(image.convert("RGB")))
        axis.axis("off")
        if boxes is not None and len(boxes) > 0 and index == 0:
            _draw_boxes(axis, boxes, image.size, args)

    text_axis = figure.add_subplot(grid[1, :])
    text_axis.axis("off")
    text_axis.text(0, 1, detail_text, fontsize=9, va="top", ha="left", family="monospace")
    figure.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    return figure


def _train_question_answer(sample: dict) -> tuple[str, str]:
    question_parts = []
    answer_parts = []
    for turn in sample.get("conversations", []):
        value = str(turn.get("value", ""))
        if turn.get("from") in {"human", "user"}:
            question_parts.append(value.replace("<image>", "").strip())
        elif turn.get("from") in {"gpt", "assistant"}:
            answer_parts.append(value.replace('{"fovea"}', "[FOVEA]"))
    return "\n".join(question_parts), "\n".join(answer_parts)


def visualize_train(args: argparse.Namespace, output_dir: Path, seed_offset: int) -> None:
    data_path = Path(args.data_path)
    image_folder = Path(args.image_folder)
    dataframe = pd.read_parquet(data_path)
    index_path = data_path.with_name(data_path.stem + "_index.json")
    index_map = json.loads(index_path.read_text()) if index_path.exists() else None
    indices = select_indices(len(dataframe), args, seed_offset)

    print(f"\n[train] {len(dataframe)} samples; visualizing {len(indices)}")
    for output_index, dataset_index in enumerate(indices):
        sample = dataframe.iloc[dataset_index].to_dict()
        image_path = image_folder / sample["image"]
        if not image_path.exists():
            print(f"  SKIP {dataset_index}: image not found: {image_path}")
            continue
        source = index_map[dataset_index]["source"] if index_map else "train"
        original_index = index_map[dataset_index]["orig_idx"] if index_map else dataset_index
        question, answer = _train_question_answer(sample)
        boxes = sample.get("fovea_query_boxes", [])
        title = f"Task: train | sample={dataset_index} | source={source} | original={original_index}\nFovea boxes: {len(boxes)}"
        figure = render_sample([Image.open(image_path).convert("RGB")], question, answer, title, args, boxes)
        output_path = output_dir / "train" / f"sample_{output_index:05d}_{source}_{original_index}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"  Saved: {output_path}")


def init_eval_task(task_name: str, args: argparse.Namespace, task_manager):
    from lmms_eval.tasks import get_task_dict

    task_dict = get_task_dict(task_name, task_manager=task_manager, task_type=args.model_type)
    if task_name not in task_dict:
        raise ValueError(f"Task '{task_name}' not found: {sorted(task_dict)}")
    return task_dict[task_name]


def format_eval_answer(task, document: dict) -> str:
    target = task.doc_to_target(document)
    if isinstance(target, str) and target in document:
        target = document[target]
    elif isinstance(target, str) and document.get("answers"):
        target = None
    if target is None and document.get("answers"):
        counts = Counter(str(answer) for answer in document["answers"])
        return "Reference answers: " + ", ".join(f"{answer} ({count})" for answer, count in counts.most_common())
    if isinstance(target, (list, tuple)):
        return ", ".join(str(item) for item in target)
    return "" if target is None else str(target)


def visualize_eval_task(task_name: str, args: argparse.Namespace, output_dir: Path, seed_offset: int, task_manager) -> None:
    task = init_eval_task(task_name, args, task_manager)
    docs = task.task_docs
    indices = select_indices(len(docs), args, seed_offset)
    print(f"\n[{task_name}] {len(docs)} samples; visualizing {len(indices)}")

    for output_index, document_index in enumerate(indices):
        document = docs[document_index]
        images = _as_images(task.doc_to_visual(document))
        if not images:
            print(f"  SKIP {document_index}: task returned no PIL images")
            continue
        question = str(task.doc_to_text(document))
        answer = format_eval_answer(task, document)
        title = f"Task: {task_name} | sample={document_index} | images={len(images)}"
        figure = render_sample(images, question, answer, title, args)
        output_path = output_dir / task_name / f"sample_{output_index:05d}_{document_index}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_tasks = [task_name for task_name in args.task if task_name != "train"]
    task_manager = None
    if eval_tasks:
        from lmms_eval.tasks import TaskManager

        task_manager = TaskManager(model_name=args.model_name)

    for task_index, task_name in enumerate(args.task):
        if task_name == "train":
            visualize_train(args, output_dir, task_index)
        else:
            visualize_eval_task(task_name, args, output_dir, task_index, task_manager)


if __name__ == "__main__":
    main()
