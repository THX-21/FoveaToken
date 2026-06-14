#!/usr/bin/env python
"""Visualize training data quality — image + QA with fovea boxes overlaid."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize training data samples.")
    parser.add_argument("--data_path", default="data/vgr/preprocessed/vgr_shortcot.parquet")
    parser.add_argument("--image_folder", default="data/llava_next/llava_next_raw_format")
    parser.add_argument("--index", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--output_dir", default="outputs/data_visualize")
    parser.add_argument("--box_color", default="#FF4444")
    parser.add_argument("--box_alpha", type=float, default=0.3)
    parser.add_argument("--box_linewidth", type=float, default=2.0)
    parser.add_argument("--title_width", type=int, default=120)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def wrap_text(text: str, width: int = 100) -> str:
    lines = []
    for block in text.splitlines() or [""]:
        block = " ".join(block.split())
        if not block:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(block, width=width) or [""])
    return "\n".join(lines)


def highlight_fovea(text: str) -> str:
    """Replace <fovea> with a visible marker in the displayed text."""
    return text.replace("<fovea>", " [FOVEA] ")


def render_sample(sample: dict, image_folder: Path, args) -> np.ndarray | None:
    image_path = image_folder / sample["image"]
    if not image_path.exists():
        print(f"  SKIP: image not found {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(np.asarray(image))
    ax.axis("off")

    # Draw fovea query boxes
    boxes = sample.get("fovea_query_boxes", [])
    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            box = np.asarray(box).flatten()
            x1, y1, x2, y2 = float(box[0]) * img_w, float(box[1]) * img_h, float(box[2]) * img_w, float(box[3]) * img_h
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=args.box_linewidth, edgecolor=args.box_color,
                facecolor=args.box_color, alpha=args.box_alpha,
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"box {i}", fontsize=8, color=args.box_color,
                    fontweight="bold", va="bottom", ha="left")

    # Parse conversations
    convs = sample.get("conversations", [])
    question = ""
    answer = ""
    for turn in convs:
        role = turn.get("from", "")
        value = str(turn.get("value", ""))
        if role in ("human", "user"):
            question = value.replace("<image>", "").strip()
        elif role in ("gpt", "assistant"):
            answer = value

    n_fovea = answer.count("<fovea>")
    n_boxes = len(boxes)

    title_lines = [
        f"Sample: {sample['image']}",
        f"Fovea tokens: {n_fovea}  |  Boxes: {n_boxes}  |  Preprocessed: {sample.get('fovea_preprocessed', 'N/A')}",
        f"",
        f"[QUESTION]",
        wrap_text(question, args.title_width),
        f"",
        f"[ANSWER]",
        wrap_text(highlight_fovea(answer), args.title_width),
    ]
    title = "\n".join(title_lines)
    ax.set_title(title, fontsize=9, loc="left", pad=12, fontfamily="monospace")

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.data_path)
    image_folder = Path(args.image_folder)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {len(df)} samples, {len(df.columns)} columns: {list(df.columns)}")
    print(f"Image folder: {image_folder}")
    print(f"Output: {out_dir}")

    for idx in args.index:
        if idx >= len(df):
            print(f"Index {idx} out of range ({len(df)} samples)")
            continue
        print(f"\n=== Sample {idx} ===")
        row = df.iloc[idx]
        sample = {col: row[col] for col in df.columns}

        fig = render_sample(sample, image_folder, args)
        if fig is None:
            continue

        out_path = out_dir / f"sample_{idx:05d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
