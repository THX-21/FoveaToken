#!/usr/bin/env python
"""Visualize one training sample with image, boxes, and conversation text."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one training sample.")
    parser.add_argument("--data_path", default="data/vgr/preprocessed", help="Parquet file or directory.")
    parser.add_argument("--image_folder", default="data/vgr/llava_next_raw_format", help="Image root.")
    parser.add_argument("--index", type=int, default=0, help="Global sample index across parquet files.")
    parser.add_argument("--output_dir", default="outputs/train_data_visualize", help="Output root.")
    return parser.parse_args()


def load_records(data_path: str) -> list[dict]:
    path = Path(data_path)
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {path}.")
    records: list[dict] = []
    for item in paths:
        frame = pd.read_parquet(item)
        records.extend(frame.to_dict("records"))
    return records


def normalize_boxes(value) -> list[list[float]]:
    boxes = []
    if value is None:
        return boxes
    for box in value:
        if hasattr(box, "tolist"):
            box = box.tolist()
        boxes.append([float(v) for v in box])
    return boxes


def render_boxed_image(image: Image.Image, boxes: list[list[float]]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    colors = [
        (255, 80, 80),
        (80, 180, 255),
        (255, 180, 60),
        (80, 220, 120),
        (200, 120, 255),
    ]
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[idx % len(colors)]
        left = int(round(x1 * width))
        top = int(round(y1 * height))
        right = int(round(x2 * width))
        bottom = int(round(y2 * height))
        draw.rectangle((left, top, right, bottom), outline=color, width=4)
        label = f"{idx:02d}"
        draw.rectangle((left, max(0, top - 24), left + 32, top), fill=color)
        draw.text((left + 6, max(0, top - 22)), label, fill=(0, 0, 0))
    return canvas


def build_text_block(record: dict, boxes: list[list[float]]) -> str:
    lines = []
    lines.append(f"image: {record.get('image')}")
    lines.append(f"num_fovea: {len(boxes)}")
    for idx, box in enumerate(boxes):
        lines.append(f"box[{idx:02d}]: [{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]")
    lines.append("")
    for turn_idx, turn in enumerate(record.get("conversations", [])):
        role = str(turn.get("from", "unknown")).upper()
        value = str(turn.get("value", "")).strip()
        lines.append(f"[{turn_idx:02d}] {role}")
        lines.append(value)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_text_panel(text: str, width: int = 1200, padding: int = 24) -> Image.Image:
    font = ImageFont.load_default()
    wrapped_lines = []
    for line in text.splitlines():
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=90) or [""])
    line_height = 18
    height = padding * 2 + max(1, len(wrapped_lines)) * line_height
    image = Image.new("RGB", (width, height), color=(250, 248, 243))
    draw = ImageDraw.Draw(image)
    y = padding
    for line in wrapped_lines:
        draw.text((padding, y), line, fill=(20, 20, 20), font=font)
        y += line_height
    return image


def main() -> None:
    args = parse_args()
    records = load_records(args.data_path)
    if not (0 <= args.index < len(records)):
        raise IndexError(f"index out of range: {args.index}, total={len(records)}")

    record = records[args.index]
    image_name = record.get("image")
    if isinstance(image_name, list):
        if len(image_name) != 1:
            raise ValueError("Only one image per sample is supported.")
        image_name = image_name[0]
    image_path = Path(args.image_folder) / str(image_name)
    image = Image.open(image_path).convert("RGB")
    boxes = normalize_boxes(record.get("fovea_query_boxes"))

    out_dir = Path(args.output_dir) / f"sample_{args.index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image.save(out_dir / "image_raw.png")
    render_boxed_image(image, boxes).save(out_dir / "image_boxes.png")

    text = build_text_block(record, boxes)
    (out_dir / "conversation.txt").write_text(text, encoding="utf-8")
    render_text_panel(text).save(out_dir / "conversation.png")

    serializable = dict(record)
    serializable["fovea_query_boxes"] = boxes
    if hasattr(serializable.get("conversations"), "tolist"):
        serializable["conversations"] = serializable["conversations"].tolist()
    (out_dir / "sample.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out_dir)


if __name__ == "__main__":
    main()
