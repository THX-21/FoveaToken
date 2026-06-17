#!/usr/bin/env python
"""Offline VGR fovea-trigger preprocessing.

This rewrites VGR parquet conversations from `<SOT>box<EOT><image>` into
Qwen's built-in `<|vision_start|>` trigger and stores the matched boxes in
`fovea_query_boxes` for training.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOVEA_TOKEN = "<|vision_start|>"
DEFAULT_IMAGE_TOKEN = "<image>"
SOT_EOT_IMAGE_RE = re.compile(r"<SOT>\s*(\[[^\]]+\])\s*<EOT>\s*<image>")
ORPHAN_VGR_TAG_RE = re.compile(r"<SOT>|<EOT>")


def parse_vgr_box(box_text: str) -> tuple[float, float, float, float]:
    values = json.loads(box_text)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"Invalid VGR box: {box_text}")
    x1, y1, x2, y2 = [float(v) for v in values]
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid non-positive VGR box: {box_text}")
    return x1, y1, x2, y2


def replace_vgr_regions_with_fovea(text: str) -> tuple[str, list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    saw_vgr_markup = bool(SOT_EOT_IMAGE_RE.search(text) or ORPHAN_VGR_TAG_RE.search(text))

    def replace(match: re.Match) -> str:
        box = parse_vgr_box(match.group(1))
        queries.append({"box": box})
        return FOVEA_TOKEN

    cleaned_text = ORPHAN_VGR_TAG_RE.sub("", SOT_EOT_IMAGE_RE.sub(replace, text))
    if saw_vgr_markup:
        cleaned_text = cleaned_text.replace(DEFAULT_IMAGE_TOKEN, "")
    return cleaned_text, queries


def parquet_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    paths = [path / "vgr_shortcot.parquet", path / "vgr_longcot.parquet"]
    paths = [item for item in paths if item.exists()]
    if not paths:
        raise FileNotFoundError(f"No VGR parquet files found under {path}.")
    return paths


def preprocess_record(record: dict[str, Any], image_folder: Path) -> dict[str, Any]:
    image_field = record.get("image")
    if image_field is None:
        raise ValueError("VGR record is missing image.")
    image_names = image_field if isinstance(image_field, list) else [image_field]
    if len(image_names) != 1:
        raise ValueError("VGR preprocessing expects one image per sample.")

    conversations = copy.deepcopy(record["conversations"])
    query_boxes: list[list[float]] = []
    for sentence in conversations:
        if sentence.get("from") not in {"gpt", "assistant"}:
            continue
        value, parsed_queries = replace_vgr_regions_with_fovea(sentence["value"])
        sentence["value"] = value
        query_boxes.extend([list(query["box"]) for query in parsed_queries])

    record = dict(record)
    record["conversations"] = conversations
    record["fovea_query_boxes"] = query_boxes
    record["fovea_preprocessed"] = True
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline preprocess VGR parquets into fixed-fovea training parquets.")
    parser.add_argument("--input", default="data/vgr/data", help="Input VGR parquet file or directory.")
    parser.add_argument("--output", default="data/vgr/preprocessed", help="Output parquet file or directory.")
    parser.add_argument("--image_folder", default="data/vgr/llava_next_raw_format", help="Folder containing VGR raw images.")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    image_folder = Path(args.image_folder)

    inputs = parquet_paths(input_path)
    output_is_file = output_path.suffix == ".parquet"
    if len(inputs) > 1 and output_is_file:
        raise ValueError("--output must be a directory when --input is a directory with multiple parquets.")
    if not output_is_file:
        output_path.mkdir(parents=True, exist_ok=True)

    for src in inputs:
        dst = output_path if output_is_file else output_path / src.name
        frame = pd.read_parquet(src)
        records = frame.to_dict("records")
        processed = []
        total_queries = 0
        print(f"[preprocess] {src} -> {dst} records={len(records)}", flush=True)
        for idx, record in enumerate(records):
            item = preprocess_record(record, image_folder)
            total_queries += len(item["fovea_query_boxes"])
            processed.append(item)
            if args.log_every > 0 and (idx + 1) % args.log_every == 0:
                print(f"[preprocess] {src.name}: {idx + 1}/{len(records)} queries={total_queries}", flush=True)
        pd.DataFrame(processed).to_parquet(dst, index=False)
        print(f"[preprocess] wrote {dst} records={len(processed)} queries={total_queries}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
