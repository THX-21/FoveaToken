#!/usr/bin/env python
"""Prepare VLM-R3 SFT data for Fovea training."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any

FOVEA_TOOL_CALL = '{"fovea"}'
BBOX_RE = re.compile(r'\{\s*"bbox_2d"\s*:\s*\[[^\]]+\]\s*\}')
FOVEA_PROMPT = (
    "\nYou need to first think about the reasoning process in your mind, and then provide the answer. "
    "When thinking, you should call the \"fovea\" tool (format: {\"fovea\"}) to focus on key areas in the image. "
    "The reasoning process and the answer are included in the <think> </think> and <answer> </answer> tags respectively."
)


def validate_box(box: Any) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"Invalid VLM-R3 box: {box!r}")
    x1, y1, x2, y2 = (float(value) for value in box)
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid normalized VLM-R3 box: {box!r}")
    return [x1, y1, x2, y2]


def archive_image_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for name in archive.namelist():
        if "/dataset/imgs_" not in name or name.endswith("/"):
            continue
        image_name = Path(name).name
        if image_name in members:
            existing = archive.getinfo(members[image_name])
            candidate = archive.getinfo(name)
            if (existing.CRC, existing.file_size) != (candidate.CRC, candidate.file_size):
                raise ValueError(f"Ambiguous VLM-R3 source image name: {image_name}")
            continue
        members[image_name] = name
    if not members:
        raise FileNotFoundError("No source images found under dataset/imgs_* in the archive.")
    return members


def build_record(item: dict[str, Any], image_members: dict[str, str]) -> dict[str, Any]:
    image_name = Path(str(item["image_path"])).name
    if image_name not in image_members:
        raise FileNotFoundError(f"Image listed in JSON is missing from archive: {image_name}")

    boxes = [validate_box(box) for box in item["bbox_list"]]
    response = str(item["model_response"]).replace("\\n", "\n")
    response, replacements = BBOX_RE.subn(FOVEA_TOOL_CALL, response)
    if replacements != len(boxes):
        raise ValueError(
            f"sample_id={item.get('sample_id')} has {len(boxes)} boxes but {replacements} bbox_2d calls."
        )

    return {
        "image": f"images/{image_name}",
        "conversations": [
            {"from": "human", "value": f"<image>\n{str(item['question']).strip()}{FOVEA_PROMPT}"},
            {"from": "gpt", "value": response},
        ],
        "fovea_query_boxes": boxes,
        "source": item.get("source"),
        "sample_id": item.get("sample_id"),
    }


def main() -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("VLM-R3 preprocessing requires pandas and pyarrow.") from exc

    parser = argparse.ArgumentParser(description="Convert VLM-R3 `dataset.zip` into Fovea training parquet.")
    parser.add_argument("--input", default="data/VLM-R3-data/dataset.zip", help="Path to VLM-R3 dataset.zip.")
    parser.add_argument("--output", default="data/VLM-R3-data/preprocessed/vlir_sft_12k.parquet")
    parser.add_argument("--image_folder", default="data/VLM-R3-data/preprocessed", help="Output root for extracted source images.")
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    image_folder = Path(args.image_folder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = image_folder / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path) as archive:
        json_members = [name for name in archive.namelist() if name.endswith("/vlir_sft_12k.json")]
        if len(json_members) != 1:
            raise FileNotFoundError("Expected exactly one vlir_sft_12k.json in the archive.")
        with archive.open(json_members[0]) as stream:
            records = json.load(stream)
        image_members = archive_image_members(archive)

        processed = []
        for index, item in enumerate(records, start=1):
            record = build_record(item, image_members)
            image_path = image_dir / Path(record["image"]).name
            if not image_path.exists():
                with archive.open(image_members[image_path.name]) as source, image_path.open("wb") as destination:
                    destination.write(source.read())
            processed.append(record)
            if args.log_every > 0 and index % args.log_every == 0:
                print(f"[preprocess] {index}/{len(records)}", flush=True)

    pd.DataFrame(processed).to_parquet(output_path, index=False)
    print(f"[preprocess] wrote {output_path} records={len(processed)} images={len(image_members)}", flush=True)


if __name__ == "__main__":
    main()
