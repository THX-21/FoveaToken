#!/usr/bin/env python
"""Build fixed-fovea Visual Genome grounding parquets.

The script converts Visual Genome region descriptions into samples where the
assistant inserts `<fovea>` inside reasoning and supervises the matching box.
Small `--max_samples` dry runs are supported.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FOVEA_TOKEN = "<fovea>"


def _load_json_or_zip(path: str | os.PathLike[str]) -> Any:
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".json")]
            if len(names) != 1:
                raise ValueError(f"Expected one JSON file inside {path}, found {names}")
            with archive.open(names[0]) as handle:
                return json.load(handle)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _iter_vg_regions(region_data: Any) -> Iterable[dict[str, Any]]:
    for image_item in region_data:
        image_id = image_item.get("id", image_item.get("image_id"))
        for region in image_item.get("regions", []):
            item = dict(region)
            item.setdefault("image_id", image_id)
            yield item


def _build_vg_image_index(image_root: Path) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for path in image_root.rglob("*.jpg"):
        try:
            index[int(path.stem)] = path
        except ValueError:
            continue
    return index


def build_visual_genome_grounding(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.vg_image_root).expanduser().resolve()
    region_data = _load_json_or_zip(args.vg_region_descriptions)
    image_data = _load_json_or_zip(args.vg_image_data) if args.vg_image_data else []
    size_by_id = {
        int(item["image_id"]): (float(item["width"]), float(item["height"]))
        for item in image_data
        if item.get("image_id") is not None and item.get("width") and item.get("height")
    }
    image_index = _build_vg_image_index(image_root)
    records: list[dict[str, Any]] = []

    for region in tqdm(list(_iter_vg_regions(region_data)), desc="visual-genome"):
        if args.max_samples is not None and len(records) >= args.max_samples:
            break
        phrase = str(region.get("phrase", "")).strip()
        image_id = region.get("image_id")
        if not phrase or image_id is None:
            continue
        image_id = int(image_id)
        image_path = image_index.get(image_id)
        if image_path is None:
            continue
        width, height = size_by_id.get(image_id, Image.open(image_path).size)
        x = float(region.get("x", 0.0))
        y = float(region.get("y", 0.0))
        w = float(region.get("width", 0.0))
        h = float(region.get("height", 0.0))
        if w <= 0 or h <= 0 or width <= 0 or height <= 0:
            continue
        box = (
            max(0.0, min(1.0, x / width)),
            max(0.0, min(1.0, y / height)),
            max(0.0, min(1.0, (x + w) / width)),
            max(0.0, min(1.0, (y + h) / height)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        rel_image = str(image_path.relative_to(image_root))
        box_text = f"[{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]"
        records.append(
            {
                "id": f"vg_grounding_{image_id}_{region.get('region_id', len(records))}",
                "image": rel_image,
                "source": "jn12/VisualGenome",
                "fovea_preprocessed": True,
                "fovea_query_boxes": [list(box)],
                "fovea_supervised_substrings": [FOVEA_TOKEN, f" {box_text}"],
                "conversations": [
                    {
                        "from": "human",
                        "value": f'<image>\nDescribe where the caption "{phrase}" corresponds in the image.',
                    },
                    {
                        "from": "gpt",
                        "value": f'The caption "{phrase}" {FOVEA_TOKEN} corresponds to the image region {box_text}.',
                    },
                ],
            }
        )

    pd.DataFrame(records).to_parquet(output, index=False)
    print(f"[visual-genome] wrote {output} records={len(records)} image_root={image_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess fixed-fovea Visual Genome grounding data.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--vg_region_descriptions", default="data/visual_genome/region_descriptions.json.zip")
    parser.add_argument("--vg_image_data", default="data/visual_genome/image_data.json.zip")
    parser.add_argument("--vg_image_root", default="data/visual_genome/images")
    args = parser.parse_args()

    missing = [name for name in ("vg_region_descriptions", "vg_image_root") if not getattr(args, name)]
    if missing:
        raise ValueError(f"Visual Genome preprocessing requires: {', '.join(missing)}")
    build_visual_genome_grounding(args)


if __name__ == "__main__":
    main()
