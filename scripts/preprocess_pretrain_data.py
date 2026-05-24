#!/usr/bin/env python
"""Build Stage A/B visual-token pretraining parquets.

Stage A converts image-caption rows into image-conditioned code LM samples:
<image> + caption -> <vq> <vis_i> ... </vq>

Stage B converts Visual Genome region descriptions into VGR-style replay
samples where the assistant inserts visual-query tokens inside reasoning.

The script supports small `--max_samples` dry runs and does not require
downloading full Hugging Face datasets locally.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IBQ_REPO = str(PROJECT_ROOT / "src" / "fovea_token" / "models" / "Open-MAGVIT2")
DEFAULT_IBQ_CHECKPOINT = str(Path(DEFAULT_IBQ_REPO) / "IBQ_pretrain_16384.ckpt")
DEFAULT_IBQ_CONFIG = str(Path(DEFAULT_IBQ_REPO) / "configs" / "IBQ" / "gpu" / "pretrain_ibqgan_16384.yaml")

VQ_START_TOKEN = "<vq>"
VQ_END_TOKEN = "</vq>"
REPLAY_TOKEN = "<|replay_pad|>"


def vis_token(code_id: int) -> str:
    return f"<vis_{int(code_id)}>"


def _vq_text(codes: list[int], *, with_replay: bool) -> str:
    code_tokens = " ".join(vis_token(code) for code in codes)
    replay = "".join(REPLAY_TOKEN for _ in codes) if with_replay else ""
    return f"{VQ_START_TOKEN} {code_tokens} {VQ_END_TOKEN}{replay}"


def _save_temp_image(image: Any, directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.jpg"
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path, format="JPEG", quality=95)
        return path
    if isinstance(image, dict) and image.get("bytes") is not None:
        path.write_bytes(image["bytes"])
        return path
    if isinstance(image, dict) and image.get("path"):
        source = Path(image["path"])
        if source.exists():
            return source
    if isinstance(image, (str, os.PathLike)):
        source = Path(image)
        if source.exists():
            return source
    raise ValueError(f"Unsupported image payload type: {type(image)!r}")


def _load_ibq_codec(args: argparse.Namespace, max_latent_tokens: int | None):
    ibq_module_path = PROJECT_ROOT / "src" / "fovea_token" / "tokenizers" / "tokenization_ibq.py"
    spec = importlib.util.spec_from_file_location("fovea_token_tokenization_ibq", ibq_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load IBQ codec module from {ibq_module_path}")
    ibq_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ibq_module
    spec.loader.exec_module(ibq_module)
    return ibq_module.IBQCodec.from_paths(
        repo=args.ibq_repo,
        checkpoint=args.ibq_checkpoint,
        config=args.ibq_config,
        device=args.device,
        max_latent_tokens=max_latent_tokens,
    )


def build_coco_stage_a(args: argparse.Namespace, codec: Any) -> None:
    from datasets import load_dataset

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_dir = output.parent / "_stage_a_source_images"
    records: list[dict[str, Any]] = []

    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=True,
    )
    for row_idx, row in enumerate(dataset):
        if args.max_samples is not None and len(records) >= args.max_samples:
            break
        captions = row.get("sentences_raw") or row.get("captions") or row.get("caption")
        if isinstance(captions, str):
            captions = [captions]
        if not captions:
            continue

        image_path = _save_temp_image(row.get("image"), source_dir, f"coco_{row.get('id', row_idx)}")
        rel_image = str(image_path.relative_to(output.parent))
        codes = codec.encode_crop(str(image_path), (0.0, 0.0, 1.0, 1.0))
        answer = _vq_text(codes, with_replay=False)
        for caption_idx, caption in enumerate(captions[: args.captions_per_image]):
            records.append(
                {
                    "id": f"coco_stage_a_{row.get('id', row_idx)}_{caption_idx}",
                    "image": rel_image,
                    "source": args.dataset_name,
                    "fovea_task": "visual_code_lm",
                    "fovea_preprocessed": True,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\nThis image shows: {caption}\nGenerate visual tokens for this image.",
                        },
                        {"from": "gpt", "value": answer},
                    ],
                }
            )
            if args.max_samples is not None and len(records) >= args.max_samples:
                break

    pd.DataFrame(records).to_parquet(output, index=False)
    print(f"[stage-a] wrote {output} records={len(records)}", flush=True)


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


def build_visual_genome_stage_b(args: argparse.Namespace, codec: Any) -> None:
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

    for region in _iter_vg_regions(region_data):
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
        codes = codec.encode_crop(str(image_path), box)
        rel_image = str(image_path.relative_to(image_root))
        visual_query = _vq_text(codes, with_replay=True)
        box_text = f"[{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]"
        records.append(
            {
                "id": f"vg_stage_b_{image_id}_{region.get('region_id', len(records))}",
                "image": rel_image,
                "source": "jn12/VisualGenome",
                "fovea_preprocessed": True,
                "fovea_query_boxes": [list(box)],
                "fovea_supervised_substrings": [visual_query, box_text],
                "conversations": [
                    {
                        "from": "human",
                        "value": f'<image>\nDescribe where the caption "{phrase}" corresponds in the image.',
                    },
                    {
                        "from": "gpt",
                        "value": f'The caption "{phrase}" {visual_query} corresponds to the image region {box_text}.',
                    },
                ],
            }
        )

    pd.DataFrame(records).to_parquet(output, index=False)
    print(f"[stage-b] wrote {output} records={len(records)} image_root={image_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Stage A/B visual-token datasets.")
    parser.add_argument("--stage", choices=["coco_a", "vg_b"], required=True)
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--ibq_repo", default=DEFAULT_IBQ_REPO)
    parser.add_argument("--ibq_checkpoint", default=DEFAULT_IBQ_CHECKPOINT)
    parser.add_argument("--ibq_config", default=DEFAULT_IBQ_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--stage_a_max_visual_tokens",
        type=int,
        default=0,
        help="Maximum IBQ tokens for Stage A crops. Use 0 for no token limit.",
    )
    parser.add_argument(
        "--stage_b_max_visual_tokens",
        type=int,
        default=256,
        help="Maximum IBQ tokens for Stage B region crops.",
    )

    parser.add_argument("--dataset_name", default="Multimodal-Fatima/COCO_captions_train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true", help="Stream HF rows instead of downloading all shards.")
    parser.add_argument("--captions_per_image", type=int, default=1)

    parser.add_argument("--vg_region_descriptions", default="data/visual_genome/region_descriptions.json.zip")
    parser.add_argument("--vg_image_data", default="data/visual_genome/image_data.json.zip")
    parser.add_argument("--vg_image_root", default="data/visual_genome/images")
    args = parser.parse_args()

    if args.stage == "coco_a":
        max_tokens = None if int(args.stage_a_max_visual_tokens) <= 0 else int(args.stage_a_max_visual_tokens)
        codec = _load_ibq_codec(args, max_tokens)
        build_coco_stage_a(args, codec)
    else:
        missing = [name for name in ("vg_region_descriptions", "vg_image_root") if not getattr(args, name)]
        if missing:
            raise ValueError(f"Stage vg_b requires: {', '.join(missing)}")
        codec = _load_ibq_codec(args, int(args.stage_b_max_visual_tokens))
        build_visual_genome_stage_b(args, codec)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
