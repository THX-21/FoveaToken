#!/usr/bin/env python
"""Merge frozen Qwen base weights with Fovea trainable weights.

The low-contamination training path freezes Qwen and trains only `fovea_*`
parameters. This utility refreshes unchanged Qwen tensors from a base checkpoint
while preserving Fovea-specific tensors and metadata from the Fovea checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a Fovea checkpoint with frozen Qwen base weights.")
    parser.add_argument("--qwen_model", default="Qwen/Qwen3.5-4B", help="Base Qwen model id or local path.")
    parser.add_argument("--fovea_ckpt", default="checkpoints/fovea-vgr-qwen/checkpoint-2300")
    parser.add_argument("--output", default="checkpoints/fovea-vgr-qwen/checkpoint-2300-merged")
    return parser.parse_args()


def _resolve_snapshot(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_files_only=True)


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if not os.path.isdir(path) and "/" in path and not path.startswith((".", "~")):
        path = _resolve_snapshot(path)

    root = Path(path)
    state: dict[str, torch.Tensor] = {}
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for shard in sorted(set(index["weight_map"].values())):
            from safetensors.torch import load_file

            state.update(load_file(str(root / shard)))
        return state

    safetensors_files = sorted(root.glob("*.safetensors"))
    if safetensors_files:
        from safetensors.torch import load_file

        for item in safetensors_files:
            state.update(load_file(str(item)))
        return state

    for item in sorted(root.glob("pytorch_model*.bin")) + sorted(root.glob("model*.bin")):
        state.update(torch.load(str(item), map_location="cpu"))
    return state


def is_fovea_key(key: str) -> bool:
    return key.startswith("fovea_") or ".fovea_" in key


def merge_state_dicts(qwen_state: dict[str, torch.Tensor], fovea_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for key in sorted(set(qwen_state) | set(fovea_state)):
        if key not in qwen_state:
            merged[key] = fovea_state[key]
            continue
        if key not in fovea_state:
            merged[key] = qwen_state[key]
            continue
        if is_fovea_key(key) or qwen_state[key].shape != fovea_state[key].shape:
            merged[key] = fovea_state[key]
        else:
            merged[key] = qwen_state[key]
    return merged


def copy_metadata(src_root: Path, dst_root: Path) -> None:
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "preprocessor_config.json",
    ]
    for name in names:
        src = src_root / name
        if src.exists():
            shutil.copy2(str(src), str(dst_root / name))


def main() -> None:
    args = parse_args()
    print(f"Loading Qwen base: {args.qwen_model}")
    qwen_state = load_state_dict(args.qwen_model)
    print(f"Loading Fovea checkpoint: {args.fovea_ckpt}")
    fovea_state = load_state_dict(args.fovea_ckpt)

    merged = merge_state_dicts(qwen_state, fovea_state)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(merged, str(output / "model.safetensors"))
    copy_metadata(Path(args.fovea_ckpt), output)
    print(f"Saved merged checkpoint to {output}")


if __name__ == "__main__":
    main()
