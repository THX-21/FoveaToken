#!/usr/bin/env python
"""Merge Qwen3.5-4B base weights with Fovea-specific weights from a checkpoint.

Rule:
  - Keys only in Qwen → keep (unused by checkpoint, harmless)
  - Keys only in checkpoint (fovea layers etc.) → keep from checkpoint
  - Keys in both with SAME shape → use Qwen (fresher base weights)
  - Keys in both with DIFFERENT shape (embed_tokens, lm_head resized for <fovea>) → use checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Merge Fovea checkpoint with base Qwen3.5-4B weights.")
    p.add_argument("--qwen_model", default="Qwen/Qwen3.5-4B",
                   help="Base Qwen model id or path.")
    p.add_argument("--fovea_ckpt", default="checkpoints/fovea-vgr-qwen/checkpoint-2300",
                   help="Fovea checkpoint directory.")
    p.add_argument("--output", default="checkpoints/fovea-vgr-qwen/checkpoint-2300-merged",
                   help="Output merged checkpoint directory.")
    return p.parse_args()


def _resolve_snapshot(model_id: str) -> str:
    """Resolve a HF model ID to a local directory (uses cache, no download)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, local_files_only=True)


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load full state dict from a directory (handles safetensors & pytorch)."""
    # If it looks like a HF model ID with no local path, resolve via cache
    if not os.path.isdir(path) and "/" in path and not path.startswith((".", "~")):
        path = _resolve_snapshot(path)

    sd_path = Path(path)
    sd = {}

    # Try safetensors index first, then single file, then pytorch
    index_files = list(sd_path.glob("model.safetensors.index.json"))
    if index_files:
        with open(index_files[0]) as f:
            idx = json.load(f)
        loaded = set()
        for shard in sorted(set(idx["weight_map"].values())):
            shard_path = sd_path / shard
            if shard_path.exists():
                from safetensors.torch import load_file
                part = load_file(str(shard_path))
                for k, v in part.items():
                    sd[k] = v
                loaded.add(shard)
                print(f"  Loaded shard: {shard} ({len(part)} keys)")
        return sd

    # Single safetensors file
    st_files = list(sd_path.glob("*.safetensors"))
    if st_files:
        from safetensors.torch import load_file
        for st in st_files:
            part = load_file(str(st))
            for k, v in part.items():
                sd[k] = v
            print(f"  Loaded: {st.name} ({len(part)} keys)")
        return sd

    # Pytorch bin
    bin_files = list(sd_path.glob("pytorch_model*.bin")) or list(sd_path.glob("model*.bin"))
    for bf in sorted(bin_files):
        part = torch.load(str(bf), map_location="cpu")
        for k, v in part.items():
            sd[k] = v
        print(f"  Loaded: {bf.name} ({len(part)} keys)")
    return sd



def main():
    args = parse_args()

    print(f"=== Loading Qwen base: {args.qwen_model}")
    qwen_sd = load_state_dict(args.qwen_model)
    print(f"  Total keys: {len(qwen_sd)}")

    print(f"\n=== Loading Fovea checkpoint: {args.fovea_ckpt}")
    ckpt_sd = load_state_dict(args.fovea_ckpt)
    print(f"  Total keys: {len(ckpt_sd)}")

    # Identify weight categories
    qwen_keys = set(qwen_sd.keys())
    ckpt_keys = set(ckpt_sd.keys())

    common = qwen_keys & ckpt_keys
    qwen_only = qwen_keys - ckpt_keys
    ckpt_only = ckpt_keys - qwen_keys

    print(f"\n=== Key analysis:")
    print(f"  Common (in both): {len(common)}")
    print(f"  Qwen-only: {len(qwen_only)}")
    print(f"  Ckpt-only (fovea + extras): {len(ckpt_only)}")

    # Show ckpt-only keys
    fovea_keys = [k for k in ckpt_only if "fovea" in k.lower()]
    other_keys = [k for k in ckpt_only if "fovea" not in k.lower()]
    print(f"\n  Fovea-specific keys: {len(fovea_keys)}")
    for k in sorted(fovea_keys):
        print(f"    {k}: {list(ckpt_sd[k].shape)}")
    if other_keys:
        print(f"\n  Other ckpt-only keys: {len(other_keys)}")
        for k in sorted(other_keys):
            print(f"    {k}: {list(ckpt_sd[k].shape)}")

    # --- Merge ---
    merged = {}

    # 1. Qwen-only keys (mtp layers etc.) → keep from Qwen
    for k in qwen_only:
        merged[k] = qwen_sd[k]

    # 2. Ckpt-only keys (fovea layers) → keep from ckpt
    #    But handle lm_head specially (see step 4)
    for k in ckpt_only:
        merged[k] = ckpt_sd[k]

    # 3. Common keys → use Qwen, unless shape differs (embed_tokens)
    shape_diff = []
    for k in common:
        if qwen_sd[k].shape != ckpt_sd[k].shape:
            # Will be handled in step 4
            shape_diff.append(k)
        else:
            merged[k] = qwen_sd[k]

    # 4. Special handling: embed_tokens and lm_head
    #    Rule: first 248077 tokens from Qwen, <fovea> (index 248077) from checkpoint
    # Find the embed_tokens key
    embed_key = None
    for k in common:
        if k.endswith("embed_tokens.weight"):
            embed_key = k
            break

    if embed_key is None:
        raise RuntimeError("Could not find embed_tokens key in common keys")

    fovea_token_id = 248077  # <fovea> token index in checkpoint
    pre_fovea = fovea_token_id  # tokens 0..248076 from Qwen, 248077 from ckpt

    print(f"\n=== Embedding merge ===")
    print(f"  embed_key: {embed_key}")
    print(f"  Qwen embed shape: {list(qwen_sd[embed_key].shape)}")
    print(f"  Ckpt embed shape: {list(ckpt_sd[embed_key].shape)}")
    print(f"  Pre-fovea tokens from Qwen: 0..{pre_fovea-1}")
    print(f"  <fovea> token from ckpt: {fovea_token_id}")

    # embed_tokens: Qwen[0:248077] + ckpt[248077:]
    merged[embed_key] = torch.cat([
        qwen_sd[embed_key][:pre_fovea],
        ckpt_sd[embed_key][pre_fovea:],
    ], dim=0)
    print(f"  Merged embed_tokens: {list(merged[embed_key].shape)}")

    # lm_head
    lm_key = None
    for k in ckpt_sd:
        if k == "lm_head.weight":
            lm_key = k
            break

    if lm_key is not None:
        # Qwen has no separate lm_head (tie_word_embeddings=true), so construct from embed_tokens
        merged[lm_key] = torch.cat([
            qwen_sd[embed_key][:pre_fovea],
            ckpt_sd[lm_key][pre_fovea:],
        ], dim=0)
        print(f"  Merged lm_head: {list(merged[lm_key].shape)} (constructed from Qwen embed + ckpt lm_head)")
    else:
        print(f"  No separate lm_head in ckpt (tied).")

    if shape_diff:
        print(f"\n  Shape-mismatch handled: {shape_diff}")

    print(f"\n=== Merged state dict: {len(merged)} keys")

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "model.safetensors"
    from safetensors.torch import save_file
    save_file(merged, str(save_path))
    print(f"\n=== Saved merged weights to: {save_path}")

    # Copy config files from checkpoint
    for fname in ["config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "processor_config.json",
                   "chat_template.jinja", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "preprocessor_config.json"]:
        src = Path(args.fovea_ckpt) / fname
        if src.exists():
            shutil.copy2(str(src), str(out / fname))
    print("=== Copied config/tokenizer files from checkpoint")


if __name__ == "__main__":
    main()
