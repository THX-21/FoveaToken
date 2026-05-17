"""Local Open-MAGVIT2 visual codec with a small deterministic cache.

This module converts an image crop into discrete visual code ids. It never
downloads code or weights; callers pass explicit local paths, or use the
project-local defaults under `models/Open-MAGVIT2`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_MAGVIT2_REPO = "models/Open-MAGVIT2"
DEFAULT_MAGVIT2_CHECKPOINT = "models/Open-MAGVIT2/tokenizer_16384.pt"
DEFAULT_MAGVIT2_CONFIG = "models/Open-MAGVIT2/configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_16384.yaml"


@dataclass(frozen=True)
class OpenMAGVIT2Config:
    repo: str = DEFAULT_MAGVIT2_REPO
    checkpoint: str = DEFAULT_MAGVIT2_CHECKPOINT
    config: str = DEFAULT_MAGVIT2_CONFIG
    cache_dir: str | None = None
    codebook_size: int = 16384
    max_latent_tokens: int = 64
    downsample_factor: int = 16
    device: str | None = None


class OpenMAGVIT2Codec:
    """Strict adapter around a local Open-MAGVIT2 16384-code visual tokenizer."""

    def __init__(self, config: OpenMAGVIT2Config) -> None:
        self.config = config
        self.repo = Path(config.repo).expanduser().resolve()
        self.checkpoint = Path(config.checkpoint).expanduser().resolve()
        self.config_path = Path(config.config).expanduser().resolve()
        if not self.repo.exists():
            raise FileNotFoundError(f"Open-MAGVIT2 repo does not exist: {self.repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Open-MAGVIT2 checkpoint does not exist: {self.checkpoint}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Open-MAGVIT2 config does not exist: {self.config_path}")
        self.cache_dir = Path(config.cache_dir).expanduser().resolve() if config.cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: Any | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str = DEFAULT_MAGVIT2_REPO,
        checkpoint: str = DEFAULT_MAGVIT2_CHECKPOINT,
        config: str = DEFAULT_MAGVIT2_CONFIG,
        cache_dir: str | None = None,
        device: str | None = None,
    ) -> "OpenMAGVIT2Codec":
        return cls(OpenMAGVIT2Config(repo=repo, checkpoint=checkpoint, config=config, cache_dir=cache_dir, device=device))

    def _cache_key(self, image_path: str, box: tuple[float, float, float, float]) -> str:
        image_stat = os.stat(image_path)
        ckpt_stat = os.stat(self.checkpoint)
        config_stat = os.stat(self.config_path)
        payload = {
            "path": str(Path(image_path).resolve()),
            "mtime": image_stat.st_mtime_ns,
            "size": image_stat.st_size,
            "box": [round(float(v), 6) for v in box],
            "ckpt": str(self.checkpoint),
            "ckpt_mtime": ckpt_stat.st_mtime_ns,
            "ckpt_size": ckpt_stat.st_size,
            "config": str(self.config_path),
            "config_mtime": config_stat.st_mtime_ns,
            "config_size": config_stat.st_size,
            "max": self.config.max_latent_tokens,
            "stride": self.config.downsample_factor,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_cache(self, key: str) -> list[int] | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return [int(v) for v in json.load(handle)["codes"]]

    def _save_cache(self, key: str, codes: list[int]) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"codes": [int(v) for v in codes]}, handle)
        os.replace(tmp, path)

    def _crop_and_resize(self, image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
        width, height = image.size
        x1, y1, x2, y2 = box
        left = max(0, min(width - 1, round(x1 * width)))
        top = max(0, min(height - 1, round(y1 * height)))
        right = max(left + 1, min(width, round(x2 * width)))
        bottom = max(top + 1, min(height, round(y2 * height)))
        crop = image.crop((left, top, right, bottom)).convert("RGB")
        stride = max(1, int(self.config.downsample_factor))
        max_tokens = max(1, int(self.config.max_latent_tokens))
        latent_h = (crop.height + stride - 1) // stride
        latent_w = (crop.width + stride - 1) // stride
        if latent_h * latent_w <= max_tokens:
            return crop
        scale = (max_tokens / float(latent_h * latent_w)) ** 0.5
        target_w = max(stride, int(crop.width * scale))
        target_h = max(stride, int(crop.height * scale))
        return crop.resize((target_w, target_h), Image.Resampling.BICUBIC)

    def _load_model(self):
        if self._model is not None:
            return self._model
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError("Open-MAGVIT2 codec requires omegaconf to load the official config.") from exc

        config = OmegaConf.load(self.config_path)
        class_path = str(config.model.class_path)
        module_name, class_name = class_path.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module_name), class_name)
        model = model_cls(**config.model.init_args)
        state = torch.load(self.checkpoint, map_location="cpu")
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        self._model = model
        return model

    def _to_model_tensor(self, image: Image.Image) -> torch.Tensor:
        array = torch.from_numpy(np.array(image).astype("float32"))
        tensor = array.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 127.5 - 1.0
        return tensor.to(device=self.device)

    @torch.no_grad()
    def encode_crop(self, image_path: str, box: tuple[float, float, float, float]) -> list[int]:
        key = self._cache_key(image_path, box)
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        image = Image.open(image_path).convert("RGB")
        crop = self._crop_and_resize(image, box)
        model = self._load_model()
        images = self._to_model_tensor(crop)
        output = model.encode(images)
        if not isinstance(output, tuple) or len(output) < 3:
            raise RuntimeError("Official Open-MAGVIT2 VQModel.encode must return quant, loss, indices, ...")
        codes = output[2]
        if torch.is_tensor(codes):
            codes = codes.detach().cpu().reshape(-1).tolist()
        codes = [int(v) for v in codes]
        if len(codes) > self.config.max_latent_tokens:
            raise ValueError(f"Open-MAGVIT2 returned {len(codes)} codes, expected <= {self.config.max_latent_tokens}.")
        if any(code < 0 or code >= self.config.codebook_size for code in codes):
            raise ValueError("Open-MAGVIT2 returned a code outside the configured 16384-token codebook.")
        if not codes:
            raise ValueError("Open-MAGVIT2 returned an empty visual-code sequence.")
        self._save_cache(key, codes)
        return codes


__all__ = [
    "DEFAULT_MAGVIT2_CHECKPOINT",
    "DEFAULT_MAGVIT2_CONFIG",
    "DEFAULT_MAGVIT2_REPO",
    "OpenMAGVIT2Codec",
    "OpenMAGVIT2Config",
]
