"""Local IBQ visual tokenizer with a deterministic crop cache.

This module converts an image crop into discrete visual code ids. It never
downloads code or weights; callers pass explicit local paths, or use the
project-local SEED-Voken checkout under `src/fovea_token/models/Open-MAGVIT2`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


FOVEA_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IBQ_REPO = str(FOVEA_PACKAGE_ROOT / "models" / "Open-MAGVIT2")
DEFAULT_IBQ_CHECKPOINT = str(Path(DEFAULT_IBQ_REPO) / "IBQ_pretrain_16384.ckpt")
DEFAULT_IBQ_CONFIG = str(Path(DEFAULT_IBQ_REPO) / "configs" / "IBQ" / "gpu" / "pretrain_ibqgan_16384.yaml")
_VENDOR_SRC_ROOT = DEFAULT_IBQ_REPO
if _VENDOR_SRC_ROOT not in sys.path:
    sys.path.insert(0, _VENDOR_SRC_ROOT)

from src.IBQ.models.ibqgan import IBQ


@dataclass(frozen=True)
class IBQConfig:
    repo: str = DEFAULT_IBQ_REPO
    checkpoint: str = DEFAULT_IBQ_CHECKPOINT
    config: str = DEFAULT_IBQ_CONFIG
    cache_dir: str | None = None
    codebook_size: int = 16384
    max_latent_tokens: int = 64
    downsample_factor: int = 16
    device: str | None = None


class IBQCodec:
    """Strict adapter around the local IBQ 16384-code visual tokenizer."""

    def __init__(self, config: IBQConfig) -> None:
        self.config = config
        self.repo = Path(config.repo).expanduser().resolve()
        self.checkpoint = Path(config.checkpoint).expanduser().resolve()
        self.config_path = Path(config.config).expanduser().resolve()
        if not self.repo.exists():
            raise FileNotFoundError(f"SEED-Voken repo does not exist: {self.repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"IBQ checkpoint does not exist: {self.checkpoint}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"IBQ config does not exist: {self.config_path}")
        self.cache_dir = Path(config.cache_dir).expanduser().resolve() if config.cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: Any | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str = DEFAULT_IBQ_REPO,
        checkpoint: str = DEFAULT_IBQ_CHECKPOINT,
        config: str = DEFAULT_IBQ_CONFIG,
        cache_dir: str | None = None,
        device: str | None = None,
    ) -> "IBQCodec":
        return cls(IBQConfig(repo=repo, checkpoint=checkpoint, config=config, cache_dir=cache_dir, device=device))

    def _cache_key(self, image_path: str, box: tuple[float, float, float, float]) -> str:
        image_stat = os.stat(image_path)
        ckpt_stat = os.stat(self.checkpoint)
        payload = {
            "path": str(Path(image_path).resolve()),
            "mtime": image_stat.st_mtime_ns,
            "size": image_stat.st_size,
            "box": [round(float(v), 6) for v in box],
            "ckpt": str(self.checkpoint),
            "ckpt_mtime": ckpt_stat.st_mtime_ns,
            "ckpt_size": ckpt_stat.st_size,
            "max": self.config.max_latent_tokens,
            "stride": self.config.downsample_factor,
        }
        config_stat = os.stat(self.config_path)
        payload.update(
            {
                "config": str(self.config_path),
                "config_mtime": config_stat.st_mtime_ns,
                "config_size": config_stat.st_size,
            }
        )
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
            target_w = max(stride, ((crop.width + stride - 1) // stride) * stride)
            target_h = max(stride, ((crop.height + stride - 1) // stride) * stride)
            return crop if (target_w, target_h) == crop.size else crop.resize((target_w, target_h), Image.Resampling.BICUBIC)
        scale = (max_tokens / float(latent_h * latent_w)) ** 0.5
        target_w = max(stride, int(crop.width * scale))
        target_h = max(stride, int(crop.height * scale))
        target_w = max(stride, (target_w // stride) * stride)
        target_h = max(stride, (target_h // stride) * stride)
        return crop.resize((target_w, target_h), Image.Resampling.BICUBIC)

    def _load_model(self):
        if self._model is not None:
            return self._model
        config = OmegaConf.load(self.config_path)
        init_args = OmegaConf.to_container(config.model.init_args, resolve=True)
        init_args["lossconfig"] = {"target": "torch.nn.Identity"}
        model = IBQ(**init_args)
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
        if getattr(model, "use_ema", False):
            with model.ema_scope():
                output = model.encode(images)
        else:
            output = model.encode(images)
        if not isinstance(output, tuple) or len(output) < 3:
            raise RuntimeError("Official IBQ.encode must return quant, loss, (_, _, indices).")
        info = output[2]
        if not isinstance(info, tuple) or len(info) < 3:
            raise RuntimeError("Official IBQ.encode must return quantization info as (_, _, indices).")
        codes = info[2]
        if torch.is_tensor(codes):
            codes = codes.detach().cpu().reshape(-1).tolist()
        codes = [int(v) for v in codes]
        if len(codes) > self.config.max_latent_tokens:
            raise ValueError(f"IBQ returned {len(codes)} codes, expected <= {self.config.max_latent_tokens}.")
        if any(code < 0 or code >= self.config.codebook_size for code in codes):
            raise ValueError("IBQ returned a code outside the configured 16384-token codebook.")
        if not codes:
            raise ValueError("IBQ returned an empty visual-code sequence.")
        self._save_cache(key, codes)
        return codes


__all__ = [
    "DEFAULT_IBQ_CHECKPOINT",
    "DEFAULT_IBQ_CONFIG",
    "DEFAULT_IBQ_REPO",
    "IBQCodec",
    "IBQConfig",
]
