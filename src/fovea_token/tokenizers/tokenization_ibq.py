"""Local IBQ visual tokenizer with a deterministic crop cache.

This module converts an image crop into discrete visual code ids. It never
downloads code or weights; callers pass explicit local paths, or use the
project-local SEED-Voken checkout under `src/fovea_token/models/Open-MAGVIT2`.
"""

from __future__ import annotations

import inspect
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
    codebook_size: int = 16384
    max_latent_tokens: int | None = 64
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
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: Any | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str = DEFAULT_IBQ_REPO,
        checkpoint: str = DEFAULT_IBQ_CHECKPOINT,
        config: str = DEFAULT_IBQ_CONFIG,
        device: str | None = None,
        max_latent_tokens: int | None = 64,
    ) -> "IBQCodec":
        return cls(IBQConfig(repo=repo, checkpoint=checkpoint, config=config, device=device, max_latent_tokens=max_latent_tokens))

    def _crop_and_resize(self, image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
        width, height = image.size
        x1, y1, x2, y2 = box
        left = max(0, min(width - 1, round(x1 * width)))
        top = max(0, min(height - 1, round(y1 * height)))
        right = max(left + 1, min(width, round(x2 * width)))
        bottom = max(top + 1, min(height, round(y2 * height)))
        crop = image.crop((left, top, right, bottom)).convert("RGB")
        stride = max(1, int(self.config.downsample_factor))
        max_latent_tokens = self.config.max_latent_tokens
        if max_latent_tokens is None or int(max_latent_tokens) <= 0:
            target_w = max(stride, ((crop.width + stride - 1) // stride) * stride)
            target_h = max(stride, ((crop.height + stride - 1) // stride) * stride)
            return crop if (target_w, target_h) == crop.size else crop.resize((target_w, target_h), Image.Resampling.BICUBIC)
        max_tokens = max(1, int(max_latent_tokens))
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
        valid_params = set(inspect.signature(IBQ.__init__).parameters)
        valid_params.discard("self")
        init_args = {key: value for key, value in init_args.items() if key in valid_params}
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

    def _load_image(self, source: str | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        return Image.open(source).convert("RGB")

    def _prep_crop(self, source: str | Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
        image = self._load_image(source)
        return self._crop_and_resize(image, box)

    @torch.no_grad()
    def encode_crop(self, image_path: str, box: tuple[float, float, float, float]) -> list[int]:
        crop = self._prep_crop(image_path, box)
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
        self._validate_codes(codes)
        return codes

    @torch.no_grad()
    def encode_crops_batch(
        self, items: list[tuple[str | Image.Image, tuple[float, float, float, float]]]
    ) -> list[list[int]]:
        """Batch-encode multiple (image_source, box) pairs in one forward pass."""
        if not items:
            return []

        model = self._load_model()
        crops: list[Image.Image] = []
        orig_sizes: list[tuple[int, int]] = []

        for source, box in items:
            crop = self._prep_crop(source, box)
            crops.append(crop)
            orig_sizes.append(crop.size)  # (W, H)

        # Pad to common size
        max_w = max(s[0] for s in orig_sizes)
        max_h = max(s[1] for s in orig_sizes)
        tensors = []
        for crop_img, (w, h) in zip(crops, orig_sizes):
            if w != max_w or h != max_h:
                padded = Image.new("RGB", (max_w, max_h), (0, 0, 0))
                padded.paste(crop_img, (0, 0))
                crop_img = padded
            tensors.append(self._to_model_tensor(crop_img))
        batch = torch.cat(tensors, dim=0)

        # Use AMP for faster inference if on CUDA
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            if getattr(model, "use_ema", False):
                with model.ema_scope():
                    output = model.encode(batch)
            else:
                output = model.encode(batch)

        info = output[2]
        if not isinstance(info, tuple) or len(info) < 3:
            raise RuntimeError("Official IBQ.encode must return quantization info as (_, _, indices).")
        all_codes = info[2]
        if torch.is_tensor(all_codes):
            all_codes = all_codes.detach().cpu()

        # Split per-image codes, filtering out padding tokens
        stride = max(1, int(self.config.downsample_factor))
        max_latent_w = (max_w + stride - 1) // stride
        max_latent_h = (max_h + stride - 1) // stride
        batch_size = len(items)
        all_codes = self._reshape_batch_codes(all_codes, batch_size, max_latent_h, max_latent_w)

        results: list[list[int]] = []
        for i, (w, h) in enumerate(orig_sizes):
            latent_w = (w + stride - 1) // stride
            latent_h = (h + stride - 1) // stride
            sample_codes = all_codes[i]
            valid_codes = sample_codes[:latent_h, :latent_w].reshape(-1).tolist()
            codes = [int(v) for v in valid_codes]
            self._validate_codes(codes)
            results.append(codes)

        return results

    def _reshape_batch_codes(self, all_codes: Any, batch_size: int, latent_h: int, latent_w: int) -> torch.Tensor:
        per_image = latent_h * latent_w
        codes = torch.as_tensor(all_codes)

        if codes.ndim == 1:
            if codes.numel() != batch_size * per_image:
                raise RuntimeError(f"IBQ returned {codes.numel()} codes, expected {batch_size * per_image} for batch size {batch_size}.")
            return codes.reshape(batch_size, latent_h, latent_w)

        if codes.ndim == 2:
            if codes.shape == (batch_size, per_image):
                return codes.reshape(batch_size, latent_h, latent_w)
            if codes.shape == (batch_size * per_image, 1):
                return codes.reshape(batch_size, latent_h, latent_w)
            if codes.shape == (latent_h, latent_w) and batch_size == 1:
                return codes.unsqueeze(0)

        if codes.ndim == 3 and codes.shape[:3] == (batch_size, latent_h, latent_w):
            return codes

        raise RuntimeError(f"Unsupported IBQ code shape {tuple(codes.shape)} for batch size {batch_size}.")

    def _validate_codes(self, codes: list[int]) -> None:
        max_latent_tokens = self.config.max_latent_tokens
        if max_latent_tokens is not None and int(max_latent_tokens) > 0 and len(codes) > int(max_latent_tokens):
            raise ValueError(f"IBQ returned {len(codes)} codes, expected <= {max_latent_tokens}.")
        if any(code < 0 or code >= self.config.codebook_size for code in codes):
            raise ValueError("IBQ returned a code outside the configured 16384-token codebook.")
        if not codes:
            raise ValueError("IBQ returned an empty visual-code sequence.")


__all__ = [
    "DEFAULT_IBQ_CHECKPOINT",
    "DEFAULT_IBQ_CONFIG",
    "DEFAULT_IBQ_REPO",
    "IBQCodec",
    "IBQConfig",
]
