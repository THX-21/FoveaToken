import ast
import math
from dataclasses import dataclass
from math import gcd
from typing import Sequence

import torch
from PIL import Image


DEFAULT_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class VisionPackerConfig:
    """Minimal vision-preprocessing config used by the local image packer.

    The model's official processor may not always be available, so the training
    pipeline keeps a compact config object that is sufficient to reproduce the
    expected image normalization and patch flattening behavior locally.
    """

    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN
    image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD
    rescale_factor: float = 1.0 / 255.0


def select_best_resolution(original_size: tuple[int, int], possible_resolutions: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Pick the candidate resolution that preserves the most useful pixels.

    The heuristic mirrors common "any resolution" preprocessing behavior:
    maximize the effective image area after scaling while minimizing padded /
    wasted canvas area among ties.
    """

    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    if best_fit is None:
        raise ValueError("No candidate resolution is available.")
    return best_fit


def resize_and_pad_image(image: Image.Image, target_resolution: tuple[int, int]) -> Image.Image:
    """Resize with aspect-ratio preservation and letterbox into a fixed canvas."""

    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    resized_image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    return new_image


def parse_grid_pinpoints(grid_pinpoints: str | Sequence[Sequence[int]] | None) -> list[tuple[int, int]]:
    """Normalize user-provided anyres candidates into `(width, height)` pairs."""

    if grid_pinpoints is None:
        return []
    if isinstance(grid_pinpoints, str):
        parsed = ast.literal_eval(grid_pinpoints)
    else:
        parsed = grid_pinpoints
    return [tuple(int(dim) for dim in pair) for pair in parsed]


def infer_grid_stride(resolutions: Sequence[tuple[int, int]], fallback: int) -> int:
    """Infer the base tile stride from candidate resolutions.

    For grids like `[(672, 336), (1008, 336)]`, the GCD gives the tile edge
    length used to estimate how many tiles a candidate consumes.
    """

    dims = []
    for width, height in resolutions:
        if width > 0:
            dims.append(width)
        if height > 0:
            dims.append(height)
    if not dims:
        return fallback
    stride = dims[0]
    for value in dims[1:]:
        stride = gcd(stride, value)
    return stride or fallback


def align_resolution(value: int, multiple: int) -> int:
    """Round a spatial dimension down to the nearest valid packing multiple."""

    if multiple <= 1:
        return value
    return max(multiple, (value // multiple) * multiple)


def align_resolutions_for_packing(
    resolutions: Sequence[tuple[int, int]],
    patch_size: int,
    spatial_merge_size: int,
) -> list[tuple[int, int]]:
    """Ensure candidate canvases produce patch grids valid for spatial merging."""

    required_multiple = patch_size * max(spatial_merge_size, 1)
    aligned: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for width, height in resolutions:
        candidate = (
            align_resolution(width, required_multiple),
            align_resolution(height, required_multiple),
        )
        if candidate not in seen:
            seen.add(candidate)
            aligned.append(candidate)
    return aligned


def estimate_image_token_count(width: int, height: int, patch_size: int, spatial_merge_size: int) -> int:
    """Estimate final image placeholder count after spatial merging."""

    grid_h = height // patch_size
    grid_w = width // patch_size
    merge_area = max(spatial_merge_size, 1) ** 2
    return (grid_h * grid_w) // merge_area


def filter_resolutions_by_token_budget(
    resolutions: Sequence[tuple[int, int]],
    patch_size: int,
    spatial_merge_size: int,
    max_image_tokens: int | None,
) -> list[tuple[int, int]]:
    """Drop anyres candidates that would exceed the final image-token budget."""

    if max_image_tokens is None or max_image_tokens <= 0 or not resolutions:
        return list(resolutions)

    filtered = []
    for width, height in resolutions:
        image_token_count = estimate_image_token_count(width, height, patch_size, spatial_merge_size)
        if image_token_count <= max_image_tokens:
            filtered.append((width, height))
    return filtered


def choose_anyres_resolution(
    image_size: tuple[int, int],
    grid_pinpoints: str | Sequence[Sequence[int]],
    max_tiles: int | None,
    patch_size: int,
    spatial_merge_size: int,
) -> tuple[int, int]:
    """Resolve the final padded anyres canvas for one image."""

    resolutions = parse_grid_pinpoints(grid_pinpoints)
    if not resolutions:
        raise ValueError("image_grid_pinpoints must contain at least one resolution for anyres packing.")
    resolutions = align_resolutions_for_packing(
        resolutions,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
    )
    resolutions = filter_resolutions_by_token_budget(
        resolutions,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        max_image_tokens=max_tiles,
    )
    if not resolutions:
        raise ValueError(
            "No image_grid_pinpoints remain after applying max_image_tokens="
            f"{max_tiles}. Provide smaller pinpoints or increase the token budget."
        )
    return select_best_resolution(image_size, resolutions)


def resize_to_token_budget(
    image: Image.Image,
    config: VisionPackerConfig,
    max_image_tokens: int | None,
) -> Image.Image:
    """Downscale an image proportionally when it exceeds the merged token budget."""

    if max_image_tokens is None or max_image_tokens <= 0:
        return image

    width, height = image.size
    current_tokens = estimate_image_token_count(width, height, config.patch_size, config.spatial_merge_size)
    required_multiple = config.patch_size * max(config.spatial_merge_size, 1)
    aligned_width = align_resolution(width, required_multiple)
    aligned_height = align_resolution(height, required_multiple)
    if current_tokens <= max_image_tokens:
        if (aligned_width, aligned_height) != (width, height):
            return image.resize((aligned_width, aligned_height), Image.Resampling.BICUBIC)
        return image

    scale = math.sqrt(max_image_tokens / max(current_tokens, 1))
    target_width = align_resolution(max(1, int(width * scale)), required_multiple)
    target_height = align_resolution(max(1, int(height * scale)), required_multiple)
    if (target_width, target_height) == (width, height):
        return image
    return image.resize((target_width, target_height), Image.Resampling.BICUBIC)


def normalize_image(image: Image.Image, config: VisionPackerConfig) -> torch.Tensor:
    """Convert a PIL image into a normalized CHW float tensor."""

    image = image.convert("RGB")
    width, height = image.size
    # Build the tensor directly from PIL bytes without going through deprecated
    # typed-storage APIs. `bytearray` makes the buffer writable so PyTorch does
    # not warn about aliasing a read-only Python `bytes` object.
    byte_tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = byte_tensor.view(height, width, 3).permute(2, 0, 1).contiguous().float()
    tensor = tensor * config.rescale_factor
    mean = torch.tensor(config.image_mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.image_std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def pack_single_image(
    image: Image.Image,
    config: VisionPackerConfig,
    max_image_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """Flatten one image into vision patches plus its `(t, h, w)` grid metadata.

    Returns:
        patches:
            Shape `(grid_h * grid_w, c * t_patch * patch * patch)`.
        grid_thw:
            Shape `(3,)`, storing `(1, grid_h, grid_w)`. The temporal dimension
            is synthesized as `1` because we are packing a still image.
    """

    image = resize_to_token_budget(image, config, max_image_tokens)
    image_tensor = normalize_image(image, config)
    channels, height, width = image_tensor.shape
    if height % config.patch_size != 0 or width % config.patch_size != 0:
        raise ValueError(
            f"Image size {(width, height)} is not divisible by patch_size={config.patch_size}. "
            "Use a padded anyres resolution first."
        )

    grid_h = height // config.patch_size
    grid_w = width // config.patch_size
    merge_size = max(config.spatial_merge_size, 1)
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError(
            f"Patch grid {(grid_h, grid_w)} is not divisible by spatial_merge_size={merge_size}. "
            "Use a padded resolution compatible with spatial merging first."
        )

    # Match the official Qwen2-VL/Qwen3.5 processor patch order. A still image is
    # repeated across the temporal patch dimension, then patches are grouped by
    # spatial merge blocks before flattening.
    frames = image_tensor.unsqueeze(0).unsqueeze(0).repeat(1, config.temporal_patch_size, 1, 1, 1)
    patches = (
        frames.view(
            1,
            1,
            config.temporal_patch_size,
            channels,
            grid_h // merge_size,
            merge_size,
            config.patch_size,
            grid_w // merge_size,
            merge_size,
            config.patch_size,
        )
        .permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        .contiguous()
        .view(grid_h * grid_w, channels * config.temporal_patch_size * config.patch_size * config.patch_size)
    )
    grid_thw = torch.tensor([1, grid_h, grid_w], dtype=torch.long)
    return patches, grid_thw


def pack_anyres_image(
    image: Image.Image,
    config: VisionPackerConfig,
    grid_pinpoints: str | Sequence[Sequence[int]],
    max_tiles: int | None,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """Pack an image after choosing an anyres canvas and letterboxing to it."""

    target_resolution = choose_anyres_resolution(
        image_size=image.size,
        grid_pinpoints=grid_pinpoints,
        max_tiles=max_tiles,
        patch_size=config.patch_size,
        spatial_merge_size=config.spatial_merge_size,
    )
    image = resize_and_pad_image(image, target_resolution)
    return pack_single_image(image, config)


def maybe_infer_processor_stats(processor) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Read normalization stats from a processor, with local defaults as fallback."""

    if processor is None:
        return DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD, 1.0 / 255.0

    image_processor = getattr(processor, "image_processor", processor)
    image_mean = tuple(float(x) for x in getattr(image_processor, "image_mean", DEFAULT_IMAGE_MEAN))
    image_std = tuple(float(x) for x in getattr(image_processor, "image_std", DEFAULT_IMAGE_STD))
    rescale_factor = float(getattr(image_processor, "rescale_factor", 1.0 / 255.0))
    return image_mean, image_std, rescale_factor
