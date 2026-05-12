import math
from dataclasses import dataclass

import torch
from PIL import Image


DEFAULT_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class VisionPackerConfig:
    """Minimal vision-preprocessing config used by the local image packer."""

    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN
    image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD
    rescale_factor: float = 1.0 / 255.0


@dataclass
class ImageBlock:
    """One crop plus its location in the budgeted full-image coordinate system."""

    image: Image.Image
    row: int
    col: int
    rows: int
    cols: int
    left: int
    top: int
    right: int
    bottom: int


def split_image_into_blocks(image: Image.Image, tile_size: int) -> list[ImageBlock]:
    """Split an image into non-overlapping, evenly sized blocks.

    The number of blocks along each axis is `ceil(dim / tile_size)`. Boundaries
    are then evenly spaced over the original image, so the last block is not a
    small leftover strip. Returned block coordinates are in pixels of `image`.
    """

    if tile_size is None or tile_size <= 0:
        raise ValueError("img_slot_tile_size must be a positive integer when ImgSlot is enabled.")

    width, height = image.size
    cols = max(1, math.ceil(width / tile_size))
    rows = max(1, math.ceil(height / tile_size))
    blocks: list[ImageBlock] = []
    for row in range(rows):
        top = round(row * height / rows)
        bottom = round((row + 1) * height / rows)
        for col in range(cols):
            left = round(col * width / cols)
            right = round((col + 1) * width / cols)
            blocks.append(
                ImageBlock(
                    image=image.crop((left, top, right, bottom)),
                    row=row,
                    col=col,
                    rows=rows,
                    cols=cols,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                )
            )
    return blocks


def align_resolution(value: int, multiple: int) -> int:
    """Round a spatial dimension down to the nearest valid packing multiple."""

    if multiple <= 1:
        return value
    return max(multiple, (value // multiple) * multiple)


def estimate_image_token_count(width: int, height: int, patch_size: int, spatial_merge_size: int) -> int:
    """Estimate final image placeholder count after spatial merging."""

    grid_h = height // patch_size
    grid_w = width // patch_size
    merge_area = max(spatial_merge_size, 1) ** 2
    return (grid_h * grid_w) // merge_area


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
            "Increase max_image_tokens or use a size aligned to the patch grid."
        )

    grid_h = height // config.patch_size
    grid_w = width // config.patch_size
    merge_size = max(config.spatial_merge_size, 1)
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError(
            f"Patch grid {(grid_h, grid_w)} is not divisible by spatial_merge_size={merge_size}. "
            "Increase max_image_tokens or use a size compatible with spatial merging."
        )

    # Match Qwen2-VL/Qwen3.5 patch order. A still image is repeated across the
    # temporal patch dimension, then patches are grouped by spatial merge blocks
    # before flattening.
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


def default_processor_stats() -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return the local normal image normalization defaults."""

    return DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD, 1.0 / 255.0
