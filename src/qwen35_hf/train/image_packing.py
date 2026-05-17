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
class PackedImage:
    """Packed Qwen vision inputs plus original-coordinate token boxes."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.LongTensor
    patch_boxes: torch.Tensor


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


def build_merged_patch_boxes(
    image_width: int,
    image_height: int,
    grid_thw: torch.LongTensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Return normalized boxes for merged visual tokens in image coordinates."""

    _t, grid_h, grid_w = [int(v) for v in grid_thw.tolist()]
    merge = max(int(spatial_merge_size), 1)
    boxes: list[list[float]] = []
    for h in range(0, grid_h, merge):
        for w in range(0, grid_w, merge):
            boxes.append(
                [
                    w / max(grid_w, 1),
                    h / max(grid_h, 1),
                    min(w + merge, grid_w) / max(grid_w, 1),
                    min(h + merge, grid_h) / max(grid_h, 1),
                ]
            )
    return torch.tensor(boxes, dtype=torch.float32)


def pack_single_image_with_boxes(
    image: Image.Image,
    config: VisionPackerConfig,
    max_image_tokens: int | None = None,
) -> PackedImage:
    """Pack one image and track each merged token's normalized box."""

    image = resize_to_token_budget(image, config, max_image_tokens)
    width, height = image.size
    patches, grid_thw = pack_single_image(image, config, max_image_tokens=None)
    patch_boxes = build_merged_patch_boxes(width, height, grid_thw, config.spatial_merge_size)
    return PackedImage(pixel_values=patches, image_grid_thw=grid_thw, patch_boxes=patch_boxes)


def default_processor_stats() -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return the local normal image normalization defaults."""

    return DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD, 1.0 / 255.0
