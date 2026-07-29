from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


def normalize_patch_boxes(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        return boxes[0]
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected patch boxes with shape [N, 4] or [1, N, 4], got {tuple(boxes.shape)}")
    return boxes


@dataclass
class CropRegion:
    box: tuple[int, int, int, int]
    weight: float
    crop: Image.Image


def crop_from_normalized_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    x1 = max(0, min(width, int(round(float(box[0]) * width))))
    y1 = max(0, min(height, int(round(float(box[1]) * height))))
    x2 = max(0, min(width, int(round(float(box[2]) * width))))
    y2 = max(0, min(height, int(round(float(box[3]) * height))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box after scaling: {box}")
    return image.crop((x1, y1, x2, y2))


def crop_attended_regions(
    image: Image.Image,
    boxes: torch.Tensor,
    weights: torch.Tensor,
    threshold: float = 0.25,
    region_scale: float = 1.2,
) -> list[CropRegion]:
    if region_scale < 1:
        raise ValueError(f"region_scale must be at least 1, got {region_scale}.")
    width, height = image.size
    boxes_np = normalize_patch_boxes(boxes).detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    if weights_np.size == 0:
        return []

    max_weight = float(weights_np.max())
    if max_weight <= 0:
        return []

    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_weights: list[float] = []
    for box, weight in zip(boxes_np, weights_np):
        if float(weight) <= threshold * max_weight:
            continue
        x1 = max(0, min(width, int(round(box[0] * width))))
        y1 = max(0, min(height, int(round(box[1] * height))))
        x2 = max(0, min(width, int(round(box[2] * width))))
        y2 = max(0, min(height, int(round(box[3] * height))))
        if x2 > x1 and y2 > y1:
            kept_boxes.append((x1, y1, x2, y2))
            kept_weights.append(float(weight))
    if not kept_boxes:
        return []

    x1 = min(box[0] for box in kept_boxes)
    y1 = min(box[1] for box in kept_boxes)
    x2 = max(box[2] for box in kept_boxes)
    y2 = max(box[3] for box in kept_boxes)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    half_width = (x2 - x1) * region_scale / 2
    half_height = (y2 - y1) * region_scale / 2
    x1 = max(0, int(round(center_x - half_width)))
    y1 = max(0, int(round(center_y - half_height)))
    x2 = min(width, int(round(center_x + half_width)))
    y2 = min(height, int(round(center_y + half_height)))
    if x2 <= x1 or y2 <= y1:
        return []
    return [
        CropRegion(
            box=(x1, y1, x2, y2),
            weight=max(kept_weights),
            crop=image.crop((x1, y1, x2, y2)),
        )
    ]
