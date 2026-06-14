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


def _box_boundary_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    dx = max(box_a[0] - box_b[2], box_b[0] - box_a[2], 0)
    dy = max(box_a[1] - box_b[3], box_b[1] - box_a[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def _find_connected_components(boxes: list[tuple[int, int, int, int]], margin: float) -> list[list[int]]:
    if not boxes:
        return []

    patch_size = max(boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1])
    max_dist = margin * patch_size
    neighbors = {i: [] for i in range(len(boxes))}
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _box_boundary_distance(boxes[i], boxes[j]) <= max_dist:
                neighbors[i].append(j)
                neighbors[j].append(i)

    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(boxes)):
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        component: list[int] = []
        while queue:
            node = queue.pop()
            component.append(node)
            for nb in neighbors[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(component)
    return components


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
    threshold: float = 0.3,
    margin: float = 1.0,
    padding: float = 1.0,
) -> list[CropRegion]:
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

    patch_w = int(round((boxes_np[0, 2] - boxes_np[0, 0]) * width))
    patch_h = int(round((boxes_np[0, 3] - boxes_np[0, 1]) * height))
    pad_x = int(round(padding * patch_w))
    pad_y = int(round(padding * patch_h))

    regions: list[CropRegion] = []
    for component in _find_connected_components(kept_boxes, margin):
        x1 = max(0, min(kept_boxes[i][0] for i in component) - pad_x)
        y1 = max(0, min(kept_boxes[i][1] for i in component) - pad_y)
        x2 = min(width, max(kept_boxes[i][2] for i in component) + pad_x)
        y2 = min(height, max(kept_boxes[i][3] for i in component) + pad_y)
        regions.append(
            CropRegion(
                box=(x1, y1, x2, y2),
                weight=max(kept_weights[i] for i in component),
                crop=image.crop((x1, y1, x2, y2)),
            )
        )
    regions.sort(key=lambda item: item.weight, reverse=True)
    return regions
