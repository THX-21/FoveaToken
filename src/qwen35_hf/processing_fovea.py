from typing import Any

import numpy as np
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin


DEFAULT_IMAGE_PAD = "<|image_pad|>"
DEFAULT_VISION_START = "<|vision_start|>"
DEFAULT_VISION_END = "<|vision_end|>"


def image_token_count_from_grid(image_grid_thw, merge_size: int) -> int:
    """Convert one `(t, h, w)` image grid into text-side image token count."""
    merge_length = max(int(merge_size), 1) ** 2
    grid_prod = image_grid_thw.prod()
    if hasattr(grid_prod, "item"):
        grid_prod = grid_prod.item()
    return int(grid_prod // merge_length)


def build_visual_placeholder(
    num_image_tokens: int,
    image_token: str = DEFAULT_IMAGE_PAD,
    vision_start_token: str = DEFAULT_VISION_START,
    vision_end_token: str = DEFAULT_VISION_END,
) -> str:
    """Expand one logical image into the placeholder span expected by Fovea."""
    return f"{vision_start_token}{image_token * num_image_tokens}{vision_end_token}"


def build_mm_token_type_ids(input_ids, image_token_id: int, video_token_id: int | None = None):
    """Mark image placeholder token positions for the multimodal model."""
    if hasattr(input_ids, "new_zeros"):
        mm_token_type_ids = input_ids.new_zeros(input_ids.shape, dtype=getattr(input_ids, "dtype", None))
        mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        return mm_token_type_ids

    array_ids = np.array(input_ids)
    mm_token_type_ids = np.zeros_like(array_ids)
    mm_token_type_ids[array_ids == image_token_id] = 1
    if video_token_id is not None:
        mm_token_type_ids[array_ids == video_token_id] = 2
    return mm_token_type_ids


class FoveaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.image_token_id = getattr(tokenizer, "image_token_id", tokenizer.convert_tokens_to_ids(self.image_token))
        self.video_token_id = getattr(tokenizer, "video_token_id", tokenizer.convert_tokens_to_ids(self.video_token))
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.vision_start_token_id = getattr(tokenizer, "vision_start_token_id", tokenizer.convert_tokens_to_ids(self.vision_start_token))
        self.vision_end_token_id = getattr(tokenizer, "vision_end_token_id", tokenizer.convert_tokens_to_ids(self.vision_end_token))

    def image_token_counts_from_grids(self, image_grid_thw) -> list[int]:
        if image_grid_thw is None:
            return []
        if hasattr(image_grid_thw, "dim") and image_grid_thw.dim() == 1:
            grids = [image_grid_thw]
        else:
            grids = list(image_grid_thw)
        img_slot_token_count = getattr(self.image_processor, "img_slot_token_count", None)
        if img_slot_token_count is not None:
            return [int(img_slot_token_count) for _ in grids]
        return [image_token_count_from_grid(grid, self.image_processor.merge_size) for grid in grids]

    def uses_imgslot_placeholders(self) -> bool:
        return getattr(self.image_processor, "img_slot_token_count", None) is not None

    def build_visual_placeholder(self, num_image_tokens: int) -> str:
        return build_visual_placeholder(
            num_image_tokens,
            image_token=self.image_token,
            vision_start_token=self.vision_start_token,
            vision_end_token=self.vision_end_token,
        )

    def expand_image_pad_tokens(self, text: list[str], image_grid_thw) -> list[str]:
        image_token_counts = self.image_token_counts_from_grids(image_grid_thw)
        image_block_counts = getattr(self.image_processor, "_last_image_block_counts", None)
        index = 0
        image_index = 0
        output = text.copy()

        def take_block_counts() -> list[int]:
            nonlocal index, image_index
            if index >= len(image_token_counts):
                raise ValueError("Text contains more image placeholders than image_grid_thw entries.")
            block_count = 1
            if image_block_counts is not None:
                if image_index >= len(image_block_counts):
                    raise ValueError("Text contains more image placeholders than image block metadata entries.")
                block_count = int(image_block_counts[image_index])
            block_counts = image_token_counts[index : index + block_count]
            if len(block_counts) != block_count:
                raise ValueError("image_grid_thw contains fewer blocks than expected for image placeholder.")
            index += block_count
            image_index += 1
            return block_counts

        for i in range(len(output)):
            anchor_count = getattr(self.image_processor, "img_slot_anchor_count", None)
            anchor_placeholder = self.image_token * int(anchor_count) if anchor_count is not None else ""
            has_image_placeholder = False
            full_placeholder = f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"
            while full_placeholder in output[i]:
                has_image_placeholder = True
                block_counts = take_block_counts()
                replacement = "".join(
                    build_visual_placeholder(
                        count,
                        image_token="<|placeholder|>",
                        vision_start_token=self.vision_start_token,
                        vision_end_token=self.vision_end_token,
                    )
                    for count in block_counts
                )
                output[i] = output[i].replace(full_placeholder, replacement, 1)
            while self.image_token in output[i]:
                has_image_placeholder = True
                block_counts = take_block_counts()
                if len(block_counts) == 1:
                    replacement = "<|placeholder|>" * block_counts[0]
                else:
                    replacement = "".join(
                        build_visual_placeholder(
                            count,
                            image_token="<|placeholder|>",
                            vision_start_token=self.vision_start_token,
                            vision_end_token=self.vision_end_token,
                        )
                        for count in block_counts
                    )
                output[i] = output[i].replace(self.image_token, replacement, 1)
            output[i] = output[i].replace("<|placeholder|>", self.image_token)
            if anchor_placeholder and has_image_placeholder:
                output[i] = anchor_placeholder + output[i]
        if index != len(image_token_counts):
            raise ValueError("image_grid_thw contains more images than text placeholders.")
        return output

    def build_mm_token_type_ids(self, input_ids):
        return build_mm_token_type_ids(input_ids, self.image_token_id, self.video_token_id)

    def _normalize_image_counts_per_sample(self, images, image_counts_per_sample):
        if image_counts_per_sample is not None:
            return image_counts_per_sample
        if not isinstance(images, list):
            images = [images]
        return [len(sample_images) if isinstance(sample_images, (list, tuple)) else 1 for sample_images in images]

    def _group_image_block_counts(self, flat_image_block_counts, image_counts_per_sample):
        if flat_image_block_counts is None:
            return None
        grouped_image_block_counts = []
        flat_index = 0
        for num_images in image_counts_per_sample:
            sample_counts = flat_image_block_counts[flat_index : flat_index + num_images]
            if len(sample_counts) != num_images:
                raise ValueError("Image block metadata does not align with per-sample image grouping.")
            grouped_image_block_counts.append(sample_counts)
            flat_index += num_images
        if flat_index != len(flat_image_block_counts):
            raise ValueError("Image block metadata contains extra entries after per-sample grouping.")
        return grouped_image_block_counts

    def _tensorize_grouped_image_block_counts(self, grouped_image_block_counts):
        if grouped_image_block_counts is None:
            return None
        max_images = max((len(sample_counts) for sample_counts in grouped_image_block_counts), default=0)
        padded_counts = [
            list(sample_counts) + [0] * (max_images - len(sample_counts))
            for sample_counts in grouped_image_block_counts
        ]
        return torch.tensor(padded_counts, dtype=torch.long)

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        return_mm_token_type_ids: bool | None = None,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        image_counts_per_sample = kwargs.pop("image_counts_per_sample", None)
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        image_inputs = {}
        image_grid_thw = None
        flat_image_block_counts = getattr(self.image_processor, "_last_image_block_counts", None)
        grouped_image_block_counts = None
        if images is not None:
            image_inputs = self.image_processor(images=images, **kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]
            flat_image_block_counts = getattr(self.image_processor, "_last_image_block_counts", flat_image_block_counts)
            image_counts_per_sample = self._normalize_image_counts_per_sample(images, image_counts_per_sample)
            grouped_image_block_counts = self._group_image_block_counts(flat_image_block_counts, image_counts_per_sample)

        videos_inputs = {}
        if videos is not None and self.video_processor is not None:
            videos_inputs = self.video_processor(videos=videos, **kwargs)

        if image_grid_thw is not None:
            text = self.expand_image_pad_tokens(text, image_grid_thw)

        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if self.uses_imgslot_placeholders():
            return_mm_token_type_ids = False
        elif return_mm_token_type_ids is None:
            return_mm_token_type_ids = True

        if return_mm_token_type_ids:
            mm_token_type_ids = self.build_mm_token_type_ids(text_inputs["input_ids"])
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        image_block_counts = grouped_image_block_counts
        if image_block_counts is not None and return_tensors is not None:
            image_block_counts = self._tensorize_grouped_image_block_counts(image_block_counts)

        return BatchFeature(
            data={
                **text_inputs,
                **image_inputs,
                **({"image_block_counts": image_block_counts} if image_block_counts is not None else {}),
                **videos_inputs,
            },
            tensor_type=return_tensors,
        )


__all__ = [
    "DEFAULT_IMAGE_PAD",
    "DEFAULT_VISION_END",
    "DEFAULT_VISION_START",
    "FoveaProcessor",
    "build_mm_token_type_ids",
    "build_visual_placeholder",
    "image_token_count_from_grid",
]
