import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ..processing_fovea import (
    DEFAULT_IMAGE_PAD,
    build_mm_token_type_ids,
    build_visual_placeholder,
    image_token_count_from_grid,
)
from .image_packing import (
    VisionPackerConfig,
    default_processor_stats,
    pack_single_image,
    resize_to_token_budget,
    split_image_into_blocks,
)

Image.MAX_IMAGE_PIXELS = None


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


@dataclass
class SampleEncoding:
    """Typed view of one encoded sample.

    The current dataset returns raw dicts because that is what `Trainer` and the
    collator consume directly, but this structure documents the intended fields.
    """

    input_ids: torch.LongTensor
    labels: torch.LongTensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.LongTensor | None
    mm_token_type_ids: torch.IntTensor


def preprocess_multimodal_conversations(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize conversation text before tokenization.

    When a turn contains exactly one `<image>` token mixed into text, move it to
    the beginning of the message on its own line. This matches the prompt style
    expected by many multimodal instruction datasets and keeps template handling
    predictable.
    """

    conversations = copy.deepcopy(conversations)
    for sentence in conversations:
        value = sentence["value"]
        num_images = len(re.findall(DEFAULT_IMAGE_TOKEN, value))
        if num_images == 1 and DEFAULT_IMAGE_TOKEN in value and not value.startswith(DEFAULT_IMAGE_TOKEN):
            value = value.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            value = f"{DEFAULT_IMAGE_TOKEN}\n{value}".strip()
        sentence["value"] = value
    return conversations


def load_json_records(data_path: str) -> list[dict[str, Any]]:
    """Load the SFT dataset file.

    Expected schema:
    - top level: a JSON list
    - each item: at least `conversations`, optionally `image`
    """

    with open(data_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{data_path} must contain a JSON list.")
    return records


def replace_image_tokens_in_conversations(
    conversations: Sequence[dict[str, Any]],
    image_token_counts: Sequence[int | Sequence[int]],
) -> list[dict[str, Any]]:
    """Apply image placeholder expansion across a full conversation list.

    `image_token_counts[i]` describes how many `<|image_pad|>` tokens the i-th
    referenced image needs after vision packing. The replacement is sequential,
    so multiple images inside a sample keep their original order.
    """

    conversations = copy.deepcopy(list(conversations))
    image_token_groups = [
        [int(count) for count in counts] if isinstance(counts, (list, tuple)) else [int(counts)]
        for counts in image_token_counts
    ]
    total_placeholders = sum(sentence["value"].count(DEFAULT_IMAGE_TOKEN) for sentence in conversations)
    if total_placeholders == 1 and len(image_token_groups) > 1:
        joined_placeholder = "".join(build_visual_placeholder(count) for group in image_token_groups for count in group)
        for sentence in conversations:
            value = sentence["value"]
            if DEFAULT_IMAGE_TOKEN in value:
                sentence["value"] = value.replace(DEFAULT_IMAGE_TOKEN, joined_placeholder, 1)
                return conversations

    image_index = 0
    for sentence in conversations:
        value = sentence["value"]
        while DEFAULT_IMAGE_TOKEN in value:
            if image_index >= len(image_token_groups):
                raise ValueError("Conversation references more <image> placeholders than the sample provides.")
            placeholder = "".join(build_visual_placeholder(count) for count in image_token_groups[image_index])
            value = value.replace(DEFAULT_IMAGE_TOKEN, placeholder, 1)
            image_index += 1
        sentence["value"] = value
    if image_index != len(image_token_groups):
        raise ValueError("Sample provides more images than there are <image> placeholders in the conversation.")
    return conversations


def tokenize_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Tokenize without adding tokenizer-managed special tokens.

    ChatML tags are inserted manually in this file, so auto-added BOS/EOS would
    shift label positions and break the handcrafted prompt template.
    """

    return tokenizer(text, add_special_tokens=False).input_ids


def encode_chatml_example(
    tokenizer: PreTrainedTokenizerBase,
    conversations: Sequence[dict[str, Any]],
    image_token_counts: Sequence[int | Sequence[int]],
    system_message: str,
    img_slot_anchor_count: int | None = None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Encode one conversation into causal-LM inputs and labels.

    Labeling policy:
    - system turns: ignored
    - user turns: ignored
    - assistant turns: supervise only the answer content and turn end marker,
      not the role prefix or empty thinking scaffold

    This is the standard SFT setup where the model learns to continue the prompt
    as the assistant.
    """

    prompt_conversations = preprocess_multimodal_conversations(list(conversations))
    prompt_conversations = replace_image_tokens_in_conversations(prompt_conversations, image_token_counts)

    input_ids: list[int] = []
    labels: list[int] = []

    def append_segment(text: str, supervised_prefix_len: int | None) -> None:
        # `supervised_prefix_len=None` means the entire segment is context only.
        # Otherwise we mask the prefix tokens and train on the remainder.
        segment_ids = tokenize_text(tokenizer, text)
        input_ids.extend(segment_ids)
        if supervised_prefix_len is None:
            labels.extend([IGNORE_INDEX] * len(segment_ids))
        else:
            supervised_prefix_len = min(supervised_prefix_len, len(segment_ids))
            labels.extend([IGNORE_INDEX] * supervised_prefix_len)
            labels.extend(segment_ids[supervised_prefix_len:])

    if img_slot_anchor_count is not None and image_token_counts:
        append_segment(DEFAULT_IMAGE_PAD * int(img_slot_anchor_count), supervised_prefix_len=None)

    system_segment = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    append_segment(system_segment, supervised_prefix_len=None)

    role_prefixes = {
        "human": "<|im_start|>user\n",
        "user": "<|im_start|>user\n",
        "gpt": "<|im_start|>assistant\n",
        "assistant": "<|im_start|>assistant\n",
    }

    for sentence in prompt_conversations:
        role = sentence["from"]
        prefix = role_prefixes.get(role)
        if prefix is None:
            raise ValueError(f"Unsupported role {role!r}.")
        if role in {"human", "user"}:
            full_segment = f"{prefix}{sentence['value']}<|im_end|>\n"
            prefix_len = len(tokenize_text(tokenizer, prefix))
        else:
            # Align with the official Qwen3.5 chat template: assistant turns
            # include an explicit thinking block even when reasoning content is
            # absent. The scaffold is prompt context only; supervise the answer.
            answer_prefix = f"{prefix}<think>\n\n</think>\n\n"
            full_segment = f"{answer_prefix}{sentence['value']}<|im_end|>\n"
            prefix_len = len(tokenize_text(tokenizer, answer_prefix))
        if role in {"human", "user"}:
            append_segment(full_segment, supervised_prefix_len=None)
        else:
            append_segment(full_segment, supervised_prefix_len=prefix_len)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class VisionPacker:
    """Local normal image preprocessing adapter."""

    def __init__(
        self,
        vision_config,
        max_image_tokens: int | None = None,
        img_slot_enable: bool = False,
        img_slot_m: int = 4,
        img_slot_k: int = 64,
        img_slot_tile_size: int | None = None,
    ) -> None:
        self.max_image_tokens = max_image_tokens
        self.img_slot_enable = img_slot_enable
        self.img_slot_anchor_count = int(img_slot_m)
        self.img_slot_token_count = int(img_slot_k)
        self.img_slot_tile_size = img_slot_tile_size
        if self.img_slot_enable:
            if self.img_slot_tile_size is None:
                raise ValueError("img_slot_tile_size is required when img_slot_enable=true.")
            if self.img_slot_anchor_count <= 0 or self.img_slot_token_count <= 0:
                raise ValueError("img_slot_m and img_slot_k must be positive when ImgSlot is enabled.")

        # Some configs expose scalar patch sizes while others expose tuples.
        patch_size = vision_config.patch_size if isinstance(vision_config.patch_size, int) else int(vision_config.patch_size[0])
        temporal_patch_size = (
            vision_config.temporal_patch_size
            if isinstance(vision_config.temporal_patch_size, int)
            else int(vision_config.temporal_patch_size[0])
        )
        image_mean, image_std, rescale_factor = default_processor_stats()
        self.local_config = VisionPackerConfig(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=vision_config.spatial_merge_size,
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
        )

    def _pack_single_block(self, image: Image.Image) -> tuple[torch.Tensor, torch.LongTensor]:
        return pack_single_image(image=image, config=self.local_config, max_image_tokens=self.max_image_tokens)

    def pack(self, image: Image.Image) -> tuple[torch.Tensor, torch.LongTensor]:
        """Pack one image and return `(pixel_values, image_grid_thw)`."""

        if self.img_slot_enable:
            # Apply the token budget to the whole image first; block count and
            # boundaries are based on the budgeted normal-resolution image.
            image = resize_to_token_budget(image, self.local_config, self.max_image_tokens)
            pixel_values_list = []
            image_grid_list = []
            for block in split_image_into_blocks(image, int(self.img_slot_tile_size)):
                packed_pixels, packed_grid = self._pack_single_block(block)
                pixel_values_list.append(packed_pixels)
                image_grid_list.append(packed_grid)
            return torch.cat(pixel_values_list, dim=0), torch.stack(image_grid_list, dim=0)

        return self._pack_single_block(image)


class LazySupervisedDataset(Dataset):
    """Lazily load images and encode samples on access.

    JSON metadata is kept in memory, but image decoding and tokenization happen
    inside `__getitem__`, which avoids a large up-front preprocessing step.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: PreTrainedTokenizerBase,
        vision_packer: VisionPacker,
        image_token_id: int,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        img_slot_enable: bool = False,
    ) -> None:
        self.records = load_json_records(data_path)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.vision_packer = vision_packer
        self.image_token_id = image_token_id
        self.system_message = system_message
        self.img_slot_enable = img_slot_enable

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_name: str) -> tuple[torch.Tensor, torch.LongTensor]:
        """Load an image file relative to `image_folder` and pack it."""

        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        return self.vision_packer.pack(image)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Build one training instance.

        Output fields:
        - `input_ids`, `labels`: text-side causal-LM training tensors
        - `mm_token_type_ids`: marks image placeholder token positions
        - `pixel_values`, `image_grid_thw`: packed image data, or `None`
        """

        record = self.records[index]
        image_field = record.get("image")
        pixel_values = None
        image_grid_thw = None
        image_token_counts: list[list[int]] = []

        if image_field is not None:
            image_names = image_field if isinstance(image_field, list) else [image_field]
            pixel_values_list = []
            image_grid_list = []
            for image_name in image_names:
                packed_pixels, packed_grid = self._load_image(image_name)
                pixel_values_list.append(packed_pixels)
                if packed_grid.dim() == 1:
                    grids_for_counts = [packed_grid]
                    image_grid_list.append(packed_grid)
                else:
                    grids_for_counts = list(packed_grid)
                    image_grid_list.extend(grids_for_counts)
                current_image_token_counts: list[int] = []
                # Text-side visual span lengths: ImgSlot uses one shared
                # sentence-level A span of length m plus fixed k-token spans for
                # each block; the normal path derives counts from `(t, h, w)`
                # grid metadata after spatial merge.
                for grid in grids_for_counts:
                    if self.img_slot_enable:
                        current_image_token_counts.append(self.vision_packer.img_slot_token_count)
                    else:
                        current_image_token_counts.append(
                            image_token_count_from_grid(
                                grid,
                                self.vision_packer.local_config.spatial_merge_size,
                            )
                        )
                image_token_counts.append(current_image_token_counts)
            if pixel_values_list:
                # Multiple images or blocks in one sample are concatenated along
                # the patch axis. Keep one grid row per logical image/block.
                pixel_values = torch.cat(pixel_values_list, dim=0)
                image_grid_thw = torch.stack(image_grid_list, dim=0)

        input_ids, labels = encode_chatml_example(
            tokenizer=self.tokenizer,
            conversations=record["conversations"],
            image_token_counts=image_token_counts,
            system_message=self.system_message,
            img_slot_anchor_count=self.vision_packer.img_slot_anchor_count if self.img_slot_enable else None,
        )
        mm_token_type_ids = build_mm_token_type_ids(input_ids=input_ids, image_token_id=self.image_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


@dataclass
class DataCollatorForQwen3_5SFT:
    """Pad text fields and concatenate image fields into a trainer batch."""

    tokenizer: PreTrainedTokenizerBase
    model_max_length: int

    def _pad(self, tensors: Sequence[torch.Tensor], padding_value: int) -> torch.Tensor:
        """Right-pad a list of 1D tensors after truncating to the model limit."""

        tensors = [tensor[: self.model_max_length] for tensor in tensors]
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Merge per-sample dicts into the batch schema expected by `Trainer`."""

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        input_ids = self._pad([instance["input_ids"] for instance in instances], padding_value=self.tokenizer.pad_token_id)
        labels = self._pad([instance["labels"] for instance in instances], padding_value=IGNORE_INDEX)
        mm_token_type_ids = self._pad([instance["mm_token_type_ids"] for instance in instances], padding_value=0).int()

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "mm_token_type_ids": mm_token_type_ids,
        }

        pixel_values = [instance["pixel_values"] for instance in instances if instance["pixel_values"] is not None]
        image_grid_thw = [instance["image_grid_thw"] for instance in instances if instance["image_grid_thw"] is not None]
        if pixel_values:
            # Vision patches are already flattened per image, so batching is a
            # simple concatenation rather than padding to a rectangular tensor.
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        return batch
