import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ..processing_fovea import (
    build_mm_token_type_ids,
    build_visual_placeholder,
    image_token_count_from_grid,
)
from .image_packing import (
    VisionPackerConfig,
    default_processor_stats,
    pack_single_image_with_boxes,
    pack_single_image,
)
from ..tokenizers.tokenization_visual_query import (
    REPLAY_TOKEN,
    VQ_END_TOKEN,
    VQ_START_TOKEN,
    vis_token,
)

Image.MAX_IMAGE_PIXELS = None


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
VISUAL_CODE_LM_TASK = "visual_code_lm"
SOT_EOT_IMAGE_RE = re.compile(r"<SOT>\s*(\[[^\]]+\])\s*<EOT>\s*<image>")
ORPHAN_VGR_TAG_RE = re.compile(r"<SOT>|<EOT>")


def load_training_records(data_path: str) -> list[dict[str, Any]]:
    """Load VGR parquet records from a file or `data/vgr` directory."""

    path = Path(data_path)
    if path.is_dir():
        paths = sorted(path.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet training files found under {path}.")
    else:
        paths = [path]

    records: list[dict[str, Any]] = []
    for item in paths:
        if item.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError("Reading VGR parquet requires pandas and pyarrow.") from exc
            frame = pd.read_parquet(item)
            records.extend(frame.to_dict("records"))
        else:
            raise ValueError(f"Only VGR parquet training files are supported: {item}")
    return records


def parse_vgr_box(box_text: str) -> tuple[float, float, float, float]:
    values = json.loads(box_text)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"Invalid VGR box: {box_text}")
    x1, y1, x2, y2 = [float(v) for v in values]
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid non-positive VGR box: {box_text}")
    return x1, y1, x2, y2


def replace_vgr_regions_with_visual_queries(
    text: str,
    *,
    image_path: str,
    visual_codec: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert VGR `<SOT>box<EOT><image>` tags to `<vq> codes </vq> replay pads`.

    Assistant-side VGR data may also contain orphan `<SOT>/<EOT>` tags or stray
    plain `<image>` markers that do not correspond to a real extra image. Drop
    those leftovers here so downstream placeholder expansion only sees the
    sample-level user image placeholder.
    """

    queries: list[dict[str, Any]] = []
    saw_vgr_markup = bool(SOT_EOT_IMAGE_RE.search(text) or ORPHAN_VGR_TAG_RE.search(text))

    def replace(match: re.Match) -> str:
        box = parse_vgr_box(match.group(1))
        codes = visual_codec.encode_crop(image_path, box)
        code_tokens = " ".join(vis_token(code) for code in codes)
        replay = "".join(REPLAY_TOKEN for _ in codes)
        queries.append({"box": box, "codes": codes})
        return f"{VQ_START_TOKEN} {code_tokens} {VQ_END_TOKEN}{replay}"

    cleaned_text = ORPHAN_VGR_TAG_RE.sub("", SOT_EOT_IMAGE_RE.sub(replace, text))
    if saw_vgr_markup:
        cleaned_text = cleaned_text.replace(DEFAULT_IMAGE_TOKEN, "")
    return cleaned_text, queries


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
        joined_placeholder = "\n".join(build_visual_placeholder(count) for group in image_token_groups for count in group)
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
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Encode one conversation into causal-LM inputs and labels.

    Labeling policy:
    - system turns: ignored
    - user turns: ignored
    - assistant turns: supervise the sample-provided assistant content and turn
      end marker, not the role prefix

    This is the standard SFT setup where the model learns to continue the prompt
    as the assistant.
    """

    prompt_conversations = replace_image_tokens_in_conversations(list(conversations), image_token_counts)

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
            full_segment = f"{prefix}{sentence['value']}<|im_end|>\n"
            prefix_len = len(tokenize_text(tokenizer, prefix))
        if role in {"human", "user"}:
            append_segment(full_segment, supervised_prefix_len=None)
        else:
            append_segment(full_segment, supervised_prefix_len=prefix_len)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def apply_selective_label_substrings(
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    tokenizer: PreTrainedTokenizerBase,
    substrings: Sequence[str],
) -> torch.LongTensor:
    """Keep labels only for exact tokenized substrings.

    This is used by auxiliary grounding stages where template text should be
    context, while specific generated artifacts such as visual-query tokens and
    bbox coordinates remain supervised.
    """

    if not substrings:
        return labels
    ids = input_ids.tolist()
    selective_labels = torch.full_like(labels, IGNORE_INDEX)
    search_start = 0
    for substring in substrings:
        pattern = tokenize_text(tokenizer, str(substring))
        if not pattern:
            continue
        found = -1
        last_start = max(0, len(ids) - len(pattern))
        for start in range(search_start, last_start + 1):
            if ids[start : start + len(pattern)] == pattern:
                found = start
                break
        if found < 0:
            for start in range(0, last_start + 1):
                if ids[start : start + len(pattern)] == pattern:
                    found = start
                    break
        if found < 0:
            raise ValueError(f"Could not locate supervised substring after tokenization: {substring!r}")
        end = found + len(pattern)
        selective_labels[found:end] = input_ids[found:end]
        search_start = end
    return selective_labels


def normalize_supervised_substrings(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def build_visual_query_metadata(
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    *,
    tokenizer: PreTrainedTokenizerBase,
    query_boxes: Sequence[Sequence[float]],
) -> dict[str, torch.Tensor]:
    """Locate `<vq> ... </vq> replay` spans after tokenization."""

    vq_start_id = tokenizer.convert_tokens_to_ids(VQ_START_TOKEN)
    if not query_boxes and vq_start_id not in input_ids.tolist():
        return {
            "vq_code_positions": torch.empty((0,), dtype=torch.long),
            "vq_replay_positions": torch.empty((0,), dtype=torch.long),
            "vq_code_query_indices": torch.empty((0,), dtype=torch.long),
            "vq_boxes": torch.empty((0, 4), dtype=torch.float32),
            "vq_code_label_mask": torch.zeros_like(labels, dtype=torch.bool),
        }
    vq_end_id = tokenizer.convert_tokens_to_ids(VQ_END_TOKEN)
    replay_id = tokenizer.convert_tokens_to_ids(REPLAY_TOKEN)
    vis_start_id = tokenizer.convert_tokens_to_ids("<vis_0>")
    vis_end_id = tokenizer.convert_tokens_to_ids("<vis_16383>")
    if min(vq_start_id, vq_end_id, replay_id, vis_start_id, vis_end_id) < 0:
        raise ValueError("Visual query special tokens must be added to the tokenizer before encoding VGR.")

    ids = input_ids.tolist()
    code_positions: list[int] = []
    code_query_indices: list[int] = []
    replay_positions: list[int] = []
    code_label_mask = torch.zeros_like(labels, dtype=torch.bool)
    query_index = 0
    pos = 0
    while pos < len(ids):
        if ids[pos] != vq_start_id:
            pos += 1
            continue
        end = pos + 1
        while end < len(ids) and ids[end] != vq_end_id:
            end += 1
        if end >= len(ids):
            raise ValueError("Found <vq> without matching </vq>.")
        codes = [idx for idx in range(pos + 1, end) if vis_start_id <= ids[idx] <= vis_end_id]
        if not codes:
            raise ValueError("Found empty visual query.")
        replay_start = end + 1
        replay_end = replay_start
        while replay_end < len(ids) and ids[replay_end] == replay_id:
            replay_end += 1
        if replay_end - replay_start != len(codes):
            raise ValueError("Replay pad count must equal visual code count.")
        if query_index >= len(query_boxes):
            raise ValueError("More tokenized visual queries than parsed VGR boxes.")
        code_positions.extend(codes)
        replay_positions.extend(range(replay_start, replay_end))
        code_query_indices.extend([query_index] * len(codes))
        code_label_mask[codes] = labels[codes].ne(IGNORE_INDEX)
        labels[replay_start:replay_end] = IGNORE_INDEX
        query_index += 1
        pos = replay_end

    if query_index != len(query_boxes):
        raise ValueError("Parsed VGR boxes do not match tokenized visual queries.")

    return {
        "vq_code_positions": torch.tensor(code_positions, dtype=torch.long),
        "vq_replay_positions": torch.tensor(replay_positions, dtype=torch.long),
        "vq_code_query_indices": torch.tensor(code_query_indices, dtype=torch.long),
        "vq_boxes": torch.tensor(query_boxes, dtype=torch.float32),
        "vq_code_label_mask": code_label_mask,
    }


def empty_visual_query_metadata(labels: torch.LongTensor) -> dict[str, torch.Tensor]:
    return {
        "vq_code_positions": torch.empty((0,), dtype=torch.long),
        "vq_replay_positions": torch.empty((0,), dtype=torch.long),
        "vq_code_query_indices": torch.empty((0,), dtype=torch.long),
        "vq_boxes": torch.empty((0, 4), dtype=torch.float32),
        "vq_code_label_mask": torch.zeros_like(labels, dtype=torch.bool),
    }


class VisionPacker:
    """Local normal image preprocessing adapter."""

    def __init__(
        self,
        vision_config,
        max_image_tokens: int | None = None,
    ) -> None:
        self.max_image_tokens = max_image_tokens
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

    def pack_retrieve(self, image: Image.Image, max_image_tokens: int | None) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        packed = pack_single_image_with_boxes(image=image, config=self.local_config, max_image_tokens=max_image_tokens)
        return packed.pixel_values, packed.image_grid_thw, packed.patch_boxes

    def pack(self, image: Image.Image) -> tuple[torch.Tensor, torch.LongTensor]:
        return self._pack_single_block(image)


class LazySupervisedDataset(Dataset):
    """Lazily load images and encode samples on access.

    VGR metadata is kept in memory, but image decoding and tokenization happen
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
        retrieve_max_image_tokens: int | None = 4096,
        model_max_length: int | None = None,
    ) -> None:
        self.records = load_training_records(data_path)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.vision_packer = vision_packer
        self.image_token_id = image_token_id
        self.system_message = system_message
        self.retrieve_max_image_tokens = retrieve_max_image_tokens
        self.model_max_length = int(model_max_length or getattr(tokenizer, "model_max_length", 0) or 0)
        self._printed_overlength_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_name: str) -> tuple[torch.Tensor, torch.LongTensor]:
        """Load an image file relative to `image_folder` and pack it."""

        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        return self.vision_packer.pack(image)

    def _load_retrieve_image(self, image_name: str) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        return self.vision_packer.pack_retrieve(image, self.retrieve_max_image_tokens)

    def _build_instance(self, index: int) -> dict[str, Any]:
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
        retrieve_pixel_values = None
        retrieve_grid_thw = None
        retrieve_patch_boxes = None
        image_token_counts: list[list[int]] = []
        query_boxes: list[tuple[float, float, float, float]] = []

        is_visual_code_lm = record.get("fovea_task") == VISUAL_CODE_LM_TASK

        if image_field is not None:
            image_names = image_field if isinstance(image_field, list) else [image_field]
            pixel_values_list = []
            image_grid_list = []
            retrieve_pixels_list = []
            retrieve_grid_list = []
            retrieve_box_list = []
            for image_name in image_names:
                packed_pixels, packed_grid = self._load_image(image_name)
                if not is_visual_code_lm:
                    retrieve_pixels, retrieve_grid, retrieve_boxes = self._load_retrieve_image(image_name)
                    retrieve_pixels_list.append(retrieve_pixels)
                    retrieve_grid_list.append(retrieve_grid)
                    retrieve_box_list.append(retrieve_boxes)
                pixel_values_list.append(packed_pixels)
                image_grid_list.append(packed_grid)
                image_token_counts.append([
                    image_token_count_from_grid(
                        packed_grid,
                        self.vision_packer.local_config.spatial_merge_size,
                    )
                ])
            if pixel_values_list:
                pixel_values = torch.cat(pixel_values_list, dim=0)
                image_grid_thw = torch.stack(image_grid_list, dim=0)
                if retrieve_pixels_list:
                    retrieve_pixel_values = torch.cat(retrieve_pixels_list, dim=0)
                    retrieve_grid_thw = torch.stack(retrieve_grid_list, dim=0)
                    retrieve_patch_boxes = torch.cat(retrieve_box_list, dim=0)

        conversations = copy.deepcopy(record["conversations"])
        if not is_visual_code_lm:
            if image_field is None:
                raise ValueError("VGR visual-query training requires an image.")
            image_names = image_field if isinstance(image_field, list) else [image_field]
            if len(image_names) != 1:
                raise ValueError("VGR visual-query training expects one image per sample.")
            if not record.get("fovea_preprocessed", False) or "fovea_query_boxes" not in record:
                raise ValueError("Training requires offline-preprocessed VGR parquet with fovea_query_boxes.")
            query_boxes.extend(tuple(float(v) for v in box) for box in record["fovea_query_boxes"])

        input_ids, labels = encode_chatml_example(
            tokenizer=self.tokenizer,
            conversations=conversations,
            image_token_counts=image_token_counts,
            system_message=self.system_message,
        )
        labels = apply_selective_label_substrings(
            input_ids,
            labels,
            self.tokenizer,
            normalize_supervised_substrings(record.get("fovea_supervised_substrings")),
        )
        mm_token_type_ids = build_mm_token_type_ids(input_ids=input_ids, image_token_id=self.image_token_id)
        if is_visual_code_lm:
            vq_metadata = empty_visual_query_metadata(labels)
        else:
            vq_metadata = build_visual_query_metadata(
                input_ids,
                labels,
                tokenizer=self.tokenizer,
                query_boxes=query_boxes,
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "retrieve_pixel_values": retrieve_pixel_values,
            "retrieve_grid_thw": retrieve_grid_thw,
            "retrieve_patch_boxes": retrieve_patch_boxes,
            **vq_metadata,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        for offset in range(len(self.records)):
            current_index = (index + offset) % len(self.records)
            instance = self._build_instance(current_index)
            if self.model_max_length <= 0 or instance["input_ids"].numel() <= self.model_max_length:
                return instance
            if current_index not in self._printed_overlength_indices:
                self._printed_overlength_indices.add(current_index)
                image_name = self.records[current_index].get("image", "<no-image>")
                print(
                    "[data] skip overlength sample "
                    f"index={current_index} image={image_name} "
                    f"tokens={instance['input_ids'].numel()} model_max_length={self.model_max_length}",
                    flush=True,
                )
        raise RuntimeError(f"All {len(self.records)} training samples exceed model_max_length={self.model_max_length}.")


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
        vq_code_label_mask = self._pad([instance["vq_code_label_mask"] for instance in instances], padding_value=0).bool()

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "mm_token_type_ids": mm_token_type_ids,
            "vq_code_label_mask": vq_code_label_mask,
        }

        pixel_values = [instance["pixel_values"] for instance in instances if instance["pixel_values"] is not None]
        image_grid_thw = [instance["image_grid_thw"] for instance in instances if instance["image_grid_thw"] is not None]
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        retrieve_pixel_values = [instance["retrieve_pixel_values"] for instance in instances if instance.get("retrieve_pixel_values") is not None]
        retrieve_grid_thw = [instance["retrieve_grid_thw"] for instance in instances if instance.get("retrieve_grid_thw") is not None]
        retrieve_patch_boxes = [instance["retrieve_patch_boxes"] for instance in instances if instance.get("retrieve_patch_boxes") is not None]
        if retrieve_pixel_values:
            batch["retrieve_pixel_values"] = torch.cat(retrieve_pixel_values, dim=0)
            batch["retrieve_grid_thw"] = torch.cat(retrieve_grid_thw, dim=0)
            batch["retrieve_patch_boxes"] = torch.cat(retrieve_patch_boxes, dim=0)
            batch["retrieve_image_counts"] = torch.tensor(
                [int(instance["retrieve_grid_thw"].shape[0]) if instance.get("retrieve_grid_thw") is not None else 0 for instance in instances],
                dtype=torch.long,
            )

        code_positions = []
        replay_positions = []
        code_query_indices = []
        boxes = []
        query_offset = 0
        for batch_idx, instance in enumerate(instances):
            num_queries = int(instance["vq_boxes"].shape[0])
            if num_queries:
                boxes.append(instance["vq_boxes"])
            codes = instance["vq_code_positions"]
            if codes.numel() > 0:
                code_positions.append(torch.stack([torch.full_like(codes, batch_idx), codes], dim=1))
                replay = instance["vq_replay_positions"]
                replay_positions.append(torch.stack([torch.full_like(replay, batch_idx), replay], dim=1))
                code_query_indices.append(instance["vq_code_query_indices"] + query_offset)
            query_offset += num_queries
        batch["vq_code_positions"] = torch.cat(code_positions, dim=0) if code_positions else torch.empty((0, 2), dtype=torch.long)
        batch["vq_replay_positions"] = torch.cat(replay_positions, dim=0) if replay_positions else torch.empty((0, 2), dtype=torch.long)
        batch["vq_code_query_indices"] = torch.cat(code_query_indices, dim=0) if code_query_indices else torch.empty((0,), dtype=torch.long)
        batch["vq_boxes"] = torch.cat(boxes, dim=0) if boxes else torch.empty((0, 4), dtype=torch.float32)

        return batch
