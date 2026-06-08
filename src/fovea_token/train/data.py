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

from ..tokenizers.tokenization_fovea import FOVEA_TOKEN

Image.MAX_IMAGE_PIXELS = None


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
DEFAULT_IMAGE_PAD = "<|image_pad|>"
DEFAULT_VISION_START = "<|vision_start|>"
DEFAULT_VISION_END = "<|vision_end|>"
SOT_EOT_IMAGE_RE = re.compile(r"<SOT>\s*(\[[^\]]+\])\s*<EOT>\s*<image>")
ORPHAN_VGR_TAG_RE = re.compile(r"<SOT>|<EOT>")


def image_token_count_from_grid(image_grid_thw, merge_size: int) -> int:
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
    return f"{vision_start_token}{image_token * int(num_image_tokens)}{vision_end_token}"


def build_mm_token_type_ids(input_ids, image_token_id: int, video_token_id: int | None = None):
    mm_token_type_ids = input_ids.new_zeros(input_ids.shape)
    mm_token_type_ids[input_ids == image_token_id] = 1
    if video_token_id is not None:
        mm_token_type_ids[input_ids == video_token_id] = 2
    return mm_token_type_ids


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
    visual_codec: Any | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert VGR `<SOT>box<EOT><image>` tags to fixed `<fovea>` triggers.

    Assistant-side VGR data may also contain orphan `<SOT>/<EOT>` tags or stray
    plain `<image>` markers that do not correspond to a real extra image. Drop
    those leftovers here so downstream placeholder expansion only sees the
    sample-level user image placeholder.
    """

    queries: list[dict[str, Any]] = []
    saw_vgr_markup = bool(SOT_EOT_IMAGE_RE.search(text) or ORPHAN_VGR_TAG_RE.search(text))

    def replace(match: re.Match) -> str:
        box = parse_vgr_box(match.group(1))
        queries.append({"box": box})
        return FOVEA_TOKEN

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

    Chat-template text is rendered by the checkpoint processor, so auto-added
    BOS/EOS would shift label positions.
    """

    return tokenizer(text, add_special_tokens=False).input_ids


def _conversation_role(role: str) -> str:
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    raise ValueError(f"Unsupported role {role!r}.")


def _render_chat_template(processor, messages: Sequence[dict[str, Any]]) -> str:
    rendered = processor.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"Expected processor.apply_chat_template(..., tokenize=False) to return str, got {type(rendered)!r}.")
    return rendered


def encode_chat_template_example(
    processor,
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
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_message}]
    messages.extend(
        {"role": _conversation_role(sentence["from"]), "content": str(sentence["value"])}
        for sentence in prompt_conversations
    )

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

    rendered_prefix = ""
    for message_index, message in enumerate(messages):
        rendered_current = _render_chat_template(processor, messages[: message_index + 1])
        if not rendered_current.startswith(rendered_prefix):
            raise ValueError("Chat template rendering is not prefix-stable; cannot build assistant-only labels.")
        segment = rendered_current[len(rendered_prefix) :]
        rendered_prefix = rendered_current
        if message["role"] != "assistant":
            append_segment(segment, supervised_prefix_len=None)
            continue
        content = str(message["content"])
        content_offset = segment.find(content)
        if content_offset < 0:
            raise ValueError("Could not locate assistant content inside rendered chat template segment.")
        prefix_len = len(tokenize_text(tokenizer, segment[:content_offset]))
        append_segment(segment, supervised_prefix_len=prefix_len)

    eos_token_id = tokenizer.eos_token_id
    if messages[-1]["role"] == "assistant" and eos_token_id is not None and (not input_ids or input_ids[-1] != eos_token_id):
        input_ids.append(int(eos_token_id))
        labels.append(int(eos_token_id))

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


def build_fovea_metadata(
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    *,
    tokenizer: PreTrainedTokenizerBase,
    query_boxes: Sequence[Sequence[float]],
) -> dict[str, torch.Tensor]:
    """Locate `<fovea>` trigger tokens after tokenization."""

    fovea_id = tokenizer.convert_tokens_to_ids(FOVEA_TOKEN)
    if not query_boxes and fovea_id not in input_ids.tolist():
        return {
            "fovea_positions": torch.empty((0,), dtype=torch.long),
            "fovea_box_indices": torch.empty((0,), dtype=torch.long),
            "fovea_boxes": torch.empty((0, 4), dtype=torch.float32),
        }
    if fovea_id < 0:
        raise ValueError("The <fovea> special token must be added to the tokenizer before encoding VGR.")

    ids = input_ids.tolist()
    positions = [idx for idx, token_id in enumerate(ids) if token_id == fovea_id]
    if len(positions) != len(query_boxes):
        raise ValueError(f"Parsed VGR boxes ({len(query_boxes)}) do not match tokenized <fovea> triggers ({len(positions)}).")

    return {
        "fovea_positions": torch.tensor(positions, dtype=torch.long),
        "fovea_box_indices": torch.arange(len(positions), dtype=torch.long),
        "fovea_boxes": torch.tensor(query_boxes, dtype=torch.float32),
    }


def empty_fovea_metadata() -> dict[str, torch.Tensor]:
    return {
        "fovea_positions": torch.empty((0,), dtype=torch.long),
        "fovea_box_indices": torch.empty((0,), dtype=torch.long),
        "fovea_boxes": torch.empty((0, 4), dtype=torch.float32),
    }


class VisionPacker:
    """Official processor-backed image preprocessing adapter."""

    def __init__(
        self,
        processor,
        vision_config=None,
        max_image_tokens: int | None = None,
    ) -> None:
        self.processor = processor
        self.image_processor = processor.image_processor
        self.vision_config = vision_config
        self.max_image_tokens = max_image_tokens
        self.spatial_merge_size = int(
            getattr(self.image_processor, "merge_size", getattr(vision_config, "spatial_merge_size", 2))
        )
        patch_size = getattr(self.image_processor, "patch_size", getattr(vision_config, "patch_size", 16))
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[-1]
        self.patch_size = int(patch_size)

    def _process(self, image: Image.Image, max_image_tokens: int | None = None) -> tuple[torch.Tensor, torch.LongTensor]:
        kwargs = {}
        if max_image_tokens is not None:
            kwargs["max_pixels"] = (
                int(max_image_tokens)
                * self.spatial_merge_size
                * self.spatial_merge_size
                * self.patch_size
                * self.patch_size
            )
        inputs = self.image_processor(images=image, return_tensors="pt", **kwargs)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"].to(torch.long)
        return pixel_values, image_grid_thw[0]

    def _grid_boxes(self, grid_thw: torch.LongTensor) -> torch.Tensor:
        _t, grid_h, grid_w = [int(v) for v in grid_thw.tolist()]
        merge = max(int(self.spatial_merge_size), 1)
        rows = torch.arange(0, grid_h, merge, dtype=torch.float32)
        cols = torch.arange(0, grid_w, merge, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        return torch.stack(
            [
                col_grid / float(grid_w),
                row_grid / float(grid_h),
                (col_grid + merge).clamp_max(float(grid_w)) / float(grid_w),
                (row_grid + merge).clamp_max(float(grid_h)) / float(grid_h),
            ],
            dim=-1,
        ).reshape(-1, 4)

    def pack_retrieve(self, image: Image.Image, max_image_tokens: int | None = None) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        pixel_values, grid_thw = self._process(image, max_image_tokens=max_image_tokens)
        return pixel_values, grid_thw, self._grid_boxes(grid_thw)

    def pack(self, image: Image.Image) -> tuple[torch.Tensor, torch.LongTensor]:
        return self._process(image, max_image_tokens=self.max_image_tokens)


class LazySupervisedDataset(Dataset):
    """Lazily load images and encode samples on access.

    VGR metadata is kept in memory, but image decoding and tokenization happen
    inside `__getitem__`, which avoids a large up-front preprocessing step.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor,
        tokenizer: PreTrainedTokenizerBase,
        vision_packer: VisionPacker,
        image_token_id: int,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        retrieve_max_image_tokens: int | None = 4096,
        model_max_length: int | None = None,
    ) -> None:
        self.records = load_training_records(data_path)
        self.image_folder = image_folder
        self.processor = processor
        self.tokenizer = tokenizer
        self.vision_packer = vision_packer
        self.image_token_id = image_token_id
        self.system_message = system_message
        self.retrieve_max_image_tokens = retrieve_max_image_tokens
        self.model_max_length = int(model_max_length or getattr(tokenizer, "model_max_length", 0) or 0)
        self._printed_overlength_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _open_image(self, image_value: Any) -> Image.Image:
        if isinstance(image_value, dict):
            image_bytes = image_value.get("bytes")
            if image_bytes is not None:
                import io

                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_path = image_value.get("path")
            if image_path:
                return Image.open(os.path.join(self.image_folder, str(image_path))).convert("RGB")
            raise ValueError("Unsupported image dict: expected `bytes` or `path`.")
        if not isinstance(image_value, str):
            raise ValueError(f"Unsupported image value type: {type(image_value)!r}")
        return Image.open(os.path.join(self.image_folder, image_value)).convert("RGB")

    def _load_image(self, image_value: Any) -> tuple[torch.Tensor, torch.LongTensor]:
        """Load one image from inline bytes or from `image_folder` and pack it."""

        image = self._open_image(image_value)
        return self.vision_packer.pack(image)

    def _load_retrieve_image(self, image_value: Any) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        image = self._open_image(image_value)
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

        if image_field is not None:
            image_values = image_field if isinstance(image_field, list) else [image_field]
            pixel_values_list = []
            image_grid_list = []
            retrieve_pixels_list = []
            retrieve_grid_list = []
            retrieve_box_list = []
            for image_value in image_values:
                packed_pixels, packed_grid = self._load_image(image_value)
                retrieve_pixels, retrieve_grid, retrieve_boxes = self._load_retrieve_image(image_value)
                retrieve_pixels_list.append(retrieve_pixels)
                retrieve_grid_list.append(retrieve_grid)
                retrieve_box_list.append(retrieve_boxes)
                pixel_values_list.append(packed_pixels)
                image_grid_list.append(packed_grid)
                image_token_counts.append([
                    image_token_count_from_grid(
                        packed_grid,
                        self.vision_packer.spatial_merge_size,
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
        query_boxes.extend(tuple(float(v) for v in box) for box in record.get("fovea_query_boxes", []))
        if query_boxes:
            if image_field is None:
                raise ValueError("Fovea training samples with fovea_query_boxes require an image.")
            image_values = image_field if isinstance(image_field, list) else [image_field]
            if len(image_values) != 1:
                raise ValueError("Fovea training expects one image per sample when fovea_query_boxes are present.")

        input_ids, labels = encode_chat_template_example(
            processor=self.processor,
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
        fovea_metadata = build_fovea_metadata(
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
            **fovea_metadata,
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

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "mm_token_type_ids": mm_token_type_ids,
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

        fovea_positions = []
        fovea_box_indices = []
        boxes = []
        query_offset = 0
        for batch_idx, instance in enumerate(instances):
            num_queries = int(instance["fovea_boxes"].shape[0])
            if num_queries:
                boxes.append(instance["fovea_boxes"])
            positions = instance["fovea_positions"]
            if positions.numel() > 0:
                keep = positions < self.model_max_length
                positions = positions[keep]
                indices = instance["fovea_box_indices"][keep]
                if positions.numel() > 0:
                    fovea_positions.append(torch.stack([torch.full_like(positions, batch_idx), positions], dim=1))
                    fovea_box_indices.append(indices + query_offset)
            query_offset += num_queries
        batch["fovea_positions"] = torch.cat(fovea_positions, dim=0) if fovea_positions else torch.empty((0, 2), dtype=torch.long)
        batch["fovea_box_indices"] = torch.cat(fovea_box_indices, dim=0) if fovea_box_indices else torch.empty((0,), dtype=torch.long)
        batch["fovea_boxes"] = torch.cat(boxes, dim=0) if boxes else torch.empty((0, 4), dtype=torch.float32)

        return batch
