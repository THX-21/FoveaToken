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

from ..fovea_crop import crop_from_normalized_box
from ..tokenizers.tokenization_fovea import FOVEA_TOOL_CALL

Image.MAX_IMAGE_PIXELS = None


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
DEFAULT_IMAGE_PAD = "<|image_pad|>"
DEFAULT_VISION_START = "<|vision_start|>"
DEFAULT_VISION_END = "<|vision_end|>"
SOT_EOT_IMAGE_RE = re.compile(r"<SOT>\s*(\[[^\]]+\])\s*<EOT>\s*<image>")
ORPHAN_VGR_TAG_RE = re.compile(r"<SOT>|<EOT>")
THINK_START = "<think>"
THINK_END = "</think>"
FOVEA_TOOL_LINE_INDENT_RE = re.compile(r'(?m)^[ \t]+(?=\{"fovea"\})')


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


def build_visual_payload(
    num_image_tokens: int,
    image_token: str = DEFAULT_IMAGE_PAD,
    vision_start_token: str = DEFAULT_VISION_START,
    vision_end_token: str = DEFAULT_VISION_END,
) -> str:
    return f"{vision_start_token}{image_token * int(num_image_tokens)}{vision_end_token}"


def append_visual_placeholders_to_fovea(
    conversations: Sequence[dict[str, Any]],
    crop_token_counts: Sequence[int],
) -> list[dict[str, Any]]:
    updated = copy.deepcopy(list(conversations))
    crop_cursor = 0
    for sentence in updated:
        value = str(sentence.get("value", ""))
        if sentence.get("from") not in {"gpt", "assistant"} or FOVEA_TOOL_CALL not in value:
            sentence["value"] = value
            continue
        value = FOVEA_TOOL_LINE_INDENT_RE.sub("", value)
        pieces: list[str] = []
        start = 0
        while True:
            idx = value.find(FOVEA_TOOL_CALL, start)
            if idx < 0:
                pieces.append(value[start:])
                break
            pieces.append(value[start : idx + len(FOVEA_TOOL_CALL)])
            if crop_cursor >= len(crop_token_counts):
                raise ValueError("Missing crop token counts for Fovea placeholders.")
            pieces.append("\n" + build_visual_payload(crop_token_counts[crop_cursor]))
            crop_cursor += 1
            start = idx + len(FOVEA_TOOL_CALL)
        sentence["value"] = "".join(pieces)
    if crop_cursor != len(crop_token_counts):
        raise ValueError("Unused crop token counts remain after expanding Fovea placeholders.")
    return updated


def build_mm_token_type_ids(input_ids, image_token_id: int, video_token_id: int | None = None):
    mm_token_type_ids = input_ids.new_zeros(input_ids.shape)
    mm_token_type_ids[input_ids == image_token_id] = 1
    if video_token_id is not None:
        mm_token_type_ids[input_ids == video_token_id] = 2
    return mm_token_type_ids


def load_training_records(data_path: str) -> list[dict[str, Any]]:
    """Load preprocessed training parquet records from a file or directory."""

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
                raise ImportError("Reading training parquet requires pandas and pyarrow.") from exc
            frame = pd.read_parquet(item)
            records.extend(frame.to_dict("records"))
        else:
            raise ValueError(f"Only parquet training files are supported: {item}")
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


def replace_vgr_regions_with_fovea(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert VGR `<SOT>box<EOT><image>` tags to Fovea JSON tool calls.

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
        return FOVEA_TOOL_CALL

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


def _assistant_role_prefix_token_len(tokenizer: PreTrainedTokenizerBase) -> int:
    """Token length of the assistant role prefix emitted by the Qwen chat template."""

    return len(tokenize_text(tokenizer, "<|im_start|>assistant\n"))


def _assistant_turn_suffix_token_len(tokenizer: PreTrainedTokenizerBase) -> int:
    """Token length of the assistant turn suffix emitted by the Qwen chat template."""

    return len(tokenize_text(tokenizer, "<|im_end|>\n"))


def normalize_assistant_think_text(text: str) -> str:
    """Normalize leading assistant think blocks to Qwen-style newlines."""

    if not text.startswith(THINK_START):
        return text

    remainder = text[len(THINK_START) :]
    if not remainder.startswith("\n"):
        remainder = "\n" + remainder

    if THINK_END not in remainder:
        return THINK_START + remainder

    think_body, suffix = remainder.split(THINK_END, 1)
    think_body = think_body.rstrip("\n")
    if suffix:
        suffix = suffix.lstrip("\n")
        return f"{THINK_START}{think_body}\n{THINK_END}\n\n{suffix}"
    return f"{THINK_START}{think_body}\n{THINK_END}"


def normalize_conversation_think_format(conversations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(list(conversations))
    for sentence in normalized:
        if sentence.get("from") not in {"gpt", "assistant"}:
            continue
        value = sentence.get("value")
        if isinstance(value, str):
            sentence["value"] = normalize_assistant_think_text(value)
    return normalized


def encode_chat_template_example(
    processor,
    tokenizer: PreTrainedTokenizerBase,
    conversations: Sequence[dict[str, Any]],
    image_token_counts: Sequence[int | Sequence[int]],
    system_message: str,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Encode one conversation into causal-LM inputs and labels.

    The checkpoint chat template requires at least one user query, so we only
    render cumulative prefixes after the first user message exists. This keeps
    the prompt fully template-compatible while still letting us build
    assistant-only supervision by diffing valid rendered prefixes.
    """

    prompt_conversations = replace_image_tokens_in_conversations(list(conversations), image_token_counts)
    messages: list[dict[str, Any]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.extend(
        {"role": _conversation_role(sentence["from"]), "content": str(sentence["value"])}
        for sentence in prompt_conversations
    )

    input_ids: list[int] = []
    labels: list[int] = []

    def append_segment(text: str, supervised_prefix_len: int | None, supervised_suffix_len: int = 0) -> None:
        segment_ids = tokenize_text(tokenizer, text)
        input_ids.extend(segment_ids)
        if supervised_prefix_len is None:
            labels.extend([IGNORE_INDEX] * len(segment_ids))
        else:
            supervised_prefix_len = min(supervised_prefix_len, len(segment_ids))
            supervised_suffix_len = min(supervised_suffix_len, max(0, len(segment_ids) - supervised_prefix_len))
            supervised_end = len(segment_ids) - supervised_suffix_len
            labels.extend([IGNORE_INDEX] * supervised_prefix_len)
            labels.extend(segment_ids[supervised_prefix_len:supervised_end])
            labels.extend([IGNORE_INDEX] * supervised_suffix_len)

    first_user_index = next((idx for idx, message in enumerate(messages) if message["role"] == "user"), None)
    if first_user_index is None:
        raise ValueError("Chat-template training samples must contain at least one user message.")

    rendered_prefix = _render_chat_template(processor, messages[: first_user_index + 1])
    append_segment(rendered_prefix, supervised_prefix_len=None)

    for message_index in range(first_user_index + 1, len(messages)):
        message = messages[message_index]
        rendered_current = _render_chat_template(processor, messages[: message_index + 1])
        if not rendered_current.startswith(rendered_prefix):
            raise ValueError("Chat template rendering is not prefix-stable; cannot build assistant-only labels.")
        segment = rendered_current[len(rendered_prefix) :]
        rendered_prefix = rendered_current
        if message["role"] != "assistant":
            append_segment(segment, supervised_prefix_len=None)
            continue
        prefix_len = _assistant_role_prefix_token_len(tokenizer)
        if message["content"].startswith(THINK_START + "\n"):
            prefix_len += len(tokenize_text(tokenizer, THINK_START + "\n"))
        suffix_len = _assistant_turn_suffix_token_len(tokenizer)
        # Supervise only the final <|im_end|>, not the trailing newline.
        append_segment(segment, supervised_prefix_len=prefix_len, supervised_suffix_len=max(0, suffix_len - 1))

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    labels_tensor = mask_visual_placeholder_labels(input_ids_tensor, labels_tensor, tokenizer)
    return input_ids_tensor, labels_tensor


def mask_visual_placeholder_labels(
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    tokenizer: PreTrainedTokenizerBase,
) -> torch.LongTensor:
    masked = labels.clone()
    start_ids = tokenize_text(tokenizer, DEFAULT_VISION_START)
    pad_ids = tokenize_text(tokenizer, DEFAULT_IMAGE_PAD)
    end_ids = tokenize_text(tokenizer, DEFAULT_VISION_END)
    if len(start_ids) != 1 or len(pad_ids) != 1 or len(end_ids) != 1:
        raise ValueError("Qwen visual placeholder markers are expected to be single tokens.")
    start_id, pad_id, end_id = start_ids[0], pad_ids[0], end_ids[0]
    ids = input_ids.tolist()
    idx = 0
    while idx < len(ids):
        if ids[idx] != start_id or idx + 1 >= len(ids) or ids[idx + 1] != pad_id:
            idx += 1
            continue
        end = idx + 2
        while end < len(ids) and ids[end] == pad_id:
            end += 1
        if end < len(ids) and ids[end] == end_id:
            # Mask the full visual placeholder span including <|vision_start|>.
            masked[idx : end + 1] = IGNORE_INDEX
            idx = end + 1
            continue
        idx += 1
    return masked


def apply_selective_label_substrings(
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    tokenizer: PreTrainedTokenizerBase,
    substrings: Sequence[str],
) -> torch.LongTensor:
    """Keep labels only for exact tokenized substrings.

    This is used by auxiliary grounding stages where template text should be
    context, while specific generated artifacts remain supervised.
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
    """Locate Fovea JSON tool calls in the tokenized sequence."""

    if not query_boxes:
        return {
            "fovea_positions": torch.empty((0,), dtype=torch.long),
            "fovea_box_indices": torch.empty((0,), dtype=torch.long),
            "fovea_boxes": torch.empty((0, 4), dtype=torch.float32),
        }
    ids = input_ids.tolist()
    trigger_ids = tokenize_text(tokenizer, FOVEA_TOOL_CALL + "\n")
    positions = []
    for start in range(0, len(ids) - len(trigger_ids) + 1):
        end = start + len(trigger_ids)
        if ids[start:end] == trigger_ids and all(labels[index].item() != IGNORE_INDEX for index in range(start, end)):
            positions.append(end - 1)
    if len(positions) != len(query_boxes):
        raise ValueError(
            "Assistant-side Fovea tool calls do not match parsed query boxes: "
            f"boxes={len(query_boxes)}, triggers={len(positions)}. "
            "Fovea tool calls must be in supervised assistant text."
        )

    return {
        "fovea_positions": torch.tensor(positions, dtype=torch.long),
        "fovea_box_indices": torch.arange(len(positions), dtype=torch.long),
        "fovea_boxes": torch.tensor(query_boxes, dtype=torch.float32),
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

    def _process(
        self,
        image: Image.Image,
        max_image_tokens: int | None = None,
        min_image_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        kwargs = {}
        if min_image_tokens is not None and max_image_tokens is not None and min_image_tokens > max_image_tokens:
            raise ValueError("min_image_tokens cannot exceed max_image_tokens.")
        if min_image_tokens is not None:
            kwargs["min_pixels"] = (
                int(min_image_tokens)
                * self.spatial_merge_size
                * self.spatial_merge_size
                * self.patch_size
                * self.patch_size
            )
        if max_image_tokens is not None:
            if "min_pixels" not in kwargs:
                min_pixels = getattr(self.image_processor, "min_pixels", None)
                if min_pixels is None:
                    size = getattr(self.image_processor, "size", None)
                    if size is not None:
                        min_pixels = size.get("shortest_edge")
                if min_pixels is not None:
                    kwargs["min_pixels"] = int(min_pixels)
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
        fovea_crop_max_image_tokens: int | None = 1024,
        fovea_crop_min_image_tokens: int | None = 64,
        model_max_length: int | None = None,
    ) -> None:
        self.records = load_training_records(data_path)
        self.image_folder = image_folder
        self.processor = processor
        self.tokenizer = tokenizer
        self.vision_packer = vision_packer
        self.image_token_id = image_token_id
        self.system_message = system_message
        self.fovea_crop_max_image_tokens = fovea_crop_max_image_tokens
        self.fovea_crop_min_image_tokens = fovea_crop_min_image_tokens
        self.model_max_length = int(model_max_length or getattr(tokenizer, "model_max_length", 0) or 0)

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
        image_token_counts: list[list[int]] = []
        crop_token_counts: list[int] = []
        query_boxes: list[tuple[float, float, float, float]] = []
        source_images: list[Image.Image] = []

        if image_field is not None:
            image_values = image_field if isinstance(image_field, list) else [image_field]
            pixel_values_list = []
            image_grid_list = []
            for image_value in image_values:
                pil_image = self._open_image(image_value)
                source_images.append(pil_image)
                packed_pixels, packed_grid = self.vision_packer.pack(pil_image)
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

        conversations = normalize_conversation_think_format(record["conversations"])
        query_boxes.extend(tuple(float(v) for v in box) for box in record.get("fovea_query_boxes", []))
        if query_boxes:
            if image_field is None:
                raise ValueError("Fovea training samples with fovea_query_boxes require an image.")
            image_values = image_field if isinstance(image_field, list) else [image_field]
            if len(image_values) != 1:
                raise ValueError("Fovea training expects one image per sample when fovea_query_boxes are present.")
            crop_pixel_values_list = []
            crop_grid_list = []
            for box in query_boxes:
                crop_image = crop_from_normalized_box(source_images[0], box)
                crop_pixels, crop_grid = self.vision_packer._process(
                    crop_image,
                    max_image_tokens=self.fovea_crop_max_image_tokens,
                    min_image_tokens=self.fovea_crop_min_image_tokens,
                )
                crop_pixel_values_list.append(crop_pixels)
                crop_grid_list.append(crop_grid)
                crop_token_counts.append(image_token_count_from_grid(crop_grid, self.vision_packer.spatial_merge_size))
            if crop_pixel_values_list:
                pixel_values = torch.cat([pixel_values, *crop_pixel_values_list], dim=0)
                image_grid_thw = torch.cat([image_grid_thw, torch.stack(crop_grid_list, dim=0)], dim=0)
            conversations = append_visual_placeholders_to_fovea(conversations, crop_token_counts)

        input_ids, labels = encode_chat_template_example(
            processor=self.processor,
            tokenizer=self.tokenizer,
            conversations=conversations,
            image_token_counts=image_token_counts,
            system_message=self.system_message,
        )
        fovea_metadata = build_fovea_metadata(
            input_ids,
            labels,
            tokenizer=self.tokenizer,
            query_boxes=query_boxes,
        )
        labels = apply_selective_label_substrings(
            input_ids,
            labels,
            self.tokenizer,
            normalize_supervised_substrings(record.get("fovea_supervised_substrings")),
        )
        mm_token_type_ids = build_mm_token_type_ids(input_ids=input_ids, image_token_id=self.image_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            **fovea_metadata,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        instance = self._build_instance(index)
        if self.model_max_length > 0 and instance["input_ids"].numel() > self.model_max_length:
            image_name = self.records[index].get("image", "<no-image>")
            print(
                "[data] overlength sample (will be truncated by collator) "
                f"index={index} image={image_name} "
                f"tokens={instance['input_ids'].numel()} model_max_length={self.model_max_length}",
                flush=True,
            )
        return instance


@dataclass
class DataCollatorForFoveaSFT:
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
            batch["num_images_per_sample"] = [int(grid.shape[0]) for grid in image_grid_thw]

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
