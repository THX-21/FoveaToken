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
from transformers.image_processing_utils import select_best_resolution

from ..tokenizers.tokenization_fovea import FOVEA_TOKEN

Image.MAX_IMAGE_PIXELS = None


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
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
        joined_placeholder = "\n".join(DEFAULT_IMAGE_TOKEN * count for group in image_token_groups for count in group)
        for sentence in conversations:
            value = sentence["value"]
            if DEFAULT_IMAGE_TOKEN in value:
                sentence["value"] = value.replace(DEFAULT_IMAGE_TOKEN, joined_placeholder, 1)
                return conversations

    for sentence in conversations:
        value = sentence["value"]
        if DEFAULT_IMAGE_TOKEN not in value:
            continue
        pieces = value.split(DEFAULT_IMAGE_TOKEN)
        if len(pieces) - 1 > len(image_token_groups):
            raise ValueError("Conversation references more <image> placeholders than the sample provides.")
        rebuilt = [pieces[0]]
        for idx, piece in enumerate(pieces[1:]):
            if idx >= len(image_token_groups):
                raise ValueError("Conversation references more <image> placeholders than the sample provides.")
            placeholder = "".join(DEFAULT_IMAGE_TOKEN * count for count in image_token_groups[idx])
            rebuilt.append(placeholder)
            rebuilt.append(piece)
        sentence["value"] = "".join(rebuilt)
        image_token_groups = image_token_groups[len(pieces) - 1 :]
    if image_token_groups:
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
    """Encode one conversation using the LLaVA-v1/Vicuna conversation style.

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

    role_prefixes = {
        "human": "USER: ",
        "user": "USER: ",
        "gpt": "ASSISTANT: ",
        "assistant": "ASSISTANT: ",
    }
    if system_message:
        append_segment(f"{system_message}\n\n", supervised_prefix_len=None)

    for sentence in prompt_conversations:
        role = sentence["from"]
        prefix = role_prefixes.get(role)
        if prefix is None:
            raise ValueError(f"Unsupported role {role!r}.")
        if role in {"human", "user"}:
            full_segment = f"{prefix}{sentence['value']}\n"
            prefix_len = len(tokenize_text(tokenizer, prefix))
        else:
            eos = tokenizer.eos_token or "</s>"
            full_segment = f"{prefix}{sentence['value']}{eos}\n"
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

    box_indices = list(range(len(positions)))

    return {
        "fovea_positions": torch.tensor(positions, dtype=torch.long),
        "fovea_box_indices": torch.tensor(box_indices, dtype=torch.long),
        "fovea_boxes": torch.tensor(query_boxes, dtype=torch.float32),
    }


def empty_fovea_metadata() -> dict[str, torch.Tensor]:
    return {
        "fovea_positions": torch.empty((0,), dtype=torch.long),
        "fovea_box_indices": torch.empty((0,), dtype=torch.long),
        "fovea_boxes": torch.empty((0, 4), dtype=torch.float32),
    }


class VisionPacker:
    """LLaVA-NeXT image preprocessing adapter."""

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

    def _process(self, image: Image.Image) -> tuple[torch.Tensor, torch.LongTensor, int]:
        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["image_sizes"].to(torch.long)
        count = self.input_feature_count(image_sizes[0])
        return pixel_values, image_sizes, count

    def _grid_boxes(self, rows: int, cols: int) -> torch.Tensor:
        row_starts = torch.arange(rows, dtype=torch.float32)
        col_starts = torch.arange(cols, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(row_starts, col_starts, indexing="ij")
        return torch.stack(
            [
                col_grid / float(cols),
                row_grid / float(rows),
                (col_grid + 1) / float(cols),
                (row_grid + 1) / float(rows),
            ],
            dim=-1,
        ).reshape(-1, 4)

    def _pooled_grid_boxes(self, rows: int, cols: int, pool_size: int) -> torch.Tensor:
        pool_size = max(1, int(pool_size))
        pooled_rows = (int(rows) + pool_size - 1) // pool_size
        pooled_cols = (int(cols) + pool_size - 1) // pool_size
        row_starts = torch.arange(pooled_rows, dtype=torch.float32)
        col_starts = torch.arange(pooled_cols, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(row_starts, col_starts, indexing="ij")
        return torch.stack(
            [
                (col_grid * pool_size) / float(cols),
                (row_grid * pool_size) / float(rows),
                torch.clamp((col_grid + 1) * pool_size, max=float(cols)) / float(cols),
                torch.clamp((row_grid + 1) * pool_size, max=float(rows)) / float(rows),
            ],
            dim=-1,
        ).reshape(-1, 4)

    def _highres_unpadded_shape(self, image_size: torch.Tensor) -> tuple[int, int]:
        orig_height, orig_width = [int(v) for v in image_size.tolist()]
        image_grid_pinpoints = getattr(self.image_processor, "image_grid_pinpoints", None)
        if image_grid_pinpoints is None:
            image_grid_pinpoints = getattr(self.processor, "image_grid_pinpoints", None)
        if image_grid_pinpoints is None:
            raise ValueError("LLaVA-NeXT image_grid_pinpoints are required for exact retrieval boxes.")

        vision_image_size = int(getattr(self.vision_config, "image_size", 336) if self.vision_config is not None else 336)
        vision_patch_size = int(getattr(self.vision_config, "patch_size", getattr(self.processor, "patch_size", 14)))
        grid_per_tile = vision_image_size // vision_patch_size
        best_height, best_width = select_best_resolution([orig_height, orig_width], image_grid_pinpoints)
        grid_height = (int(best_height) // vision_image_size) * grid_per_tile
        grid_width = (int(best_width) // vision_image_size) * grid_per_tile

        original_aspect_ratio = orig_width / max(orig_height, 1)
        current_aspect_ratio = grid_width / max(grid_height, 1)
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(orig_height * (grid_width / max(orig_width, 1)), 7))
            padding = max(0, (grid_height - new_height) // 2)
            grid_height = max(1, grid_height - 2 * padding)
        else:
            new_width = int(round(orig_width * (grid_height / max(orig_height, 1)), 7))
            padding = max(0, (grid_width - new_width) // 2)
            grid_width = max(1, grid_width - 2 * padding)
        return int(grid_height), int(grid_width)

    def input_feature_count(self, image_size: torch.Tensor | Sequence[int]) -> int:
        if not isinstance(image_size, torch.Tensor):
            image_size = torch.tensor(image_size, dtype=torch.long)
        vision_image_size = int(getattr(self.vision_config, "image_size", 336) if self.vision_config is not None else 336)
        vision_patch_size = int(getattr(self.vision_config, "patch_size", getattr(self.processor, "patch_size", 14)))
        base_grid = vision_image_size // vision_patch_size
        base_pool = int(getattr(getattr(self.processor, "config", None), "fovea_input_base_pool", 2))
        highres_pool = int(getattr(getattr(self.processor, "config", None), "fovea_input_highres_pool", 4))
        base_rows = (base_grid + base_pool - 1) // base_pool
        base_cols = (base_grid + base_pool - 1) // base_pool
        highres_rows, highres_cols = self._highres_unpadded_shape(image_size)
        pooled_rows = (highres_rows + highres_pool - 1) // highres_pool
        pooled_cols = (highres_cols + highres_pool - 1) // highres_pool
        return int(base_rows * base_cols + pooled_rows * (pooled_cols + 1))

    def retrieval_feature_boxes(self, image_size: torch.Tensor) -> torch.Tensor:
        strategy = getattr(self.processor, "vision_feature_select_strategy", "default")
        if strategy != "default":
            raise ValueError("Exact LLaVA-NeXT retrieval boxes currently require vision_feature_select_strategy='default'.")
        highres_rows, highres_cols = self._highres_unpadded_shape(image_size)
        retrieve_pool = int(getattr(getattr(self.processor, "config", None), "fovea_retrieve_pool", 2))
        return self._pooled_grid_boxes(highres_rows, highres_cols, retrieve_pool)

    def pack_retrieve(self, image: Image.Image) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        pixel_values, image_sizes, _count = self._process(image)
        return pixel_values, image_sizes, self.retrieval_feature_boxes(image_sizes[0])

    def pack(self, image: Image.Image) -> tuple[torch.Tensor, torch.LongTensor, int]:
        return self._process(image)


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
        model_max_length: int | None = None,
    ) -> None:
        self.records = load_training_records(data_path)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.vision_packer = vision_packer
        self.image_token_id = image_token_id
        self.system_message = system_message
        self.model_max_length = int(model_max_length or getattr(tokenizer, "model_max_length", 0) or 0)
        self._printed_overlength_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_name: str) -> tuple[torch.Tensor, torch.LongTensor, int]:
        """Load an image file relative to `image_folder` and pack it."""

        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        return self.vision_packer.pack(image)

    def _load_retrieve_image(self, image_name: str) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        return self.vision_packer.pack_retrieve(image)

    def _build_instance(self, index: int) -> dict[str, Any]:
        """Build one training instance.

        Output fields:
        - `input_ids`, `labels`: text-side causal-LM training tensors
        - `pixel_values`, `image_sizes`: LLaVA-NeXT image data, or `None`
        """

        record = self.records[index]
        image_field = record.get("image")
        pixel_values = None
        image_sizes = None
        retrieve_pixel_values = None
        retrieve_image_sizes = None
        retrieve_patch_boxes = None
        image_token_counts: list[list[int]] = []
        query_boxes: list[tuple[float, float, float, float]] = []

        if image_field is not None:
            image_names = image_field if isinstance(image_field, list) else [image_field]
            pixel_values_list = []
            image_size_list = []
            retrieve_pixels_list = []
            retrieve_size_list = []
            retrieve_box_list = []
            for image_name in image_names:
                packed_pixels, packed_image_sizes, image_token_count = self._load_image(image_name)
                retrieve_pixels, retrieve_sizes, retrieve_boxes = self._load_retrieve_image(image_name)
                retrieve_pixels_list.append(retrieve_pixels)
                retrieve_size_list.append(retrieve_sizes)
                retrieve_box_list.append(retrieve_boxes)
                pixel_values_list.append(packed_pixels)
                image_size_list.append(packed_image_sizes)
                image_token_counts.append([image_token_count])
            if pixel_values_list:
                pixel_values = torch.cat(pixel_values_list, dim=0)
                image_sizes = torch.cat(image_size_list, dim=0)
                if retrieve_pixels_list:
                    retrieve_pixel_values = torch.cat(retrieve_pixels_list, dim=0)
                    retrieve_image_sizes = torch.cat(retrieve_size_list, dim=0)
                    retrieve_patch_boxes = torch.cat(retrieve_box_list, dim=0)

        conversations = copy.deepcopy(record["conversations"])
        if image_field is None:
            raise ValueError("Fovea training requires an image.")
        image_names = image_field if isinstance(image_field, list) else [image_field]
        if len(image_names) != 1:
            raise ValueError("Fovea training expects one image per sample.")
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
        fovea_metadata = build_fovea_metadata(
            input_ids,
            labels,
            tokenizer=self.tokenizer,
            query_boxes=query_boxes,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "retrieve_pixel_values": retrieve_pixel_values,
            "retrieve_image_sizes": retrieve_image_sizes,
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
class DataCollatorForLlavaNextSFT:
    """Pad text fields and concatenate image fields into a trainer batch."""

    tokenizer: PreTrainedTokenizerBase
    model_max_length: int

    def _pad(self, tensors: Sequence[torch.Tensor], padding_value: int) -> torch.Tensor:
        """Right-pad a list of 1D tensors after truncating to the model limit."""

        tensors = [tensor[: self.model_max_length] for tensor in tensors]
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    def _cat_padded_pixel_values(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        max_patches = max(int(tensor.shape[1]) for tensor in tensors)
        padded = []
        for tensor in tensors:
            if int(tensor.shape[1]) == max_patches:
                padded.append(tensor)
                continue
            pad_shape = (tensor.shape[0], max_patches - tensor.shape[1], *tensor.shape[2:])
            pad = tensor.new_zeros(pad_shape)
            padded.append(torch.cat([tensor, pad], dim=1))
        return torch.cat(padded, dim=0)

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Merge per-sample dicts into the batch schema expected by `Trainer`."""

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        input_ids = self._pad([instance["input_ids"] for instance in instances], padding_value=self.tokenizer.pad_token_id)
        labels = self._pad([instance["labels"] for instance in instances], padding_value=IGNORE_INDEX)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        pixel_values = [instance["pixel_values"] for instance in instances if instance["pixel_values"] is not None]
        image_sizes = [instance["image_sizes"] for instance in instances if instance["image_sizes"] is not None]
        if pixel_values:
            batch["pixel_values"] = self._cat_padded_pixel_values(pixel_values)
            batch["image_sizes"] = torch.cat(image_sizes, dim=0)

        retrieve_pixel_values = [instance["retrieve_pixel_values"] for instance in instances if instance.get("retrieve_pixel_values") is not None]
        retrieve_image_sizes = [instance["retrieve_image_sizes"] for instance in instances if instance.get("retrieve_image_sizes") is not None]
        retrieve_patch_boxes = [instance["retrieve_patch_boxes"] for instance in instances if instance.get("retrieve_patch_boxes") is not None]
        if retrieve_pixel_values:
            batch["retrieve_pixel_values"] = self._cat_padded_pixel_values(retrieve_pixel_values)
            batch["retrieve_image_sizes"] = torch.cat(retrieve_image_sizes, dim=0)
            batch["retrieve_patch_boxes"] = torch.cat(retrieve_patch_boxes, dim=0)
            batch["retrieve_image_counts"] = torch.tensor(
                [int(instance["retrieve_image_sizes"].shape[0]) if instance.get("retrieve_image_sizes") is not None else 0 for instance in instances],
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
