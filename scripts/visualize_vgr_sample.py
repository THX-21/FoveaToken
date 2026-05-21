#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import textwrap
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "lmms-eval")]

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from fovea_token.train.data import replace_vgr_regions_with_visual_queries
from fovea_token.tokenizers.tokenization_ibq import (
    DEFAULT_IBQ_CHECKPOINT,
    DEFAULT_IBQ_CONFIG,
    DEFAULT_IBQ_REPO,
    IBQCodec,
)


SOT_MARKER = "<SOT>"
EOT_MARKER = "<EOT><image>"
DEFAULT_OPEN_MAGVIT2_REPO = str(PROJECT_ROOT / "src" / "fovea_token" / "models" / "Open-MAGVIT2")
DEFAULT_OPEN_MAGVIT2_CHECKPOINT = str(PROJECT_ROOT / "src" / "fovea_token" / "models" / "Open-MAGVIT2" / "Open_MAGVIT2_pretrain256_16384.ckpt")
DEFAULT_OPEN_MAGVIT2_CONFIG = str(PROJECT_ROOT / "src" / "fovea_token" / "models" / "Open-MAGVIT2" / "configs" / "Open-MAGVIT2" / "gpu" / "pretrain_lfqgan_256_16384.yaml")


class OpenMAGVIT2Codec:
    def __init__(
        self,
        *,
        repo: str,
        checkpoint: str,
        config: str,
        device: str | None,
        codebook_size: int = 16384,
        max_latent_tokens: int = 256,
        downsample_factor: int = 16,
    ) -> None:
        self.repo = Path(repo).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.config_path = Path(config).expanduser().resolve()
        if not self.repo.exists():
            raise FileNotFoundError(f"Open-MAGVIT2 repo does not exist: {self.repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Open-MAGVIT2 checkpoint does not exist: {self.checkpoint}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Open-MAGVIT2 config does not exist: {self.config_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.codebook_size = int(codebook_size)
        self.max_latent_tokens = int(max_latent_tokens)
        self.downsample_factor = int(downsample_factor)
        self._model = None

    def _crop_and_resize(self, image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
        width, height = image.size
        x1, y1, x2, y2 = box
        left = max(0, min(width - 1, round(x1 * width)))
        top = max(0, min(height - 1, round(y1 * height)))
        right = max(left + 1, min(width, round(x2 * width)))
        bottom = max(top + 1, min(height, round(y2 * height)))
        crop = image.crop((left, top, right, bottom)).convert("RGB")
        stride = max(1, self.downsample_factor)
        latent_h = (crop.height + stride - 1) // stride
        latent_w = (crop.width + stride - 1) // stride
        if latent_h * latent_w > self.max_latent_tokens:
            scale = (self.max_latent_tokens / float(latent_h * latent_w)) ** 0.5
            target_w = max(stride, int(crop.width * scale))
            target_h = max(stride, int(crop.height * scale))
        else:
            target_w = crop.width
            target_h = crop.height
        target_w = max(stride, ((target_w + stride - 1) // stride) * stride)
        target_h = max(stride, ((target_h + stride - 1) // stride) * stride)
        return crop if (target_w, target_h) == crop.size else crop.resize((target_w, target_h), Image.Resampling.BICUBIC)

    def _to_model_tensor(self, image: Image.Image) -> torch.Tensor:
        array = torch.from_numpy(np.array(image).astype("float32"))
        tensor = array.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 127.5 - 1.0
        return tensor.to(device=self.device)

    def _load_model(self):
        if self._model is not None:
            return self._model
        config = OmegaConf.load(self.config_path)
        model_cfg = config.model
        class_path = str(model_cfg.class_path)
        module_name, class_name = class_path.rsplit(".", 1)
        model_cls = getattr(import_module(module_name), class_name)
        init_args = OmegaConf.to_container(model_cfg.init_args, resolve=True)
        init_args["lossconfig"] = {"target": "torch.nn.Identity"}
        valid_params = set(inspect.signature(model_cls.__init__).parameters)
        valid_params.discard("self")
        init_args = {key: value for key, value in init_args.items() if key in valid_params}
        model = model_cls(**init_args)
        state = torch.load(self.checkpoint, map_location="cpu")
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        self._model = model
        return model

    @torch.no_grad()
    def encode_crop(self, image_path: str, box: tuple[float, float, float, float]) -> list[int]:
        image = Image.open(image_path).convert("RGB")
        crop = self._crop_and_resize(image, box)
        model = self._load_model()
        images = self._to_model_tensor(crop)
        if getattr(model, "use_ema", False):
            with model.ema_scope():
                output = model.encode(images)
        else:
            output = model.encode(images)
        if not isinstance(output, tuple) or len(output) < 3:
            raise RuntimeError("Open-MAGVIT2.encode must return quant, loss, indices.")
        codes = output[2]
        if torch.is_tensor(codes):
            codes = codes.detach().cpu().reshape(-1).tolist()
        codes = [int(v) for v in codes]
        if len(codes) > self.max_latent_tokens:
            raise ValueError(f"Open-MAGVIT2 returned {len(codes)} codes, expected <= {self.max_latent_tokens}.")
        if any(code < 0 or code >= self.codebook_size for code in codes):
            raise ValueError("Open-MAGVIT2 returned a code outside the configured 16384-token codebook.")
        if not codes:
            raise ValueError("Open-MAGVIT2 returned an empty visual-code sequence.")
        return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one VGR training sample and its <vis_i> replacement.")
    parser.add_argument("--data-path", type=str, default="data/vgr/data/vgr_shortcot.parquet")
    parser.add_argument("--image-folder", type=str, default="data/vgr/llava_next_raw_format")
    parser.add_argument("--sample-index", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="outputs/vgr_visualize")
    parser.add_argument("--visual-codec", type=str, choices=("ibq", "open_magvit2"), default="ibq")
    parser.add_argument("--ibq-repo", type=str, default=DEFAULT_IBQ_REPO)
    parser.add_argument("--ibq-checkpoint", type=str, default=DEFAULT_IBQ_CHECKPOINT)
    parser.add_argument("--ibq-config", type=str, default=DEFAULT_IBQ_CONFIG)
    parser.add_argument("--open-magvit2-repo", type=str, default=DEFAULT_OPEN_MAGVIT2_REPO)
    parser.add_argument("--open-magvit2-checkpoint", type=str, default=DEFAULT_OPEN_MAGVIT2_CHECKPOINT)
    parser.add_argument("--open-magvit2-config", type=str, default=DEFAULT_OPEN_MAGVIT2_CONFIG)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-text-width", type=int, default=110)
    return parser.parse_args()


def load_record(data_path: Path, sample_index: int) -> dict:
    frame = pd.read_parquet(data_path)
    if sample_index < 0 or sample_index >= len(frame):
        raise IndexError(f"sample_index={sample_index} out of range, dataset size={len(frame)}")
    return frame.iloc[sample_index].to_dict()


def find_assistant_text(record: dict) -> str:
    conversations = record.get("conversations")
    if conversations is not None and not isinstance(conversations, list):
        try:
            conversations = list(conversations)
        except TypeError:
            conversations = None
    if conversations:
        for turn in conversations:
            value = turn.get("value", "") if isinstance(turn, dict) else ""
            role = turn.get("from") if isinstance(turn, dict) else None
            if role in {"assistant", "gpt"} and SOT_MARKER in value:
                return value
        for turn in conversations:
            value = turn.get("value", "") if isinstance(turn, dict) else ""
            role = turn.get("from") if isinstance(turn, dict) else None
            if role in {"assistant", "gpt"}:
                return value
    for key in ("answer", "response", "assistant", "text"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Could not find assistant text in the sample record.")


def find_image_path(record: dict, image_folder: Path) -> Path:
    image_value = record.get("image") or record.get("image_path") or record.get("images")
    if isinstance(image_value, list):
        if not image_value:
            raise ValueError("Sample contains empty image list.")
        image_value = image_value[0]
    if not isinstance(image_value, str) or not image_value:
        raise ValueError("Could not find image path in the sample record.")
    image_path = image_folder / image_value
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    return image_path


def normalize_text_block(text: str, max_width: int) -> str:
    paragraphs = []
    for block in text.splitlines():
        stripped = block.rstrip()
        if not stripped:
            paragraphs.append("")
            continue
        if len(stripped) <= max_width:
            paragraphs.append(stripped)
            continue
        paragraphs.append("\n".join(textwrap.wrap(stripped, width=max_width, break_long_words=False, replace_whitespace=False)))
    return "\n".join(paragraphs)


def render_text_panel(title: str, body: str, width: int = 1600, padding: int = 24) -> Image.Image:
    font = ImageFont.load_default()
    lines = [title, "", *body.splitlines()]
    line_height = 18
    height = padding * 2 + line_height * max(1, len(lines))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = padding
    for idx, line in enumerate(lines):
        color = "#111111" if idx != 0 else "#004b87"
        draw.text((padding, y), line, fill=color, font=font)
        y += line_height
    return canvas


def draw_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    rgb = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rgb)
    width, height = rgb.size
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    line_width = max(3, min(width, height) // 200)
    draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=line_width)
    draw.text((left + 6, max(0, top - 16)), f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]", fill=(255, 0, 0), font=ImageFont.load_default())
    return rgb


def crop_from_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    return image.crop((left, top, right, bottom)).convert("RGB")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float().clamp(-1, 1)
    tensor = ((tensor + 1.0) * 127.5).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def infer_latent_hw(length: int) -> tuple[int, int]:
    side_h = int(math.floor(math.sqrt(length)))
    while side_h > 1 and length % side_h != 0:
        side_h -= 1
    return side_h, length // side_h


def decode_ibq_codes(codec: IBQCodec, codes: list[int]) -> Image.Image:
    model = codec._load_model()
    length = len(codes)
    side_h, side_w = infer_latent_hw(length)
    code_tensor = torch.tensor(codes, device=codec.device, dtype=torch.long)
    quant = model.quantize.get_codebook_entry(code_tensor, (1, side_h, side_w, model.quantize.e_dim))
    with torch.no_grad():
        decoded = model.decode(quant)[0]
    return tensor_to_pil(decoded)


def decode_open_magvit2_codes(codec: OpenMAGVIT2Codec, codes: list[int]) -> Image.Image:
    model = codec._load_model()
    length = len(codes)
    side_h, side_w = infer_latent_hw(length)
    code_tensor = torch.tensor(codes, device=codec.device, dtype=torch.long).view(1, -1)
    quant = model.quantize.get_codebook_entry(
        code_tensor,
        (1, side_h, side_w, model.quantize.codebook_dim),
        order="post",
    )
    with torch.no_grad():
        decoded = model.decode(quant)[0]
    return tensor_to_pil(decoded)


def fit_image(image: Image.Image, max_size: tuple[int, int], background: str = "white") -> Image.Image:
    max_width, max_height = max_size
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image size must be positive.")
    scale = min(max_width / image.width, max_height / image.height)
    scale = min(scale, 1.0)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (max_width, max_height), background)
    offset = ((max_width - resized.width) // 2, (max_height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def stack_vertical(images: list[Image.Image], gap: int = 20, background: str = "#f5f5f5") -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    y = 0
    for image in images:
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    return canvas


def stack_horizontal(images: list[Image.Image], gap: int = 20, background: str = "#f5f5f5") -> Image.Image:
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for image in images:
        canvas.paste(image, (x, (height - image.height) // 2))
        x += image.width + gap
    return canvas


def make_labeled_image(title: str, image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = fit_image(image, size)
    header = render_text_panel(title, "", width=fitted.width, padding=18)
    return stack_vertical([header, fitted], gap=0, background="white")


def make_query_overview(
    raw_text: str,
    replaced_text: str,
    overlay: Image.Image,
    crop: Image.Image,
    recon: Image.Image,
    info_text: str,
    *,
    max_text_width: int,
) -> Image.Image:
    left_column = stack_vertical(
        [
            render_text_panel("Original assistant text", normalize_text_block(raw_text, max_text_width), width=1200),
            render_text_panel("Replaced assistant text (<vis_i> form)", normalize_text_block(replaced_text, max_text_width), width=1200),
            render_text_panel("Query details", normalize_text_block(info_text, max_text_width), width=1200),
        ],
        gap=20,
        background="#f5f5f5",
    )
    right_column = stack_vertical(
        [
            make_labeled_image("Original image with query box", overlay, (960, 640)),
            stack_horizontal(
                [
                    make_labeled_image("Cropped region", crop, (460, 360)),
                    make_labeled_image("Decoded from <vis_i>", recon, (460, 360)),
                ],
                gap=20,
                background="#f5f5f5",
            ),
        ],
        gap=20,
        background="#f5f5f5",
    )
    return stack_horizontal([left_column, right_column], gap=24, background="#f5f5f5")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    image_folder = Path(args.image_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record = load_record(data_path, args.sample_index)
    image_path = find_image_path(record, image_folder)
    raw_text = find_assistant_text(record)

    if args.visual_codec == "ibq":
        codec = IBQCodec.from_paths(
            repo=args.ibq_repo,
            checkpoint=args.ibq_checkpoint,
            config=args.ibq_config,
            device=args.device,
        )
    else:
        codec = OpenMAGVIT2Codec(
            repo=args.open_magvit2_repo,
            checkpoint=args.open_magvit2_checkpoint,
            config=args.open_magvit2_config,
            device=args.device,
        )

    replaced_text, queries = replace_vgr_regions_with_visual_queries(
        raw_text,
        image_path=str(image_path),
        visual_codec=codec,
    )
    if not queries:
        raise ValueError("No <SOT>...[x1,y1,x2,y2]...<EOT><image> region found in this sample.")

    image = Image.open(image_path).convert("RGB")

    sample_dir = output_dir / f"sample_{args.sample_index:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f"data_path: {data_path}",
        f"image_path: {image_path}",
        f"visual_codec: {args.visual_codec}",
        f"num_queries: {len(queries)}",
    ]
    render_text_panel("Sample summary", "\n".join(summary_lines)).save(sample_dir / "00_summary.png")

    for idx, query in enumerate(queries):
        box = tuple(float(v) for v in query["box"])
        codes = [int(v) for v in query["codes"]]
        overlay = draw_box(image, box)
        crop = crop_from_box(image, box)
        if args.visual_codec == "ibq":
            recon = decode_ibq_codes(codec, codes)
        else:
            recon = decode_open_magvit2_codes(codec, codes)

        info = "\n".join([
            f"query_index: {idx}",
            f"box: [{box[0]:.6f}, {box[1]:.6f}, {box[2]:.6f}, {box[3]:.6f}]",
            f"num_codes: {len(codes)}",
            "codes:",
            " ".join(f"<vis_{code}>" for code in codes),
        ])
        overview = make_query_overview(
            raw_text,
            replaced_text,
            overlay,
            crop,
            recon,
            info,
            max_text_width=args.max_text_width,
        )
        overview.save(sample_dir / f"query_{idx:02d}_overview.png")

    print(f"Saved visualization to: {sample_dir}")


if __name__ == "__main__":
    main()
