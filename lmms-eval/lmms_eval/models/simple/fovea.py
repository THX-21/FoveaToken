import re
from pathlib import Path
from typing import List, Optional, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.image_processing_utils import ImageProcessingMixin

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

from qwen35_hf import FoveaForConditionalGeneration, FoveaTokenizer, FoveaProcessor
from qwen35_hf.train.data import VisionPacker


class LocalVisionImageProcessor(ImageProcessingMixin):
    """Small image processor wrapper around the local Qwen3.5 vision packer."""

    model_input_names = ["pixel_values", "image_grid_thw", "image_block_offsets"]

    def __init__(self, vision_packer: VisionPacker) -> None:
        self.vision_packer = vision_packer
        self.merge_size = vision_packer.local_config.spatial_merge_size
        self.img_slot_token_count = vision_packer.img_slot_token_count if vision_packer.img_slot_enable else None
        self._last_image_block_counts = []

    def __call__(self, images=None, **kwargs):
        if images is None:
            return {}
        if isinstance(images, Image.Image):
            images = [images]

        pixel_values = []
        image_grid_thw = []
        image_block_offsets = []
        image_block_counts = []
        for image_index, image in enumerate(images):
            packed = self.vision_packer.pack(image)
            if self.vision_packer.img_slot_enable:
                packed_pixels, packed_grid, packed_offsets = packed
                packed_offsets = packed_offsets.clone()
                packed_offsets[:, 0] = image_index
                image_block_offsets.extend(list(packed_offsets))
            else:
                packed_pixels, packed_grid = packed
            pixel_values.append(packed_pixels)
            if packed_grid.dim() == 1:
                image_grid_thw.append(packed_grid)
                image_block_counts.append(1)
            else:
                image_grid_thw.extend(list(packed_grid))
                image_block_counts.append(int(packed_grid.shape[0]))

        self._last_image_block_counts = image_block_counts

        model_inputs = {
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.stack(image_grid_thw, dim=0),
        }
        if image_block_offsets:
            model_inputs["image_block_offsets"] = torch.stack(image_block_offsets, dim=0)
        return model_inputs


class LocalNoOpVideoProcessor(BaseVideoProcessor):
    model_input_names = []

    def __call__(self, videos=None, **kwargs):
        if videos is not None:
            raise ValueError("Fovea local adapter currently supports image inputs only.")
        return {}


@register_model("fovea")
class Fovea(Qwen3_VL):
    """lmms-eval adapter for the local Fovea implementation."""

    @staticmethod
    def _pick_torch_dtype():
        if torch.cuda.is_available():
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "float32"

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        peft: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "sdpa",
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        max_image_tokens: int | None = 128,
        img_slot_enable: bool = True,
        img_slot_k: int | None = None,
        img_slot_delta: int | None = None,
        img_slot_beta: float | None = None,
        img_slot_lambda: float | None = None,
        img_slot_tile_size: int | None = 1024,
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")
        if isinstance(img_slot_enable, str):
            img_slot_enable = img_slot_enable.lower() in {"1", "true", "yes"}
        if img_slot_k is not None:
            img_slot_k = int(img_slot_k)
        if img_slot_delta is not None:
            img_slot_delta = int(img_slot_delta)
        if img_slot_beta is not None:
            img_slot_beta = float(img_slot_beta)
        if img_slot_lambda is not None:
            img_slot_lambda = float(img_slot_lambda)
        if img_slot_enable and img_slot_tile_size is None:
            raise ValueError("img_slot_tile_size is required when img_slot_enable=true.")
        self.img_slot_enable = bool(img_slot_enable)

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            resolved_device = device
            if resolved_device in {None, "cuda"} and device_map not in {None, "auto"}:
                resolved_device = device_map
            self._device = torch.device(resolved_device)
            self.device_map = device_map if device_map else resolved_device

        model_kwargs = {
            "torch_dtype": self._pick_torch_dtype(),
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if self.img_slot_enable:
            model_kwargs["img_slot_enable"] = True
        if img_slot_k is not None:
            model_kwargs["img_slot_k"] = img_slot_k
        if img_slot_delta is not None:
            model_kwargs["img_slot_delta"] = img_slot_delta
        if img_slot_beta is not None:
            model_kwargs["img_slot_beta"] = img_slot_beta
        if img_slot_lambda is not None:
            model_kwargs["img_slot_lambda"] = img_slot_lambda
        if img_slot_tile_size is not None:
            model_kwargs["img_slot_tile_size"] = int(img_slot_tile_size)

        self._model = FoveaForConditionalGeneration.from_pretrained(pretrained, **model_kwargs)
        self._model.config.img_slot_enable = self.img_slot_enable
        if img_slot_tile_size is not None:
            self._model.config.img_slot_tile_size = int(img_slot_tile_size)
        config_img_slot_k = int(self._model.config.img_slot_k)
        if peft is not None:
            from peft import PeftModel

            self._load_deepspeed_trainables(peft)
            self._model = PeftModel.from_pretrained(self._model, peft)
        self._model = self._model.eval()

        self._tokenizer = FoveaTokenizer.from_pretrained(pretrained)
        vision_packer = VisionPacker(
            vision_config=self._model.config.vision_config,
            max_image_tokens=max_image_tokens,
            img_slot_enable=self.img_slot_enable,
            img_slot_k=config_img_slot_k,
            img_slot_tile_size=None if img_slot_tile_size is None else int(img_slot_tile_size),
        )
        self.processor = FoveaProcessor(
            image_processor=LocalVisionImageProcessor(vision_packer),
            tokenizer=self._tokenizer,
            video_processor=LocalNoOpVideoProcessor(),
            chat_template=getattr(self._tokenizer, "chat_template", None),
        )

        self.enable_thinking = enable_thinking
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _load_deepspeed_trainables(self, checkpoint_path: str) -> None:
        """Load non-LoRA trainables saved in the Deepspeed checkpoint.

        PEFT's adapter file contains LoRA and ImgSlot modules_to_save tensors.
        The vision tower is still stored in Deepspeed's model state file under
        ``global_step*/mp_rank_00_model_states.pt``.
        """

        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_dir():
            return

        latest_file = checkpoint / "latest"
        if latest_file.exists():
            global_step_name = latest_file.read_text().strip()
            global_step_dir = checkpoint / global_step_name
        else:
            global_steps = sorted(checkpoint.glob("global_step*"))
            global_step_dir = global_steps[-1] if global_steps else None
        if global_step_dir is None:
            return

        state_path = global_step_dir / "mp_rank_00_model_states.pt"
        if not state_path.exists():
            return

        eval_logger.info(f"Loading non-LoRA trainables from {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        module_state = state.get("module", {})
        trainable_state = {}
        prefix = "base_model.model."
        for key, tensor in module_state.items():
            if not key.startswith(prefix):
                continue
            raw_key = key[len(prefix) :]
            if raw_key.startswith("model.visual"):
                trainable_state[raw_key] = tensor

        if not trainable_state:
            eval_logger.warning(f"No vision trainables found in {state_path}")
            return

        incompatible = self._model.load_state_dict(trainable_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            eval_logger.warning(f"Unexpected non-LoRA trainable keys while loading {state_path}: {unexpected[:20]}")
        eval_logger.info(f"Loaded {len(trainable_state) - len(unexpected)} vision tensors from Deepspeed checkpoint")

    def _preprocess_chunk(self, chunk):
        """Build prompts and local processor inputs without HF image processing."""

        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
        until = [item for item in until if item != "\n\n"]

        if isinstance(contexts, tuple):
            contexts = list(contexts)

        batched_messages = []
        image_inputs = []
        image_counts_per_sample = []
        logical_image_placeholder = self.processor.build_visual_placeholder(1)
        for i, context in enumerate(contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            message = [{"role": "system", "content": self.system_prompt}]

            if self.reasoning_prompt:
                context = context.strip() + self.reasoning_prompt
                contexts[i] = context

            processed_visuals = []
            if visual_list[i] is not None:
                sample_visuals = [visual_list[i]] if isinstance(visual_list[i], (Image.Image, str)) else visual_list[i]
                for visual in sample_visuals:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        raise ValueError("Fovea local adapter currently supports image inputs only.")
                    if isinstance(visual, Image.Image):
                        processed_visuals.append({"type": "image", "image": visual})

            if self.img_slot_enable:
                sample_text = context
                sample_image_count = len(processed_visuals)
                if self.interleave_visuals:
                    image_placeholders = re.findall(r"<image \d+>", context)
                    text_parts = re.split(r"<image \d+>", context)
                    sample_text = text_parts[0] if text_parts else ""
                    ordered_images = []
                    for placeholder_idx, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            ordered_images.append(processed_visuals[image_idx]["image"])
                            sample_text += logical_image_placeholder
                        if placeholder_idx + 1 < len(text_parts):
                            sample_text += text_parts[placeholder_idx + 1]
                    image_inputs.extend(ordered_images)
                    sample_image_count = len(ordered_images)
                else:
                    sample_text = logical_image_placeholder * sample_image_count + context
                    image_inputs.extend(part["image"] for part in processed_visuals)

                message.append(
                    {
                        "role": "user",
                        "content": sample_text,
                    }
                )
                image_counts_per_sample.append(sample_image_count)
            elif self.interleave_visuals is False:
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
                image_inputs.extend(part["image"] for part in processed_visuals)
                image_counts_per_sample.append(len(processed_visuals))
            else:
                image_placeholders = re.findall(r"<image \d+>", context)
                content_parts = []
                text_parts = re.split(r"<image \d+>", context)
                sample_image_count = 0
                if text_parts[0]:
                    content_parts.append({"type": "text", "text": text_parts[0]})

                for placeholder_idx, placeholder in enumerate(image_placeholders):
                    img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                    image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                    if processed_visuals and image_idx < len(processed_visuals):
                        content_parts.append(processed_visuals[image_idx])
                        image_inputs.append(processed_visuals[image_idx]["image"])
                        sample_image_count += 1
                    if placeholder_idx + 1 < len(text_parts) and text_parts[placeholder_idx + 1]:
                        content_parts.append({"type": "text", "text": text_parts[placeholder_idx + 1]})

                message.append(
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                )
                image_counts_per_sample.append(sample_image_count)

            batched_messages.append(message)

        texts = self._apply_chat_template(batched_messages)
        processor_kwargs = {
            "text": texts,
            "images": image_inputs or None,
            "image_counts_per_sample": image_counts_per_sample,
            "return_mm_token_type_ids": not self.img_slot_enable,
            "return_tensors": "pt",
        }
        if self.batch_size > 1:
            processor_kwargs.update({"padding": True, "padding_side": "left"})
        inputs = self.processor(**processor_kwargs)
        if self.img_slot_enable:
            inputs.pop("mm_token_type_ids", None)
            gen_kwargs = dict(gen_kwargs)

        return inputs, contexts, gen_kwargs, until
