import re
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

from qwen35_hf import Qwen3_5ForConditionalGeneration, Qwen3_5Tokenizer, Qwen3VLProcessor
from qwen35_hf.train.data import VisionPacker


class LocalVisionImageProcessor(ImageProcessingMixin):
    """Small image processor wrapper around the local Qwen3.5 vision packer."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, vision_packer: VisionPacker) -> None:
        self.vision_packer = vision_packer
        self.merge_size = vision_packer.local_config.spatial_merge_size

    def __call__(self, images=None, **kwargs):
        if images is None:
            return {}
        if isinstance(images, Image.Image):
            images = [images]

        pixel_values = []
        image_grid_thw = []
        for image in images:
            packed_pixels, packed_grid = self.vision_packer.pack(image)
            pixel_values.append(packed_pixels)
            image_grid_thw.append(packed_grid)

        return {
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.stack(image_grid_thw, dim=0),
        }


class LocalNoOpVideoProcessor(BaseVideoProcessor):
    model_input_names = []

    def __call__(self, videos=None, **kwargs):
        if videos is not None:
            raise ValueError("qwen35_hf local adapter currently supports image inputs only.")
        return {}


@register_model("qwen35_hf")
class Qwen35HF(Qwen3_VL):
    """lmms-eval adapter for the local qwen35_hf implementation."""

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        peft: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        processor_backend: str = "local",
        image_aspect_ratio: str = "normal",
        image_grid_pinpoints: str | None = None,
        max_image_tokens: int | None = 128,
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(pretrained, **model_kwargs)
        if peft is not None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, peft)
        self._model = self._model.eval()

        self._tokenizer = Qwen3_5Tokenizer.from_pretrained(pretrained)
        vision_packer = VisionPacker(
            vision_config=self._model.config.vision_config,
            processor=None,
            processor_backend=processor_backend,
            image_aspect_ratio=image_aspect_ratio,
            image_grid_pinpoints=image_grid_pinpoints,
            max_image_tokens=max_image_tokens,
        )
        self.processor = Qwen3VLProcessor(
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
                        raise ValueError("qwen35_hf local adapter currently supports image inputs only.")
                    if isinstance(visual, Image.Image):
                        processed_visuals.append({"type": "image", "image": visual})

            if self.interleave_visuals is False:
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
                image_inputs.extend(part["image"] for part in processed_visuals)
            else:
                image_placeholders = re.findall(r"<image \d+>", context)
                content_parts = []
                text_parts = re.split(r"<image \d+>", context)
                if text_parts[0]:
                    content_parts.append({"type": "text", "text": text_parts[0]})

                for placeholder_idx, placeholder in enumerate(image_placeholders):
                    img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                    image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                    if processed_visuals and image_idx < len(processed_visuals):
                        content_parts.append(processed_visuals[image_idx])
                        image_inputs.append(processed_visuals[image_idx]["image"])
                    if placeholder_idx + 1 < len(text_parts) and text_parts[placeholder_idx + 1]:
                        content_parts.append({"type": "text", "text": text_parts[placeholder_idx + 1]})

                message.append(
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                )

            batched_messages.append(message)

        texts = self._apply_chat_template(batched_messages)
        processor_kwargs = {
            "text": texts,
            "images": image_inputs or None,
            "return_mm_token_type_ids": True,
            "return_tensors": "pt",
        }
        if self.batch_size > 1:
            processor_kwargs.update({"padding": True, "padding_side": "left"})
        inputs = self.processor(**processor_kwargs)

        return inputs, contexts, gen_kwargs, until
