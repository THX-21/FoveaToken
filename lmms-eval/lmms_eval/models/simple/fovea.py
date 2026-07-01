import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

from fovea_token import FoveaForConditionalGeneration
from fovea_token.train.data import VisionPacker
from fovea_token.tokenizers.tokenization_fovea import sync_fovea_token_ids


MODEL_WEIGHT_FILENAMES = {
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


@register_model("fovea")
class Fovea(Qwen3_VL):
    """lmms-eval adapter for the local Fovea implementation."""

    @staticmethod
    def _pick_torch_dtype():
        if torch.cuda.is_available():
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "float32"

    @staticmethod
    def _as_local_path(path: Optional[str]) -> Optional[Path]:
        if not path:
            return None
        candidate = Path(path).expanduser()
        return candidate if candidate.exists() else None

    @classmethod
    def _is_full_checkpoint_dir(cls, path: Optional[str]) -> bool:
        candidate = cls._as_local_path(path)
        if not candidate or not candidate.is_dir() or not (candidate / "config.json").exists():
            return False
        return any((candidate / filename).exists() for filename in MODEL_WEIGHT_FILENAMES)

    @classmethod
    def _checkpoint_mode(cls, pretrained: str) -> str:
        return "full" if cls._is_full_checkpoint_dir(pretrained) else "base"

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "sdpa",
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        max_image_tokens: int | None = 512,
        fovea_crop_max_image_tokens: int | None = 1024,
        disable_fovea_retrieval: Optional[bool] = False,
        fovea_use_aux_head: Optional[bool] = False,
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
            resolved_device = device
            if resolved_device in {None, "cuda"} and device_map not in {None, "auto"}:
                resolved_device = device_map
            self._device = torch.device(resolved_device)
            self.device_map = device_map if device_map else resolved_device

        eval_logger.info(f"Resolved Fovea checkpoint mode: {self._checkpoint_mode(pretrained)}")
        model_kwargs = {
            "torch_dtype": self._pick_torch_dtype(),
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self._model = FoveaForConditionalGeneration.from_pretrained(pretrained, **model_kwargs)
        tokenizer_source = pretrained
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        sync_fovea_token_ids(self._model.config, self._tokenizer, self._model)
        self._model.config.image_token_id = self._tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self._model.config.video_token_id = self._tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self._model.config.vision_start_token_id = self._tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self._model.config.vision_end_token_id = self._tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self._model = self._model.eval()
        self.processor = AutoProcessor.from_pretrained(tokenizer_source)
        self.processor.tokenizer = self._tokenizer
        vision_packer = VisionPacker(
            processor=self.processor,
            vision_config=self._model.config.vision_config,
            max_image_tokens=max_image_tokens,
        )
        self.vision_packer = vision_packer
        self._model.config.fovea_crop_max_image_tokens = int(fovea_crop_max_image_tokens)
        self._model.config.fovea_use_aux_head = bool(fovea_use_aux_head)
        self.disable_fovea_retrieval = bool(disable_fovea_retrieval)

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

    def _build_generate_kwargs(self, gen_kwargs):
        generate_kwargs = super()._build_generate_kwargs(gen_kwargs)
        if self.disable_fovea_retrieval:
            generate_kwargs["disable_fovea_retrieval"] = True
        return generate_kwargs

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

            if self.interleave_visuals is False:
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
            "return_mm_token_type_ids": True,
            "return_tensors": "pt",
        }
        if self.batch_size > 1:
            processor_kwargs.update({"padding": True, "padding_side": "left"})
        inputs = self.processor(**processor_kwargs)
        return inputs, contexts, gen_kwargs, until, image_inputs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=1, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._preprocess_chunk, chunks[0]) if chunks else None

            for idx in range(len(chunks)):
                inputs, contexts, gen_kwargs, until, image_inputs = future.result()
                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                cont = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                    source_images=image_inputs,
                    image_processor=self.processor.image_processor,
                )
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
