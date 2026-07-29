import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.progress import make_progress
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL

from fovea_token import FoveaForConditionalGeneration
from fovea_token.train.data import VisionPacker
from fovea_token.train.sft import FOVEA_EXTRA_WEIGHTS_NAME, load_fovea_extra
from fovea_token.tokenizers.tokenization_fovea import FOVEA_TOOL_CALL, sync_fovea_trigger_ids


MODEL_WEIGHT_FILENAMES = {
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
FOVEA_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)
FOVEA_REASONING_TAG_PAIRS = (("<think>", "</think>"), ("<analysis>", "</analysis>"))
FOVEA_TRAILING_CHAT_TOKENS_RE = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>|</s>)\s*$")
FOVEA_BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FOVEA_TOOL_CALL_RE = re.compile(r'\{\s*"fovea"\s*\}')
FOVEA_JSON_ANSWER_RE = re.compile(r'\{\s*"(?:answer|option)"\s*:\s*"?([^"}\n]+)"?\s*\}', re.IGNORECASE)
FOVEA_CODE_OPTION_RE = re.compile(r"```(?:[A-Za-z]+)?\s*\n\s*([A-Ea-e])\s*\n?```\s*$")
FOVEA_ANSWER_MARKER_LINE_RE = re.compile(
    r"^\s*(?:\*{1,2})?(?:(?:the\s+)?final\s+(?:answer|value)|the\s+(?:correct\s+)?(?:answer|option)|(?:correct|best)\s+option|answer)(?:\*{1,2})?\s*(?:is)?\s*[:：]?\s*(.*)$",
    flags=re.IGNORECASE,
)
FOVEA_STANDALONE_OPTION_RE = re.compile(r"^\s*\*{0,2}\(?([A-Ea-e])\)?\*{0,2}[.,]?\s*$")


def add_think_prefill(texts: list[str], enabled: bool) -> list[str]:
    if not enabled:
        return texts
    return [text + "<think>\n" for text in texts]


def extract_fovea_final_answer(response: str) -> str:
    """Return the Fovea response text that benchmark scorers should consume."""

    response = FOVEA_TOOL_CALL_RE.sub("", response)
    answer_tags = FOVEA_ANSWER_TAG_RE.findall(response)
    if answer_tags:
        result = answer_tags[-1]
    else:
        result = response
        for start_tag, end_tag in FOVEA_REASONING_TAG_PAIRS:
            while start_tag in result and end_tag in result:
                start = result.find(start_tag)
                end = result.find(end_tag, start)
                result = result[:start] + result[end + len(end_tag) :]
            if end_tag in result and start_tag not in result:
                result = result.rsplit(end_tag, 1)[-1]

        json_answers = FOVEA_JSON_ANSWER_RE.findall(result)
        boxed_answers = FOVEA_BOXED_ANSWER_RE.findall(result)
        code_option = FOVEA_CODE_OPTION_RE.search(result)
        if json_answers:
            result = json_answers[-1]
        elif boxed_answers:
            result = boxed_answers[-1]
        elif code_option:
            result = code_option.group(1).upper()
        else:
            lines = result.splitlines()
            marked_answers = []
            for index, line in enumerate(lines):
                marker = FOVEA_ANSWER_MARKER_LINE_RE.fullmatch(line)
                if not marker:
                    continue
                answer = marker.group(1).strip()
                if not answer:
                    answer = next((candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()), "")
                if answer:
                    marked_answers.append(answer)

            if marked_answers:
                result = marked_answers[-1]
            else:
                standalone = next((line for line in reversed(lines) if line.strip()), "")
                option = FOVEA_STANDALONE_OPTION_RE.fullmatch(standalone)
                if option:
                    result = option.group(1).upper()

    return FOVEA_TRAILING_CHAT_TOKENS_RE.sub("", result).strip().rstrip(".,")


@register_model("fovea")
class Fovea(Qwen2_5_VL):
    """lmms-eval adapter for the local Fovea implementation."""

    preserve_raw_resps = True

    @staticmethod
    def postprocess_response_for_scoring(response: str, task_name: str | None = None) -> str:
        return extract_fovea_final_answer(response)

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
    def _is_lora_checkpoint_dir(cls, path: Optional[str]) -> bool:
        candidate = cls._as_local_path(path)
        return bool(candidate and candidate.is_dir() and (candidate / "adapter_config.json").exists())

    @classmethod
    def _checkpoint_mode(cls, pretrained: str) -> str:
        if cls._is_lora_checkpoint_dir(pretrained):
            return "lora"
        return "full" if cls._is_full_checkpoint_dir(pretrained) else "base"

    @staticmethod
    def _lora_base_model_name(adapter_path: str) -> str:
        adapter_config = Path(adapter_path).expanduser() / "adapter_config.json"
        data = json.loads(adapter_config.read_text())
        base_model = data.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"LoRA checkpoint {adapter_path} does not define base_model_name_or_path.")
        return base_model

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "sdpa",
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        prefill_think: Optional[bool] = False,
        max_image_tokens: int | None = 512,
        fovea_crop_min_image_tokens: int | None = 64,
        fovea_crop_max_image_tokens: int | None = 1024,
        fovea_crop_threshold: float = 0.25,
        fovea_crop_region_scale: float = 1.2,
        fovea_crop_image_scale: float = 2.0,
        disable_fovea_retrieval: Optional[bool] = False,
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

        checkpoint_mode = self._checkpoint_mode(pretrained)
        eval_logger.info(f"Resolved Fovea checkpoint mode: {checkpoint_mode}")
        model_kwargs = {
            "torch_dtype": self._pick_torch_dtype(),
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        tokenizer_source = pretrained
        if checkpoint_mode == "lora":
            from peft import PeftModel

            base_model_name = self._lora_base_model_name(pretrained)
            tokenizer_source = base_model_name
            self._model = FoveaForConditionalGeneration.from_pretrained(base_model_name, **model_kwargs)
            if (Path(pretrained).expanduser() / FOVEA_EXTRA_WEIGHTS_NAME).exists():
                load_fovea_extra(self._model, pretrained)
            self._model = PeftModel.from_pretrained(self._model, pretrained)
        else:
            self._model = FoveaForConditionalGeneration.from_pretrained(pretrained, **model_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        fovea_model = self._model.get_base_model() if checkpoint_mode == "lora" else self._model
        sync_fovea_trigger_ids(fovea_model.config, self._tokenizer)
        fovea_model.config.image_token_id = self._tokenizer.convert_tokens_to_ids("<|image_pad|>")
        fovea_model.config.video_token_id = self._tokenizer.convert_tokens_to_ids("<|video_pad|>")
        fovea_model.config.vision_start_token_id = self._tokenizer.convert_tokens_to_ids("<|vision_start|>")
        fovea_model.config.vision_end_token_id = self._tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self._model = self._model.eval()
        self.processor = AutoProcessor.from_pretrained(tokenizer_source)
        self.processor.tokenizer = self._tokenizer
        vision_packer = VisionPacker(
            processor=self.processor,
            vision_config=fovea_model.config.vision_config,
            max_image_tokens=max_image_tokens,
        )
        self.vision_packer = vision_packer
        fovea_model.config.fovea_crop_max_image_tokens = int(fovea_crop_max_image_tokens)
        fovea_model.config.fovea_crop_min_image_tokens = int(fovea_crop_min_image_tokens)
        fovea_model.config.fovea_crop_threshold = float(fovea_crop_threshold)
        fovea_model.config.fovea_crop_region_scale = float(fovea_crop_region_scale)
        fovea_model.config.fovea_crop_image_scale = float(fovea_crop_image_scale)
        self.disable_fovea_retrieval = bool(disable_fovea_retrieval)

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.prefill_think = bool(prefill_think)

        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self._config = fovea_model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.log_responses = os.environ.get("LMMS_EVAL_FOVEA_LOG_RESPONSES", "").lower() in {"1", "true", "yes"}
        self.log_response_chars = int(os.environ.get("LMMS_EVAL_FOVEA_LOG_RESPONSE_CHARS", "500"))

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
        current = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            **gen_kwargs,
        }
        do_sample = current.get("temperature", 0) > 0
        generate_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": do_sample,
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            value = current.get(key)
            if value is not None and (do_sample or key == "num_beams"):
                generate_kwargs[key] = value
        if self.disable_fovea_retrieval:
            generate_kwargs["disable_fovea_retrieval"] = True
        return generate_kwargs

    def _apply_chat_template(self, batched_messages):
        return self.processor.apply_chat_template(
            batched_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_processor_image_kwargs(self):
        max_image_tokens = getattr(self.vision_packer, "max_image_tokens", None)
        if max_image_tokens is None:
            return {}

        image_processor = self.processor.image_processor
        processor_kwargs = {
            "max_pixels": (
                int(max_image_tokens)
                * int(self.vision_packer.spatial_merge_size) ** 2
                * int(self.vision_packer.patch_size) ** 2
            )
        }
        min_pixels = getattr(image_processor, "min_pixels", None)
        if min_pixels is None:
            size = getattr(image_processor, "size", None)
            if size is not None:
                min_pixels = size.get("shortest_edge")
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        return processor_kwargs

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

        texts = add_think_prefill(self._apply_chat_template(batched_messages), self.prefill_think)
        processor_kwargs = {
            "text": texts,
            "images": image_inputs or None,
            "return_mm_token_type_ids": True,
            "return_tensors": "pt",
        }
        processor_kwargs.update(self._build_processor_image_kwargs())
        if self.batch_size > 1:
            processor_kwargs.update({"padding": True, "padding_side": "left"})
        inputs = self.processor(**processor_kwargs)
        return inputs, contexts, gen_kwargs, until, image_inputs, doc_id, task, split

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = make_progress(
            total=len(requests),
            disable=(self.rank != 0),
            desc=f"Rank {self.rank} Responding",
        )
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=1, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._preprocess_chunk, chunks[0]) if chunks else None

            for idx in range(len(chunks)):
                inputs, contexts, gen_kwargs, until, image_inputs, doc_ids, tasks, splits = future.result()
                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                started_at = time.perf_counter()
                cont = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                    source_images=image_inputs,
                    image_processor=self.processor.image_processor,
                )
                elapsed = time.perf_counter() - started_at
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

                for ans, context, generated_ids, doc_id, task, split in zip(answers, contexts, generated_ids_trimmed, doc_ids, tasks, splits):
                    if self.log_responses:
                        snippet = ans.replace("\n", "\\n")[: self.log_response_chars]
                        eval_logger.info(
                            "Rank {} doc={} task={} split={} response_tokens={} chars={} fovea_count={} elapsed={:.2f}s text={}",
                            self.rank,
                            doc_id,
                            task,
                            split,
                            int(generated_ids.numel()),
                            len(ans),
                            ans.count(FOVEA_TOOL_CALL),
                            elapsed,
                            snippet,
                        )
                    res.append(
                        GenerationResult(
                            text=ans,
                            token_counts=TokenCounts(output_tokens=len(generated_ids)),
                        )
                    )
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
