import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from safetensors.torch import load_file as safe_load_file
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from fovea_token import FoveaForConditionalGeneration
from fovea_token.configuration_fovea import sync_expanded_image_grid_pinpoints
from fovea_token.train.data import VisionPacker
from fovea_token.train.sft import get_visual_module
from fovea_token.tokenizers.tokenization_fovea import add_fovea_tokens, sync_fovea_token_ids


VISION_TOWER_WEIGHTS_NAME = "vision_tower.safetensors"
MODEL_WEIGHT_FILENAMES = {
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


@register_model("fovea")
class Fovea(lmms):
    """lmms-eval adapter for the local Fovea implementation."""

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

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
    def _is_peft_adapter_dir(cls, path: Optional[str]) -> bool:
        candidate = cls._as_local_path(path)
        return bool(candidate and (candidate / "adapter_config.json").exists())

    @classmethod
    def _is_full_checkpoint_dir(cls, path: Optional[str]) -> bool:
        candidate = cls._as_local_path(path)
        if not candidate or not candidate.is_dir() or not (candidate / "config.json").exists():
            return False
        return any((candidate / filename).exists() for filename in MODEL_WEIGHT_FILENAMES)

    @classmethod
    def _resolve_checkpoint_sources(cls, pretrained: str, peft: Optional[str]) -> tuple[str, Optional[str], str]:
        if peft is not None:
            return pretrained, peft, "peft"
        if cls._is_peft_adapter_dir(pretrained):
            raise ValueError(
                "`pretrained` points to a PEFT adapter directory. Pass the base model as `pretrained` and the adapter path as `peft`."
            )
        if cls._is_full_checkpoint_dir(pretrained):
            return pretrained, None, "full"
        return pretrained, None, "base"

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
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
        max_image_tokens: int | None = 512,
        use_fovea: Optional[bool] = False,
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

        load_pretrained, load_peft, checkpoint_mode = self._resolve_checkpoint_sources(pretrained, peft)
        eval_logger.info(f"Resolved Fovea checkpoint mode: {checkpoint_mode}")
        model_kwargs = {
            "torch_dtype": self._pick_torch_dtype(),
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self._model = FoveaForConditionalGeneration.from_pretrained(load_pretrained, **model_kwargs)
        tokenizer_source = load_pretrained
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        add_fovea_tokens(self._tokenizer)
        if len(self._tokenizer) != self._model.get_input_embeddings().weight.shape[0]:
            self._model.resize_token_embeddings(len(self._tokenizer))
        sync_fovea_token_ids(self._model.config, self._tokenizer)
        if load_peft is not None:
            from peft import PeftModel

            self._load_deepspeed_trainables(load_peft)
            self._model = PeftModel.from_pretrained(self._model, load_peft)
            base_model = self._model.get_base_model()
            if hasattr(base_model, "_repair_fovea_init"):
                base_model._repair_fovea_init()
            sync_fovea_token_ids(base_model.config, self._tokenizer)
        self._model = self._model.eval()
        self.processor = AutoProcessor.from_pretrained(tokenizer_source)
        self.processor.tokenizer = self._tokenizer
        sync_expanded_image_grid_pinpoints(self._model.config, self.processor)
        self.processor.config = self._model.config
        self.processor.patch_size = getattr(self._model.config.vision_config, "patch_size", self.processor.patch_size)
        self.processor.vision_feature_select_strategy = self._model.config.vision_feature_select_strategy
        self._model.config.image_token_index = self._tokenizer.convert_tokens_to_ids(getattr(self.processor, "image_token", "<image>"))
        vision_packer = VisionPacker(
            processor=self.processor,
            vision_config=self._model.config.vision_config,
            max_image_tokens=max_image_tokens,
        )
        self.vision_packer = vision_packer
        self.processor._get_number_of_features = self._get_pooled_number_of_features
        self.use_fovea = use_fovea

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

    def _apply_chat_template(self, batched_messages):
        texts = []
        for messages in batched_messages:
            messages = self._normalize_messages_for_template(messages)
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(self._apply_thinking_prefill(text))
                continue
            parts = []
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "system":
                    parts.append(str(content).strip())
                    continue
                prefix = "USER: " if role == "user" else "ASSISTANT: "
                if isinstance(content, list):
                    text = "".join("<image>\n" if item.get("type") == "image" else str(item.get("text", "")) for item in content)
                else:
                    text = str(content)
                parts.append(f"{prefix}{text}")
            text = "\n".join(part for part in parts if part) + "\nASSISTANT: "
            texts.append(self._apply_thinking_prefill(text))
        return texts

    @staticmethod
    def _normalize_messages_for_template(messages):
        normalized = []
        for message in messages:
            item = dict(message)
            content = item.get("content", "")
            if isinstance(content, str):
                item["content"] = [{"type": "text", "text": content}]
            normalized.append(item)
        return normalized

    def _apply_thinking_prefill(self, text: str) -> str:
        if self.enable_thinking:
            return text + "<think>"
        return text + "<think>\n\n</think>"

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Fovea")

    def _build_eos_token_ids(self):
        eos_token_ids = []
        for token_id in (self.tokenizer.eos_token_id,):
            if token_id is None or token_id < 0 or token_id in eos_token_ids:
                continue
            eos_token_ids.append(int(token_id))
        return eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids

    def _get_pooled_number_of_features(self, orig_height: int, orig_width: int, _height: int, _width: int) -> int:
        count = self.vision_packer.input_feature_count((orig_height, orig_width))
        if getattr(self.processor, "vision_feature_select_strategy", "default") == "default":
            count += 1
        return count

    def _build_generate_kwargs(self, gen_kwargs):
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        pad_token_id = self.tokenizer.pad_token_id
        if current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)
        generate_kwargs = {
            "eos_token_id": self._build_eos_token_ids(),
            "pad_token_id": pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": current["do_sample"],
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            val = current.get(key)
            if val is not None:
                generate_kwargs[key] = val
        return generate_kwargs

    def _strip_thinking(self, answer):
        return answer

    def _load_deepspeed_trainables(self, checkpoint_path: str) -> None:
        """Load legacy non-LoRA vision trainables saved alongside a LoRA checkpoint."""

        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_dir():
            return

        safe_path = checkpoint / VISION_TOWER_WEIGHTS_NAME
        if safe_path.exists():
            eval_logger.info(f"Loading vision tower weights from {safe_path}")
            vision_module = get_visual_module(self._model)
            vision_state = safe_load_file(str(safe_path))
            vision_module.load_state_dict(vision_state, strict=True)
            eval_logger.info(f"Loaded {len(vision_state)} vision tensors from {safe_path}")
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
            "return_tensors": "pt",
        }
        if self.batch_size > 1:
            processor_kwargs.update({"padding": True, "padding_side": "left"})
        inputs = self.processor(**processor_kwargs)
        retrieve_pixels = []
        retrieve_sizes = []
        retrieve_boxes = []
        retrieve_counts = []
        image_cursor = 0
        for count in image_counts_per_sample:
            retrieve_counts.append(count)
            for image in image_inputs[image_cursor : image_cursor + count]:
                pixels, image_sizes, boxes = self.vision_packer.pack_retrieve(image)
                retrieve_pixels.append(pixels)
                retrieve_sizes.append(image_sizes)
                retrieve_boxes.append(boxes)
            image_cursor += count
        if retrieve_pixels:
            inputs["retrieve_pixel_values"] = torch.cat(retrieve_pixels, dim=0)
            inputs["retrieve_image_sizes"] = torch.cat(retrieve_sizes, dim=0)
            inputs["retrieve_patch_boxes"] = torch.cat(retrieve_boxes, dim=0)
            inputs["retrieve_image_counts"] = torch.tensor(retrieve_counts, dtype=torch.long)

        return inputs, contexts, gen_kwargs, until

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
                inputs, contexts, gen_kwargs, until = future.result()
                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                cont = self.model.generate(**inputs, **generate_kwargs)
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    ans = self._strip_thinking(ans)
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Fovea")
