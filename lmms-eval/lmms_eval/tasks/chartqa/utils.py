import re

from lmms_eval.api.filter import Filter


def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_process_results(doc, results):
    pred = results[0]
    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


class FinalAnswerFilter(Filter):
    def _extract_final_answer(self, text: str) -> str:
        text = str(text).strip()
        if not text:
            return ""

        # Strip special tokens like <|im_end|>, <|endoftext|>
        text = re.sub(r"<\|[^<>|]+\|>", "", text)

        answer_tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        if answer_tags:
            text = answer_tags[-1].strip()
        elif "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()
        elif "</analysis>" in text:
            text = text.rsplit("</analysis>", 1)[-1].strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            text = lines[-1]

        match = re.search(r"(?i)(?:final answer|answer)\s*(?:is|:)\s*(.+)$", text)
        if match:
            text = match.group(1).strip()

        text = text.strip().strip("`").strip("*").strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[ \t]+([,.;:!?])$", r"\1", text)
        text = re.sub(r"[.;:,!?]+$", "", text).strip()
        return text

    def apply(self, resps, docs):
        filtered_resps = []
        for inst in resps:
            filtered_resps.append([self._extract_final_answer(resp) for resp in inst])
        return filtered_resps
