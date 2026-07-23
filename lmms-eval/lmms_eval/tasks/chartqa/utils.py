import re


_RATIO_RE = re.compile(r"^\s*(-?\d[\d,]*(?:\.\d+)?)\s*[:/]\s*(-?\d[\d,]*(?:\.\d+)?)\s*$")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
_JSON_ANSWER_RE = re.compile(r'^\s*\{\s*"answer"\s*:\s*"?(-?\d[\d,]*(?:\.\d+)?%?)"?\s*\}\s*$', re.IGNORECASE)


def normalize_chartqa_answer(answer):
    """Normalize standalone numeric answers without interpreting prose."""

    if not isinstance(answer, str):
        return answer

    answer = answer.strip()
    ratio_match = _RATIO_RE.fullmatch(answer)
    if ratio_match:
        numerator = float(ratio_match.group(1).replace(",", ""))
        denominator = float(ratio_match.group(2).replace(",", ""))
        if denominator:
            return str(numerator / denominator)

    json_match = _JSON_ANSWER_RE.fullmatch(answer)
    if json_match:
        answer = json_match.group(1)

    if not _NUMBER_RE.fullmatch(answer):
        return answer
    return answer.rstrip("%").replace(",", "")


def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_process_results(doc, results):
    pred = normalize_chartqa_answer(results[0])
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

    prediction_float = _to_float(normalize_chartqa_answer(prediction))
    target_float = _to_float(normalize_chartqa_answer(target))
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
