import json
import re

from loguru import logger

from lmms_eval.api.metrics import anls
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def _clean_docvqa_prediction(text: str) -> str:
    text = str(text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    else:
        text = re.sub(r"(?is)^\s*<think>\s*", "", text, count=1)
    text = text.replace("<|im_end|>", " ")
    text = text.replace("<|endoftext|>", " ")
    text = re.sub(r"(?is)^\s*(?:final\s+answer|answer)\s*[:：]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def docvqa_val_process_results(doc, results):
    pred = _clean_docvqa_prediction(results[0])
    return anls(references=doc["answers"], predictions=[pred])


def docvqa_test_process_results(doc, results):
    pred = _clean_docvqa_prediction(results[0])
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}, "submission": {"questionId": int(questionId), "answer": pred}}


def docvqa_test_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("docvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")
