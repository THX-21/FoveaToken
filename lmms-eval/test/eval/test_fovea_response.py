from types import SimpleNamespace

from lmms_eval.evaluator import _postprocess_response_for_scoring
from lmms_eval.loggers.evaluation_tracker import _drop_redundant_resps
from lmms_eval.models.simple.fovea import add_think_prefill, extract_fovea_final_answer


def test_fovea_answer_tag_wins_over_reasoning():
    response = "<think>reasoning A B C</think>\n<answer>C</answer><|im_end|>"
    assert extract_fovea_final_answer(response) == "C"


def test_fovea_answer_falls_back_after_thinking():
    response = "<think>reasoning</think>\nThe answer is: B D<|im_end|>"
    assert extract_fovea_final_answer(response) == "B D"


def test_fovea_answer_extracts_final_answer_prefix():
    assert extract_fovea_final_answer("<think>reasoning</think>\nFinal Answer: B") == "B"


def test_fovea_answer_ignores_tool_calls_after_final_answer():
    response = "reasoning\nFinal Answer: A\n{\"fovea\"}\n"
    assert extract_fovea_final_answer(response) == "A"


def test_fovea_answer_extracts_option_and_boxed_forms():
    assert extract_fovea_final_answer("The correct option is (C).") == "(C)"
    assert extract_fovea_final_answer("reasoning\\boxed{B}") == "B"
    assert extract_fovea_final_answer('reasoning {"answer": A}') == "A"
    assert extract_fovea_final_answer("reasoning\n```python\nD\n```") == "D"


def test_fovea_answer_reads_option_on_the_line_after_marker():
    response = "reasoning\nFinal Answer\n**A**\n{\"fovea\"}"
    assert extract_fovea_final_answer(response) == "**A**"


def test_fovea_answer_extracts_final_value_variants():
    assert extract_fovea_final_answer("The final answer is $10.4$.") == "$10.4$"
    assert extract_fovea_final_answer("Final value: 0.21") == "0.21"


def test_fovea_answer_keeps_plain_response():
    assert extract_fovea_final_answer("A plain response<|im_end|>") == "A plain response"


def test_evaluator_uses_fovea_postprocessor_only_when_available():
    fovea_model = SimpleNamespace(postprocess_response_for_scoring=lambda response, task_name: f"{task_name}:" + response)
    plain_model = SimpleNamespace()
    assert _postprocess_response_for_scoring(fovea_model, "raw", [["<think>", "</think>"]], "chartqa") == "chartqa:raw"
    assert _postprocess_response_for_scoring(plain_model, "<think>x</think> answer", [["<think>", "</think>"]]) == "answer"


def test_fovea_log_sample_keeps_raw_response():
    sample = {"resps": "<think>x</think><answer>A</answer>", "filtered_resps": "A", "preserve_raw_resps": True}
    _drop_redundant_resps(sample)
    assert sample["resps"] == "<think>x</think><answer>A</answer>"
    assert "preserve_raw_resps" not in sample


def test_fovea_prefill_is_input_only():
    prompt = "<|im_start|>assistant\n"
    assert add_think_prefill([prompt], enabled=True) == ["<|im_start|>assistant\n<think>\n"]
    assert add_think_prefill([prompt], enabled=False) == [prompt]
