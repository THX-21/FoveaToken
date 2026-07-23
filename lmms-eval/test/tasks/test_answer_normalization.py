from lmms_eval.tasks.chartqa.utils import normalize_chartqa_answer, relaxed_correctness
from lmms_eval.tasks.mathvista.mathvista_evals import MathVistaEvaluator, extract_standalone_number, extract_terminal_answer
from lmms_eval.tasks.textvqa.utils import normalize_textvqa_answer
from lmms_eval.tasks.xlrs.mcq_utils import extract_characters_regex


def test_chartqa_normalizes_standalone_numeric_formats():
    assert normalize_chartqa_answer("96.4%") == "96.4"
    assert normalize_chartqa_answer("1,541.08") == "1541.08"
    assert normalize_chartqa_answer("62:29") == str(62 / 29)
    assert normalize_chartqa_answer('{"answer": 43.4}') == "43.4"
    assert relaxed_correctness("96.4%", "96.4")


def test_chartqa_does_not_parse_prose_with_multiple_numbers():
    answer = "The values are 31.16 and 30.68."
    assert normalize_chartqa_answer(answer) == answer


def test_xlrs_uses_the_last_explicit_answer_only():
    assert extract_characters_regex("Reasoning mentions A and C. Final Answer: B") == "B"
    assert extract_characters_regex("{\"answer\": \"D\"}") == "D"
    assert extract_characters_regex("The result is unclear") == ""


def test_mathvista_extracts_explicit_terminal_answers():
    assert extract_terminal_answer("work\nFinal Answer: $1,541.08$<|im_end|>") == "$1,541.08$"
    assert extract_standalone_number("\\boxed{197.3}") == "197.3"
    evaluator = MathVistaEvaluator.__new__(MathVistaEvaluator)
    problem = {"question_type": "free_form", "answer_type": "float", "choices": [], "query": ""}
    assert evaluator.extract_answer("The calculation is complete. Final Answer: $13.8$", problem) == "13.8"
    multiple_choice = {"question_type": "multi_choice", "answer_type": "text", "choices": ["yes", "no"], "query": ""}
    assert evaluator.extract_answer("A", multiple_choice) == "A"
    assert evaluator.extract_answer("(B) no", multiple_choice) == "B"
    assert evaluator.safe_equal("51.05", "51.04", precision=2)
    assert not evaluator.safe_equal("51.06", "51.04", precision=2)


def test_textvqa_removes_explicit_answer_wrappers_only():
    assert normalize_textvqa_answer("<answer>Coca-Cola</answer><|im_end|>") == "Coca-Cola"
    assert normalize_textvqa_answer("Final Answer: Thai Airways") == "Thai Airways"
