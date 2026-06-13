import re
from typing import List, Optional, Union


def strip_reasoning_tags(text: str, tag_pairs: List[List[str]]) -> str:
    """Remove reasoning tag blocks from model output.

    Args:
        text: Raw model output string
        tag_pairs: List of [start_tag, end_tag] pairs,
                   e.g. [["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]

    Returns:
        Cleaned text with reasoning blocks removed.
    """
    result = text
    for start_tag, end_tag in tag_pairs:
        while start_tag in result and end_tag in result:
            start = result.find(start_tag)
            end = result.find(end_tag, start)
            if start != -1 and end != -1:
                result = result[:start] + result[end + len(end_tag) :]
            else:
                break
        # Some chat templates prefill the opening reasoning tag in the prompt,
        # so the model completion may contain only the closing tag plus the
        # final answer. In that case, keep the suffix after the final closing
        # tag so downstream scorers see the answer instead of the reasoning.
        if end_tag in result and start_tag not in result:
            result = result.rsplit(end_tag, 1)[-1]
    return result.strip()


def parse_reasoning_tags_config(cli_value: Optional[str] = None, task_value: Optional[object] = None) -> Optional[List[List[str]]]:
    """Resolve reasoning_tags from CLI + task config.

    Priority: task_value > cli_value.
    "none" / None = disabled.
    """
    import json

    effective = task_value if task_value is not None else cli_value
    if effective is None or effective == "none" or effective is False:
        return None
    if isinstance(effective, str):
        return json.loads(effective)
    return effective


def restore_prefilled_reasoning_prefix(text: str, tag_pairs: Optional[List[List[str]]], enable_thinking: Optional[bool] = None) -> str:
    """Restore a prefilled reasoning prefix for log display."""
    if not isinstance(text, str) or not tag_pairs:
        return text
    start_tag, _ = tag_pairs[0]
    if start_tag in text:
        return text
    if enable_thinking is False:
        prefix = f"{start_tag}\n\n</think>\n\n"
        return f"{prefix}{text}" if text else prefix.rstrip()
    return f"{start_tag}\n{text}" if text else start_tag
