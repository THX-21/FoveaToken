#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable

TASK_PAIRS = [
    ("Complex reasoning", "Anomaly Detection and Interpretation"),
    ("Complex reasoning", "Environmental condition reasoning"),
    ("Complex reasoning", "Route planning"),
    ("Counting", "Counting with changing detection"),
    ("Counting", "Counting with complex reasoning"),
    ("Counting", "Overall counting"),
    ("Counting", "Regional counting"),
    ("Land use classification", "Overall Land use classification"),
    ("Land use classification", "Regional Land use classification"),
    ("Object properties", "Object classification"),
    ("Object properties", "Object color"),
    ("Object properties", "Object motion state"),
    ("Object spatial relationship", "Object spatial relationship"),
]

COLUMN_MAP = {
    ("Object properties", "Object classification"): "OC",
    ("Counting", "Regional counting"): "RC",
    ("Land use classification", "Overall Land use classification"): "OLUC",
    ("Land use classification", "Regional Land use classification"): "RLUC",
    ("Counting", "Overall counting"): "OCC",
    ("Object properties", "Object color"): "OCL",
    ("Object properties", "Object motion state"): "OMS",
    ("Object spatial relationship", "Object spatial relationship"): "OSR",
    ("Complex reasoning", "Anomaly Detection and Interpretation"): "AD",
    ("Complex reasoning", "Environmental condition reasoning"): "ECR",
    ("Complex reasoning", "Route planning"): "RP",
    ("Counting", "Counting with changing detection"): "RCCD",
    ("Counting", "Counting with complex reasoning"): "CCR",
}

COLUMN_ORDER = [
    "OC", "RC", "OLUC", "RLUC", "OCC", "OCL", "OMS", "OSR", "AD", "ECR", "RP", "RCCD", "CCR", "Micro Avg.", "Macro Avg."
]

TASK_ORDER = [
    "Complex reasoning",
    "Counting",
    "Land use classification",
    "Object properties",
    "Object spatial relationship",
]


def normalize_answer(value: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    letters = sorted({char for char in value.upper() if char in "ABCDE"})
    return "".join(letters)


def compute_accuracy(items: list[dict]) -> tuple[float, int, int]:
    total = len(items)
    if total == 0:
        return 0.0, 0, 0
    correct = 0
    for item in items:
        pred = normalize_answer(item.get("pred_answer", ""))
        answer = normalize_answer(item.get("answer", ""))
        correct += int(set(pred) == set(answer))
    return correct / total, correct, total


def iter_jsonl_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc


def load_results(path: Path) -> list[dict]:
    results = []
    for record in iter_jsonl_records(path):
        payload = record.get("xlrs_micro_score") or record.get("xlrs_macro_score")
        if not isinstance(payload, dict):
            continue
        results.append(payload)
    return results


def collect_sample_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    files = sorted(path.rglob("*_samples_xlrs-lite.jsonl"))
    if not files:
        raise FileNotFoundError(f"No '*_samples_xlrs-lite.jsonl' files found under {path}")
    return files


def summarize(results: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {pair: [] for pair in TASK_PAIRS}
    task_grouped: dict[str, list[dict]] = {task: [] for task in TASK_ORDER}

    for item in results:
        pair = (item["category"], item["sub_category"])
        if pair not in grouped:
            continue
        grouped[pair].append(item)
        task_grouped[item["category"]].append(item)

    subtask_acc = {}
    macro_values = []
    for pair, items in grouped.items():
        acc, correct, total = compute_accuracy(items)
        subtask_acc[pair] = {
            "abbr": COLUMN_MAP[pair],
            "acc": acc,
            "correct": correct,
            "total": total,
        }
        macro_values.append(acc)

    task_acc = {}
    for task in TASK_ORDER:
        acc, correct, total = compute_accuracy(task_grouped[task])
        task_acc[task] = {"acc": acc, "correct": correct, "total": total}

    overall_acc, overall_correct, overall_total = compute_accuracy(results)
    macro_acc = sum(macro_values) / len(macro_values) if macro_values else 0.0
    return {
        "subtasks": subtask_acc,
        "tasks": task_acc,
        "overall": {
            "micro_acc": overall_acc,
            "macro_acc": macro_acc,
            "correct": overall_correct,
            "total": overall_total,
        },
    }


def format_pct(acc: float) -> str:
    return f"{acc * 100:.1f}"


def build_markdown_table(model_name: str, summary: dict) -> str:
    value_map = {info["abbr"]: format_pct(info["acc"]) for info in summary["subtasks"].values()}
    micro_avg = format_pct(summary["overall"]["micro_acc"])
    macro_avg = format_pct(summary["overall"]["macro_acc"])
    row = [model_name] + [value_map.get(column, "-") for column in COLUMN_ORDER[:-2]] + [micro_avg, macro_avg]
    header = ["Sub-tasks (L-3 Capability)"] + COLUMN_ORDER
    sep = ["---"] * len(header)
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        "| " + " | ".join(row) + " |",
    ])


def print_verbose_report(model_name: str, sample_file: Path, summary: dict) -> None:
    print(f"=== {model_name} ===")
    print(f"File: {sample_file}")
    print(
        f"Overall Micro Acc {summary['overall']['micro_acc']:.4f} "
        f"({summary['overall']['correct']}/{summary['overall']['total']}), "
        f"Macro Acc {summary['overall']['macro_acc']:.4f}"
    )
    print()
    for task in TASK_ORDER:
        task_info = summary["tasks"][task]
        print(f"{task}: {task_info['acc']:.4f} ({task_info['correct']}/{task_info['total']})")
        for pair in TASK_PAIRS:
            if pair[0] != task:
                continue
            info = summary["subtasks"][pair]
            print(f"  - {info['abbr']:<4} {pair[1]}: {info['acc']:.4f} ({info['correct']}/{info['total']})")
        print()
    print(build_markdown_table(model_name, summary))


def infer_model_name(sample_file: Path) -> str:
    return sample_file.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate XLRS-lite sample logs into accuracy tables.")
    parser.add_argument("path", help="Path to one *_samples_xlrs-lite.jsonl file or a directory containing them")
    parser.add_argument("--model-name", help="Override displayed model name (valid for single-file mode)")
    parser.add_argument("--latest-only", action="store_true", help="When path is a directory, keep only the newest sample file per parent directory")
    parser.add_argument("--table-only", action="store_true", help="Only print markdown table rows")
    args = parser.parse_args()

    sample_files = collect_sample_files(Path(args.path))
    if args.latest_only:
        latest = {}
        for file in sample_files:
            key = file.parent
            if key not in latest or file.name > latest[key].name:
                latest[key] = file
        sample_files = sorted(latest.values())

    if args.table_only:
        header = ["Sub-tasks (L-3 Capability)"] + COLUMN_ORDER
        sep = ["---"] * len(header)
        print("| " + " | ".join(header) + " |")
        print("| " + " | ".join(sep) + " |")
        for sample_file in sample_files:
            model_name = args.model_name if (args.model_name and len(sample_files) == 1) else infer_model_name(sample_file)
            summary = summarize(load_results(sample_file))
            value_map = {info["abbr"]: format_pct(info["acc"]) for info in summary["subtasks"].values()}
            micro_avg = format_pct(summary["overall"]["micro_acc"])
            macro_avg = format_pct(summary["overall"]["macro_acc"])
            row = [model_name] + [value_map.get(column, "-") for column in COLUMN_ORDER[:-2]] + [micro_avg, macro_avg]
            print("| " + " | ".join(row) + " |")
        return

    for index, sample_file in enumerate(sample_files):
        model_name = args.model_name if (args.model_name and len(sample_files) == 1) else infer_model_name(sample_file)
        summary = summarize(load_results(sample_file))
        if index:
            print()
        print_verbose_report(model_name, sample_file, summary)


if __name__ == "__main__":
    main()
