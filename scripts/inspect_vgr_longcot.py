#!/usr/bin/env python
"""Interactively print VGR longcot samples for data inspection."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REGION_RE = re.compile(r"<SOT>\s*(\[[^\]]+\])\s*<EOT>\s*<image>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Press Enter to print VGR longcot samples one by one.")
    parser.add_argument("--parquet", default="data/vgr/vgr_longcot.parquet")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--context", type=int, default=180, help="Characters around each region tag to print.")
    parser.add_argument("--compact", action="store_true", help="Only print snippets around region tags.")
    return parser.parse_args()


def think_state(text: str, pos: int) -> str:
    before = text[:pos].lower()
    last_open = before.rfind("<think>")
    last_close = before.rfind("</think>")
    if last_open != -1 and last_open > last_close:
        return "inside <think>"
    if last_close > last_open:
        return "after </think>"
    return "before/no <think>"


def print_sample(row, index: int, context_chars: int, compact: bool) -> None:
    print("\n" + "=" * 100)
    print(f"index: {index}")
    print(f"image: {row['image']}")
    conversations = row["conversations"]
    assistant_text = "\n".join(turn["value"] for turn in conversations if turn.get("from") in {"gpt", "assistant"})
    matches = list(REGION_RE.finditer(assistant_text))
    print(f"num_regions: {len(matches)}")
    for region_idx, match in enumerate(matches):
        pos_ratio = match.start() / max(len(assistant_text), 1)
        print(f"  region[{region_idx}] box={match.group(1)} pos={pos_ratio:.2%} state={think_state(assistant_text, match.start())}")

    print("-" * 100)
    for turn_idx, turn in enumerate(conversations):
        role = turn.get("from")
        value = turn.get("value", "")
        print(f"[turn {turn_idx}] {role}")
        if not compact:
            print(value)
            continue
        if role in {"human", "user"}:
            print(value)
            continue
        if not matches:
            print(value[:2000])
            continue
        for region_idx, match in enumerate(REGION_RE.finditer(value)):
            start = max(0, match.start() - context_chars)
            end = min(len(value), match.end() + context_chars)
            snippet = value[start:end].replace("\n", "\\n")
            print(f"  snippet around region[{region_idx}]:")
            print(f"  ...{snippet}...")


def main() -> None:
    args = parse_args()
    path = Path(args.parquet)
    frame = pd.read_parquet(path)
    print(f"loaded {path} rows={len(frame)}")
    print("Press Enter for next sample, type an index to jump, q to quit.")

    index = int(args.start)
    while 0 <= index < len(frame):
        command = input(f"\nnext index={index}> ").strip()
        if command.lower() in {"q", "quit", "exit"}:
            break
        if command:
            try:
                index = int(command)
            except ValueError:
                print("Enter a row index, blank for next, or q to quit.")
                continue
            if not (0 <= index < len(frame)):
                print(f"Index out of range: 0..{len(frame) - 1}")
                continue
        print_sample(frame.iloc[index], index, args.context, args.compact)
        index += 1


if __name__ == "__main__":
    main()
