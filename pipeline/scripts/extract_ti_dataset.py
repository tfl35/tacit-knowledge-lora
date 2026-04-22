#!/usr/bin/env python3
"""
Dataset extraction utility.
Extracts TI-only examples from combined datasets.

Usage:
    python extract_ti_dataset.py input.json output.json
    python extract_ti_dataset.py input.json output.json --count 350
"""

import argparse
import json
import sys


TI_PROMPT_PREFIX = "You are an experienced talent market intelligence analyst"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input JSON (combined dataset)")
    p.add_argument("output", help="Output JSON (TI-only)")
    p.add_argument("--count", type=int, default=None,
                   help="Expected example count (warn if different)")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    print(f"Input: {len(data)} total examples")

    # Extract TI examples
    ti_examples = []
    for ex in data:
        for turn in ex["conversations"]:
            if turn["from"] == "system" and turn["value"].startswith(TI_PROMPT_PREFIX):
                ti_examples.append(ex)
                break

    print(f"Extracted: {len(ti_examples)} TI examples")

    if args.count and len(ti_examples) != args.count:
        print(f"  WARNING: Expected {args.count}, got {len(ti_examples)}")

    # Stats
    single = sum(1 for ex in ti_examples
                 if sum(1 for t in ex["conversations"] if t["from"] == "gpt") == 1)
    multi = len(ti_examples) - single
    total_words = sum(len(t["value"].split())
                      for ex in ti_examples
                      for t in ex["conversations"]
                      if t["from"] == "gpt")

    print(f"  Single-turn: {single}, Multi-turn: {multi}")
    print(f"  Total assistant words: {total_words:,}")

    with open(args.output, "w") as f:
        json.dump(ti_examples, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
