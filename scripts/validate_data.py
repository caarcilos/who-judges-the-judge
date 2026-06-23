#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from persuasion_judge_audit.io import load_examples
from persuasion_judge_audit.models import LABELS, ValidationError


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the benchmark dataset.")
    parser.add_argument(
        "examples",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "examples.jsonl",
    )
    args = parser.parse_args()
    try:
        examples = load_examples(args.examples)
    except (OSError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts = Counter(example.gold_label for example in examples)
    print(f"Valid dataset: {len(examples)} examples")
    for label in LABELS:
        print(f"  {label.value}: {counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

