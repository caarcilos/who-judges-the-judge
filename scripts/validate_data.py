#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from persuasion_judge_audit.io import load_examples
from persuasion_judge_audit.models import LABELS, ValidationError
from scripts.dataset_options import DATASETS, resolve_examples_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the benchmark dataset.")
    parser.add_argument(
        "examples_path",
        nargs="?",
        type=Path,
        help="Backward-compatible custom examples JSONL path",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="core",
        help="Named benchmark split to validate",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        help="Custom examples JSONL file; mutually exclusive with --dataset hard",
    )
    args = parser.parse_args()
    try:
        custom_examples = args.examples or args.examples_path
        if args.examples and args.examples_path:
            raise ValueError(
                "pass a custom examples path either positionally or with "
                "--examples, not both"
            )
        examples_path, dataset_name = resolve_examples_path(
            dataset=args.dataset,
            examples=custom_examples,
        )
        examples = load_examples(examples_path)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts = Counter(example.gold_label for example in examples)
    print(f"Valid dataset: {dataset_name} ({examples_path})")
    print(f"Examples: {len(examples)}")
    for label in LABELS:
        print(f"  {label.value}: {counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
