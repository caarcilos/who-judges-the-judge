#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from persuasion_judge_audit.io import load_examples, load_predictions
from persuasion_judge_audit.metrics import score
from persuasion_judge_audit.models import ValidationError
from persuasion_judge_audit.reporting import format_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare structured judge predictions with manual labels."
    )
    parser.add_argument("predictions", type=Path, help="Prediction JSONL file")
    parser.add_argument(
        "--examples",
        type=Path,
        default=ROOT / "data" / "examples.jsonl",
        help="Gold-label JSONL file",
    )
    parser.add_argument("--json-out", type=Path, help="Optional detailed JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = score(load_examples(args.examples), load_predictions(args.predictions))
    except (OSError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_report(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nDetailed report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

