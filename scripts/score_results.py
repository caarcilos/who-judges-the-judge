#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from persuasion_judge_audit.io import load_examples, load_predictions
from persuasion_judge_audit.metrics import score
from persuasion_judge_audit.models import ValidationError
from persuasion_judge_audit.reporting import format_report
from scripts.dataset_options import DATASETS, resolve_examples_path


def _dataset_from_metadata(predictions_path: Path) -> str | None:
    metadata_path = predictions_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    dataset = metadata.get("dataset")
    if isinstance(dataset, str) and dataset in DATASETS:
        return dataset
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare structured judge predictions with manual labels."
    )
    parser.add_argument("predictions", type=Path, help="Prediction JSONL file")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        help="Named benchmark split to score against; inferred from run metadata when omitted",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        help="Custom gold-label JSONL file; mutually exclusive with --dataset hard",
    )
    parser.add_argument("--json-out", type=Path, help="Optional detailed JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset = args.dataset or (
            "core" if args.examples else _dataset_from_metadata(args.predictions) or "core"
        )
        examples_path, dataset_name = resolve_examples_path(
            dataset=dataset,
            examples=args.examples,
        )
        report = score(load_examples(examples_path), load_predictions(args.predictions))
    except (OSError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Dataset: {dataset_name} ({examples_path})\n")
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
