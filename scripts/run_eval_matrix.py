#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("core", "hard")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model: str
    file_slug: str
    reasoning_effort: str | None = None


MATRIX = (
    ModelSpec(
        name="gpt-5-nano",
        provider="openai",
        model="gpt-5-nano",
        file_slug="openai-gpt-5-nano",
    ),
    ModelSpec(
        name="gpt-5.5",
        provider="openai",
        model="gpt-5.5",
        file_slug="openai-gpt-5.5",
    ),
    ModelSpec(
        name="gpt-oss-20b@medium",
        provider="together",
        model="openai/gpt-oss-20b",
        file_slug="together-gpt-oss-20b-reasoning-medium",
        reasoning_effort="medium",
    ),
)


def _stem(spec: ModelSpec, dataset: str, limit: int | None) -> str:
    stem = f"{spec.file_slug}-{dataset}"
    if limit is not None:
        stem += f"-limit-{limit}"
    return stem


def build_run_command(
    spec: ModelSpec,
    *,
    dataset: str,
    output_path: Path,
    limit: int | None,
    overwrite: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_judge.py",
        "--provider",
        spec.provider,
        "--dataset",
        dataset,
        "--model",
        spec.model,
        "--output",
        str(output_path),
    ]
    if spec.reasoning_effort is not None:
        command.extend(["--reasoning-effort", spec.reasoning_effort])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if overwrite:
        command.append("--overwrite")
    return command


def build_score_command(predictions_path: Path, report_path: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/score_results.py",
        str(predictions_path),
        "--json-out",
        str(report_path),
    ]


def iter_matrix_commands(
    *,
    datasets: tuple[str, ...],
    runs_dir: Path,
    reports_dir: Path,
    limit: int | None,
    overwrite: bool,
) -> list[tuple[list[str], list[str]]]:
    commands = []
    for spec in MATRIX:
        for dataset in datasets:
            stem = _stem(spec, dataset, limit)
            predictions_path = runs_dir / f"{stem}.jsonl"
            report_path = reports_dir / f"{stem}.json"
            commands.append(
                (
                    build_run_command(
                        spec,
                        dataset=dataset,
                        output_path=predictions_path,
                        limit=limit,
                        overwrite=overwrite,
                    ),
                    build_score_command(predictions_path, report_path),
                )
            )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and score the 3-model judge comparison matrix."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the commands. Without this, only print the plan.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Pass --limit to each judge run for a low-cost smoke matrix",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to each judge run",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / "runs",
        help="Directory for prediction JSONL files",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports",
        help="Directory for scored JSON reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2

    commands = iter_matrix_commands(
        datasets=tuple(args.datasets),
        runs_dir=args.runs_dir,
        reports_dir=args.reports_dir,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    if not args.execute:
        print("Dry run. Add --execute to run this matrix.\n")
        for run_command, score_command in commands:
            print(shlex.join(run_command))
            print(shlex.join(score_command))
            print()
        return 0

    for index, (run_command, score_command) in enumerate(commands, start=1):
        total = len(commands)
        print(f"\n[{index}/{total}] {shlex.join(run_command)}", flush=True)
        run_result = subprocess.run(run_command, cwd=ROOT)
        if run_result.returncode != 0:
            return run_result.returncode

        print(shlex.join(score_command), flush=True)
        score_result = subprocess.run(score_command, cwd=ROOT)
        if score_result.returncode != 0:
            return score_result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
