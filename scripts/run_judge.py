#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from persuasion_judge_audit.io import load_examples, load_predictions
from persuasion_judge_audit.models import JudgePrediction, ValidationError
from runners.base import RunnerError
from runners.config import load_env_file, require_env
from runners.openai_runner import OpenAIJudgeRunner
from runners.together_runner import REASONING_EFFORTS, TogetherJudgeRunner
from scripts.dataset_options import DATASETS, resolve_examples_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real model judge and write schema-valid predictions."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "together"],
        default="openai",
        help="Judge provider",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="core",
        help="Named benchmark split to run",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        help="Custom input examples JSONL; mutually exclusive with --dataset hard",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=ROOT / "prompts" / "judge_prompt.md",
        help="Judge prompt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination predictions JSONL",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Environment file containing provider API keys",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID to evaluate; required to avoid accidental default-model runs",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        help="Reasoning effort for Together GPT-OSS models",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N examples for a smoke test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_dict(prediction: JudgePrediction) -> dict[str, object]:
    return {
        "id": prediction.id,
        "label": prediction.label.value,
        "confidence": prediction.confidence,
        "rationale": prediction.rationale,
    }


def write_predictions(path: Path, predictions: list[JudgePrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(_prediction_dict(prediction), ensure_ascii=False) + "\n"
        for prediction in predictions
    )
    path.write_text(content, encoding="utf-8")


def write_metadata(
    path: Path,
    *,
    provider: str,
    model: str,
    dataset: str,
    reasoning_effort: str | None,
    examples_path: Path,
    prompt_path: Path,
    count: int,
) -> Path:
    metadata_path = path.with_suffix(".metadata.json")
    metadata = {
        "provider": provider,
        "model": model,
        "dataset": dataset,
        "reasoning_effort": reasoning_effort,
        "api_surface": "responses" if provider == "openai" else "chat.completions",
        "structured_output": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "examples": str(examples_path),
        "examples_sha256": _sha256(examples_path),
        "prompt": str(prompt_path),
        "prompt_sha256": _sha256(prompt_path),
        "prediction_count": count,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def _create_runner(
    *,
    provider: str,
    model: str,
    prompt_path: Path,
    reasoning_effort: str | None,
    on_prediction,
):
    if provider == "openai":
        if reasoning_effort is not None:
            raise RunnerError("--reasoning-effort is only supported for Together")
        return OpenAIJudgeRunner(
            api_key=require_env("OPENAI_API_KEY"),
            model=model,
            prompt_path=prompt_path,
            on_prediction=on_prediction,
        )
    if provider == "together":
        if model.startswith("openai/gpt-oss-") and reasoning_effort is None:
            raise RunnerError(
                "Together GPT-OSS runs must pass --reasoning-effort "
                "so the setting is explicit in command history and metadata."
            )
        return TogetherJudgeRunner(
            api_key=require_env("TOGETHER_API_KEY"),
            model=model,
            prompt_path=prompt_path,
            reasoning_effort=reasoning_effort,
            on_prediction=on_prediction,
        )
    raise RunnerError(f"unsupported provider: {provider}")


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2
    if args.output.exists() and not args.overwrite:
        print(
            f"error: {args.output} already exists; pass --overwrite to replace it",
            file=sys.stderr,
        )
        return 2

    try:
        examples_path, dataset_name = resolve_examples_path(
            dataset=args.dataset,
            examples=args.examples,
        )
        load_env_file(args.env_file)
        model = args.model
        examples = load_examples(examples_path)
        if args.limit is not None:
            examples = examples[: args.limit]

        def show_progress(
            index: int, total: int, prediction: JudgePrediction
        ) -> None:
            print(
                f"[{index:>2}/{total}] {prediction.id}: "
                f"{prediction.label.value} ({prediction.confidence:.2f})",
                file=sys.stderr,
            )

        runner = _create_runner(
            provider=args.provider,
            model=model,
            prompt_path=args.prompt,
            reasoning_effort=args.reasoning_effort,
            on_prediction=show_progress,
        )
        predictions = runner.predict(examples)
        write_predictions(args.output, predictions)

        # Read through the normal loader so a successful command guarantees that
        # the generated file satisfies the same schema used by the scorer.
        validated = load_predictions(args.output)
        if len(validated) != len(examples):
            raise RunnerError("generated prediction count does not match input count")
        metadata_path = write_metadata(
            args.output,
            provider=args.provider,
            model=model,
            dataset=dataset_name,
            reasoning_effort=args.reasoning_effort,
            examples_path=examples_path,
            prompt_path=args.prompt,
            count=len(validated),
        )
    except (OSError, RunnerError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {len(validated)} predictions to {args.output}")
    print(f"Wrote run metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
