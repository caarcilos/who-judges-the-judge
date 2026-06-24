from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "core": ROOT / "data" / "examples.jsonl",
    "hard": ROOT / "data" / "examples_hard.jsonl",
}


def resolve_examples_path(
    *,
    dataset: str,
    examples: Path | None,
) -> tuple[Path, str]:
    """Return the example file path and a human-readable dataset name."""
    if examples is not None:
        if dataset != "core":
            raise ValueError("pass either --dataset or --examples, not both")
        return examples, "custom"
    return DATASETS[dataset], dataset

