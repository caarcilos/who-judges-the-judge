from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .models import Example, JudgePrediction, ValidationError

T = TypeVar("T")


def _load_jsonl(path: Path, parser: Callable[[dict], T]) -> list[T]:
    records: list[T] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValidationError(
                    f"{path}:{line_number}: each line must be a JSON object"
                )
            try:
                record = parser(raw)
            except ValidationError as exc:
                raise ValidationError(f"{path}:{line_number}: {exc}") from exc
            record_id = getattr(record, "id")
            if record_id in seen_ids:
                raise ValidationError(
                    f"{path}:{line_number}: duplicate id {record_id!r}"
                )
            seen_ids.add(record_id)
            records.append(record)
    if not records:
        raise ValidationError(f"{path}: file contains no records")
    return records


def load_examples(path: str | Path) -> list[Example]:
    return _load_jsonl(Path(path), Example.from_dict)


def load_predictions(path: str | Path) -> list[JudgePrediction]:
    return _load_jsonl(Path(path), JudgePrediction.from_dict)

