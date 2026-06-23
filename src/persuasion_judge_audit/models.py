from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Label(str, Enum):
    PERSUASION_ATTEMPT = "persuasion_attempt"
    REFUSAL = "refusal"
    NEUTRAL_INFORMATION = "neutral_information"
    SAFE_REDIRECTION = "safe_redirection"


LABELS = tuple(Label)


class ValidationError(ValueError):
    """Raised when an input record does not match the benchmark schema."""


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field!r} must be a non-empty string")
    return value.strip()


def _parse_label(value: Any, field: str) -> Label:
    try:
        return Label(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(label.value for label in LABELS)
        raise ValidationError(f"{field!r} must be one of: {choices}") from exc


@dataclass(frozen=True)
class Example:
    id: str
    scenario: str
    assistant_response: str
    gold_label: Label
    annotator_notes: str
    challenge_tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Example":
        tags = record.get("challenge_tags", [])
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag.strip() for tag in tags
        ):
            raise ValidationError("'challenge_tags' must be a list of strings")
        return cls(
            id=_require_string(record, "id"),
            scenario=_require_string(record, "scenario"),
            assistant_response=_require_string(record, "assistant_response"),
            gold_label=_parse_label(record.get("gold_label"), "gold_label"),
            annotator_notes=_require_string(record, "annotator_notes"),
            challenge_tags=tuple(tag.strip() for tag in tags),
        )


@dataclass(frozen=True)
class JudgePrediction:
    id: str
    label: Label
    confidence: float
    rationale: str

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "JudgePrediction":
        confidence = record.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValidationError("'confidence' must be a number from 0 to 1")
        return cls(
            id=_require_string(record, "id"),
            label=_parse_label(record.get("label"), "label"),
            confidence=float(confidence),
            rationale=_require_string(record, "rationale"),
        )

