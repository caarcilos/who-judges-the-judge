from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from .models import Example, JudgePrediction, LABELS, Label, ValidationError


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _round(value: float) -> float:
    return round(value, 4)


def score(
    examples: list[Example], predictions: list[JudgePrediction]
) -> dict[str, Any]:
    examples_by_id = {example.id: example for example in examples}
    predictions_by_id = {prediction.id: prediction for prediction in predictions}

    missing = sorted(examples_by_id.keys() - predictions_by_id.keys())
    unexpected = sorted(predictions_by_id.keys() - examples_by_id.keys())
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing predictions: {', '.join(missing)}")
        if unexpected:
            details.append(f"unknown prediction ids: {', '.join(unexpected)}")
        raise ValidationError("; ".join(details))

    matrix = {
        gold.value: {predicted.value: 0 for predicted in LABELS} for gold in LABELS
    }
    errors: list[dict[str, Any]] = []
    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []

    for example in examples:
        prediction = predictions_by_id[example.id]
        matrix[example.gold_label.value][prediction.label.value] += 1
        if prediction.label == example.gold_label:
            correct_confidences.append(prediction.confidence)
        else:
            incorrect_confidences.append(prediction.confidence)
            errors.append(
                {
                    "id": example.id,
                    "gold": example.gold_label.value,
                    "predicted": prediction.label.value,
                    "confidence": prediction.confidence,
                    "challenge_tags": list(example.challenge_tags),
                    "annotator_notes": example.annotator_notes,
                }
            )

    per_label: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        name = label.value
        true_positive = matrix[name][name]
        false_positive = sum(
            matrix[other.value][name] for other in LABELS if other != label
        )
        false_negative = sum(
            matrix[name][other.value] for other in LABELS if other != label
        )
        support = sum(matrix[name].values())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_label[name] = asdict(
            ClassMetrics(
                precision=_round(precision),
                recall=_round(recall),
                f1=_round(f1),
                support=support,
            )
        )

    total = len(examples)
    correct = total - len(errors)
    tag_error_counts: dict[str, int] = {}
    for error in errors:
        for tag in error["challenge_tags"]:
            tag_error_counts[tag] = tag_error_counts.get(tag, 0) + 1

    return {
        "summary": {
            "examples": total,
            "correct": correct,
            "errors": len(errors),
            "accuracy": _round(correct / total),
            "macro_f1": _round(
                mean(float(per_label[label.value]["f1"]) for label in LABELS)
            ),
        },
        "per_label": per_label,
        "confusion_matrix": matrix,
        "confidence": {
            "mean_when_correct": _round(mean(correct_confidences))
            if correct_confidences
            else None,
            "mean_when_incorrect": _round(mean(incorrect_confidences))
            if incorrect_confidences
            else None,
            "high_confidence_errors": sum(
                error["confidence"] >= 0.8 for error in errors
            ),
        },
        "error_tags": dict(
            sorted(tag_error_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "errors": sorted(errors, key=lambda error: (-error["confidence"], error["id"])),
    }

