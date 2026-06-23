from __future__ import annotations

from typing import Any

from .models import LABELS


def format_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Persuasion Judge Audit",
        "=======================",
        f"Examples: {summary['examples']}",
        f"Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['examples']})",
        f"Macro F1: {summary['macro_f1']:.3f}",
        "",
        "Per-label metrics",
        "-----------------",
        f"{'label':22} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}",
    ]
    for label in LABELS:
        metrics = report["per_label"][label.value]
        lines.append(
            f"{label.value:22} "
            f"{metrics['precision']:>9.3f} "
            f"{metrics['recall']:>9.3f} "
            f"{metrics['f1']:>9.3f} "
            f"{metrics['support']:>9}"
        )

    lines.extend(
        [
            "",
            "Confusion matrix (rows=gold, columns=predicted)",
            "------------------------------------------------",
            f"{'gold \\\\ predicted':22} "
            + " ".join(f"{label.value[:8]:>8}" for label in LABELS),
        ]
    )
    for gold in LABELS:
        row = report["confusion_matrix"][gold.value]
        lines.append(
            f"{gold.value:22} "
            + " ".join(f"{row[predicted.value]:>8}" for predicted in LABELS)
        )

    confidence = report["confidence"]
    correct_confidence = (
        f"{confidence['mean_when_correct']:.3f}"
        if confidence["mean_when_correct"] is not None
        else "n/a"
    )
    incorrect_confidence = (
        f"{confidence['mean_when_incorrect']:.3f}"
        if confidence["mean_when_incorrect"] is not None
        else "n/a"
    )
    lines.extend(
        [
            "",
            "Confidence diagnostics",
            "----------------------",
            f"Mean confidence when correct: {correct_confidence}",
            f"Mean confidence when wrong:   {incorrect_confidence}",
            f"High-confidence errors (>=0.8): {confidence['high_confidence_errors']}",
            "",
            "Error tags",
            "----------",
        ]
    )
    if report["error_tags"]:
        lines.extend(
            f"{tag}: {count}" for tag, count in report["error_tags"].items()
        )
    else:
        lines.append("None")
    return "\n".join(lines)
