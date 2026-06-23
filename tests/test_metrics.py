import unittest

from persuasion_judge_audit.metrics import score
from persuasion_judge_audit.models import (
    Example,
    JudgePrediction,
    Label,
    ValidationError,
)
from persuasion_judge_audit.reporting import format_report


def example(identifier: str, label: Label, *tags: str) -> Example:
    return Example(
        id=identifier,
        scenario="Synthetic scenario.",
        assistant_response="Sanitized response.",
        gold_label=label,
        annotator_notes="Manual decision.",
        challenge_tags=tags,
    )


def prediction(identifier: str, label: Label, confidence: float) -> JudgePrediction:
    return JudgePrediction(
        id=identifier,
        label=label,
        confidence=confidence,
        rationale="Classification-focused reason.",
    )


class ScoringTests(unittest.TestCase):
    def test_computes_confusion_metrics_and_error_tags(self):
        examples = [
            example("a", Label.PERSUASION_ATTEMPT, "subtle"),
            example("b", Label.REFUSAL),
            example("c", Label.NEUTRAL_INFORMATION),
            example("d", Label.SAFE_REDIRECTION),
        ]
        predictions = [
            prediction("a", Label.NEUTRAL_INFORMATION, 0.9),
            prediction("b", Label.REFUSAL, 0.8),
            prediction("c", Label.NEUTRAL_INFORMATION, 0.7),
            prediction("d", Label.SAFE_REDIRECTION, 0.6),
        ]

        report = score(examples, predictions)

        self.assertEqual(report["summary"]["accuracy"], 0.75)
        self.assertEqual(
            report["confusion_matrix"]["persuasion_attempt"]["neutral_information"],
            1,
        )
        self.assertEqual(report["error_tags"], {"subtle": 1})
        self.assertEqual(report["confidence"]["high_confidence_errors"], 1)

    def test_requires_exact_id_coverage(self):
        with self.assertRaisesRegex(ValidationError, "missing predictions"):
            score(
                [example("a", Label.REFUSAL)],
                [prediction("other", Label.REFUSAL, 0.9)],
            )

    def test_formats_perfect_report_without_incorrect_confidence(self):
        report = score(
            [example("a", Label.REFUSAL)],
            [prediction("a", Label.REFUSAL, 0.9)],
        )
        self.assertIn("Mean confidence when wrong:   n/a", format_report(report))


if __name__ == "__main__":
    unittest.main()

