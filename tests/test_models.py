import unittest

from persuasion_judge_audit.models import (
    Example,
    JudgePrediction,
    Label,
    ValidationError,
)


class ModelValidationTests(unittest.TestCase):
    def test_parses_valid_example(self):
        example = Example.from_dict(
            {
                "id": "x",
                "scenario": "Synthetic scenario.",
                "assistant_response": "A neutral response.",
                "gold_label": "neutral_information",
                "annotator_notes": "No advocacy.",
                "challenge_tags": ["plain_information"],
            }
        )
        self.assertEqual(example.gold_label, Label.NEUTRAL_INFORMATION)
        self.assertEqual(example.challenge_tags, ("plain_information",))

    def test_parses_example_field_aliases(self):
        example = Example.from_dict(
            {
                "id": "x",
                "scenario": "Synthetic scenario.",
                "response": "A neutral response.",
                "gold_label": "neutral_information",
                "notes": "No advocacy.",
                "challenge_tags": ["plain_information"],
            }
        )

        self.assertEqual(example.assistant_response, "A neutral response.")
        self.assertEqual(example.annotator_notes, "No advocacy.")

    def test_rejects_unknown_label(self):
        with self.assertRaisesRegex(ValidationError, "must be one of"):
            JudgePrediction.from_dict(
                {
                    "id": "x",
                    "label": "other",
                    "confidence": 0.5,
                    "rationale": "Reason.",
                }
            )

    def test_rejects_confidence_outside_unit_interval(self):
        with self.assertRaisesRegex(ValidationError, "from 0 to 1"):
            JudgePrediction.from_dict(
                {
                    "id": "x",
                    "label": "refusal",
                    "confidence": 1.1,
                    "rationale": "Reason.",
                }
            )


if __name__ == "__main__":
    unittest.main()
