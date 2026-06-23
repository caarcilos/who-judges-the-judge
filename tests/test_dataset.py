import unittest
from collections import Counter
from pathlib import Path

from persuasion_judge_audit.io import load_examples, load_predictions
from persuasion_judge_audit.metrics import score
from persuasion_judge_audit.models import LABELS

ROOT = Path(__file__).resolve().parents[1]


class IncludedArtifactTests(unittest.TestCase):
    def test_dataset_is_balanced_and_large_enough(self):
        examples = load_examples(ROOT / "data" / "examples.jsonl")
        counts = Counter(example.gold_label for example in examples)
        self.assertEqual(len(examples), 64)
        self.assertEqual({counts[label] for label in LABELS}, {16})

    def test_example_predictions_cover_dataset_and_score(self):
        examples = load_examples(ROOT / "data" / "examples.jsonl")
        predictions = load_predictions(ROOT / "data" / "example_predictions.jsonl")
        report = score(examples, predictions)
        self.assertEqual(report["summary"]["examples"], 64)
        self.assertEqual(report["summary"]["correct"], 54)
        self.assertEqual(report["summary"]["accuracy"], 0.8438)


if __name__ == "__main__":
    unittest.main()

