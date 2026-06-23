import json
import tempfile
import unittest
from pathlib import Path

from persuasion_judge_audit.io import load_predictions
from persuasion_judge_audit.models import ValidationError


class JsonlLoadingTests(unittest.TestCase):
    def test_rejects_duplicate_ids(self):
        record = {
            "id": "duplicate",
            "label": "refusal",
            "confidence": 0.8,
            "rationale": "Direct refusal.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "duplicate id"):
                load_predictions(path)

    def test_reports_line_number_for_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("{}\nnot-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, r":1:|:2:"):
                load_predictions(path)


if __name__ == "__main__":
    unittest.main()

