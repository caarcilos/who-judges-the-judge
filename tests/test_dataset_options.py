import json
import tempfile
import unittest
from pathlib import Path

from scripts.dataset_options import DATASETS, resolve_examples_path
from scripts.score_results import _dataset_from_metadata

ROOT = Path(__file__).resolve().parents[1]


class DatasetOptionTests(unittest.TestCase):
    def test_named_hard_dataset_resolves_to_hard_file(self):
        path, name = resolve_examples_path(dataset="hard", examples=None)

        self.assertEqual(name, "hard")
        self.assertEqual(path, ROOT / "data" / "examples_hard.jsonl")

    def test_custom_examples_override_default_core_dataset(self):
        custom_path = ROOT / "data" / "custom.jsonl"
        path, name = resolve_examples_path(dataset="core", examples=custom_path)

        self.assertEqual(name, "custom")
        self.assertEqual(path, custom_path)

    def test_custom_examples_cannot_be_combined_with_hard_dataset(self):
        with self.assertRaisesRegex(ValueError, "either --dataset or --examples"):
            resolve_examples_path(
                dataset="hard",
                examples=ROOT / "data" / "custom.jsonl",
            )

    def test_score_dataset_can_be_inferred_from_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions_path = root / "run.jsonl"
            metadata_path = root / "run.metadata.json"
            metadata_path.write_text(
                json.dumps({"dataset": "hard"}), encoding="utf-8"
            )

            self.assertEqual(_dataset_from_metadata(predictions_path), "hard")

    def test_unknown_metadata_dataset_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions_path = root / "run.jsonl"
            metadata_path = root / "run.metadata.json"
            metadata_path.write_text(
                json.dumps({"dataset": "not-a-split"}), encoding="utf-8"
            )

            self.assertIsNone(_dataset_from_metadata(predictions_path))

    def test_dataset_registry_tracks_current_splits(self):
        self.assertEqual(set(DATASETS), {"core", "hard"})


if __name__ == "__main__":
    unittest.main()

