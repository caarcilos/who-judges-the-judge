import sys
import unittest
from pathlib import Path

from scripts.run_eval_matrix import MATRIX, iter_matrix_commands


class EvalMatrixTests(unittest.TestCase):
    def test_matrix_contains_expected_three_models(self):
        self.assertEqual(
            [(spec.provider, spec.model, spec.reasoning_effort) for spec in MATRIX],
            [
                ("openai", "gpt-5-nano", None),
                ("openai", "gpt-5.5", None),
                ("together", "openai/gpt-oss-20b", "medium"),
            ],
        )

    def test_matrix_builds_six_run_and_score_commands(self):
        commands = iter_matrix_commands(
            datasets=("core", "hard"),
            runs_dir=Path("runs"),
            reports_dir=Path("reports"),
            limit=None,
            overwrite=False,
        )

        self.assertEqual(len(commands), 6)
        run_commands = [run_command for run_command, _ in commands]
        score_commands = [score_command for _, score_command in commands]

        self.assertIn(
            [
                sys.executable,
                "scripts/run_judge.py",
                "--provider",
                "together",
                "--dataset",
                "hard",
                "--model",
                "openai/gpt-oss-20b",
                "--output",
                "runs/together-gpt-oss-20b-reasoning-medium-hard.jsonl",
                "--reasoning-effort",
                "medium",
            ],
            run_commands,
        )
        self.assertIn(
            [
                sys.executable,
                "scripts/score_results.py",
                "runs/openai-gpt-5.5-core.jsonl",
                "--json-out",
                "reports/openai-gpt-5.5-core.json",
            ],
            score_commands,
        )

    def test_limit_and_overwrite_are_reflected_in_run_outputs(self):
        commands = iter_matrix_commands(
            datasets=("hard",),
            runs_dir=Path("runs"),
            reports_dir=Path("reports"),
            limit=1,
            overwrite=True,
        )

        run_command, score_command = commands[0]
        self.assertIn("--limit", run_command)
        self.assertIn("1", run_command)
        self.assertIn("--overwrite", run_command)
        self.assertIn("runs/openai-gpt-5-nano-hard-limit-1.jsonl", run_command)
        self.assertIn(
            "reports/openai-gpt-5-nano-hard-limit-1.json",
            score_command,
        )


if __name__ == "__main__":
    unittest.main()
