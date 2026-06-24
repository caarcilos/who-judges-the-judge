import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from persuasion_judge_audit.io import load_examples, load_predictions
from persuasion_judge_audit.models import Example, Label
from runners.base import RunnerError
from runners.config import load_env_file, require_env
from runners.openai_runner import OpenAIJudgeRunner
from runners.together_runner import TOGETHER_BASE_URL, TogetherJudgeRunner
from scripts.run_judge import write_predictions

ROOT = Path(__file__).resolve().parents[1]


def example(identifier: str = "case-1") -> Example:
    return Example(
        id=identifier,
        scenario="The user asks for a synthetic message.",
        assistant_response="I cannot help with that.",
        gold_label=Label.REFUSAL,
        annotator_notes="Manual label must not be sent to the judge.",
        challenge_tags=("plain_refusal",),
    )


class FakeResponses:
    def __init__(self, output: dict):
        self.output = output
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.output))


class EchoResponses:
    def create(self, **kwargs):
        payload = json.loads(kwargs["input"].split("\n", 1)[1])
        output = {
            "id": payload["id"],
            "label": "neutral_information",
            "confidence": 0.5,
            "rationale": "Mock structured prediction.",
        }
        return SimpleNamespace(output_text=json.dumps(output))


class FakeChatCompletions:
    def __init__(self, output: dict):
        self.output = output
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.output))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class EchoChatCompletions:
    def create(self, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"].split("\n", 1)[1])
        output = {
            "id": payload["id"],
            "label": "neutral_information",
            "confidence": 0.5,
            "rationale": "Mock structured prediction.",
        }
        message = SimpleNamespace(content=json.dumps(output))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RunnerTests(unittest.TestCase):
    def test_loads_dotenv_without_overriding_existing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                'OPENAI_API_KEY="from-file"\nTOGETHER_API_KEY=together-test\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ, {"OPENAI_API_KEY": "already-set"}, clear=True
            ):
                load_env_file(path)
                self.assertEqual(require_env("OPENAI_API_KEY"), "already-set")
                self.assertEqual(os.environ["TOGETHER_API_KEY"], "together-test")

    def test_openai_runner_uses_prompt_and_strict_schema(self):
        responses = FakeResponses(
            {
                "id": "case-1",
                "label": "refusal",
                "confidence": 0.91,
                "rationale": "The assistant directly declines.",
            }
        )
        client = SimpleNamespace(responses=responses)
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.md"
            prompt_path.write_text("Classify the response.", encoding="utf-8")
            runner = OpenAIJudgeRunner(
                api_key="test-key",
                model="gpt-test",
                prompt_path=prompt_path,
                client=client,
            )
            predictions = runner.predict([example()])

        self.assertEqual(predictions[0].label, Label.REFUSAL)
        request = responses.calls[0]
        self.assertEqual(request["model"], "gpt-test")
        self.assertEqual(request["instructions"], "Classify the response.")
        self.assertFalse(request["store"])
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertNotIn("gold_label", request["input"])
        self.assertNotIn("annotator_notes", request["input"])

    def test_openai_runner_anchors_prediction_id_to_input_example(self):
        responses = FakeResponses(
            {
                "id": "case-1-classification",
                "label": "refusal",
                "confidence": 0.9,
                "rationale": "Direct refusal.",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.md"
            prompt_path.write_text("Classify.", encoding="utf-8")
            runner = OpenAIJudgeRunner(
                api_key="test-key",
                model="gpt-test",
                prompt_path=prompt_path,
                client=SimpleNamespace(responses=responses),
            )
            predictions = runner.predict([example()])

        self.assertEqual(predictions[0].id, "case-1")

    def test_together_runner_uses_chat_schema_and_reasoning_effort(self):
        completions = FakeChatCompletions(
            {
                "id": "case-1-classification",
                "label": "refusal",
                "confidence": 0.91,
                "rationale": "The assistant directly declines.",
            }
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.md"
            prompt_path.write_text("Classify the response.", encoding="utf-8")
            runner = TogetherJudgeRunner(
                api_key="test-key",
                model="openai/gpt-oss-20b",
                prompt_path=prompt_path,
                reasoning_effort="medium",
                client=client,
            )
            predictions = runner.predict([example()])

        self.assertEqual(predictions[0].id, "case-1")
        self.assertEqual(predictions[0].label, Label.REFUSAL)
        request = completions.calls[0]
        self.assertEqual(request["model"], "openai/gpt-oss-20b")
        self.assertEqual(request["reasoning_effort"], "medium")
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(request["messages"][0]["content"], "Classify the response.")
        self.assertEqual(request["messages"][1]["role"], "user")
        self.assertEqual(
            request["response_format"]["json_schema"]["name"], "judge_prediction"
        )
        self.assertNotIn("gold_label", request["messages"][1]["content"])
        self.assertNotIn("annotator_notes", request["messages"][1]["content"])

    def test_together_client_uses_together_base_url(self):
        openai_class = Mock()
        fake_openai = SimpleNamespace(OpenAI=openai_class)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            TogetherJudgeRunner._create_client("test-key")

        openai_class.assert_called_once_with(
            api_key="test-key",
            base_url=TOGETHER_BASE_URL,
        )

    def test_full_dataset_round_trips_through_standard_loader(self):
        examples = load_examples(ROOT / "data" / "examples.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "prompt.md"
            output_path = root / "predictions.jsonl"
            prompt_path.write_text("Classify.", encoding="utf-8")
            runner = OpenAIJudgeRunner(
                api_key="test-key",
                model="gpt-test",
                prompt_path=prompt_path,
                client=SimpleNamespace(responses=EchoResponses()),
            )
            write_predictions(output_path, runner.predict(examples))
            loaded = load_predictions(output_path)

        self.assertEqual(len(loaded), 64)
        self.assertEqual(
            [prediction.id for prediction in loaded],
            [benchmark_example.id for benchmark_example in examples],
        )

    def test_full_dataset_round_trips_through_together_runner(self):
        examples = load_examples(ROOT / "data" / "examples.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "prompt.md"
            output_path = root / "predictions.jsonl"
            prompt_path.write_text("Classify.", encoding="utf-8")
            runner = TogetherJudgeRunner(
                api_key="test-key",
                model="openai/gpt-oss-20b",
                prompt_path=prompt_path,
                reasoning_effort="medium",
                client=SimpleNamespace(
                    chat=SimpleNamespace(completions=EchoChatCompletions()),
                ),
            )
            write_predictions(output_path, runner.predict(examples))
            loaded = load_predictions(output_path)

        self.assertEqual(len(loaded), 64)
        self.assertEqual(
            [prediction.id for prediction in loaded],
            [benchmark_example.id for benchmark_example in examples],
        )


if __name__ == "__main__":
    unittest.main()
