# Persuasion Judge Audit

A small Python research-engineering toy repo for auditing whether LLM judges
correctly classify sanitized persuasion-risk responses.

The motivating question is simple: if an eval relies on an LLM judge, where can
that judge itself fail? This repo does not reproduce FAR.AI's Attempt to
Persuade Eval and does not reuse its code; it is only a compact, independent
exercise inspired by that kind of evaluator-reliability question.

For the current three-model comparison, figures, and short interpretation, see
[`RESULTS.md`](RESULTS.md).

## What is here

- synthetic, sanitized examples with manual labels
- a small four-class schema:
  `persuasion_attempt`, `refusal`, `neutral_information`, `safe_redirection`
- offline scoring against prediction JSONL
- optional real-model runners for OpenAI and Together
- a hard split intended to expose boundary mistakes
- generated result figures under [`reports/figures`](reports/figures)

This is intentionally a toy benchmark, not a comprehensive persuasion eval.

## Quick start

The offline scoring path needs only Python 3.10+:

```bash
python scripts/validate_data.py
python scripts/validate_data.py --dataset hard
python scripts/score_results.py data/example_predictions.jsonl
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Or, with dev dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Running model judges

Create a local `.env` from the example and add only the provider keys you need:

```bash
cp .env.example .env
```

Run one model:

```bash
python scripts/run_judge.py \
  --provider openai \
  --dataset hard \
  --model gpt-5-nano \
  --output runs/openai-gpt-5-nano-hard.jsonl
```

Score it:

```bash
python scripts/score_results.py runs/openai-gpt-5-nano-hard.jsonl \
  --json-out reports/openai-gpt-5-nano-hard.json
```

Run the small comparison matrix used in `RESULTS.md`:

```bash
python scripts/run_eval_matrix.py --execute
```

For a cheap smoke test:

```bash
python scripts/run_eval_matrix.py --limit 1 --execute
```

The matrix covers:

| Provider | Model | Datasets |
|---|---|---|
| OpenAI | `gpt-5-nano` | core + hard |
| OpenAI | `gpt-5.5` | core + hard |
| Together | `openai/gpt-oss-20b`, reasoning `medium` | core + hard |

Model names are always explicit on the command line to avoid accidental default
runs. Raw runs are gitignored; selected scored reports and figures are committed
so the current results can be inspected without API keys.

## Regenerating figures

```bash
python -m pip install -e ".[plots]"
python scripts/plot_results.py
```

The plot script reads the committed report JSON files in `reports/` and writes
figures to `reports/figures/`.

## Safety

This project intentionally uses sanitized synthetic examples rather than
publishing harmful persuasion prompts or realistic manipulation scripts. The
aim is to study evaluator reliability, not to improve persuasive capability.

Placeholders such as `POSITION_X`, `GROUP_A`, `PRODUCT_Y`, `RISKY_CLAIM_Z`,
`PUBLIC_FIGURE_A`, and `POLICY_B` keep examples abstract.

## Limitations

- tiny synthetic datasets
- single-annotator gold labels
- self-reported confidence is not calibrated probability
- hosted openweight results are endpoint-specific
- results are illustrative, not statistical claims

## License

MIT
