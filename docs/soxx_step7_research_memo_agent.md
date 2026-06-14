# SOXX/SOXL Step 7 Claude Research Memo Agent

Status: implemented.

## Goal

Add the safest first LLM component: a post-run research memo agent that reads completed artifacts and produces a cited `research_memo.md`.

Step 7 answers:

```text
Can an LLM summarize deterministic SOXX/SOXL run artifacts without changing the backtest, forecast, graph, or validation logic?
```

This step does not add new features, market data, model changes, experiment proposals, or graph nodes by default.

## What This Step Adds

- A memo agent module: `soxx_mvp/agents/memo_agent.py`
- A post-run CLI: `scripts/run_research_memo.py`
- A Claude/Sonnet provider using the Anthropic Python SDK
- A deterministic template provider for tests and local smoke runs
- A prompt contract: `prompts/research_memo_v1.md`
- Step 7 tests for artifact loading, leakage blocking, citation validation, provider parsing, and CLI smoke behavior

## Artifact Inputs

Required:

```text
metrics.json
latest_forecast.json
backtest_report.md
leakage_report.json
artifact_manifest.json
```

Optional:

```text
validation_report.md
```

Output:

```text
research_memo.md
```

The memo cites artifact paths directly. It also states that feature attribution is unavailable because the current pipeline does not produce a feature attribution artifact.

## Run

From `SOXX_agentic/`, first create a Step 7 artifact directory with the existing graph runner:

```bash
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_step7_sample \
  --run-id step7-sample
```

Then generate the memo into that same artifact directory:

```bash
ANTHROPIC_API_KEY=... \
./.venv/bin/python scripts/run_research_memo.py \
  --artifact-dir artifacts/soxx_step7_sample
```

Override the model:

```bash
ANTHROPIC_API_KEY=... \
./.venv/bin/python scripts/run_research_memo.py \
  --artifact-dir artifacts/soxx_step7_sample \
  --model claude-sonnet-4-6
```

Deterministic local smoke run without an API call:

```bash
./.venv/bin/python scripts/run_research_memo.py \
  --artifact-dir artifacts/soxx_step7_sample \
  --provider template
```

## Guardrails

- The agent only receives compact facts parsed from existing artifacts.
- Every substantive memo claim must cite one or more known artifact filenames.
- Unknown artifact citations fail validation.
- Missing required artifacts fail before any memo is written.
- Leakage failures block memo generation.
- Provider output is validated before writing `research_memo.md`.
- The memo is written atomically after validation.

## Acceptance Criteria

Step 7 is complete when:

```text
pytest passes
template memo generation works without Anthropic credentials
Claude provider is covered by mocked SDK tests
missing required artifacts fail clearly
leakage failures block memo generation
unsupported or unknown citations fail validation
existing deterministic, LangGraph, and LangSmith tests still pass
```
