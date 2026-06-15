# SOXX/SOXL Agentic Research Lab

Deterministic research pipeline for SOXX/SOXL semiconductor ETF forecasting with point-in-time validation, walk-forward backtesting, LangGraph orchestration, and optional LangSmith tracing.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

For tests:

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest
```

## Run The Graph

Sample-data run:

```bash
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_sample
```

Live or cached market-data run:

```bash
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --output-dir artifacts/soxx_live
```

## Optional LangSmith Tracing

Copy `.env.example` to `.env`, fill in your LangSmith API key, then load it:

```bash
set -a
source .env
set +a
```

Then run the graph with optional tags:

```bash
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_langsmith_sample \
  --trace-tag step6
```

Generated artifacts and cached market data are ignored by git.

## Generate A Research Memo

Create a Step 7 artifact directory with the graph runner:

```bash
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_step7_sample \
  --run-id step7-sample
```

Then generate a cited post-run memo from that artifact directory:

```bash
ANTHROPIC_API_KEY=... \
./.venv/bin/python scripts/run_research_memo.py \
  --artifact-dir artifacts/soxx_step7_sample
```

For a deterministic smoke run without an API call:

```bash
./.venv/bin/python scripts/run_research_memo.py \
  --artifact-dir artifacts/soxx_step7_sample \
  --provider template
```
