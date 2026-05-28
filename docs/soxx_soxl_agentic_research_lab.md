# Point-in-Time Agentic Semiconductor ETF Research System

## SOXX Forecasting with SOXL Leverage-Aware Translation

### Project Goal

Build a modern agentic research platform that forecasts semiconductor-sector ETF movement using point-in-time data, multi-agent research workflows, deterministic backtesting, LangGraph orchestration, LangSmith tracing, and custom MCP tools.

The project is designed to showcase strong agentic development skills for teams working on agentic modeling, quant research automation, financial AI systems, or research infrastructure.

This is **not** a simple “LLM predicts a stock price” project. The goal is to build an auditable research system where agents gather, validate, structure, critique, and explain information, while deterministic code performs feature generation, forecasting, leakage checks, and backtesting.

---

## Core Thesis

A serious financial agentic system should not allow an LLM to directly make unsupported trading decisions. Instead, the system should separate responsibilities:

```text
Agents: retrieve, extract, summarize, critique, supervise, and explain
Deterministic code: validate data, materialize features, run models, backtest, audit leakage
Human/supervisor layer: approve, retry, reroute, or stop risky workflows
```

The project demonstrates that agentic modeling can be made:

```text
point-in-time
traceable
auditable
reproducible
low-leakage
experiment-driven
risk-aware
```

---

## Why SOXX First, SOXL Second

The original idea was to predict SOXL directly. However, SOXL is a 3x daily leveraged ETF, which makes it a harder and noisier target.

SOXL is affected by:

```text
daily leverage reset
path dependency
volatility decay
financing and fee drag
larger drawdowns
larger gap risk
higher noise
```

Therefore, the cleaner design is:

```text
1. Forecast the semiconductor sector using SOXX or the underlying semiconductor index.
2. Forecast risk/volatility/drawdown of the sector.
3. Translate the SOXX forecast into a SOXL-aware signal using leverage, compounding, and volatility-risk adjustment.
```

### Recommended Target Setup

```text
Primary target: SOXX 1-day and 5-day forward return
Secondary target: SOXX realized volatility / drawdown risk
Derived output: SOXL expected move and risk-adjusted trade signal
```

### Interview Framing

> I initially considered predicting SOXL directly, but SOXL is a daily-reset leveraged product. Directly modeling its price would mix semiconductor-sector signal with leverage path-dependency and volatility decay. I therefore separated the problem into two layers: first, forecast semiconductor-sector return using SOXX or the underlying index; second, translate that forecast into a SOXL-aware risk-adjusted signal.

---

## What “Point-in-Time” Means

A point-in-time platform only uses information that was actually available at the moment the prediction would have been made.

For example, suppose the system predicts after market close on:

```text
2026-05-07 16:00:00 America/New_York
```

The system is allowed to use:

```text
SOXX/SOXL price history up to the May 7 close
ETF holdings published before the cutoff time
SEC filings accepted before the cutoff time
news articles published before the cutoff time
macro data released before the cutoff time
earnings reports already released before the cutoff time
```

The system is not allowed to use:

```text
May 8 price movement
news published after the cutoff
filings released after the cutoff
revised macro data unavailable at the cutoff
future ETF holdings
future analyst upgrades/downgrades
current company fundamentals when backtesting a past date
```

### Why This Matters

Without point-in-time control, backtests can accidentally cheat.

Example: if the system backtests a prediction for January 2024 but uses today’s SOXX or SOXL holdings, it has leaked future knowledge. The ETF composition may have changed.

Another example: a company files a 10-Q on May 8, but the filing covers the quarter ending March 31. If the model is predicting on April 15, it cannot use that filing because the filing was not available yet.

The system must always ask:

> As of this prediction time, what did we actually know?

### Example Point-in-Time Tool Calls

```python
get_etf_holdings(
    etf="SOXX",
    as_of_time="2026-05-07T16:00:00-04:00"
)
```

```python
get_sec_filings(
    ticker="NVDA",
    forms=["10-Q", "10-K", "8-K"],
    available_before="2026-05-07T16:00:00-04:00"
)
```

```python
search_news(
    ticker="AMD",
    published_before="2026-05-07T16:00:00-04:00"
)
```

```python
get_macro_data(
    series="10Y Treasury Yield",
    available_before="2026-05-07T16:00:00-04:00"
)
```

### Example LangSmith Trace Metadata

```json
{
  "prediction_timestamp": "2026-05-07T16:00:00-04:00",
  "allowed_data_cutoff": "2026-05-07T16:00:00-04:00",
  "accepted_holdings_snapshot": "dated_before_cutoff",
  "accepted_price_data": "up_to_market_close",
  "rejected_articles": 3,
  "rejected_reason": "published_after_cutoff",
  "rejected_filings": 1,
  "rejected_filing_reason": "available_after_prediction_time"
}
```

---

## High-Level System Architecture

```text
User / Research Request
        |
        v
LangGraph Orchestrator
        |
        |-- Resolve universe and holdings
        |-- Pull market data
        |-- Pull SEC filings and fundamentals
        |-- Pull macro data
        |-- Retrieve news and events
        |-- Enforce point-in-time constraints
        |-- Extract structured event signals
        |-- Materialize features
        |-- Run forecasting models
        |-- Run leakage audit and backtest
        |-- Run critic/supervisor review
        |-- Generate final research memo
        v
Auditable Output
        |
        |-- Forecast
        |-- Confidence
        |-- Risk flags
        |-- Feature attribution
        |-- Backtest metrics
        |-- Source citations / artifact links
        |-- LangSmith trace
```

---

## Recommended Agent Roles

### 1. Orchestrator / LangGraph Controller

Owns the workflow state.

Responsibilities:

```text
track as_of_time
track prediction horizon
track universe snapshot
track artifact URIs
track feature versions
track model config
track LangSmith run IDs
control graph routing
```

This should be mostly deterministic.

---

### 2. Universe and Holdings Agent

Builds the ETF universe.

Responsibilities:

```text
pull SOXX holdings
optionally pull SOXL holdings for instrument comparison
map holdings to tickers, sectors, CIKs, and weights
compute top-N weighted universe
snapshot holdings point-in-time
hash the holdings artifact
```

Why it matters:

```text
prevents survivorship bias
prevents using future holdings
links company-level signals to ETF-level exposure
```

---

### 3. Temporal Data Guard

Prevents future leakage.

Responsibilities:

```text
verify every artifact timestamp
reject news after as_of_time
reject filings after as_of_time
reject future price data
reject revised macro data unavailable at prediction time
log accepted and rejected artifacts
```

This is one of the most important components for a credible quant-style project.

---

### 4. SEC Filings Agent

Retrieves and structures company fundamental information.

Responsibilities:

```text
resolve ticker to CIK
retrieve latest 10-K, 10-Q, and 8-K filings before cutoff
retrieve company facts / XBRL concepts
extract relevant sections
produce structured filing signals
```

Possible extracted signals:

```text
revenue growth
gross margin change
inventory growth
capex trend
R&D trend
customer concentration risk
supply-chain risk
guidance commentary
risk-factor changes
```

---

### 5. Earnings / News / Event Agent

Extracts structured event signals from unstructured text.

Responsibilities:

```text
retrieve company-specific news
retrieve sector-wide semiconductor news
dedupe articles
extract structured event labels
link each extracted event to evidence
assign confidence
```

Possible event labels:

```text
earnings_guidance_up
earnings_guidance_down
ai_demand_strength
export_control_risk
inventory_glut
capex_acceleration
supply_chain_disruption
foundry_capacity_constraint
customer_concentration_risk
valuation_pressure
```

---

### 6. Macro and Market Regime Agent

Adds macro and regime context.

Responsibilities:

```text
pull interest rates
pull yield curve features
pull volatility features
pull dollar index / FX proxies if available
pull credit/risk sentiment proxies
pull semiconductor-sector momentum
identify volatility regime
```

The macro agent should be mostly deterministic.

---

### 7. Feature Engineering Agent

Converts raw artifacts into model-ready features.

Responsibilities:

```text
materialize features from market data
materialize weighted company-level features
materialize event-count and event-intensity features
materialize macro regime features
validate feature schemas
store feature matrix with version/hash
```

Example features:

```text
SOXX 5-day return
SOXX 20-day realized volatility
weighted top-10 constituent return
weighted positive AI-demand event score
weighted export-control risk score
weighted inventory growth
weighted capex acceleration
VIX regime indicator
rate-change feature
semiconductor breadth feature
```

---

### 8. Hypothesis Generator Agent

Proposes research hypotheses.

Responsibilities:

```text
suggest candidate features
suggest interactions
suggest ablations
suggest alternative targets
summarize why a feature might matter
```

Example hypotheses:

```text
Weighted inventory growth across semiconductor holdings predicts short-term downside risk.
AI capex commentary from top holdings predicts next-week semiconductor momentum.
Export-control news increases realized volatility more than it predicts direction.
SOXL should be avoided when SOXX signal is positive but volatility regime is extremely high.
```

This agent should not directly make trading decisions.

---

### 9. Forecasting and Backtest Agent

Runs deterministic models and evaluations.

Responsibilities:

```text
train models
run walk-forward validation
compare baselines
compute metrics
run transaction cost sensitivity
run drawdown analysis
run leakage checks
run ablation studies
```

Candidate models:

```text
logistic regression
elastic net
random forest
LightGBM / XGBoost
simple momentum baseline
mean-reversion baseline
volatility-regime baseline
```

Target types:

```text
SOXX 1-day return
SOXX 5-day return
SOXX direction
SOXX next-5-day realized volatility
SOXX next-5-day drawdown risk
SOXL leverage-aware derived return
```

---

### 10. Bull / Bear Critic Agents

Challenge the forecast from different perspectives.

Bull critic asks:

```text
What supports the positive semiconductor view?
Are earnings revisions improving?
Are top-weighted holdings showing momentum?
Is AI demand strengthening?
```

Bear critic asks:

```text
What contradicts the positive view?
Is the signal driven by one company only?
Is volatility too high for SOXL exposure?
Are there export-control or macro risks?
Is the backtest overfit?
```

These agents must be constrained to use existing artifacts and metrics. They should not hallucinate new facts.

---

### 11. Research Memo Agent

Produces the final output.

Responsibilities:

```text
summarize the forecast
explain the evidence
show risk flags
show confidence
show model metrics
show backtest comparison
cite artifacts
include LangSmith trace link
state whether SOXL signal is actionable, reduced-confidence, or rejected
```

---

## MCP Server Design

MCP should be used to expose typed tools and resources to the agent system. The goal is not to use random generic tools, but to create domain-specific finance/research MCP servers.

---

### 1. `etf_holdings_mcp`

Tools:

```text
get_etf_holdings(etf, as_of_time)
get_index_holdings(index_name, as_of_time)
compute_weighted_universe(etf, top_n, as_of_time)
map_tickers_to_ciks(tickers)
```

Resources:

```text
etf://SOXX/holdings/2026-05-07
etf://SOXL/holdings/2026-05-07
universe://semiconductor/top20/2026-05-07
```

---

### 2. `market_data_mcp`

Tools:

```text
get_ohlcv(ticker, start, end, interval)
get_adjusted_returns(tickers, start, end)
get_realized_vol(ticker, window)
get_drawdown(ticker, window)
get_sector_breadth(tickers, as_of_time)
```

Resources:

```text
prices://SOXX/daily/2024-01-01_2026-05-07
prices://SOXL/daily/2024-01-01_2026-05-07
returns://semiconductor_universe/top20/2026-05-07
```

---

### 3. `sec_filings_mcp`

Tools:

```text
resolve_ticker_to_cik(ticker)
get_latest_filings(cik, forms, before)
get_companyfacts(cik, concepts, as_of_time)
extract_filing_section(cik, accession, section)
extract_xbrl_features(cik, concepts, as_of_time)
```

Resources:

```text
sec://CIK0001045810/10Q/latest_before/2026-05-07
sec://CIK0001045810/companyfacts/asof/2026-05-07
```

---

### 4. `macro_mcp`

Tools:

```text
get_fred_series(series_id, start, end)
get_release_calendar(series_id)
get_vintage_series(series_id, vintage_date)
compute_rate_change_features(series_id, as_of_time)
```

Resources:

```text
fred://DGS10/observations
fred://VIXCLS/observations
fred://macro_snapshot/2026-05-07
```

---

### 5. `news_event_mcp`

Tools:

```text
search_company_news(ticker, start, end, published_before)
search_sector_news(query, start, end, published_before)
dedupe_articles(article_ids)
extract_event_signal(article_ids, schema)
```

Resources:

```text
news://NVDA/2026-04-01_2026-05-07
news://semiconductor_sector/2026-04-01_2026-05-07
```

---

### 6. `feature_store_mcp`

Tools:

```text
materialize_features(as_of_time, universe_hash, feature_set)
get_feature_matrix(run_id)
validate_schema(dataset_id)
check_missingness(dataset_id)
compute_feature_hash(dataset_id)
```

Resources:

```text
features://soxx/v3/asof=2026-05-07
features://soxx/v3/schema.json
```

---

### 7. `backtest_mcp`

Tools:

```text
run_walkforward(config)
score_signal(signal_id)
run_leakage_audit(dataset_id)
compare_against_baselines(run_id)
run_transaction_cost_sensitivity(run_id)
compute_drawdown_metrics(run_id)
```

Resources:

```text
backtest://run/abc123/metrics.json
backtest://run/abc123/equity_curve.parquet
backtest://run/abc123/leakage_report.json
```

---

### 8. `research_memory_mcp`

Tools:

```text
save_hypothesis(text, artifacts)
get_prior_hypotheses(topic)
mark_hypothesis_rejected(reason)
retrieve_failed_experiments(topic)
```

Resources:

```text
memory://hypotheses/semiconductor_inventory_cycle
memory://failed_experiments/soxx_event_features
```

---

## LangGraph Design

LangGraph should be the deterministic control plane.

### Example State Object

```python
from typing import TypedDict, Literal

class ResearchState(TypedDict):
    as_of_time: str
    horizon: Literal["1d", "5d", "20d"]
    primary_target: Literal["SOXX_return", "SOXX_direction", "SOXX_volatility"]
    derived_target: Literal["SOXL_expected_return", "SOXL_risk_adjusted_signal"]
    universe_snapshot_uri: str
    universe_hash: str
    raw_artifacts: list[str]
    accepted_artifacts: list[str]
    rejected_artifacts: list[str]
    extracted_events_uri: str
    feature_matrix_uri: str
    leakage_audit_uri: str
    model_predictions: list[dict]
    backtest_metrics: dict
    risk_flags: list[str]
    supervisor_decision: dict
    final_memo_uri: str
```

### Graph Flow

```text
START
  -> ResolveUniverse
  -> PullMarketData
  -> PullSECData
  -> PullMacroData
  -> RetrieveNewsAndEvents
  -> TemporalGuard
  -> ExtractStructuredSignals
  -> MaterializeFeatures
  -> RunForecastModels
  -> RunBacktests
  -> SupervisorReview
  -> CriticReview
  -> GenerateResearchMemo
END
```

---

## LangSmith Usage

LangSmith should be used for observability, debugging, evaluation, and reproducibility.

Track:

```text
every graph run
every node input and output
every tool call
every rejected artifact
every structured extraction
every model output
every backtest result
every Claude supervisor intervention
every final memo
```

### Recommended Metadata

```json
{
  "project": "soxx-soxl-agentic-research-lab",
  "as_of_time": "2026-05-07T16:00:00-04:00",
  "horizon": "1d",
  "primary_target": "SOXX_1d_forward_return",
  "derived_target": "SOXL_expected_return",
  "universe_hash": "sha256:...",
  "feature_set": "soxx_v3_point_in_time",
  "data_snapshot": "snapshot_20260507",
  "prompt_versions": {
    "event_extractor": "v2.1",
    "critic": "v1.4",
    "supervisor": "v1.2"
  },
  "model_versions": {
    "extractor_model": "claude_or_openai_model_name",
    "forecast_model": "lightgbm_v0.3"
  }
}
```

---

## Dynamic Claude Intervention Layer

A strong addition is to include Claude as a dynamic supervisory intervention layer that monitors LangGraph execution.

Claude should not be the trader. Claude should be the runtime supervisor.

### Core Idea

```text
LangGraph pipeline
  |
  | streams node status, artifacts, tool calls, confidence, errors
  v
Monitoring / Supervisor layer
  |
  | Claude reviews state summaries, risk flags, and traces
  v
Intervention decision
  |
  | approve / retry / reroute / request human review / stop
  v
LangGraph resumes with updated state
```

### Claude Supervisor Responsibilities

```text
monitor graph progress
review artifact quality
review timestamp violations
review schema failures
review missing high-weight holdings
review suspicious features
review backtest instability
review overfitting risk
request retries
route to fallback tools
lower confidence
request human review
stop unsafe or low-quality runs
```

### Intervention Points

Claude should intervene only at gated checkpoints, not after every node.

Good checkpoints:

```text
After data ingestion
After event extraction
After feature materialization
After leakage audit
After backtest
Before final memo
```

### Constrained Action Space

Claude should only be allowed to choose from predefined actions:

```text
approve
retry_current_node
retry_specific_previous_node
route_to_fallback_source
drop_suspicious_feature
lower_confidence
request_human_review
stop_run
```

This keeps the system auditable and controlled.

### Example Supervisor Input

```json
{
  "as_of_time": "2026-05-07T16:00:00-04:00",
  "current_node": "ExtractStructuredSignals",
  "completed_nodes": [
    "ResolveUniverse",
    "PullMarketData",
    "PullSECFilings",
    "RetrieveNews"
  ],
  "risk_flags": [
    "NVDA news source count unusually high",
    "Two articles published after cutoff were rejected",
    "SEC filing for AVGO missing"
  ],
  "artifact_summary": {
    "holdings_snapshot": "valid",
    "market_data": "valid",
    "news_extraction": "partial",
    "sec_data": "missing one top-10 constituent"
  },
  "allowed_actions": [
    "approve",
    "retry_node",
    "route_to_fallback_data",
    "request_human_review",
    "stop_run"
  ]
}
```

### Example Supervisor Output

```json
{
  "decision": "retry_node",
  "target_node": "PullSECFilings",
  "reason": "AVGO is a high-weight constituent and missing SEC data may bias weighted fundamental features.",
  "state_update": {
    "required_tickers": ["AVGO"],
    "fallback_source_allowed": true
  },
  "human_review_required": false
}
```

### Example LangGraph Supervisor Node

```python
def supervisor_node(state):
    review = claude_supervisor.invoke({
        "state_summary": summarize_state(state),
        "risk_flags": state["risk_flags"],
        "allowed_actions": [
            "approve",
            "retry_node",
            "route_to_fallback",
            "request_human_review",
            "stop_run"
        ],
    })

    return {
        "supervisor_decision": review["decision"],
        "supervisor_reason": review["reason"],
        "state_updates": review.get("state_update", {})
    }
```

### Example Routing Function

```python
def route_after_supervisor(state):
    decision = state["supervisor_decision"]

    if decision == "approve":
        return "next_stage"

    if decision == "retry_node":
        return state["state_updates"]["target_node"]

    if decision == "route_to_fallback":
        return "FallbackDataNode"

    if decision == "request_human_review":
        return "HumanReviewNode"

    if decision == "stop_run":
        return "END"
```

### Human Review Node

```python
from langgraph.types import interrupt

def human_review_node(state):
    decision = interrupt({
        "message": "Supervisor requested review",
        "reason": state["supervisor_reason"],
        "state_summary": summarize_state(state),
        "options": ["approve", "retry", "stop"]
    })

    return {"human_decision": decision}
```

### Best Interview Framing

> The system has two control loops. The inner loop is the LangGraph financial research pipeline. The outer loop is a Claude-based supervisory layer that monitors traces, risk flags, schema failures, and data-quality issues. It can dynamically pause execution, reroute to fallback tools, retry failed nodes, or escalate to human review. All interventions are recorded in LangSmith.

---

## Claude Code Role

Claude Code can be used as the development and debugging environment.

Claude Code responsibilities:

```text
inspect failed LangSmith traces
generate patches to LangGraph nodes
update MCP tool schemas
add regression tests for failed runs
write new evaluators
improve prompts after failures
refactor graph logic
create new synthetic test cases
```

Runtime Claude responsibilities:

```text
monitor graph state
review risk flags
approve/retry/reroute/stop
produce structured intervention decisions
request human review when needed
```

Important distinction:

```text
Claude Code builds and improves the system.
Claude runtime supervisor monitors and controls graph execution.
```

---

## Modeling Plan

### Feature Groups

#### Market / Technical Features

```text
SOXX returns
SOXL returns
SMH returns
top-holding weighted returns
realized volatility
rolling drawdown
volume spikes
gap risk
semiconductor breadth
momentum / mean-reversion features
```

#### ETF / Instrument Features

```text
SOXL leverage mapping
SOXL volatility decay proxy
SOXL compounding-path risk
SOXL drawdown risk
SOXL volume/liquidity features
```

#### Fundamental Features

```text
weighted revenue growth
weighted gross margin change
weighted inventory growth
weighted capex growth
weighted R&D growth
weighted free cash flow trend
filing recency
8-K event flags
```

#### LLM-Extracted Event Features

```text
weighted AI demand strength
weighted guidance up/down score
weighted export-control risk
weighted inventory-glut risk
weighted supply-chain disruption risk
weighted capex acceleration score
weighted customer concentration risk
weighted earnings-call positivity/negativity
```

#### Macro / Regime Features

```text
10Y Treasury yield
2Y Treasury yield
yield curve slope
VIX / volatility proxy
credit-risk proxy
dollar strength proxy
inflation surprise proxy
manufacturing regime proxy
risk-on/risk-off regime
```

---

## Forecasting and Evaluation

### Forecast Targets

```text
SOXX_1d_forward_return
SOXX_5d_forward_return
SOXX_1d_direction
SOXX_5d_direction
SOXX_next_5d_realized_volatility
SOXX_next_5d_drawdown
SOXL_expected_1d_return
SOXL_risk_adjusted_signal
```

### Baselines

```text
buy and hold
SOXX momentum
SOXX mean reversion
lagged return
volatility-regime baseline
market beta baseline
random forest without LLM features
LightGBM without LLM features
```

### Metrics

```text
hit rate
accuracy
precision/recall for direction
Brier score
log loss
RMSE / MAE for returns
information coefficient
Sharpe ratio
Sortino ratio
max drawdown
turnover
transaction-cost-adjusted return
calibration error
baseline uplift
```

### Ablation Studies

```text
price-only baseline
price + macro
price + macro + holdings
price + macro + holdings + SEC fundamentals
price + macro + holdings + SEC + LLM event features
full system with Claude supervisor intervention
```

---

## SOXX-to-SOXL Translation Layer

After forecasting SOXX, translate the result into a SOXL-aware output.

### Simple Approximation

```text
expected_SOXL_daily_return ≈ 3 × expected_SOXX_daily_return
```

### Risk-Aware Adjustment

Adjust confidence downward when:

```text
SOXX realized volatility is high
SOXX forecast confidence is low
constituent concentration is high
news risk is high
macro regime is unfavorable
recent SOXL drawdown is large
model disagreement is high
```

### Example Output

```text
SOXX forecast: +0.45% next day
Naive SOXL equivalent: +1.35%
Volatility regime: high
Risk-adjusted SOXL signal: weak bullish
Confidence: medium-low
Action: do not produce high-conviction leveraged signal
```

This is more credible than simply predicting SOXL price directly.

---

## Example End-to-End Tool Trace

```python
etf.get_etf_holdings(
    etf="SOXX",
    as_of_time="2026-05-07T16:00:00-04:00"
)
```

```python
market.get_adjusted_returns(
    tickers=["SOXX", "SOXL", "NVDA", "AMD", "AVGO", "QCOM"],
    start="2024-01-01",
    end="2026-05-07"
)
```

```python
sec.resolve_ticker_to_cik(
    ticker="NVDA"
)
```

```python
sec.get_companyfacts(
    cik="0001045810",
    concepts=[
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "InventoryNet",
        "ResearchAndDevelopmentExpense",
        "PaymentsToAcquirePropertyPlantAndEquipment"
    ],
    as_of_time="2026-05-07T16:00:00-04:00"
)
```

```python
news.search_company_news(
    ticker="NVDA",
    start="2026-04-01",
    end="2026-05-07",
    published_before="2026-05-07T16:00:00-04:00"
)
```

```python
events.extract_event_signal(
    document_ids=["news:...", "sec:..."],
    schema="semiconductor_event_v2"
)
```

```python
features.materialize_features(
    as_of_time="2026-05-07T16:00:00-04:00",
    universe_hash="sha256:...",
    feature_set="soxx_v3_point_in_time"
)
```

```python
backtest.run_walkforward(
    target="SOXX_1d_forward_return",
    feature_set="soxx_v3_point_in_time",
    train_start="2021-01-01",
    test_start="2024-01-01",
    test_end="2026-05-07",
    baselines=[
        "SOXX_momentum",
        "SOXX_buy_hold",
        "lagged_return",
        "vol_regime"
    ]
)
```

```python
audit.run_leakage_audit(
    dataset_id="features://soxx/v3/asof=2026-05-07",
    target_col="SOXX_1d_forward_return"
)
```

---

## Final Demo Output

The demo should produce the following artifacts:

```text
1. LangSmith trace
2. Point-in-time data lineage report
3. Feature matrix artifact
4. Leakage audit report
5. Backtest report
6. Ablation study table
7. Claude supervisor intervention log
8. Final research memo
9. SOXX forecast
10. SOXL leverage-aware translated signal
```

### Example Final Research Memo Structure

```text
Title: Semiconductor ETF Forecast as of 2026-05-07 Close

1. Executive Summary
2. Forecast
3. SOXX Signal
4. SOXL Leverage-Aware Translation
5. Supporting Evidence
6. Contradictory Evidence
7. Feature Attribution
8. Risk Flags
9. Backtest Context
10. Supervisor Interventions
11. Final Confidence
12. Appendix: Data Artifacts and LangSmith Trace
```

---

## Step-by-Step Implementation Roadmap

Build the project in layers. The main rule is:

```text
First make a boring deterministic backtest that cannot cheat.
Then add agents around it.
Only wrap tools as MCP servers after their Python interfaces stabilize.
```

### Step 1: Deterministic Market-Data MVP

Status: implemented.

Current files:

Paths are relative to `SOXX_agentic/`.

```text
configs/soxx_mvp.json
scripts/run_soxx_mvp.py
soxx_mvp/data.py
soxx_mvp/features.py
soxx_mvp/models.py
soxx_mvp/backtest.py
soxx_mvp/metrics.py
soxx_mvp/io_utils.py
docs/soxx_step1_mvp.md
```

Current flow:

```text
configs/soxx_mvp.json
  -> scripts/run_soxx_mvp.py
  -> soxx_mvp/data.py
  -> soxx_mvp/features.py
  -> soxx_mvp/backtest.py
  -> soxx_mvp/models.py
  -> soxx_mvp/metrics.py
  -> artifacts/
```

What this step does:

```text
pull SOXX/SOXL/SMH daily prices
cache market data locally
build fixed market features
create SOXX 1-day and 5-day forward targets
train simple deterministic models
run bounded walk-forward backtests
write features, predictions, metrics, and latest forecast artifacts
```

Important limitation:

```text
This is not agentic yet. Feature definitions and model config are hand-written.
```

Done criteria:

```text
sample-data run succeeds
live-data run succeeds
features.csv is written
predictions_h1.csv and predictions_h5.csv are written
metrics.json is written
latest_forecast.json is written
```

### Step 2: Full Walk-Forward Backtest and Baselines

Status: implemented.

Goal: turn the MVP from a smoke test into a meaningful baseline.

Add:

```text
full historical walk-forward option
buy-and-hold baseline
momentum baseline
mean-reversion baseline
volatility-regime baseline
transaction-cost sensitivity
separate train/validation/test date windows
```

Files:

```text
configs/backtest_full.json
soxx_mvp/baselines.py
soxx_mvp/reports.py
tests/test_backtest_baselines.py
```

Acceptance criteria:

```text
metrics compare model vs baselines
results include drawdown, turnover, costs, and hit rate
strategy performance is not reported without baseline context
```

### Step 3: Point-in-Time Validation and Leakage Audit

Status: implemented.

Goal: make the backtest credible before adding LLMs.

Add:

```text
as_of_date validation
feature availability timestamps
target-shift checks
training-window checks
future-data leakage checks
artifact hashes
validation report
```

Files:

```text
soxx_mvp/temporal.py
soxx_mvp/leakage.py
soxx_mvp/artifacts.py
tests/test_temporal_guard.py
tests/test_leakage_audit.py
```

Acceptance criteria:

```text
no training row can use a target label unavailable at prediction time
no feature timestamp can be later than the prediction timestamp
validation fails loudly on intentionally leaked synthetic data
```

### Step 4: Config-Driven Feature Selection

Status: implemented.

Goal: prepare for agents to propose feature sets without letting them bypass rules.

Change from:

```text
features.py defines all features and feature_columns() selects all approved inputs
```

To:

```text
feature registry defines allowed features
config selects a subset of allowed features
code rejects unknown or unavailable features
```

Likely files:

```text
configs/feature_sets.json
soxx_mvp/feature_registry.py
tests/test_feature_schema.py
```

Acceptance criteria:

```text
config can choose feature subsets
unknown feature names fail validation
feature matrix includes only approved columns
backtest records feature_set hash
```

### Step 5: LangGraph Wrapper Around Deterministic Nodes

Goal: introduce graph orchestration without changing the model logic.

Wrap the existing deterministic steps as graph nodes:

```text
LoadConfig
PullMarketData
MaterializeFeatures
ValidatePointInTimeData
RunForecastModels
RunBacktest
WriteArtifacts
```

Likely files:

```text
src/graph/state.py
src/graph/nodes.py
src/graph/main_graph.py
src/graph/routing.py
configs/graph.yaml
```

Acceptance criteria:

```text
graph run produces the same artifacts as scripts/run_soxx_mvp.py
node inputs and outputs are structured
failures stop at the correct node
```

### Step 6: LangSmith Tracing

Goal: make every run inspectable and reproducible.

Trace:

```text
config path and config hash
as_of_date
feature set
feature matrix hash
model config
prediction artifacts
backtest metrics
leakage-audit result
```

Likely files:

```text
src/utils/tracing.py
src/graph/nodes.py
configs/langsmith.yaml
```

Acceptance criteria:

```text
every graph run has a trace
every artifact path is logged
every model metric is attached to trace metadata
```

### Step 7: Research Memo Agent

Goal: add the safest first LLM component.

The memo agent reads existing artifacts only:

```text
metrics.json
latest_forecast.json
feature attribution report
leakage report
backtest report
```

It produces:

```text
research_memo.md
```

Rules:

```text
no new market facts
no unsupported claims
must cite artifact paths
must separate forecast from backtest context
must state limitations
```

Likely files:

```text
src/agents/memo_agent.py
src/prompts/research_memo_v1.md
tests/test_memo_artifact_citations.py
```

Acceptance criteria:

```text
memo summarizes actual metrics
memo cites artifacts
memo does not invent unseen data
```

### Step 8: Hypothesis and Feature Proposal Agent

Goal: let an agent influence what experiments get tested, not directly make predictions.

The agent may propose:

```text
feature subsets
new allowed feature candidates
model family from an approved list
target horizon
ablation plan
evaluation window
```

The agent may not:

```text
invent arbitrary feature columns
skip validation
train a model directly
change the target after seeing test results
override leakage failures
```

Flow:

```text
HypothesisAgent proposes experiment_config
  -> config validator checks allowed values
  -> deterministic backtest runs
  -> metrics decide whether the idea survives
```

Likely files:

```text
src/agents/hypothesis_agent.py
src/prompts/hypothesis_agent_v1.md
soxx_mvp/experiment_config.py
tests/test_experiment_config_validation.py
```

Acceptance criteria:

```text
invalid proposed feature fails validation
accepted proposal becomes a saved config
backtest output is comparable to baselines
failed hypotheses are saved, not hidden
```

### Step 9: Macro and SEC Data Integration

Goal: add non-price structured features before unstructured news.

Add:

```text
FRED macro series
vintage/release-date checks where possible
SEC ticker-to-CIK mapping
latest filings before cutoff
companyfacts/XBRL features for top holdings
weighted fundamental features
```

Likely files:

```text
src/data/macro.py
src/data/sec.py
src/features/fundamentals.py
src/features/macro.py
tests/test_sec_available_before.py
tests/test_macro_release_dates.py
```

Acceptance criteria:

```text
features include available_before timestamps
top-holding weighted features are reproducible
missing company data is reported, not silently filled
```

### Step 10: LLM Event Extraction Agent

Goal: use LLMs where they add real value: structuring unstructured news/events.

Add:

```text
timestamped news retrieval
article deduplication
event schema
LLM extraction with evidence spans
event confidence scores
weighted ETF-level event features
```

Example event features:

```text
weighted_ai_demand_score_20d
weighted_export_control_risk_20d
weighted_inventory_glut_risk_20d
weighted_guidance_up_score_20d
```

Likely files:

```text
src/agents/news_event_agent.py
src/prompts/event_extractor_v1.md
src/features/event_features.py
tests/test_event_schema.py
tests/test_event_timestamp_filter.py
```

Acceptance criteria:

```text
every event has source URL, published timestamp, evidence, and confidence
events after as_of_time are rejected
backtest compares with and without LLM event features
```

### Step 11: MCP Wrappers Around Stable Tools

Goal: expose stable functionality as typed tools after Python functions work.

Do not build MCP first. Wrap only mature functions:

```text
market_data_mcp
feature_store_mcp
backtest_mcp
macro_mcp
sec_filings_mcp
news_event_mcp
research_memory_mcp
```

Likely files:

```text
src/mcp_servers/market_data_mcp/
src/mcp_servers/feature_store_mcp/
src/mcp_servers/backtest_mcp/
src/mcp_servers/sec_filings_mcp/
src/mcp_servers/macro_mcp/
src/mcp_servers/news_event_mcp/
```

Acceptance criteria:

```text
MCP tools return typed JSON
MCP tools enforce as_of_time
MCP tools expose artifact URIs instead of large blobs when appropriate
graph can call MCP tools instead of direct Python functions
```

### Step 12: Critic and Supervisor Agents

Goal: add governed agentic control after the deterministic system works.

Critic agents:

```text
bull critic
bear critic
leakage critic
overfit critic
```

Supervisor actions:

```text
approve
retry_current_node
route_to_fallback_source
drop_suspicious_feature
lower_confidence
request_human_review
stop_run
```

Likely files:

```text
src/agents/critic_agents.py
src/graph/supervisor.py
src/prompts/supervisor_v1.md
configs/supervisor_policy.yaml
tests/test_supervisor_policy.py
```

Acceptance criteria:

```text
supervisor has constrained action space
all interventions are logged
human review can interrupt and resume graph execution
supervisor cannot freely rewrite code or bypass validation
```

### Step 13: Final Demo and Portfolio Polish

Goal: make the project easy to understand and inspect.

Deliverables:

```text
README with architecture diagram
demo notebook
example LangSmith trace
sample research memo
backtest report
leakage audit report
ablation table
resume bullet
```

Acceptance criteria:

```text
one command runs deterministic MVP
one command runs graph workflow
one example trace can be opened and explained
memo links back to artifacts
limitations are explicit
```

---

## Repository Structure

```text
soxx-soxl-agentic-research-lab/

  README.md
  pyproject.toml
  .env.example

  configs/
    graph.yaml
    model_config.yaml
    feature_sets.yaml
    supervisor_policy.yaml

  src/
    graph/
      main_graph.py
      state.py
      nodes.py
      routing.py
      supervisor.py

    agents/
      universe_agent.py
      sec_agent.py
      news_event_agent.py
      macro_agent.py
      feature_agent.py
      critic_agents.py
      memo_agent.py

    mcp_servers/
      etf_holdings_mcp/
      market_data_mcp/
      sec_filings_mcp/
      macro_mcp/
      news_event_mcp/
      feature_store_mcp/
      backtest_mcp/
      research_memory_mcp/

    features/
      build_features.py
      schemas.py
      validation.py
      leakage_audit.py

    models/
      train.py
      predict.py
      baselines.py
      evaluation.py

    backtest/
      walkforward.py
      metrics.py
      transaction_costs.py
      reports.py

    prompts/
      event_extractor_v1.md
      supervisor_v1.md
      bull_critic_v1.md
      bear_critic_v1.md
      research_memo_v1.md

    utils/
      time.py
      hashing.py
      artifact_store.py
      logging.py

  notebooks/
    01_data_exploration.ipynb
    02_feature_ablation.ipynb
    03_backtest_review.ipynb

  tests/
    test_temporal_guard.py
    test_feature_schema.py
    test_leakage_audit.py
    test_supervisor_policy.py
    test_backtest_baselines.py

  artifacts/
    .gitkeep

  docs/
    architecture.md
    point_in_time_design.md
    supervisor_layer.md
    mcp_tooling.md
    demo_script.md
```

---

## Key Design Principles

```text
Use agents for ambiguity.
Use deterministic code for correctness.
Use point-in-time data for honest evaluation.
Use LangGraph for controlled orchestration.
Use LangSmith for tracing and evaluation.
Use MCP for typed, auditable tool interfaces.
Use Claude as supervisor, not trader.
Use SOXX as the economic target.
Use SOXL as the leverage-aware instrument layer.
```

---

## Main Pitfalls to Avoid

```text
Do not let the LLM directly make unsupported buy/sell decisions.
Do not backtest with current ETF holdings.
Do not use filings before they were actually accepted.
Do not use macro data revisions unavailable at prediction time.
Do not scrape random sources when official sources exist.
Do not report only profit; include drawdown, turnover, costs, and baselines.
Do not hide failed hypotheses.
Do not over-agent deterministic tasks.
Do not allow Claude to freely rewrite the graph during runtime.
```

---

## Strong Resume Bullet

> Built a point-in-time agentic financial research platform for SOXX/SOXL semiconductor ETF forecasting using LangGraph orchestration, LangSmith tracing/evaluation, custom MCP servers for SEC/FRED/ETF/market data, LLM-based event extraction, deterministic feature pipelines, leakage audits, Claude-based runtime supervision, and walk-forward backtesting.

---

## Strong Interview Pitch

> I built a point-in-time agentic semiconductor ETF research system. Instead of asking an LLM to directly predict SOXL, I separated the problem into a cleaner SOXX sector-forecasting layer and a SOXL leverage-aware translation layer. LangGraph controls the workflow, MCP servers expose typed financial data tools, LangSmith traces every tool call and artifact, and a Claude supervisor monitors execution to retry, reroute, stop, or escalate when data-quality or leakage risks appear. The result is not just a trading demo, but an auditable agentic research platform with reproducible backtests and traceable decisions.

---

## One-Sentence Summary

This project demonstrates how to build a governed, point-in-time, multi-agent financial research system where agents retrieve and structure information, deterministic models evaluate signals, Claude supervises graph execution, and every decision is traceable through LangSmith.



v0.1 deterministic sklearn backtest
v0.2 point-in-time validation + leakage audit
v0.3 LangGraph wrapper
v0.4 LangSmith tracing
v0.5 memo agent
v0.6 hypothesis/feature agent
v0.7 news/event extraction
v0.8 MCP wrappers
v1.0 supervisor + critic agents
