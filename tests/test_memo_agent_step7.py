from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from soxx_mvp.agents import memo_agent
from soxx_mvp.agents.memo_agent import (
    ClaudeMemoProvider,
    MemoArtifactBundle,
    MemoValidationError,
    MissingMemoArtifactError,
    TemplateMemoProvider,
    generate_research_memo,
    validate_memo_draft,
)


def _write_bundle(tmp_path: Path, *, leakage_status: str = "passed") -> Path:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    metrics = {
        "run_id": "step7-test",
        "sample_data": True,
        "feature_row_count": 120,
        "first_feature_date": "2020-01-02",
        "last_feature_date": "2020-12-31",
        "feature_set": {
            "name": "all_market_v1",
            "hash": "sha256:test-feature-set",
            "selected_feature_count": 2,
            "selected_feature_columns": ["soxx_ret_1d", "soxx_vol_20d"],
        },
        "point_in_time_validation": {
            "feature_set": {"status": "passed", "error_count": 0},
            "leakage": {"status": leakage_status, "error_count": 0},
        },
        "horizons": {
            "1d": {
                "horizon": 1,
                "prediction_count": 10,
                "selected_feature_count": 2,
                "strategy_comparison": [
                    {
                        "horizon": 1,
                        "split": "test",
                        "strategy": "model",
                        "transaction_cost_bps": 5.0,
                        "prediction_count": 10,
                        "accuracy": 0.6,
                        "mean_strategy_return": 0.001,
                        "cumulative_return": 0.12,
                        "sharpe": 1.2,
                        "max_drawdown": -0.05,
                        "average_turnover": 0.25,
                    },
                    {
                        "horizon": 1,
                        "split": "test",
                        "strategy": "momentum",
                        "transaction_cost_bps": 5.0,
                        "prediction_count": 10,
                        "accuracy": 0.55,
                        "mean_strategy_return": 0.0005,
                        "cumulative_return": 0.06,
                        "sharpe": 0.7,
                        "max_drawdown": -0.08,
                        "average_turnover": 0.2,
                    },
                ],
            }
        },
    }
    latest_forecast = {
        "forecasts": [
            {
                "as_of_date": "2020-12-31",
                "horizon": 1,
                "n_train": 60,
                "naive_soxl_expected_return": 0.03,
                "pred_direction": 1,
                "pred_return": 0.01,
                "prob_up": 0.62,
                "train_end_date": "2020-12-30",
                "train_start_date": "2020-10-01",
            }
        ]
    }
    leakage_report = {
        "status": leakage_status,
        "error_count": 0,
        "as_of_date": "2020-12-31",
        "checks": [],
        "errors": [],
    }
    manifest = {
        "run_id": "step7-test",
        "config_hash": "sha256:test-config",
        "feature_set": metrics["feature_set"],
        "artifact_count": 5,
        "artifacts": {
            "metrics.json": "sha256:metrics",
            "latest_forecast.json": "sha256:forecast",
            "backtest_report.md": "sha256:report",
            "leakage_report.json": "sha256:leakage",
            "validation_report.md": "sha256:validation",
        },
        "source_files": {},
    }

    _write_json(artifact_dir / "metrics.json", metrics)
    _write_json(artifact_dir / "latest_forecast.json", latest_forecast)
    (artifact_dir / "backtest_report.md").write_text("# Backtest\n", encoding="utf-8")
    _write_json(artifact_dir / "leakage_report.json", leakage_report)
    _write_json(artifact_dir / "artifact_manifest.json", manifest)
    (artifact_dir / "validation_report.md").write_text("# Validation\n", encoding="utf-8")
    return artifact_dir


def _valid_draft(**overrides: Any) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "headline": "SOXX/SOXL post-run memo",
        "overview": [
            {
                "text": "The run completed with a registered feature set.",
                "citations": ["metrics.json", "artifact_manifest.json"],
            }
        ],
        "latest_forecast": [
            {
                "text": "The latest forecast artifact contains the current horizon outputs.",
                "citations": ["latest_forecast.json"],
            }
        ],
        "backtest_context": [
            {
                "text": "Historical model context is available in the backtest artifacts.",
                "citations": ["metrics.json", "backtest_report.md"],
            }
        ],
        "validation": [
            {
                "text": "The leakage report passed.",
                "citations": ["leakage_report.json", "validation_report.md"],
            }
        ],
        "limitations": [
            {
                "text": "No feature attribution report is available for this run.",
                "citations": ["artifact_manifest.json"],
            }
        ],
        "disclaimer": "This memo is for research review only and is not investment advice.",
    }
    draft.update(overrides)
    return draft


def test_memo_artifact_bundle_loads_required_and_optional_artifacts(tmp_path: Path) -> None:
    artifact_dir = _write_bundle(tmp_path)

    bundle = MemoArtifactBundle.from_dir(artifact_dir)

    assert bundle.run_id == "step7-test"
    assert bundle.metrics["feature_set"]["name"] == "all_market_v1"
    assert "metrics.json" in bundle.citation_names
    assert "validation_report.md" in bundle.citation_names


def test_missing_required_memo_artifact_fails(tmp_path: Path) -> None:
    artifact_dir = _write_bundle(tmp_path)
    (artifact_dir / "metrics.json").unlink()

    with pytest.raises(MissingMemoArtifactError, match="metrics.json"):
        MemoArtifactBundle.from_dir(artifact_dir)


def test_leakage_failure_blocks_memo_generation(tmp_path: Path) -> None:
    artifact_dir = _write_bundle(tmp_path, leakage_status="failed")

    with pytest.raises(MemoValidationError, match="leakage status"):
        MemoArtifactBundle.from_dir(artifact_dir)


def test_memo_draft_requires_citations(tmp_path: Path) -> None:
    bundle = MemoArtifactBundle.from_dir(_write_bundle(tmp_path))
    draft = _valid_draft(overview=[{"text": "Unsupported claim", "citations": []}])

    with pytest.raises(MemoValidationError, match="citation"):
        validate_memo_draft(draft, bundle)


def test_memo_draft_rejects_unknown_citations(tmp_path: Path) -> None:
    bundle = MemoArtifactBundle.from_dir(_write_bundle(tmp_path))
    draft = _valid_draft(overview=[{"text": "Unsupported claim", "citations": ["unknown.json"]}])

    with pytest.raises(MemoValidationError, match="unknown artifact"):
        validate_memo_draft(draft, bundle)


def test_fake_claude_provider_output_writes_cited_memo(tmp_path: Path) -> None:
    artifact_dir = _write_bundle(tmp_path)
    output_path = artifact_dir / "research_memo.md"

    memo_path = generate_research_memo(
        artifact_dir=artifact_dir,
        output_path=output_path,
        provider=_FakeClaudeProvider(),
    )

    memo_text = memo_path.read_text(encoding="utf-8")
    assert "SOXX/SOXL Research Memo" in memo_text
    assert "0.620" in memo_text
    assert "1.200" in memo_text
    assert "feature attribution report available: `False`".lower() in memo_text.lower()
    assert str((artifact_dir / "metrics.json").resolve()) in memo_text
    assert "not investment advice" in memo_text


def test_claude_provider_uses_anthropic_messages_api(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_anthropic = _FakeAnthropicModule(json.dumps(_valid_draft()))
    monkeypatch.setattr(memo_agent.importlib, "import_module", lambda name: fake_anthropic)

    provider = ClaudeMemoProvider(model="claude-test-sonnet")
    draft = provider.create_memo(
        facts={"artifact_paths": {"metrics.json": "/tmp/metrics.json"}},
        prompt="System prompt",
    )

    assert draft["headline"] == "SOXX/SOXL post-run memo"
    assert fake_anthropic.client.messages.call["model"] == "claude-test-sonnet"
    assert fake_anthropic.client.messages.call["system"] == "System prompt"
    assert "output_config" not in fake_anthropic.client.messages.call
    assert "Return only valid JSON" in fake_anthropic.client.messages.call["messages"][0]["content"]


def test_template_provider_cli_smoke(tmp_path: Path) -> None:
    artifact_dir = _write_bundle(tmp_path)
    output_path = artifact_dir / "research_memo.md"
    project_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            "scripts/run_research_memo.py",
            "--artifact-dir",
            str(artifact_dir),
            "--provider",
            "template",
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=True,
    )

    memo_text = output_path.read_text(encoding="utf-8")
    assert "step7-test" in memo_text
    assert "Template" not in memo_text
    assert "not investment advice" in memo_text


class _FakeClaudeProvider:
    name = "claude"

    def create_memo(self, *, facts: dict[str, Any], prompt: str) -> dict[str, Any]:
        assert facts["run"]["run_id"] == "step7-test"
        assert "post-run research memo" in prompt.lower()
        return _valid_draft()


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.call: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.call = kwargs
        return _FakeResponse(self.text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


class _FakeAnthropicModule:
    def __init__(self, text: str) -> None:
        self.client = _FakeClient(text)

    def Anthropic(self) -> _FakeClient:
        return self.client


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
