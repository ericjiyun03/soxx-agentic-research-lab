from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from soxx_mvp.io_utils import ensure_dir, project_root, read_json


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_PROMPT_PATH = project_root() / "prompts" / "research_memo_v1.md"
REQUIRED_ARTIFACTS = (
    "metrics.json",
    "latest_forecast.json",
    "backtest_report.md",
    "leakage_report.json",
    "artifact_manifest.json",
)
OPTIONAL_ARTIFACTS = ("validation_report.md",)
MEMO_SECTIONS = (
    "overview",
    "latest_forecast",
    "backtest_context",
    "validation",
    "limitations",
)

MEMO_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "headline",
        "overview",
        "latest_forecast",
        "backtest_context",
        "validation",
        "limitations",
        "disclaimer",
    ],
    "properties": {
        "headline": {"type": "string"},
        "overview": {"$ref": "#/$defs/claim_list"},
        "latest_forecast": {"$ref": "#/$defs/claim_list"},
        "backtest_context": {"$ref": "#/$defs/claim_list"},
        "validation": {"$ref": "#/$defs/claim_list"},
        "limitations": {"$ref": "#/$defs/claim_list"},
        "disclaimer": {"type": "string"},
    },
    "$defs": {
        "claim_list": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
            },
        }
    },
}


class MemoAgentError(RuntimeError):
    """Base error for research memo generation."""


class MissingMemoArtifactError(MemoAgentError):
    """Raised when a required run artifact is missing."""


class MemoValidationError(MemoAgentError):
    """Raised when artifact state or provider output is unsafe to render."""


class MemoProvider(Protocol):
    name: str

    def create_memo(self, *, facts: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        """Return a validated-memo candidate as a JSON-like dictionary."""


@dataclass(frozen=True)
class MemoArtifactBundle:
    artifact_dir: Path
    metrics: dict[str, Any]
    latest_forecast: dict[str, Any]
    backtest_report: str
    leakage_report: dict[str, Any]
    artifact_manifest: dict[str, Any]
    validation_report: str | None
    artifact_paths: dict[str, Path]

    @classmethod
    def from_dir(cls, artifact_dir: Path | str) -> "MemoArtifactBundle":
        resolved_dir = Path(artifact_dir).expanduser().resolve()
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            raise MissingMemoArtifactError(f"Artifact directory does not exist: {resolved_dir}")

        artifact_paths: dict[str, Path] = {}
        missing: list[str] = []
        for filename in REQUIRED_ARTIFACTS:
            path = resolved_dir / filename
            if not path.exists():
                missing.append(filename)
            else:
                artifact_paths[filename] = path
        if missing:
            raise MissingMemoArtifactError(
                "Missing required memo artifacts: " + ", ".join(sorted(missing))
            )

        for filename in OPTIONAL_ARTIFACTS:
            path = resolved_dir / filename
            if path.exists():
                artifact_paths[filename] = path

        bundle = cls(
            artifact_dir=resolved_dir,
            metrics=_read_json_object(artifact_paths["metrics.json"]),
            latest_forecast=_read_json_object(artifact_paths["latest_forecast.json"]),
            backtest_report=artifact_paths["backtest_report.md"].read_text(encoding="utf-8"),
            leakage_report=_read_json_object(artifact_paths["leakage_report.json"]),
            artifact_manifest=_read_json_object(artifact_paths["artifact_manifest.json"]),
            validation_report=(
                artifact_paths["validation_report.md"].read_text(encoding="utf-8")
                if "validation_report.md" in artifact_paths
                else None
            ),
            artifact_paths=artifact_paths,
        )
        bundle.validate_artifact_state()
        return bundle

    @property
    def run_id(self) -> str:
        return str(self.metrics.get("run_id") or self.artifact_manifest.get("run_id") or "")

    @property
    def citation_names(self) -> set[str]:
        return set(self.artifact_paths)

    def citation_path(self, artifact_name: str) -> Path:
        return self.artifact_paths[artifact_name]

    def validate_artifact_state(self) -> None:
        leakage_status = str(self.leakage_report.get("status", "")).lower()
        if leakage_status != "passed":
            raise MemoValidationError(
                f"Cannot generate research memo when leakage status is {leakage_status or 'unknown'}"
            )

        metrics_leakage = (
            self.metrics.get("point_in_time_validation", {})
            .get("leakage", {})
            .get("status")
        )
        if metrics_leakage is not None and str(metrics_leakage).lower() != "passed":
            raise MemoValidationError(
                "Cannot generate research memo when metrics leakage summary is not passed"
            )


@dataclass(frozen=True)
class TemplateMemoProvider:
    name: str = "template"

    def create_memo(self, *, facts: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        del prompt
        run = facts.get("run", {})
        feature_set = facts.get("feature_set", {})
        forecasts = facts.get("latest_forecasts", [])
        horizons = facts.get("backtest_horizons", [])

        run_id = str(run.get("run_id") or "unknown")
        forecast_text = "Latest forecast artifacts were loaded for the configured horizons."
        if forecasts:
            labels = ", ".join(f"{item.get('horizon')}d" for item in forecasts)
            forecast_text = f"Latest forecast rows are available for horizons {labels}."

        backtest_text = "Backtest metrics were loaded for the configured horizons."
        if horizons:
            labels = ", ".join(str(item.get("horizon")) for item in horizons)
            backtest_text = f"Backtest context is available for horizons {labels}."

        return {
            "headline": f"SOXX/SOXL research memo for run {run_id}",
            "overview": [
                {
                    "text": (
                        "This memo summarizes the completed deterministic run "
                        f"{run_id} using feature set {feature_set.get('name', 'unknown')}."
                    ),
                    "citations": ["metrics.json", "artifact_manifest.json"],
                }
            ],
            "latest_forecast": [
                {
                    "text": forecast_text,
                    "citations": ["latest_forecast.json"],
                }
            ],
            "backtest_context": [
                {
                    "text": backtest_text,
                    "citations": ["metrics.json", "backtest_report.md"],
                }
            ],
            "validation": [
                {
                    "text": "The leakage audit passed for the artifacts used by this memo.",
                    "citations": ["leakage_report.json"],
                }
            ],
            "limitations": [
                {
                    "text": (
                        "No feature attribution report was generated for this run, "
                        "so this memo does not make feature-level attribution claims."
                    ),
                    "citations": ["artifact_manifest.json"],
                }
            ],
            "disclaimer": "This memo is for research review only and is not investment advice.",
        }


@dataclass(frozen=True)
class ClaudeMemoProvider:
    model: str = DEFAULT_CLAUDE_MODEL
    max_tokens: int = 2500
    name: str = "claude"

    def create_memo(self, *, facts: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        try:
            anthropic = importlib.import_module("anthropic")
        except ModuleNotFoundError as exc:
            raise MemoAgentError(
                "The anthropic package is not installed. Run: "
                "./.venv/bin/python -m pip install -r requirements.txt"
            ) from exc

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=prompt,
                messages=[
                    {
                        "role": "user",
                        "content": _claude_user_payload(facts),
                    }
                ],
            )
        except Exception as exc:
            raise MemoAgentError(f"Claude memo provider failed: {exc}") from exc
        return _parse_provider_json(_extract_response_text(response))


def generate_research_memo(
    *,
    artifact_dir: Path | str,
    output_path: Path | str | None = None,
    provider: MemoProvider | None = None,
    prompt_path: Path | str = DEFAULT_PROMPT_PATH,
) -> Path:
    bundle = MemoArtifactBundle.from_dir(artifact_dir)
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    facts = build_memo_facts(bundle)
    selected_provider = provider or TemplateMemoProvider()
    draft = selected_provider.create_memo(facts=facts, prompt=prompt)
    validate_memo_draft(draft, bundle)
    markdown = render_research_memo(
        bundle=bundle,
        facts=facts,
        draft=draft,
        provider_name=selected_provider.name,
    )

    final_path = Path(output_path) if output_path is not None else bundle.artifact_dir / "research_memo.md"
    if not final_path.is_absolute():
        final_path = (Path.cwd() / final_path).resolve()
    ensure_dir(final_path.parent)
    temporary_path = final_path.with_name(f".{final_path.name}.tmp")
    temporary_path.write_text(markdown, encoding="utf-8")
    temporary_path.replace(final_path)
    return final_path


def build_memo_facts(bundle: MemoArtifactBundle) -> dict[str, Any]:
    metrics = bundle.metrics
    forecasts = bundle.latest_forecast.get("forecasts", [])
    if not isinstance(forecasts, list):
        forecasts = []

    return {
        "run": {
            "run_id": bundle.run_id,
            "sample_data": bool(metrics.get("sample_data")),
            "feature_row_count": metrics.get("feature_row_count"),
            "first_feature_date": metrics.get("first_feature_date"),
            "last_feature_date": metrics.get("last_feature_date"),
        },
        "feature_set": _compact_feature_set(metrics.get("feature_set")),
        "latest_forecasts": [_compact_forecast(item) for item in forecasts if isinstance(item, dict)],
        "backtest_horizons": _compact_horizons(metrics.get("horizons")),
        "validation": {
            "leakage_status": bundle.leakage_report.get("status"),
            "leakage_error_count": bundle.leakage_report.get("error_count"),
            "validation_report_available": bundle.validation_report is not None,
        },
        "artifact_paths": {
            name: str(path.resolve())
            for name, path in sorted(bundle.artifact_paths.items())
        },
        "feature_attribution_report": {
            "available": False,
            "reason": "No feature attribution artifact is produced by the current pipeline.",
        },
    }


def validate_memo_draft(draft: Mapping[str, Any], bundle: MemoArtifactBundle) -> None:
    if not isinstance(draft, Mapping):
        raise MemoValidationError("Memo provider output must be a JSON object")

    headline = draft.get("headline")
    if not isinstance(headline, str) or not headline.strip():
        raise MemoValidationError("Memo provider output must include a non-empty headline")

    disclaimer = draft.get("disclaimer")
    if not isinstance(disclaimer, str) or not disclaimer.strip():
        raise MemoValidationError("Memo provider output must include a non-empty disclaimer")

    allowed = bundle.citation_names
    for section_name in MEMO_SECTIONS:
        claims = draft.get(section_name)
        if not isinstance(claims, list) or not claims:
            raise MemoValidationError(f"Memo section {section_name} must contain at least one claim")
        for idx, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                raise MemoValidationError(f"Memo section {section_name}[{idx}] must be an object")
            text = claim.get("text")
            if not isinstance(text, str) or not text.strip():
                raise MemoValidationError(f"Memo section {section_name}[{idx}] must include text")
            citations = claim.get("citations")
            if not isinstance(citations, list) or not citations:
                raise MemoValidationError(
                    f"Memo section {section_name}[{idx}] must include at least one citation"
                )
            for citation in citations:
                if not isinstance(citation, str) or not citation.strip():
                    raise MemoValidationError(
                        f"Memo section {section_name}[{idx}] has an invalid citation"
                    )
                if citation not in allowed:
                    raise MemoValidationError(
                        f"Memo section {section_name}[{idx}] cites unknown artifact {citation}"
                    )


def render_research_memo(
    *,
    bundle: MemoArtifactBundle,
    facts: Mapping[str, Any],
    draft: Mapping[str, Any],
    provider_name: str,
) -> str:
    lines: list[str] = [
        "# SOXX/SOXL Research Memo",
        "",
        f"- Run ID: `{bundle.run_id}`",
        f"- Artifact directory: `{bundle.artifact_dir}`",
        f"- Memo provider: `{provider_name}`",
        "- Scope: post-run analysis of existing artifacts only",
        "",
        "## Source Artifacts",
        "",
    ]
    for artifact_name, path in sorted(bundle.artifact_paths.items()):
        lines.append(f"- {_artifact_link(artifact_name, path)}")
    lines.append("")

    lines.extend(_render_run_facts(facts))
    lines.extend(_render_forecast_table(facts))
    lines.extend(_render_backtest_table(facts))
    lines.extend(_render_validation_facts(facts))

    lines.extend(["## Analyst Memo", "", f"### {str(draft['headline']).strip()}", ""])
    for section_name, title in (
        ("overview", "Overview"),
        ("latest_forecast", "Latest Forecast"),
        ("backtest_context", "Backtest Context"),
        ("validation", "Point-in-Time Checks"),
        ("limitations", "Limitations"),
    ):
        lines.extend([f"#### {title}", ""])
        for claim in draft[section_name]:
            lines.append(f"- {claim['text'].strip()} {_render_citations(claim['citations'], bundle)}")
        lines.append("")

    lines.extend(
        [
            "## Disclaimer",
            "",
            str(draft["disclaimer"]).strip(),
            "",
        ]
    )
    return "\n".join(lines)


def _render_run_facts(facts: Mapping[str, Any]) -> list[str]:
    run = facts.get("run", {})
    feature_set = facts.get("feature_set", {})
    return [
        "## Run Summary",
        "",
        f"- Feature rows: `{run.get('feature_row_count', '')}`",
        f"- Feature date range: `{run.get('first_feature_date', '')}` to `{run.get('last_feature_date', '')}`",
        f"- Feature set: `{feature_set.get('name', '')}`",
        f"- Feature set hash: `{feature_set.get('hash', '')}`",
        "",
    ]


def _render_forecast_table(facts: Mapping[str, Any]) -> list[str]:
    forecasts = facts.get("latest_forecasts", [])
    lines = ["## Latest Forecast Facts", ""]
    if not forecasts:
        return [*lines, "No latest forecast rows were available.", ""]

    lines.extend(
        [
            "| Horizon | As Of | Train End | Prob Up | Pred Direction | Pred Return | Naive SOXL Expected Return |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for forecast in forecasts:
        lines.append(
            "| {horizon} | {as_of} | {train_end} | {prob_up} | {direction} | {pred_return} | {soxl_return} |".format(
                horizon=_format_value(forecast.get("horizon")),
                as_of=_format_value(forecast.get("as_of_date")),
                train_end=_format_value(forecast.get("train_end_date")),
                prob_up=_format_float(forecast.get("prob_up"), digits=3),
                direction=_format_value(forecast.get("pred_direction")),
                pred_return=_format_float(forecast.get("pred_return"), digits=5),
                soxl_return=_format_float(
                    forecast.get("naive_soxl_expected_return"),
                    digits=5,
                ),
            )
        )
    lines.append("")
    return lines


def _render_backtest_table(facts: Mapping[str, Any]) -> list[str]:
    horizons = facts.get("backtest_horizons", [])
    lines = ["## Backtest Facts", ""]
    if not horizons:
        return [*lines, "No backtest horizon rows were available.", ""]

    lines.extend(
        [
            "| Horizon | Split | Model N | Model Accuracy | Model Sharpe | Model Cum Return | Best Baseline | Baseline Sharpe |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for horizon in horizons:
        model = horizon.get("model_row") or {}
        baseline = horizon.get("best_baseline_row") or {}
        lines.append(
            "| {horizon_name} | {split} | {n} | {accuracy} | {model_sharpe} | {model_cum} | {baseline_name} | {baseline_sharpe} |".format(
                horizon_name=_format_value(horizon.get("horizon")),
                split=_format_value(horizon.get("split")),
                n=_format_value(model.get("prediction_count")),
                accuracy=_format_float(model.get("accuracy"), digits=3),
                model_sharpe=_format_float(model.get("sharpe"), digits=3),
                model_cum=_format_float(model.get("cumulative_return"), digits=3),
                baseline_name=_format_value(baseline.get("strategy")),
                baseline_sharpe=_format_float(baseline.get("sharpe"), digits=3),
            )
        )
    lines.append("")
    return lines


def _render_validation_facts(facts: Mapping[str, Any]) -> list[str]:
    validation = facts.get("validation", {})
    attribution = facts.get("feature_attribution_report", {})
    return [
        "## Validation Facts",
        "",
        f"- Leakage status: `{validation.get('leakage_status', '')}`",
        f"- Leakage error count: `{validation.get('leakage_error_count', '')}`",
        f"- Validation report available: `{validation.get('validation_report_available', False)}`",
        f"- Feature attribution report available: `{attribution.get('available', False)}`",
        "",
    ]


def _compact_feature_set(raw_value: Any) -> dict[str, Any]:
    if not isinstance(raw_value, Mapping):
        return {}
    return {
        "name": raw_value.get("name"),
        "hash": raw_value.get("hash"),
        "selected_feature_count": raw_value.get("selected_feature_count"),
        "selected_feature_columns": list(raw_value.get("selected_feature_columns") or []),
    }


def _compact_forecast(raw_value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "horizon",
        "as_of_date",
        "train_start_date",
        "train_end_date",
        "n_train",
        "prob_up",
        "pred_direction",
        "pred_return",
        "naive_soxl_expected_return",
    )
    return {key: raw_value.get(key) for key in keys}


def _compact_horizons(raw_value: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_value, Mapping):
        return []

    compacted: list[dict[str, Any]] = []
    for horizon_name, horizon_metrics in sorted(raw_value.items()):
        if not isinstance(horizon_metrics, Mapping):
            continue
        strategy_rows = horizon_metrics.get("strategy_comparison") or []
        if not isinstance(strategy_rows, list):
            strategy_rows = []
        selected_rows = [row for row in strategy_rows if isinstance(row, Mapping)]
        split = _preferred_split(selected_rows)
        split_rows = [row for row in selected_rows if row.get("split") == split]
        model_row = _compact_strategy_row(
            next((row for row in split_rows if row.get("strategy") == "model"), {})
        )
        baselines = [
            _compact_strategy_row(row)
            for row in split_rows
            if row.get("strategy") and row.get("strategy") != "model"
        ]
        best_baseline = _best_by_sharpe(baselines)
        compacted.append(
            {
                "horizon": horizon_name,
                "split": split,
                "prediction_count": horizon_metrics.get("prediction_count"),
                "selected_feature_count": horizon_metrics.get("selected_feature_count"),
                "model_row": model_row,
                "best_baseline_row": best_baseline,
            }
        )
    return compacted


def _preferred_split(rows: list[Mapping[str, Any]]) -> str:
    available = {str(row.get("split")) for row in rows if row.get("split")}
    for split in ("test", "validation", "all"):
        if split in available:
            return split
    return sorted(available)[0] if available else ""


def _compact_strategy_row(raw_value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "strategy",
        "split",
        "prediction_count",
        "accuracy",
        "mean_strategy_return",
        "cumulative_return",
        "sharpe",
        "max_drawdown",
        "average_turnover",
    )
    return {key: raw_value.get(key) for key in keys}


def _best_by_sharpe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=_strategy_sharpe)


def _strategy_sharpe(row: Mapping[str, Any]) -> float:
    value = row.get("sharpe")
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _extract_response_text(response: Any) -> str:
    texts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    payload = "\n".join(texts).strip()
    if not payload:
        raise MemoValidationError("Claude response did not include text content")
    return payload


def _parse_provider_json(payload: str) -> dict[str, Any]:
    payload = _strip_json_fence(payload)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MemoValidationError(f"Memo provider returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MemoValidationError("Memo provider JSON must be an object")
    return parsed


def _claude_user_payload(facts: Mapping[str, Any]) -> str:
    return (
        "Return only valid JSON. Do not wrap the JSON in Markdown fences.\n\n"
        + json.dumps(
            {
                "facts": facts,
                "allowed_citations": list(facts.get("artifact_paths", {}).keys()),
                "output_schema": MEMO_OUTPUT_SCHEMA,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _strip_json_fence(payload: str) -> str:
    stripped = payload.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise MemoValidationError(f"Artifact must be a JSON object: {path}")
    return payload


def _render_citations(citations: list[str], bundle: MemoArtifactBundle) -> str:
    links = [_artifact_link(name, bundle.citation_path(name)) for name in citations]
    return "Sources: " + ", ".join(links)


def _artifact_link(name: str, path: Path) -> str:
    return f"[{name}](<{path.resolve()}>)"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _format_float(value: Any, *, digits: int) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def default_claude_model(env: Mapping[str, str] | None = None) -> str:
    values = env if env is not None else os.environ
    return values.get("SOXX_MEMO_MODEL") or DEFAULT_CLAUDE_MODEL
