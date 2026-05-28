from __future__ import annotations

from typing import Any


def render_backtest_report(metrics: dict[str, Any]) -> str:
    lines: list[str] = [
        "# SOXX/SOXL Deterministic Backtest Report",
        "",
        f"- Run ID: `{metrics.get('run_id', '')}`",
        f"- Sample data: `{metrics.get('sample_data', False)}`",
        f"- Feature rows: `{metrics.get('feature_row_count', 0)}`",
        f"- Feature date range: `{metrics.get('first_feature_date', '')}` to `{metrics.get('last_feature_date', '')}`",
        f"- Feature set: `{metrics.get('feature_set', {}).get('name', '')}`",
        f"- Feature set hash: `{metrics.get('feature_set', {}).get('hash', '')}`",
        "",
    ]

    for horizon, horizon_metrics in metrics.get("horizons", {}).items():
        lines.extend([f"## Horizon {horizon}", ""])
        comparison_rows = horizon_metrics.get("strategy_comparison", [])
        if not comparison_rows:
            lines.extend(["No strategy comparison rows were generated.", ""])
            continue

        lines.extend(
            [
                "| Split | Strategy | N | Accuracy | Mean Return | Cumulative Return | Sharpe | Max Drawdown | Turnover |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison_rows:
            lines.append(
                "| {split} | {strategy} | {n} | {accuracy:.3f} | {mean_return:.5f} | "
                "{cumulative_return:.3f} | {sharpe:.3f} | {max_drawdown:.3f} | {turnover:.3f} |".format(
                    split=row["split"],
                    strategy=row["strategy"],
                    n=int(row["prediction_count"]),
                    accuracy=float(row.get("accuracy", 0.0)),
                    mean_return=float(row.get("mean_strategy_return", 0.0)),
                    cumulative_return=float(row.get("cumulative_return", 0.0)),
                    sharpe=float(row.get("sharpe", 0.0)),
                    max_drawdown=float(row.get("max_drawdown", 0.0)),
                    turnover=float(row.get("average_turnover", 0.0)),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- The model strategy is compared against deterministic baselines on the same prediction dates.",
            "- Transaction costs are applied on position changes and reset at split boundaries.",
            "- SOXL output remains a derived leverage-aware translation of the SOXX forecast.",
            "",
        ]
    )
    return "\n".join(lines)


def render_validation_report(
    *,
    config_report: dict[str, Any],
    feature_report: dict[str, Any],
    leakage_report: dict[str, Any],
    feature_set_report: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = [
        "# SOXX/SOXL Point-in-Time Validation Report",
        "",
        f"- Overall status: `{leakage_report.get('status', 'unknown')}`",
        f"- As-of date: `{leakage_report.get('as_of_date', '')}`",
        f"- Feature rows: `{feature_report.get('summary', {}).get('row_count', 0)}`",
        f"- Feature date range: `{feature_report.get('summary', {}).get('first_feature_date', '')}` "
        f"to `{feature_report.get('summary', {}).get('last_feature_date', '')}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Errors |",
        "| --- | --- | ---: |",
        f"| config_dates | {config_report.get('status', '')} | {int(config_report.get('error_count', 0))} |",
        f"| feature_rows | {feature_report.get('status', '')} | {int(feature_report.get('error_count', 0))} |",
    ]

    if feature_set_report is not None:
        lines.append(
            "| feature_set | {status} | {errors} |".format(
                status=feature_set_report.get("status", ""),
                errors=int(feature_set_report.get("error_count", 0)),
            )
        )

    for check in leakage_report.get("checks", []):
        lines.append(
            "| {name} | {status} | {errors} |".format(
                name=check.get("name", ""),
                status=check.get("status", ""),
                errors=int(check.get("error_count", 0)),
            )
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "- Feature availability timestamps were checked against each feature row date.",
            "- The selected model feature set was checked against the approved feature registry.",
            "- Target availability timestamps were checked against each walk-forward prediction date.",
            "- Model feature columns were checked to keep target, actual, and strategy fields out of training.",
            "- Training windows were checked so labels are not used before they would be known.",
            "",
        ]
    )

    errors = list(leakage_report.get("errors", []))
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")

    return "\n".join(lines)
