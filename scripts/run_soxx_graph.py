#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soxx_mvp.io_utils import project_root
from soxx_mvp.pipeline import RunOptions, initial_pipeline_state, print_run_summary
from soxx_mvp.tracing import invoke_soxx_graph_with_tracing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic SOXX/SOXL MVP through LangGraph.")
    parser.add_argument("--config", type=Path, default=project_root() / "configs" / "soxx_mvp.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true", help="Refresh cached live price data.")
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use deterministic synthetic data instead of live market data. Useful for smoke tests.",
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable HTTPS certificate verification for local Python installs with missing cert bundles.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier. Defaults to the current UTC timestamp.",
    )
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="Optional LangSmith project override. Defaults to LANGSMITH_PROJECT.",
    )
    parser.add_argument(
        "--trace-tag",
        action="append",
        default=[],
        help="Optional LangSmith trace tag. May be supplied more than once.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    initial_state = initial_pipeline_state(
        RunOptions(
            config_path=args.config,
            output_dir=args.output_dir,
            refresh=args.refresh,
            sample_data=args.sample_data,
            insecure_ssl=args.insecure_ssl,
            run_id=args.run_id,
        )
    )
    final_state = invoke_soxx_graph_with_tracing(
        initial_state,
        langsmith_project=args.langsmith_project,
        trace_tags=args.trace_tag,
    )
    if final_state.get("status") == "failed":
        print(
            "{node} failed with {error_type}: {message}".format(
                node=final_state.get("failed_node", "unknown node"),
                error_type=final_state.get("error_type", "Error"),
                message=final_state.get("error_message", ""),
            ),
            file=sys.stderr,
        )
        return 1

    print_run_summary(final_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
