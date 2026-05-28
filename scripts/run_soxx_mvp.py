#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soxx_mvp.io_utils import project_root
from soxx_mvp.pipeline import RunOptions, print_run_summary, run_deterministic_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic SOXX/SOXL MVP backtest.")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = run_deterministic_pipeline(
        RunOptions(
            config_path=args.config,
            output_dir=args.output_dir,
            refresh=args.refresh,
            sample_data=args.sample_data,
            insecure_ssl=args.insecure_ssl,
            run_id=args.run_id,
        )
    )
    print_run_summary(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
