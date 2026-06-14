#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soxx_mvp.agents.memo_agent import (  # noqa: E402
    ClaudeMemoProvider,
    MemoAgentError,
    TemplateMemoProvider,
    default_claude_model,
    generate_research_memo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cited post-run SOXX/SOXL research memo from existing artifacts."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to research_memo.md inside --artifact-dir.",
    )
    parser.add_argument(
        "--provider",
        choices=["claude", "template"],
        default="claude",
        help="Memo provider. Template mode is deterministic and does not call an external API.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Claude model name. Defaults to SOXX_MEMO_MODEL or claude-sonnet-4-6.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provider = _provider_from_args(args)
    output_path = args.output or args.artifact_dir / "research_memo.md"

    try:
        memo_path = generate_research_memo(
            artifact_dir=args.artifact_dir,
            output_path=output_path,
            provider=provider,
        )
    except MemoAgentError as exc:
        print(f"research memo failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote research memo to {memo_path}")
    return 0


def _provider_from_args(args: argparse.Namespace) -> ClaudeMemoProvider | TemplateMemoProvider:
    if args.provider == "template":
        return TemplateMemoProvider()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required when --provider claude is used")

    return ClaudeMemoProvider(model=args.model or default_claude_model())


if __name__ == "__main__":
    raise SystemExit(main())
