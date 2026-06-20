from __future__ import annotations

"""Command-line entry point for running Codepilot evaluation suites."""

import argparse
import asyncio
import json
from pathlib import Path

from codepilot.runtime import CreateAgentSessionOptions

from .service import EvaluationService
from .types import EvalRunOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m codepilot.evaluation",
        description="Run a Codepilot evaluation suite and write audit artifacts.",
    )
    parser.add_argument("suite", type=Path, help="Case JSON file or suite directory")
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path("benchmarks/fixtures"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(".codepilot/evals"),
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", dest="model_id", required=True)
    parser.add_argument("--benchmark-name", default="")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--remove-workspaces",
        action="store_true",
        help="Delete isolated case workspaces after each result is saved.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    service = EvaluationService()
    result = await service.run_suite(
        args.suite,
        EvalRunOptions(
            fixtures_root=args.fixtures_root,
            artifact_root=args.artifact_root,
            benchmark_name=args.benchmark_name or args.suite.stem,
            keep_workspace=not args.remove_workspaces,
            session_options=CreateAgentSessionOptions(
                workspace_dir=args.fixtures_root,
                provider=args.provider,
                model_id=args.model_id,
                system_prompt=args.system_prompt,
            ),
        ),
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {result.artifact_dir}")
    return 0 if all(item.overall == "passed" for item in result.results) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
