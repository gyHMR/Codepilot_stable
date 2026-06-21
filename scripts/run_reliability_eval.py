from __future__ import annotations

"""Run the Codepilot reliability benchmark suite.

Example:
    python scripts/run_reliability_eval.py --provider openai --model gpt-4o-mini
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from codepilot.evaluation.service import EvaluationService
from codepilot.evaluation.types import EvalRunOptions
from codepilot.runtime import CreateAgentSessionOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Codepilot reliability evaluation cases.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmarks" / "reliability",
        help="Reliability suite directory or case JSON file.",
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=ROOT / "benchmarks" / "fixtures",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / ".codepilot" / "evals",
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", dest="model_id", required=True)
    parser.add_argument("--eval-id", default=None)
    parser.add_argument("--benchmark-name", default="reliability")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--remove-workspaces",
        action="store_true",
        help="Delete isolated case workspaces after saving case artifacts.",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    service = EvaluationService()
    result = await service.run_suite(
        args.suite,
        EvalRunOptions(
            fixtures_root=args.fixtures_root,
            artifact_root=args.artifact_root,
            eval_id=args.eval_id,
            benchmark_name=args.benchmark_name,
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
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
