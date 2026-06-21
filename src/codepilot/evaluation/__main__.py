from __future__ import annotations

"""Codepilot 轻量评估命令行入口。"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from codepilot.runtime import (
    CreateAgentSessionOptions,
    WorkspaceResourceLoader,
)

from .service import EvaluationService
from .types import EvalRunOptions


MODULES = ("all", "context", "memory", "planning", "security")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m codepilot.evaluation",
        description="Run lightweight Codepilot benchmarks and experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="Run deterministic evaluation tests.",
    )
    check.add_argument("module", nargs="?", choices=MODULES, default="all")

    run = subparsers.add_parser("run", help="Run real-model benchmarks.")
    _add_run_arguments(run, experiment=False)

    experiment = subparsers.add_parser(
        "experiment",
        help="Compare one module with its feature on and off.",
    )
    _add_run_arguments(experiment, experiment=True)
    experiment.add_argument("--repeat", type=int, default=3)

    report = subparsers.add_parser(
        "report",
        help="Print an existing evaluation summary.",
    )
    report.add_argument("eval_dir", type=Path)
    return parser


def _add_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    experiment: bool,
) -> None:
    choices = ("context", "memory", "planning") if experiment else MODULES
    parser.add_argument("module", choices=choices)
    parser.add_argument("--provider")
    parser.add_argument("--model", dest="model_id")
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
    parser.add_argument("--eval-id")
    parser.add_argument("--remove-workspaces", action="store_true")


async def _run_command(args: argparse.Namespace) -> int:
    service = EvaluationService()
    suite = _suite_path(args.module)
    options = EvalRunOptions(
        fixtures_root=args.fixtures_root,
        artifact_root=args.artifact_root,
        eval_id=args.eval_id,
        benchmark_name=args.module,
        keep_workspace=not args.remove_workspaces,
        session_options=_session_options(args),
    )
    if args.command == "experiment":
        result = await service.run_experiment(
            suite,
            options,
            module=args.module,
            repeat=args.repeat,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Artifacts: {result['artifact_dir']}")
        return 0

    result = await service.run_suite(suite, options)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {result.artifact_dir}")
    return 0 if all(item.overall == "passed" for item in result.results) else 1


def _session_options(args: argparse.Namespace) -> CreateAgentSessionOptions:
    if bool(args.provider) != bool(args.model_id):
        raise ValueError("--provider and --model must be provided together")
    if args.provider and args.model_id:
        return CreateAgentSessionOptions(
            workspace_dir=Path.cwd(),
            provider=args.provider,
            model_id=args.model_id,
        )
    resources = WorkspaceResourceLoader(Path.cwd()).load()
    if resources.model is not None:
        return CreateAgentSessionOptions(
            workspace_dir=Path.cwd(),
            model=resources.model.to_model(),
            get_api_key=resources.model.build_api_key_resolver(),
        )
    if resources.settings.provider and resources.settings.model_id:
        return CreateAgentSessionOptions(
            workspace_dir=Path.cwd(),
            provider=resources.settings.provider,
            model_id=resources.settings.model_id,
        )
    raise ValueError(
        "No project model config found; provide --provider and --model"
    )


def _suite_path(module: str) -> Path:
    root = Path("benchmarks/evaluation")
    return root if module == "all" else root / module


def _run_check(module: str) -> int:
    tests = [
        "test/test_evaluation_metrics.py",
        "test/test_evaluation.py",
    ]
    if module != "all":
        tests.extend(["-k", module])
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        check=False,
    ).returncode


def _show_report(eval_dir: Path) -> int:
    path = eval_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    report = eval_dir / "report.md"
    if report.is_file():
        print(f"Report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            return _run_check(args.module)
        if args.command == "report":
            return _show_report(args.eval_dir)
        return asyncio.run(_run_command(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
