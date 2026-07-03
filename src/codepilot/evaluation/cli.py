from __future__ import annotations

# 新手导读：evaluation CLI 提供 check/run/experiment/report 等评估命令。
# 关注点：它面向项目展示和 benchmark，不参与普通 Agent 运行。

"""Command line interface for evaluation v2."""

import argparse
import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path

from codepilot.runtime import WorkspaceResourceLoader
from codepilot.runtime.contracts import CreateAgentSessionOptions

from .artifacts import EvaluationArtifacts
from .experiments import (
    experiment_variants,
    run_context_ab,
    run_security_ab,
)
from .loader import load_eval_suite
from .reports import render_comparison_markdown
from .schema import EvalRunOptions
from .service import EvaluationService


MODULES = ("all", "planning", "context", "memory", "security", "tool")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m codepilot.evaluation",
        description="Run Codepilot evaluation v2.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="Run deterministic eval tests.")
    check.add_argument("module", nargs="?", choices=MODULES, default="all")

    run = subparsers.add_parser("run", help="Run real-model benchmark cases.")
    _add_run_args(run, modules=MODULES)

    experiment = subparsers.add_parser(
        "experiment",
        help="Run on/off ablation for model-backed modules.",
    )
    _add_run_args(experiment, modules=("memory", "planning"))
    experiment.add_argument("--repeat", type=int, default=2)

    ab = subparsers.add_parser("ab", help="Run deterministic A/B experiments.")
    ab.add_argument("module", choices=("context", "security"))
    ab.add_argument("--cases", type=Path)
    ab.add_argument("--artifact-root", type=Path, default=Path(".codepilot/evals"))
    ab.add_argument("--eval-id")

    report = subparsers.add_parser("report", help="Print an existing summary.")
    report.add_argument("eval_dir", type=Path)
    return parser


def _add_run_args(
    parser: argparse.ArgumentParser,
    *,
    modules: tuple[str, ...],
) -> None:
    parser.add_argument("module", choices=modules)
    parser.add_argument("--provider")
    parser.add_argument("--model", dest="model_id")
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path("benchmarks/fixtures"),
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path("benchmarks/evaluation_v2"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(".codepilot/evals"),
    )
    parser.add_argument("--eval-id")
    parser.add_argument("--include-tag", action="append", default=[], dest="include_tags")
    parser.add_argument(
        "--workspace-policy",
        choices=("failed", "all", "none"),
        default="failed",
    )


async def _run_command(args: argparse.Namespace) -> int:
    service = EvaluationService()
    suite_path = _suite_path(args.module, args.suite_root)
    options = EvalRunOptions(
        fixtures_root=args.fixtures_root,
        artifact_root=args.artifact_root,
        eval_id=args.eval_id,
        include_tags=list(args.include_tags),
        workspace_policy=args.workspace_policy,
        session_options=_session_options(args),
    )
    if args.command == "experiment":
        return await _run_experiment(service, suite_path, options, args)
    result = await service.run_suite(suite_path, options)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {result.artifact_dir}")
    return 0 if all(item.passed for item in result.results) else 1


async def _run_experiment(
    service: EvaluationService,
    suite_path: Path,
    options: EvalRunOptions,
    args: argparse.Namespace,
) -> int:
    off, on = experiment_variants(args.module)
    eval_id = options.eval_id or f"{args.module}_{uuid.uuid4().hex[:8]}"
    root = Path(options.artifact_root) / eval_id
    comparison = {
        "schema_version": 1,
        "module": args.module,
        "variants": [off, on],
        "repeats": args.repeat,
        "variant_dirs": {},
    }
    for variant in (off, on):
        overrides = _variant_overrides(args.module, variant)
        for index in range(1, args.repeat + 1):
            repeat_options = EvalRunOptions(
                fixtures_root=options.fixtures_root,
                artifact_root=root / "variants" / variant,
                eval_id=f"repeat-{index}",
                include_tags=options.include_tags,
                workspace_policy=options.workspace_policy,
                session_options=options.session_options,
                runtime_overrides=overrides,
            )
            result = await service.run_suite(suite_path, repeat_options)
            comparison["variant_dirs"].setdefault(variant, []).append(
                result.artifact_dir
            )
    artifacts = EvaluationArtifacts(options.artifact_root, eval_id)
    artifacts.initialize(args.module, case_count=0)
    artifacts.write_comparison(comparison, render_comparison_markdown(comparison))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Artifacts: {artifacts.root}")
    return 0


def _variant_overrides(module: str, variant: str) -> dict[str, bool]:
    if module == "memory":
        return {"memory_enabled": variant == "on"}
    if module == "planning":
        return {"task_control_enabled": variant == "on"}
    return {}


def _run_ab(args: argparse.Namespace) -> int:
    cases = _load_ab_cases(args)
    comparison = (
        run_context_ab(cases)
        if args.module == "context"
        else run_security_ab(cases)
    )
    eval_id = args.eval_id or f"{args.module}_ab_{uuid.uuid4().hex[:8]}"
    artifacts = EvaluationArtifacts(args.artifact_root, eval_id)
    artifacts.initialize(args.module, case_count=len(cases))
    artifacts.write_comparison(comparison, render_comparison_markdown(comparison))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Artifacts: {artifacts.root}")
    return 0


def _load_ab_cases(args: argparse.Namespace) -> list[dict]:
    if args.cases is not None:
        payload = json.loads(args.cases.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return list(payload.get("cases") or [])
        if isinstance(payload, list):
            return payload
        raise ValueError("A/B cases must be a list or object with cases")
    if args.module == "context":
        return [
            {
                "id": "context-smoke",
                "expected": {"key_context": ["src/app.py"]},
                "candidates": [
                    {"id": "docs/noise.md", "path": "docs/noise.md", "tokens": 100},
                    {"id": "src/app.py", "path": "src/app.py", "tokens": 50},
                ],
                "selected": [
                    {"id": "src/app.py", "path": "src/app.py", "tokens": 50}
                ],
                "budget_tokens": 100,
            }
        ]
    return [
        {
            "id": "security-smoke",
            "dangerous_tools": ["write", "bash"],
            "benign_tools": ["read"],
        }
    ]


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
    raise ValueError("No project model config found; provide --provider and --model")


def _suite_path(module: str, root: Path) -> Path:
    return root if module == "all" else root / module


def _run_check(module: str) -> int:
    tests = [
        "test/test_observability_v2.py",
        "test/test_evaluation_v2.py",
    ]
    if module != "all":
        tests.extend(["-k", module])
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        check=False,
    ).returncode


def _show_report(eval_dir: Path) -> int:
    summary = eval_dir / "summary.json"
    comparison = eval_dir / "comparison.json"
    path = summary if summary.is_file() else comparison
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation summary not found: {eval_dir}")
    print(path.read_text(encoding="utf-8"))
    report = eval_dir / "report.md"
    if report.is_file():
        print(f"Report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            return _run_check(args.module)
        if args.command == "ab":
            return _run_ab(args)
        if args.command == "report":
            return _show_report(args.eval_dir)
        return asyncio.run(_run_command(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
