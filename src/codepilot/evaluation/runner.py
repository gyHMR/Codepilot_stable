from __future__ import annotations

# 新手导读：runner.py 执行单个或一组 benchmark，并收集运行证据。
# 关注点：它把 runtime 运行结果转换成可评分材料。

"""Evaluation v2 runner.

The runner is deliberately an adapter around the public RuntimeService API.  It
does not patch agent internals; after execution it reads run artifacts and turns
them into typed evidence.
"""

import asyncio
import shutil
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

from codepilot.observability import build_run_trace
from codepilot.runtime import RuntimeService
from codepilot.runtime.contracts import UserInput

from .artifacts import EvaluationArtifacts
from .evidence import (
    EvalEvidence,
    evidence_from_traces,
    workspace_diff,
    workspace_snapshot,
)
from .loader import load_eval_suite
from .reports import build_summary, render_markdown
from .schema import CheckResult, EvalCase, EvalResult, EvalRunOptions, EvalSuiteResult
from .scorers import score_metrics


RuntimeFactory = Callable[[], RuntimeService]


class EvaluationRunner:
    """Run v2 cases and write v2 artifacts."""

    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory = RuntimeService,
    ) -> None:
        self.runtime_factory = runtime_factory

    async def run_suite(
        self,
        suite_path: Path | str,
        options: EvalRunOptions,
    ) -> EvalSuiteResult:
        cases = _filter_cases(load_eval_suite(suite_path), options.include_tags)
        eval_id = options.eval_id or f"eval_{uuid.uuid4().hex[:12]}"
        artifacts = EvaluationArtifacts(options.artifact_root, eval_id)
        artifacts.initialize(Path(suite_path).stem or "evaluation", case_count=len(cases))
        results: list[EvalResult] = []
        for case in cases:
            result = await self.run_case(case, options, artifacts=artifacts)
            results.append(result)
        summary = build_summary(results)
        artifacts.write_summary(summary, render_markdown(results, summary))
        return EvalSuiteResult(
            eval_id=eval_id,
            results=results,
            artifact_dir=str(artifacts.root),
            summary=summary,
        )

    async def run_case(
        self,
        case: EvalCase,
        options: EvalRunOptions,
        *,
        artifacts: EvaluationArtifacts | None = None,
    ) -> EvalResult:
        start = time.perf_counter()
        own_artifacts = artifacts is None
        if artifacts is None:
            eval_id = options.eval_id or f"eval_{uuid.uuid4().hex[:12]}"
            artifacts = EvaluationArtifacts(options.artifact_root, eval_id)
            artifacts.initialize(case.module, case_count=1)
        workspace = _prepare_workspace(case, options)
        baseline = workspace_snapshot(workspace)
        runtime = self.runtime_factory()
        run_ids: list[str] = []
        checks: list[CheckResult] = []
        traces = []
        final_text = ""
        passed = False
        error: str | None = None
        try:
            session_options = replace(
                options.session_options,
                workspace_dir=workspace,
                **options.runtime_overrides,
            )
            handle = runtime.create_session(session_options)
            steps = case.steps if case.type == "scenario" else []
            prompts = (
                steps
                if steps
                else [replace_step_prompt(case.prompt)]
            )
            for step in steps:
                if step.kind == "modify_file":
                    _apply_modify_file(workspace, step.path, step.content or step.text)
                elif step.kind == "restart":
                    handle = runtime.create_session(
                        replace(session_options, session_id=handle.session_id)
                    )
                elif step.kind == "prompt":
                    await _run_prompt(runtime, handle.session_id, step.text)
                elif step.kind in {"verify", "inspect"} and step.check:
                    checks.append(_run_check(workspace, step.check, final_text=final_text))
            if not steps:
                for prompt_step in prompts:
                    await _run_prompt(runtime, handle.session_id, prompt_step.text)
            runs = runtime.list_runs(handle.session_id)
            run_ids = [
                str(item.get("run_id"))
                for item in runs
                if isinstance(item.get("run_id"), str)
            ]
            for run_id in run_ids:
                result = runtime.get_run_result(handle.session_id, run_id)
                events = runtime.get_run_events(handle.session_id, run_id)
                traces.append(build_run_trace(events, result=result))
            message = runtime.get_latest_assistant_message(handle.session_id)
            final_text = getattr(message, "content", "") if message is not None else ""
            checks.extend(
                _run_case_checks(workspace, case, final_text=final_text)
            )
            passed = all(check.passed for check in checks) if checks else True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            checks.append(
                CheckResult(
                    name="evaluation.execution",
                    passed=False,
                    summary=error,
                )
            )
        changes = workspace_diff(workspace, baseline)
        evidence = evidence_from_traces(
            case_id=case.id,
            module=case.module,
            traces=traces,
            expected=case.expected,
            task_passed=passed,
            workspace_changes=changes,
            final_text=final_text,
        )
        if not traces and not checks:
            evidence = EvalEvidence(
                case_id=case.id,
                module=case.module,
                task_passed=passed,
                expected=case.expected,
                workspace_changes=changes,
                final_text=final_text,
            )
        metrics = score_metrics(evidence, case.metrics)
        result = EvalResult(
            case_id=case.id,
            module=case.module,
            passed=passed,
            metrics=metrics,
            checks=checks,
            run_ids=run_ids,
            error=error,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        diff_text = "\n".join(changes) + ("\n" if changes else "")
        artifacts.write_case(case, result, evidence, workspace_diff=diff_text)
        _cleanup_workspace(case, options, workspace, passed=passed)
        if own_artifacts:
            summary = build_summary([result])
            artifacts.write_summary(summary, render_markdown([result], summary))
        return result


def replace_step_prompt(text: str):
    from .schema import EvalStep

    return EvalStep(kind="prompt", text=text)


async def _run_prompt(
    runtime: RuntimeService,
    session_id: str,
    text: str,
) -> None:
    await asyncio.wait_for(
        runtime.run_message(session_id, UserInput(text=text)),
        timeout=None,
    )


def _prepare_workspace(case: EvalCase, options: EvalRunOptions) -> Path:
    source = Path(options.fixtures_root) / case.fixture
    if not source.exists():
        raise FileNotFoundError(f"Fixture not found: {source}")
    root = Path(options.artifact_root) / "_workspaces" / f"{case.id}_{uuid.uuid4().hex[:8]}"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(source, root)
    for step in case.setup:
        if step.kind == "modify_file":
            if step.source:
                src = source / step.source
                dst = root / step.path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif step.path:
                _apply_modify_file(root, step.path, step.content or step.text)
    return root


def _apply_modify_file(workspace: Path, path: str, content: str) -> None:
    target = workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _run_case_checks(
    workspace: Path,
    case: EvalCase,
    *,
    final_text: str,
) -> list[CheckResult]:
    return [_run_check(workspace, {"kind": check.kind, **check.options}, final_text=final_text) for check in case.checks]


def _run_check(
    workspace: Path,
    check: dict,
    *,
    final_text: str,
) -> CheckResult:
    kind = str(check.get("kind") or "")
    if kind == "command":
        command = str(check.get("command") or "")
        completed = subprocess.run(
            command,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(check.get("timeout_seconds") or 60),
        )
        return CheckResult(
            name=f"command:{command}",
            passed=completed.returncode == 0,
            summary=(completed.stdout or completed.stderr)[-1000:],
            expected=0,
            actual=completed.returncode,
        )
    if kind == "file_contains":
        path = workspace / str(check.get("path") or "")
        text = str(check.get("text") or "")
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        return CheckResult(
            name=f"file_contains:{path.name}",
            passed=text in actual,
            summary=f"expected text {'found' if text in actual else 'missing'}",
            expected=text,
            actual=actual[-1000:],
        )
    if kind == "file_exists":
        path = workspace / str(check.get("path") or "")
        return CheckResult(
            name=f"file_exists:{path.name}",
            passed=path.exists(),
            summary=str(path),
            expected=True,
            actual=path.exists(),
        )
    if kind == "final_contains":
        text = str(check.get("text") or "")
        return CheckResult(
            name="final_contains",
            passed=text in final_text,
            summary=f"expected text {'found' if text in final_text else 'missing'}",
            expected=text,
            actual=final_text[-1000:],
        )
    return CheckResult(
        name=kind or "unknown",
        passed=True,
        summary="unsupported check treated as informational",
    )


def _cleanup_workspace(
    case: EvalCase,
    options: EvalRunOptions,
    workspace: Path,
    *,
    passed: bool,
) -> None:
    policy = options.workspace_policy
    keep = policy == "all" or (policy == "failed" and not passed)
    if not keep and workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)


def _filter_cases(cases: list[EvalCase], include_tags: list[str]) -> list[EvalCase]:
    requested = {tag.strip() for tag in include_tags if tag.strip()}
    if not requested:
        return cases
    return [case for case in cases if requested.issubset(set(case.tags))]


__all__ = ["EvaluationRunner"]
