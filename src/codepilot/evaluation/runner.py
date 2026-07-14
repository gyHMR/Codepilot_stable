from __future__ import annotations

# 新手导读：runner.py 执行单个或一组 benchmark，并收集运行证据。
# 关注点：它把 runtime 运行结果转换成可评分材料。

"""Evaluation v2 runner.

The runner drives the same UserAction -> RuntimeFrame spine used by interfaces.
It does not patch agent internals; after execution it reads run artifacts and
turns them into typed evidence.
"""

import asyncio
import hashlib
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from codepilot.observability import RunTrace, build_run_trace, load_run_trace
from codepilot.protocols import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from codepilot.runtime import RuntimeGateway
from codepilot.runtime.actions import (
    ApprovalRequiredFrame,
    FailedFrame,
    PromptSubmitted,
    RunFinishedFrame,
    RunPausedFrame,
)
from codepilot.sessions.memory import MemoryRecord

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


RuntimeFactory = Callable[[], Any]


@dataclass(frozen=True)
class PromptRunObservation:
    final_text: str
    run_id: str | None = None
    trace: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class CaseRunObservation:
    variant: str
    workspace: Path
    passed: bool
    checks: list[CheckResult]
    traces: list[Any]
    run_ids: list[str]
    workspace_changes: list[str]
    final_text: str = ""
    error: str | None = None


class EvaluationRunner:
    """Run v2 cases and write v2 artifacts."""

    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory = RuntimeGateway,
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
        if _is_context_compression_case(case):
            return await self._run_context_compression_case(
                case,
                options,
                artifacts=artifacts,
            )
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
            handle = runtime.open_session(session_options)
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
                    handle = runtime.open_session(
                        replace(session_options, session_id=handle.session_id)
                    )
                elif step.kind == "prompt":
                    observation = await _run_prompt(
                        runtime,
                        handle.session_id,
                        step.text,
                        workspace=workspace,
                    )
                    final_text = observation.final_text
                    if observation.run_id:
                        run_ids.append(observation.run_id)
                    if observation.trace is not None:
                        traces.append(observation.trace)
                    if observation.error:
                        raise RuntimeError(observation.error)
                elif step.kind in {"verify", "inspect"} and step.check:
                    checks.append(_run_check(workspace, step.check, final_text=final_text))
            if not steps:
                for prompt_step in prompts:
                    observation = await _run_prompt(
                        runtime,
                        handle.session_id,
                        prompt_step.text,
                        workspace=workspace,
                    )
                    final_text = observation.final_text
                    if observation.run_id:
                        run_ids.append(observation.run_id)
                    if observation.trace is not None:
                        traces.append(observation.trace)
                    if observation.error:
                        raise RuntimeError(observation.error)
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

    async def _run_context_compression_case(
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

        observations: list[CaseRunObservation] = []
        for variant in ("raw", "compressed"):
            variant_case = replace(case, id=f"{case.id}_{variant}")
            workspace = _prepare_workspace(variant_case, options)
            seeded_messages = _context_profile_messages(workspace, case)
            session_options = _context_compression_session_options(
                case,
                options,
                workspace=workspace,
                variant=variant,
                seeded_messages=seeded_messages,
            )
            observations.append(
                await self._run_case_once(
                    case,
                    options,
                    workspace=workspace,
                    session_options=session_options,
                    variant=variant,
                )
            )

        by_variant = {observation.variant: observation for observation in observations}
        raw = by_variant["raw"]
        compressed = by_variant["compressed"]
        passed = raw.passed and compressed.passed
        checks = [
            _prefix_check(observation.variant, check)
            for observation in observations
            for check in observation.checks
        ]
        traces = [trace for observation in observations for trace in observation.traces]
        run_ids = [run_id for observation in observations for run_id in observation.run_ids]
        final_text = compressed.final_text or raw.final_text
        workspace_changes = [
            f"{observation.variant}: {change}"
            for observation in observations
            for change in observation.workspace_changes
        ]
        errors = [
            f"{observation.variant}: {observation.error}"
            for observation in observations
            if observation.error
        ]
        evidence = evidence_from_traces(
            case_id=case.id,
            module=case.module,
            traces=traces,
            expected=case.expected,
            task_passed=passed,
            workspace_changes=workspace_changes,
            final_text=final_text,
        )
        evidence = replace(
            evidence,
            variant_passed={
                "raw": raw.passed,
                "compressed": compressed.passed,
            },
            variant_context_tokens={
                observation.variant: _variant_context_tokens(observation)
                for observation in observations
            },
        )
        metrics = score_metrics(evidence, case.metrics)
        result = EvalResult(
            case_id=case.id,
            module=case.module,
            passed=passed,
            metrics=metrics,
            checks=checks,
            run_ids=run_ids,
            error="; ".join(errors) if errors else None,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        diff_text = "\n".join(workspace_changes) + ("\n" if workspace_changes else "")
        artifacts.write_case(case, result, evidence, workspace_diff=diff_text)
        for observation in observations:
            _cleanup_workspace(case, options, observation.workspace, passed=observation.passed)
        if own_artifacts:
            summary = build_summary([result])
            artifacts.write_summary(summary, render_markdown([result], summary))
        return result

    async def _run_case_once(
        self,
        case: EvalCase,
        options: EvalRunOptions,
        *,
        workspace: Path,
        session_options: Any,
        variant: str,
    ) -> CaseRunObservation:
        baseline = workspace_snapshot(workspace)
        runtime = self.runtime_factory()
        run_ids: list[str] = []
        checks: list[CheckResult] = []
        traces = []
        final_text = ""
        passed = False
        error: str | None = None
        try:
            handle = runtime.open_session(session_options)
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
                    handle = runtime.open_session(
                        replace(session_options, session_id=handle.session_id)
                    )
                elif step.kind == "prompt":
                    observation = await _run_prompt(
                        runtime,
                        handle.session_id,
                        step.text,
                        workspace=workspace,
                    )
                    final_text = observation.final_text
                    if observation.run_id:
                        run_ids.append(observation.run_id)
                    if observation.trace is not None:
                        traces.append(observation.trace)
                    if observation.error:
                        raise RuntimeError(observation.error)
                elif step.kind in {"verify", "inspect"} and step.check:
                    checks.append(_run_check(workspace, step.check, final_text=final_text))
            if not steps:
                for prompt_step in prompts:
                    observation = await _run_prompt(
                        runtime,
                        handle.session_id,
                        prompt_step.text,
                        workspace=workspace,
                    )
                    final_text = observation.final_text
                    if observation.run_id:
                        run_ids.append(observation.run_id)
                    if observation.trace is not None:
                        traces.append(observation.trace)
                    if observation.error:
                        raise RuntimeError(observation.error)
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
        return CaseRunObservation(
            variant=variant,
            workspace=workspace,
            passed=passed,
            checks=checks,
            traces=traces,
            run_ids=run_ids,
            workspace_changes=workspace_diff(workspace, baseline),
            final_text=final_text,
            error=error,
        )


def replace_step_prompt(text: str):
    from .schema import EvalStep

    return EvalStep(kind="prompt", text=text)


def _is_context_compression_case(case: EvalCase) -> bool:
    return (
        case.module == "context"
        and str(case.context_profile.get("kind") or "").strip() == "compression"
    )


def _context_compression_session_options(
    case: EvalCase,
    options: EvalRunOptions,
    *,
    workspace: Path,
    variant: str,
    seeded_messages: list[Any],
) -> Any:
    profile = case.context_profile
    messages = [*list(options.session_options.messages), *seeded_messages]
    context_window = _positive_int(
        profile.get("model_context_window"),
        default=8192,
    )
    if variant == "raw":
        context_window = max(context_window, 1_000_000)
    return replace(
        options.session_options,
        workspace_dir=workspace,
        messages=messages,
        model_context_window=context_window,
        model_max_output_tokens=_positive_int(
            profile.get("model_max_output_tokens"),
            default=512,
        ),
        **options.runtime_overrides,
    )


def _context_profile_messages(workspace: Path, case: EvalCase) -> list[Any]:
    profile = case.context_profile
    history_groups = _positive_int(profile.get("history_groups"), default=18)
    chars_per_group = _positive_int(profile.get("chars_per_group"), default=1000)
    focus_tail_groups = min(
        history_groups,
        _positive_int(profile.get("focus_tail_groups"), default=5),
    )
    key_paths = _profile_paths(profile.get("key_paths")) or _profile_paths(
        case.expected.get("key_context")
    )
    noise_paths = _profile_paths(profile.get("noise_paths")) or _profile_paths(
        case.expected.get("stale_or_noise_context")
    )
    if not key_paths:
        key_paths = ["README.md"]
    if not noise_paths:
        noise_paths = list(reversed(key_paths))

    messages: list[Any] = []
    for index in range(history_groups):
        is_focus = index >= history_groups - focus_tail_groups
        paths = key_paths if is_focus else noise_paths
        path = paths[index % len(paths)]
        text = _context_load_text(
            workspace,
            path,
            index=index,
            pressure=_context_profile_pressure(case),
            is_focus=is_focus,
            target_chars=chars_per_group,
        )
        messages.append(
            UserMessage(
                content=(
                    f"[eval context load {index + 1}/{history_groups}] "
                    f"Historical {'focused' if is_focus else 'background'} note for {path}."
                ),
                metadata={
                    "eval_context_load": True,
                    "session_message_id": f"eval-{case.id}-{index}-user",
                },
            )
        )
        messages.append(
            ToolResultMessage(
                tool_call_id=f"eval-context-load-{case.id}-{index}",
                tool_name="read",
                content=[TextContent(text=text)],
                status="success",
                affected_paths=[path],
                workspace_changed=False,
                metadata={
                    "eval_context_load": True,
                    "session_message_id": f"eval-{case.id}-{index}-tool",
                    "read_paths": [path],
                },
            )
        )
    handoff = _context_handoff_text(case, key_paths)
    if handoff:
        messages.append(
            UserMessage(
                content=handoff,
                metadata={
                    "eval_context_load": True,
                    "eval_handoff": True,
                    "session_message_id": f"eval-{case.id}-handoff",
                },
            )
        )
    return messages


def _context_load_text(
    workspace: Path,
    path: str,
    *,
    index: int,
    pressure: str,
    is_focus: bool,
    target_chars: int,
) -> str:
    source_text = _read_workspace_excerpt(workspace, path)
    heading = (
        f"Context pressure={pressure}; note={index + 1}; "
        f"kind={'focused-key-context' if is_focus else 'background-noise'}; path={path}\n"
    )
    body = source_text or f"No file content was available for {path}."
    padding_unit = (
        "\nHistorical analysis: verify the current repository before trusting this note. "
        "Prefer tests and source files over stale summaries."
    )
    text = heading + body
    while len(text) < target_chars:
        text += padding_unit
    return text[:target_chars]


def _context_handoff_text(case: EvalCase, key_paths: list[str]) -> str:
    if not key_paths:
        return ""
    return (
        "[eval context handoff] The next task should verify the repository directly. "
        "Likely relevant paths: "
        + ", ".join(key_paths[:6])
        + f". User request: {case.prompt}"
    )


def _read_workspace_excerpt(workspace: Path, path: str, *, limit: int = 1800) -> str:
    target = workspace / path
    if not target.is_file():
        return ""
    try:
        return target.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _tool_result_text(message: ToolResultMessage) -> str:
    return "".join(getattr(block, "text", "") for block in message.content)


def _profile_paths(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip().replace("\\", "/")
        if text and text not in result:
            result.append(text)
    return result


def _context_profile_pressure(case: EvalCase) -> str:
    pressure = str(case.context_profile.get("pressure") or "tight").strip()
    return pressure if pressure in {"tight", "critical"} else "tight"


def _positive_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _prefix_check(variant: str, check: CheckResult) -> CheckResult:
    return replace(check, name=f"{variant}:{check.name}")


def _variant_context_tokens(observation: CaseRunObservation) -> dict[str, int]:
    contexts = [
        context
        for trace in observation.traces
        for context in getattr(trace, "contexts", [])
    ]
    if not contexts:
        return {"tokens_before": 0, "tokens_after": 0, "context_count": 0}
    first = contexts[0]
    return {
        "tokens_before": int(getattr(first, "tokens_before", 0) or 0),
        "tokens_after": int(getattr(first, "tokens_after", 0) or 0),
        "context_count": len(contexts),
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


async def _run_prompt(
    runtime: Any,
    session_id: str,
    text: str,
    *,
    workspace: Path,
) -> PromptRunObservation:
    final_message: AssistantMessage | None = None
    final_text = ""
    run_id: str | None = None
    result: Any | None = None
    error: str | None = None
    events: list[dict[str, Any]] = []

    async def consume() -> None:
        nonlocal final_message, final_text, run_id, result, error
        async for frame in runtime.dispatch(session_id, PromptSubmitted(text=text)):
            if hasattr(frame, "event"):
                event = dict(getattr(frame, "event"))
                events.append(event)
            elif isinstance(frame, (RunFinishedFrame, RunPausedFrame)):
                result = frame.record
                run_id = _optional_text(_field(frame.record, "run_id"))
                final_text = _optional_text(_field(frame.record, "final_text")) or ""
                final_message = _field(frame.record, "final_message")
                if not final_text:
                    final_text = _message_text(final_message)
            elif isinstance(frame, ApprovalRequiredFrame):
                run_id = _optional_text(_field(frame.approval, "run_id")) or run_id
            elif isinstance(frame, FailedFrame):
                error = _frame_error_message(frame)

    await asyncio.wait_for(consume(), timeout=None)
    fallback = build_run_trace(events, result=result) if run_id or events else None
    trace = _load_persisted_trace(workspace, run_id or _trace_run_id(fallback)) or fallback
    return PromptRunObservation(
        final_text=final_text or _message_text(final_message),
        run_id=run_id or _trace_run_id(trace),
        trace=trace,
        error=error,
    )


def _trace_run_id(trace: RunTrace | None) -> str | None:
    if trace is None:
        return None
    return _optional_text(trace.run_id)


def _load_persisted_trace(workspace: Path, run_id: str | None) -> RunTrace | None:
    if not run_id:
        return None
    trace_path = workspace / ".codepilot" / "runs" / run_id / "trace.json"
    if not trace_path.is_file():
        return None
    try:
        return load_run_trace(trace_path)
    except Exception:
        return None


def _message_text(message: AssistantMessage | None) -> str:
    if message is None:
        return ""
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "\n".join(parts)


def _field(value: object, name: str):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _frame_error_message(frame: FailedFrame) -> str:
    error = frame.error
    message = _field(error, "message") or _field(error, "error") or error
    return str(message)


def _prepare_workspace(case: EvalCase, options: EvalRunOptions) -> Path:
    source = Path(options.fixtures_root) / case.fixture
    if not source.exists():
        raise FileNotFoundError(f"Fixture not found: {source}")
    root = (
        Path(options.artifact_root)
        / "_workspaces"
        / f"case_{_hash_text(case.id)}_{uuid.uuid4().hex[:8]}"
    )
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
    _seed_structured_memory(root)
    _initialize_workspace_git(root)
    return root


def _seed_structured_memory(workspace: Path) -> None:
    """Copy fixture memory seeds into the runtime's current memory store."""

    source = workspace / ".codepilot" / "memory" / "memories.jsonl"
    if not source.is_file():
        return
    records: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if isinstance(raw, dict):
            records.append(_canonical_memory_seed(raw))
    if not records:
        return
    target = workspace / ".codepilot" / "memory" / "memories.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


def _canonical_memory_seed(raw: dict[str, Any]) -> dict[str, Any]:
    return MemoryRecord.from_dict(raw).to_dict()


def _initialize_workspace_git(workspace: Path) -> None:
    if (workspace / ".git").exists():
        return
    if not _run_git(workspace, "init"):
        return
    exclude = workspace / ".git" / "info" / "exclude"
    if exclude.parent.is_dir():
        existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        additions = "\n.codepilot/\n__pycache__/\n.pytest_cache/\n"
        exclude.write_text(existing.rstrip() + additions, encoding="utf-8", newline="\n")
    _run_git(workspace, "config", "user.email", "eval@example.local")
    _run_git(workspace, "config", "user.name", "Codepilot Eval")
    _run_git(workspace, "add", "-A", ".")
    _run_git(workspace, "commit", "-m", "evaluation fixture baseline")


def _run_git(workspace: Path, *args: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


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
