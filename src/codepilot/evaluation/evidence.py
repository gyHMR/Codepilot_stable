from __future__ import annotations

# 新手导读：evidence.py 从运行结果中提取工具调用、上下文和文件变更证据。
# 关注点：评估尽量基于事实证据，而不是只看最终文本。

"""Typed evidence consumed by metric scorers."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codepilot.observability import RunTrace


@dataclass(frozen=True)
class ContextEvidence:
    selected_items: list[dict[str, Any]] = field(default_factory=list)
    stale_items: list[str] = field(default_factory=list)
    tokens_after: int = 0


@dataclass(frozen=True)
class ToolCallEvidence:
    tool_call_id: str
    tool_name: str
    status: str
    is_error: bool = False
    error_reason: str | None = None
    workspace_changed: bool | None = None
    affected_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskStepEvidence:
    step_id: str
    title: str
    status: str
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalEvidence:
    case_id: str
    module: str
    task_passed: bool = False
    expected: dict[str, Any] = field(default_factory=dict)
    contexts: list[ContextEvidence] = field(default_factory=list)
    tools: list[ToolCallEvidence] = field(default_factory=list)
    steps: list[TaskStepEvidence] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    workspace_changes: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    final_text: str = ""


def evidence_from_traces(
    *,
    case_id: str,
    module: str,
    traces: list[RunTrace],
    expected: dict[str, Any],
    task_passed: bool,
    workspace_changes: list[str] | None = None,
    final_text: str = "",
) -> EvalEvidence:
    contexts = [
        ContextEvidence(
            selected_items=list(context.selected_items),
            stale_items=list(context.stale_items),
            tokens_after=context.tokens_after,
        )
        for trace in traces
        for context in trace.contexts
    ]
    tools = [
        ToolCallEvidence(
            tool_call_id=tool.tool_call_id,
            tool_name=tool.tool_name,
            status=tool.status,
            is_error=tool.is_error,
            error_reason=tool.error_reason,
            workspace_changed=tool.workspace_changed,
            affected_paths=list(tool.affected_paths),
        )
        for trace in traces
        for tool in trace.tool_calls
    ]
    steps = [
        TaskStepEvidence(
            step_id=task.step_id,
            title=task.step_title,
            status=task.step_status,
            evidence_refs=list(task.evidence_refs),
        )
        for trace in traces
        for task in trace.tasks
        if task.step_id or task.step_title
    ]
    memory_ids = sorted(
        {
            memory_id
            for trace in traces
            for memory in trace.memories
            for memory_id in memory.memory_ids
        }
    )
    return EvalEvidence(
        case_id=case_id,
        module=module,
        task_passed=task_passed,
        expected=dict(expected),
        contexts=contexts,
        tools=tools,
        steps=_dedupe_steps(steps),
        memory_ids=memory_ids,
        workspace_changes=list(workspace_changes or []),
        run_ids=[trace.run_id for trace in traces if trace.run_id],
        final_text=final_text,
    )


def repeated_read_count(tools: list[ToolCallEvidence]) -> tuple[int, int]:
    paths = [
        path
        for tool in tools
        if tool.tool_name.lower() == "read"
        for path in tool.affected_paths
    ]
    counts = Counter(paths)
    repeated = sum(max(0, count - 1) for count in counts.values())
    return repeated, len(paths)


def workspace_diff(workspace: Path, baseline: dict[str, str]) -> list[str]:
    current = _snapshot(workspace)
    changes: list[str] = []
    for path in sorted(set(baseline) | set(current)):
        if path not in baseline:
            changes.append(f"A {path}")
        elif path not in current:
            changes.append(f"D {path}")
        elif baseline[path] != current[path]:
            changes.append(f"M {path}")
    return changes


def workspace_snapshot(workspace: Path) -> dict[str, str]:
    return _snapshot(workspace)


def _dedupe_steps(steps: list[TaskStepEvidence]) -> list[TaskStepEvidence]:
    by_key: dict[str, TaskStepEvidence] = {}
    for step in steps:
        key = step.step_id or step.title
        by_key[key] = step
    return list(by_key.values())


def _snapshot(workspace: Path) -> dict[str, str]:
    import hashlib

    out: dict[str, str] = {}
    if not workspace.exists():
        return out
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        if any(part in {".git", ".codepilot", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[rel.as_posix()] = digest
    return out


__all__ = [
    "ContextEvidence",
    "EvalEvidence",
    "TaskStepEvidence",
    "ToolCallEvidence",
    "evidence_from_traces",
    "repeated_read_count",
    "workspace_diff",
    "workspace_snapshot",
]
