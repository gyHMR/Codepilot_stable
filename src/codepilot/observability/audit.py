from __future__ import annotations

"""
一次 Agent Run 的统一只读审计投影模块。

本模块从 Run 的不可变产物（events.jsonl、run.json）中
加载审计数据，构建结构化的审计报告，供评估和调试使用。

核心功能：
    - load_audit_bundle: 从磁盘加载一次 Run 的完整审计数据
    - build_audit_report: 构建包含上下文、记忆、安全、任务等维度的审计报告
    - redact_artifact: 递归脱敏敏感信息（API Key、密码等）
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .summaries import build_run_report


AUDIT_SCHEMA_VERSION = "1"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "cookie",
    "credential",
    "private_key",
)
_SENSITIVE_ENV_KEYS = tuple(
    key
    for key in os.environ
    if "token" in key.lower()
    or any(part in key.lower() for part in _SENSITIVE_KEY_PARTS)
)


@dataclass(frozen=True)
class AuditBundle:
    """审计包：从一次 Run 的不可变产物中加载的证据集合。

    包含运行的事件流、状态快照、执行结果、审计报告等完整信息，
    用于评估、调试和问题排查。

    Attributes:
        run_id: 运行 ID。
        session_id: 会话 ID（可选）。
        events: 事件列表（从 events.jsonl 加载）。
        state: 状态快照（从 run.json 加载）。
        result: 执行结果（从 run.json 加载）。
        report: 审计报告（按需从 run.json 和 events.jsonl 构建）。
        workspace: 工作区路径。
        workspace_changes: 工作区变更列表。
    """
    run_id: str
    session_id: str | None
    events: list[dict[str, Any]]
    state: dict[str, Any]
    result: dict[str, Any]
    report: dict[str, Any]
    workspace: Path
    workspace_changes: list[dict[str, str]] = field(default_factory=list)


def redact_artifact(value: Any) -> Any:
    """递归脱敏：将常见的凭据字段和已知的密钥值替换为 <redacted>。"""

    secret_values = {
        secret_value
        for key in _SENSITIVE_ENV_KEYS
        if isinstance((secret_value := os.environ.get(key)), str)
        and secret_value
    }
    return _redact(value, secret_values)


def build_audit_report(
    result: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建稳定的模块级审计报告，供评估系统使用。

    报告包含以下维度：
    - run: 运行摘要（计数器、状态、停止原因等）
    - context: 上下文治理报告（压缩比、丢弃原因等）
    - memory: 记忆报告（检索次数、事件计数等）
    - security: 安全报告（工具状态、拒绝计数等）
    - task: 任务报告（步骤完成度、决策计数等）
    - recovery: 恢复报告（新鲜度历史等）
    """

    run_report = build_run_report(result, events=events)
    return redact_artifact(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "run": run_report,
            "context": _context_report(events),
            "memory": _memory_report(events),
            "security": _security_report(events),
            "task": _task_report(result, events),
            "planning": _planning_report(result, events),
            "recovery": _recovery_report(events),
            "state": dict(state or {}),
        }
    )


def load_audit_bundle(
    run_dir: str | Path,
    *,
    workspace: str | Path | None = None,
    workspace_changes: Iterable[dict[str, str]] = (),
) -> AuditBundle:
    """从磁盘加载一次 Run 的事件/状态/结果/报告。"""

    root = Path(run_dir)
    events = _read_jsonl(root / "events.jsonl")
    run = _read_json(root / "run.json")
    state = dict(run)
    result = dict(run)
    report = build_audit_report(result, events=events, state=state)
    workspace_path = Path(
        workspace
        or state.get("workspace_path")
        or root.parents[2]
    )
    return AuditBundle(
        run_id=str(result.get("run_id") or state.get("run_id") or root.name),
        session_id=_optional_str(
            result.get("session_id") or state.get("session_id")
        ),
        events=events,
        state=state,
        result=result,
        report=report,
        workspace=workspace_path,
        workspace_changes=[dict(item) for item in workspace_changes],
    )


def _context_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        event.get("report")
        for event in events
        if event.get("type") == "context_prepared"
        and isinstance(event.get("report"), dict)
    ]
    before = [_number(item.get("estimated_tokens_before")) for item in reports]
    after = [_number(item.get("estimated_tokens_after")) for item in reports]
    ratios = [
        max(0.0, (left - right) / max(left, 1))
        for left, right in zip(before, after)
    ]
    dropped_reasons: Counter[str] = Counter()
    stale_items = 0
    request_preserved = True
    section_totals: dict[str, dict[str, int]] = {}
    retrieved_ids: list[str] = []
    for report in reports:
        stale_items += len(_list(report.get("stale_items")))
        retrieved_ids.extend(
            str(item) for item in _list(report.get("retrieved_memory_ids"))
        )
        for item in _list_of_dicts(report.get("dropped_items")):
            dropped_reasons[str(item.get("reason", "unknown"))] += 1
        for section in _list_of_dicts(report.get("sections")):
            name = str(section.get("name", "unknown"))
            totals = section_totals.setdefault(
                name,
                {"budget_tokens": 0, "selected_items": 0, "candidate_items": 0},
            )
            totals["budget_tokens"] += _integer(section.get("budget_tokens"))
            totals["selected_items"] += _integer(section.get("selected_items"))
            totals["candidate_items"] += _integer(section.get("candidate_items"))
            if name == "current_request":
                request_preserved = request_preserved and (
                    _integer(section.get("candidate_items")) == 0
                    or _integer(section.get("selected_items")) > 0
                )
    return {
        "preparation_count": len(reports),
        "average_tokens_before": _average(before),
        "average_tokens_after": _average(after),
        "average_compression_ratio": _average(ratios),
        "current_request_preserved": request_preserved if reports else None,
        "stale_item_count": stale_items,
        "dropped_reason_counts": dict(sorted(dropped_reasons.items())),
        "sections": section_totals,
        "retrieved_memory_ids": sorted(set(retrieved_ids)),
        "reports": reports,
    }


def _memory_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    retrieved_ids: list[str] = []
    reasons: dict[str, list[str]] = {}
    for event in events:
        event_type = str(event.get("type", ""))
        if event_type.startswith("memory_"):
            event_counts[event_type] += 1
        if event_type == "memory_retrieved":
            retrieved_ids.extend(
                str(item) for item in _list(event.get("memoryIds"))
            )
            raw_reasons = event.get("reasons")
            if isinstance(raw_reasons, dict):
                for memory_id, values in raw_reasons.items():
                    reasons[str(memory_id)] = [
                        str(item) for item in _list(values)
                    ]
    read_calls = sum(
        event.get("type") == "tool_execution_start"
        and str(event.get("toolName", "")).lower() == "read"
        for event in events
    )
    return {
        "retrieved_memory_ids": sorted(set(retrieved_ids)),
        "retrieval_count": len(retrieved_ids),
        "retrieval_reasons": reasons,
        "event_counts": dict(sorted(event_counts.items())),
        "read_calls": read_calls,
    }


def _security_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    denied_mutations = 0
    mutation_after_denial = 0
    for event in events:
        if event.get("type") != "tool_execution_end":
            continue
        result = event.get("result")
        result_dict = result if isinstance(result, dict) else {}
        status = str(event.get("status") or result_dict.get("status") or "")
        statuses[status] += 1
        reason = (
            event.get("errorReason")
            or result_dict.get("error_code")
            or result_dict.get("error_reason")
        )
        if reason:
            reasons[str(reason)] += 1
        if status in {"denied", "approval_required"}:
            denied_mutations += 1
            changed = event.get("workspaceChanged")
            if changed is None:
                changed = result_dict.get("workspace_changed")
            if changed is True:
                mutation_after_denial += 1
    return {
        "tool_status_counts": dict(sorted(statuses.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "denied_or_approval_count": denied_mutations,
        "mutation_after_denial_count": mutation_after_denial,
        "false_allow_count": mutation_after_denial,
    }


def _task_report(
    result: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions: Counter[str] = Counter()
    completion_reasons: Counter[str] = Counter()
    step_updates = 0
    for event in events:
        event_type = event.get("type")
        if event_type == "task_decision":
            decision = event.get("decision")
            decision_dict = decision if isinstance(decision, dict) else {}
            action = decision_dict.get("action") or event.get("action")
            if action:
                decisions[str(action)] += 1
        elif event_type == "task_step_updated":
            step_updates += 1
        elif event_type == "completion_checked":
            completion = event.get("completion")
            completion_dict = completion if isinstance(completion, dict) else {}
            reason = completion_dict.get("reason") or event.get("reason")
            if reason:
                completion_reasons[str(reason)] += 1
    task = result.get("task")
    task_dict = task if isinstance(task, dict) else {}
    completed = _list(task_dict.get("completed_steps"))
    pending = _list(task_dict.get("pending_steps"))
    blocked = _list(task_dict.get("blocked_steps"))
    evidence_refs = _task_evidence_refs(events)
    return {
        "summary": task_dict,
        "completed_steps": len(completed),
        "pending_steps": len(pending),
        "blocked_steps": len(blocked),
        "completion_satisfied": task_dict.get("completion_satisfied"),
        "completion_reason": task_dict.get("completion_reason"),
        "decision_counts": dict(sorted(decisions.items())),
        "completion_reason_counts": dict(sorted(completion_reasons.items())),
        "step_update_count": step_updates,
        "evidence_ref_count": len(evidence_refs),
    }


def _task_evidence_refs(events: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for event in events:
        if event.get("type") != "task_step_updated":
            continue
        refs.extend(str(ref) for ref in _list(event.get("evidence_refs")))
        task = _dict(event.get("task"))
        for step in _list_of_dicts(task.get("steps")):
            refs.extend(str(ref) for ref in _list(step.get("evidence_refs")))
    return list(dict.fromkeys(refs))


def _planning_report(
    result: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    task = _dict(result.get("task"))
    control_signal = _dict(task.get("control_signal"))
    planning = _dict(control_signal.get("planning"))
    if not planning:
        for event in reversed(events):
            payload = _dict(event.get("planning"))
            if payload:
                planning = payload
                break
    discovery = _dict(planning.get("discovery"))
    budget = _dict(discovery.get("budget")) or _dict(planning.get("budget"))
    return {
        "phase": planning.get("phase"),
        "source": planning.get("source"),
        "discovery_status": discovery.get("status"),
        "facts_count": len(_list(discovery.get("facts"))),
        "relevant_files": _list(discovery.get("relevant_files")),
        "risks": _list(discovery.get("risks")),
        "verification_hints": _list(discovery.get("verification_hints")),
        "open_questions": _list(discovery.get("open_questions")),
        "evidence_refs": _list(discovery.get("evidence_refs")),
        "budget": budget,
        "fallback_reason": planning.get("fallbackReason")
        or planning.get("fallback_reason"),
    }


def _recovery_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    freshness = []
    for event in events:
        if event.get("type") != "context_freshness_checked":
            continue
        payload = event.get("freshness")
        if isinstance(payload, dict):
            freshness.append(payload)
    statuses = [
        str(item.get("status"))
        for item in freshness
        if item.get("status") is not None
    ]
    return {
        "freshness_history": freshness,
        "freshness_statuses": statuses,
        "stale_detected": "stale" in statuses,
        "mismatch_detected": "mismatch" in statuses,
    }


def _redact(value: Any, secret_values: set[str], key: str = "") -> Any:
    if _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(item_key): _redact(item, secret_values, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact(item, secret_values) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secret_values:
            redacted = redacted.replace(secret, "<redacted>")
        return redacted
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized == "token"
        or normalized.endswith("_token")
        or any(part in normalized for part in _SENSITIVE_KEY_PARTS)
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            values.append(value)
    return values


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in _list(value) if isinstance(item, dict)]


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditBundle",
    "build_audit_report",
    "load_audit_bundle",
    "redact_artifact",
]
