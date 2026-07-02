from __future__ import annotations

"""Verification interpretation helpers for task control."""

from collections.abc import Mapping

from codepilot.protocols import ToolResultMessage

from .task_state import TaskStep


def has_failed_verification(results: list[ToolResultMessage]) -> bool:
    return any(
        isinstance(result.verification, dict)
        and result.verification.get("status") == "failed"
        for result in results
    )


def has_passed_verification(results: list[ToolResultMessage]) -> bool:
    return any(
        isinstance(result.verification, dict)
        and result.verification.get("status") == "passed"
        for result in results
    )


def has_non_verification_error(results: list[ToolResultMessage]) -> bool:
    return any(
        (result.status != "success" or result.is_error)
        and not isinstance(result.verification, dict)
        for result in results
    )


def is_verification_step(step: TaskStep) -> bool:
    if step.kind == "verify":
        return True
    text = f"{step.title} {step.verification_hint or ''}".lower()
    markers = ("验证", "测试", "检查", "verify", "test", "pytest")
    return any(marker in text for marker in markers)


def verification_failure_note(results: list[ToolResultMessage]) -> str:
    detail = verification_failure_detail(results)
    return f"验证失败，需要修复：{detail}" if detail else "验证失败，需要修复"


def verification_failure_detail(results: list[ToolResultMessage]) -> str:
    for result in results:
        verification = result.verification
        if not isinstance(verification, Mapping):
            continue
        if verification.get("status") != "failed":
            continue
        parts: list[str] = []
        command = _compact(verification.get("command"), limit=160)
        if command:
            parts.append(f"命令 {command}")
        exit_code = verification.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            parts.append(f"exit_code={exit_code}")
        summary = _compact(verification.get("summary"), limit=220)
        if summary:
            parts.append(f"摘要 {summary}")
        return "；".join(parts)
    return ""


def verification_command(results: list[ToolResultMessage]) -> str | None:
    for result in results:
        verification = result.verification
        if not isinstance(verification, Mapping):
            continue
        if verification.get("status") != "failed":
            continue
        command = _compact(verification.get("command"), limit=160)
        if command:
            return command
    return None


def _compact(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


__all__ = [
    "has_failed_verification",
    "has_non_verification_error",
    "has_passed_verification",
    "is_verification_step",
    "verification_command",
    "verification_failure_detail",
    "verification_failure_note",
]
