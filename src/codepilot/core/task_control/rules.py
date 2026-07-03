from __future__ import annotations

# 新手导读：本文件收拢任务控制中的小规则：证据提取、验证结果判断、完成门控和重规划提示。
# 关注点：如果想调整任务控制行为，优先在这里找对应的纯函数规则。

"""Deterministic task-control rules used by the Agent loop."""

from collections.abc import Mapping

from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

from ..run_state import RunState
from .state import CompletionCheck, TaskState, TaskStep
from .tools import COMPLETE_TASK_STEP_TOOL

READ_TOOL_MARKERS = ("read", "grep", "find", "glob", "ls", "search", "status", "codegraph")
WRITE_TOOL_MARKERS = ("write", "edit", "patch", "apply")


def evidence_refs(results: list[ToolResultMessage]) -> list[str]:
    refs: list[str] = []
    for result in results:
        if result.tool_call_id:
            refs.append(f"tool:{result.tool_call_id}")
        if result.approval_id:
            refs.append(f"approval:{result.approval_id}")
        if isinstance(result.verification, dict) and result.tool_call_id:
            refs.append(f"verification:{result.tool_call_id}")
        for path in result.affected_paths:
            refs.append(f"file:{path}")
    return refs


def first_error_code(results: list[ToolResultMessage]) -> str | None:
    return next((result.error_code for result in results if result.error_code), None)


def complete_step_payload(result: ToolResultMessage) -> Mapping[str, object] | None:
    if result.tool_name != COMPLETE_TASK_STEP_TOOL:
        return None
    payload = result.metadata.get("task_control")
    if not isinstance(payload, Mapping):
        return None
    if payload.get("action") != "complete_step":
        return None
    if payload.get("valid") is False:
        return None
    return payload


def is_tool_unavailable(result: ToolResultMessage) -> bool:
    if result.status != "error" and not result.is_error:
        return False
    if result.error_code == "tool_not_found":
        return True
    text = " ".join(
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    ).lower()
    return text.startswith("tool ") and " not found" in text


def infer_action_intent(results: list[ToolResultMessage]) -> str:
    if any(complete_step_payload(result) is not None for result in results):
        return "complete_step"
    if any(
        isinstance(result.verification, dict)
        and result.verification.get("status") == "failed"
        for result in results
    ):
        return "debug_failure"
    if any(isinstance(result.verification, dict) for result in results):
        return "run_verification"
    if any(result.workspace_changed is True for result in results):
        return "edit_file"
    names = " ".join(result.tool_name.lower() for result in results)
    if any(marker in names for marker in READ_TOOL_MARKERS):
        return "read_context"
    if any(marker in names for marker in WRITE_TOOL_MARKERS):
        return "edit_file"
    return "tool_action"


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


def repair_next_action(results: list[ToolResultMessage]) -> str:
    detail = verification_failure_detail(results)
    if detail:
        return (
            f"根据验证失败证据修复实现：{detail}。"
            "先读取失败断言和相关调用链，完成最小修复后重新运行同一验证。"
        )
    return "根据最新验证失败证据定位根因，完成最小修复后重新运行相关验证"


def should_propose_revert_after_repeated_failure(
    *,
    failure_count: int,
    has_change_sets: bool,
) -> bool:
    return failure_count >= 2 and has_change_sets


def build_completion_check(task: TaskState, run: RunState) -> CompletionCheck:
    if task.blocked_step_titles():
        return CompletionCheck(
            satisfied=False,
            reason="blocked_steps",
            missing=["unblocked_steps"],
            can_continue=False,
        )

    if run.workspace_changed and not run.fresh_verification_passed:
        can_continue = task.completion_prompt_count == 0
        return CompletionCheck(
            satisfied=False,
            reason="modified_without_fresh_verification",
            missing=["fresh_verification"],
            can_continue=can_continue,
            unverified=not can_continue,
        )

    incomplete = task.pending_step_titles()
    if incomplete:
        return CompletionCheck(
            satisfied=False,
            reason="incomplete_steps",
            missing=incomplete,
            can_continue=False,
        )

    return CompletionCheck(satisfied=True, reason="all_steps_completed")


def completion_steering_message(check: CompletionCheck) -> UserMessage:
    text = (
        "工作区已经发生修改，但当前没有与最新工作区状态一致的成功验证。\n"
        "请运行最相关的测试或检查；如果环境无法验证，请明确记录原因和剩余风险。"
    )
    return UserMessage(
        content=[TextContent(text=text)],
        metadata={
            "task_completion_gate": {
                "reason": check.reason,
                "missing": list(check.missing),
            }
        },
    )


def _compact(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


__all__ = [
    "READ_TOOL_MARKERS",
    "WRITE_TOOL_MARKERS",
    "build_completion_check",
    "complete_step_payload",
    "completion_steering_message",
    "evidence_refs",
    "first_error_code",
    "has_failed_verification",
    "has_non_verification_error",
    "has_passed_verification",
    "infer_action_intent",
    "is_tool_unavailable",
    "is_verification_step",
    "repair_next_action",
    "should_propose_revert_after_repeated_failure",
    "verification_command",
    "verification_failure_note",
]
