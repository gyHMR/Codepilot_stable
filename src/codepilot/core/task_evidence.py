from __future__ import annotations

"""Evidence extraction helpers for task control."""

from collections.abc import Mapping

from codepilot.protocols import TextContent, ToolResultMessage

from .task_tools import COMPLETE_TASK_STEP_TOOL

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


__all__ = [
    "complete_step_payload",
    "evidence_refs",
    "first_error_code",
    "infer_action_intent",
    "is_tool_unavailable",
]
