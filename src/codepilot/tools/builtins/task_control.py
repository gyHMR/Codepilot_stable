from __future__ import annotations

# 新手导读：这里创建 task-control 内置工具；core 只解释它返回的 task_control 信号。
# 关注点：工具对象属于 tools 层，避免 core 直接依赖可执行工具实现契约。

"""Built-in task-control tools."""

from collections.abc import Callable
from typing import Any

from codepilot.protocols import (
    TASK_CONTROL_COMPLETE_TOOL,
    TASK_CONTROL_UPDATE_TOOL,
    TextContent,
)
from codepilot.tools.authoring import AgentTool, AgentToolResult


def create_task_control_tools(*, allow: Callable[[str], bool]) -> list[AgentTool]:
    tools: list[AgentTool] = []
    if allow(TASK_CONTROL_UPDATE_TOOL):
        tools.append(
            AgentTool(
                name=TASK_CONTROL_UPDATE_TOOL,
                label="Update task step",
                description=(
                    "Propose an evidence-backed status update for the current "
                    "task step. Use completed only when the step acceptance "
                    "criteria are satisfied and evidence_refs point to real "
                    "tool evidence."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "step_id": {
                            "type": "string",
                            "description": "Current task step id, such as step_1.",
                        },
                        "proposed_status": {
                            "type": "string",
                            "enum": ["in_progress", "completed", "blocked"],
                            "description": "Status proposed for the current step.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Short evidence-backed update summary.",
                        },
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Evidence references such as tool:read_1.",
                        },
                    },
                    "required": ["step_id", "proposed_status", "summary"],
                    "additionalProperties": False,
                },
                execute=_execute_task_update,
                runtime_managed=True,
            )
        )
    if allow(TASK_CONTROL_COMPLETE_TOOL):
        tools.append(
            AgentTool(
                name=TASK_CONTROL_COMPLETE_TOOL,
                label="Complete task step",
                description=(
                    "Mark the current task step as complete when its acceptance "
                    "criteria are satisfied. Use this for investigation, planning, "
                    "or summary steps after gathering enough evidence. Do not use "
                    "it to bypass required verification after workspace changes."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": (
                                "Short evidence-backed summary of what was completed."
                            ),
                        },
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional evidence references such as tool:read_1."
                            ),
                        },
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
                execute=_execute_complete_task_step,
                runtime_managed=True,
            )
        )
    return tools


async def _execute_task_update(
    tool_call_id: str,
    params: dict[str, Any],
    signal: Any = None,
    on_update: Any = None,
) -> AgentToolResult:
    _ = signal, on_update
    step_id = " ".join(str(params.get("step_id") or "").strip().split())
    proposed_status = str(params.get("proposed_status") or "").strip()
    summary = " ".join(str(params.get("summary") or "").strip().split())
    evidence_refs = [
        str(item).strip()
        for item in params.get("evidence_refs", [])
        if isinstance(item, str) and item.strip()
    ]
    if not step_id:
        return _task_update_error("step_id is required", "missing_step_id")
    if proposed_status not in {"in_progress", "completed", "blocked"}:
        return _task_update_error("proposed_status is invalid", "invalid_status")
    if not summary:
        return _task_update_error("summary is required", "missing_summary")
    return AgentToolResult(
        content=[TextContent(text=f"Task step update proposed: {proposed_status}")],
        metadata={
            "task_control": {
                "action": "update_step",
                "step_id": step_id[:80],
                "proposed_status": proposed_status,
                "summary": summary[:500],
                "evidence_refs": evidence_refs,
                "tool_call_id": tool_call_id,
            }
        },
    )


def _task_update_error(message: str, code: str) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=message)],
        status="error",
        is_error=True,
        error_code=code,
        metadata={
            "task_control": {
                "action": "update_step",
                "valid": False,
            }
        },
    )


async def _execute_complete_task_step(
    tool_call_id: str,
    params: dict[str, Any],
    signal: Any = None,
    on_update: Any = None,
) -> AgentToolResult:
    _ = signal, on_update
    summary = " ".join(str(params.get("summary") or "").strip().split())
    evidence_refs = [
        str(item)
        for item in params.get("evidence_refs", [])
        if isinstance(item, str) and item.strip()
    ]
    if not summary:
        return AgentToolResult(
            content=[TextContent(text="summary is required")],
            status="error",
            is_error=True,
            error_code="missing_summary",
            metadata={
                "task_control": {
                    "action": "complete_step",
                    "valid": False,
                }
            },
        )
    return AgentToolResult(
        content=[TextContent(text=f"Current task step completed: {summary}")],
        metadata={
            "task_control": {
                "action": "complete_step",
                "summary": summary[:500],
                "evidence_refs": evidence_refs,
                "tool_call_id": tool_call_id,
            }
        },
    )


__all__ = ["create_task_control_tools"]
