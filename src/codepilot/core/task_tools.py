from __future__ import annotations

"""Internal task-control tools exposed to the model during a run."""

from typing import Any

from codepilot.protocols import TextContent
from codepilot.tools import AgentTool, AgentToolResult


COMPLETE_TASK_STEP_TOOL = "complete_task_step"


def complete_task_step_tool() -> AgentTool:
    """Return the internal tool used to mark the current task step complete."""

    return AgentTool(
        name=COMPLETE_TASK_STEP_TOOL,
        label="Complete task step",
        description=(
            "Mark the current task step as complete when its acceptance criteria "
            "are satisfied. Use this for investigation, planning, or summary "
            "steps after gathering enough evidence. Do not use it to bypass "
            "required verification after workspace changes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short evidence-backed summary of what was completed.",
                },
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional evidence references such as tool:read_1.",
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        execute=_execute_complete_task_step,
        runtime_managed=True,
    )


def has_complete_task_step_tool(tools: list[AgentTool]) -> bool:
    return any(tool.name == COMPLETE_TASK_STEP_TOOL for tool in tools)


async def _execute_complete_task_step(
    tool_call_id: str,
    params: dict[str, Any],
    signal=None,
    on_update=None,
) -> AgentToolResult:
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


__all__ = [
    "COMPLETE_TASK_STEP_TOOL",
    "complete_task_step_tool",
    "has_complete_task_step_tool",
]
