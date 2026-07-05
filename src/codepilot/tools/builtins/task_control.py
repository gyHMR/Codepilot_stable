from __future__ import annotations

# 新手导读：这里创建 task-control 内置工具；core 只解释它返回的 task_control 信号。
# 关注点：工具对象属于 tools 层，避免 core 直接依赖可执行工具实现契约。

"""Built-in task-control tools."""

from collections.abc import Callable
from typing import Any

from codepilot.protocols import TASK_CONTROL_COMPLETE_TOOL, TextContent
from codepilot.tools.authoring import AgentTool, AgentToolResult


def create_task_control_tools(*, allow: Callable[[str], bool]) -> list[AgentTool]:
    if not allow(TASK_CONTROL_COMPLETE_TOOL):
        return []
    return [
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
    ]


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
