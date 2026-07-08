from __future__ import annotations

"""Built-in soft update_plan tool."""

from collections.abc import Callable
from typing import Any

from codepilot.protocols import TextContent, UPDATE_PLAN_TOOL
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata

_ITEM_STATUSES = {"pending", "in_progress", "completed"}
_MAX_PLAN_ITEMS = 10


def create_plan_tools(*, allow: Callable[[str], bool]) -> list[ToolDefinition]:
    if not allow(UPDATE_PLAN_TOOL):
        return []
    metadata = get_builtin_tool_metadata(UPDATE_PLAN_TOOL)
    if metadata is None:
        raise ValueError(f"Missing builtin metadata for {UPDATE_PLAN_TOOL}")
    return [
        ToolDefinition(
            name=UPDATE_PLAN_TOOL,
            label="Update plan",
            description=(
                "Update the visible soft plan board. This communicates progress "
                "or a proposed plan, but never completes the run."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "plan": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_PLAN_ITEMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["step", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
            metadata=metadata,
            execute=_execute_update_plan,
        )
    ]


async def _execute_update_plan(
    request: ToolCallRequest,
    signal: Any = None,
    on_update: Any = None,
) -> ToolResult:
    _ = signal, on_update
    try:
        plan_update = _validated_plan_update(request.arguments)
    except ValueError as exc:
        return ToolResult(
            content=[TextContent(text=str(exc))],
            status="error",
            is_error=True,
            error_code="invalid_plan_update",
        )
    return ToolResult(
        content=[TextContent(text="Plan updated.")],
        metadata={"plan_update": plan_update},
    )


def _validated_plan_update(params: dict[str, Any]) -> dict[str, Any]:
    plan = params.get("plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must contain at least one item")
    if len(plan) > _MAX_PLAN_ITEMS:
        raise ValueError(f"plan cannot contain more than {_MAX_PLAN_ITEMS} items")
    explanation = _clean_text(params.get("explanation")) or ""
    items: list[dict[str, str]] = []
    in_progress = 0
    for index, raw in enumerate(plan):
        if not isinstance(raw, dict):
            raise ValueError(f"plan[{index}] must be an object")
        step = _clean_text(raw.get("step"))
        if not step:
            raise ValueError(f"plan[{index}].step is required")
        status = _clean_text(raw.get("status"))
        if status not in _ITEM_STATUSES:
            raise ValueError(f"plan[{index}].status is invalid")
        if status == "in_progress":
            in_progress += 1
        items.append({"step": step, "status": status})
    if in_progress > 1:
        raise ValueError("plan can contain at most one in_progress item")
    return {"explanation": explanation, "plan": items}


def _clean_text(value: object) -> str:
    return " ".join(str(value).strip().split()) if value is not None else ""


__all__ = ["create_plan_tools"]
