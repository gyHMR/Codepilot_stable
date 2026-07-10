from __future__ import annotations

"""The single model-facing snapshot tool for the canonical task plan."""

from collections.abc import Callable
from typing import Any

from codepilot.protocols import PLAN_ITEM_LIMIT, TextContent, UPDATE_PLAN_TOOL
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata

_ITEM_STATUSES = {"pending", "in_progress", "completed"}
_PLAN_STATUSES = {"active", "completed"}
_CHANGE_REASONS = {"user_request", "repeated_execution_failure"}


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
                "Submit the complete canonical task-plan snapshot. In plan mode this creates "
                "or revises a pending proposal. In build mode it updates the active execution "
                "contract; structural changes require change_reason. execution_objective is required "
                "only when creating a new plan and must describe the software work, not plan production."
            ),
            parameters=_snapshot_schema(),
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
        if request.current_mode == "read":
            raise ValueError("read mode cannot update a plan")
        snapshot = _validate_snapshot(request.arguments, proposal=request.current_mode == "plan")
        if request.current_mode == "plan":
            for item in snapshot["items"]:
                item["status"] = "pending"
    except ValueError as exc:
        return ToolResult(
            content=[TextContent(text=str(exc))],
            status="error",
            is_error=True,
            error_code="invalid_plan_snapshot",
        )
    return ToolResult(
        content=[TextContent(text="Plan snapshot submitted.")],
        metadata={"plan_snapshot": snapshot},
    )


def _snapshot_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Required when updating an unchanged active step."},
            "step": {"type": "string"},
            "details": {"type": "string"},
            "verification": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
        },
        "required": ["step", "details", "verification", "status"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "execution_objective": {
                "type": "string",
                "description": "Required only when creating a plan. State the code or behavior build must complete, not 'write a plan'.",
            },
            "summary": {"type": "string"},
            "completion_criteria": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": PLAN_ITEM_LIMIT,
                "items": item,
            },
            "status": {
                "type": "string",
                "enum": ["active", "completed"],
                "description": "Set completed only after checking the completion criteria.",
            },
            "change_reason": {
                "type": "string",
                "enum": ["user_request", "repeated_execution_failure"],
            },
            "explanation": {"type": "string"},
        },
        "required": ["summary", "completion_criteria", "items"],
        "additionalProperties": False,
    }


def _validate_snapshot(params: dict[str, Any], *, proposal: bool) -> dict[str, Any]:
    allowed = {
        "execution_objective",
        "summary",
        "completion_criteria",
        "items",
        "status",
        "change_reason",
        "explanation",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError("unknown plan snapshot fields: " + ", ".join(unknown))

    summary = _required_text(params.get("summary"), "summary")
    criteria = params.get("completion_criteria")
    if not isinstance(criteria, list) or not 1 <= len(criteria) <= 5:
        raise ValueError("completion_criteria must contain between 1 and 5 items")
    criteria = [_required_text(item, f"completion_criteria[{index}]") for index, item in enumerate(criteria)]

    raw_items = params.get("items")
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= PLAN_ITEM_LIMIT:
        raise ValueError(f"items must contain between 1 and {PLAN_ITEM_LIMIT} items")
    items: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"items[{index}] must be an object")
        unknown_item = sorted(set(raw_item) - {"id", "step", "details", "verification", "status"})
        if unknown_item:
            raise ValueError(f"items[{index}] has unknown fields: " + ", ".join(unknown_item))
        item = {
            "step": _required_text(raw_item.get("step"), f"items[{index}].step"),
            "details": _required_text(raw_item.get("details"), f"items[{index}].details"),
            "verification": _required_text(
                raw_item.get("verification"), f"items[{index}].verification"
            ),
            "status": str(raw_item.get("status") or "").strip(),
        }
        if item["status"] not in _ITEM_STATUSES:
            raise ValueError(f"items[{index}].status is invalid")
        item_id = raw_item.get("id")
        if item_id is not None:
            item_id = _required_text(item_id, f"items[{index}].id")
            if item_id in ids:
                raise ValueError("snapshot item ids must be unique")
            ids.add(item_id)
            item["id"] = item_id
        items.append(item)

    status = params.get("status")
    if proposal and status is not None:
        raise ValueError("plan proposals cannot set status")
    if status is not None and status not in _PLAN_STATUSES:
        raise ValueError("snapshot status is invalid")
    change_reason = params.get("change_reason")
    if change_reason is not None and change_reason not in _CHANGE_REASONS:
        raise ValueError("snapshot change_reason is invalid")

    result: dict[str, Any] = {
        "summary": summary,
        "completion_criteria": criteria,
        "items": items,
        "explanation": _optional_text(params.get("explanation")),
    }
    execution_objective = _optional_text(params.get("execution_objective"))
    if execution_objective is not None:
        result["execution_objective"] = execution_objective
    if status is not None:
        result["status"] = status
    if change_reason is not None:
        result["change_reason"] = change_reason
    return result


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


__all__ = ["create_plan_tools"]
