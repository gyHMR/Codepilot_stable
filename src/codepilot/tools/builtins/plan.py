from __future__ import annotations

"""Model-facing semantic tools for the canonical session Task Plan."""

from collections.abc import Callable
from typing import Any, Literal

from codepilot.protocols import (
    CLOSE_PLAN_TOOL,
    CREATE_BUILD_PLAN_TOOL,
    PLAN_ITEM_LIMIT,
    PROPOSE_PLAN_TOOL,
    TextContent,
    UPDATE_PLAN_PROGRESS_TOOL,
)
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata

_ITEM_STATUSES = {"pending", "in_progress", "completed"}
_PLAN_STATUSES = {"active", "completed"}
_CHANGE_REASONS = {"user_request", "repeated_execution_failure"}
_PROPOSAL_DETAIL_FIELDS = {
    "task_understanding",
    "current_implementation",
    "target_design",
    "impact_scope",
    "risks_and_open_questions",
    "verification_plan",
}
_Operation = Literal[
    "propose_plan",
    "create_build_plan",
    "update_plan_progress",
    "close_plan",
]


def create_plan_tools(*, allow: Callable[[str], bool]) -> list[ToolDefinition]:
    definitions = [
        _tool_definition(
            PROPOSE_PLAN_TOOL,
            label="Propose plan",
            description=(
                "Plan mode only. This is the required and only authoritative way to publish a "
                "completed implementation plan for user approval. Call it immediately after "
                "read-only repository exploration is sufficient and the plan is ready for Build. "
                "Do not first present the full plan in ordinary assistant text, ask whether the "
                "direction is acceptable, or wait for informal approval. The runtime validates, "
                "stores, renders, and requests approval for the canonical plan; ordinary assistant "
                "text is not an approvable Task Plan. The proposal must include structured evidence "
                "and design fields, and every step must be a real Build-mode code change or "
                "verification step. Item IDs and statuses are framework-owned and must not be "
                "supplied. Do not describe planning work such as "
                "reading code, drafting the plan, replying, or waiting for approval. "
                "Include raw_user_request and interpreted_goal on every proposal, including revisions."
            ),
            operation="propose_plan",
            schema=_snapshot_schema(
                require_request=True,
                require_goal=True,
                allow_status=False,
                require_proposal_details=True,
            ),
        ),
        _tool_definition(
            CREATE_BUILD_PLAN_TOOL,
            label="Create build plan",
            description=(
                "Build mode only. Create a lightweight active Task Plan when no current Task "
                "Plan exists and the user task is complex enough to need progress tracking."
            ),
            operation="create_build_plan",
            schema=_snapshot_schema(
                require_request=True,
                require_goal=True,
                allow_status=False,
                require_proposal_details=False,
            ),
        ),
        _tool_definition(
            UPDATE_PLAN_PROGRESS_TOOL,
            label="Update plan progress",
            description=(
                "Build mode only. Update status, summary, or controlled revisions of the "
                "current active Task Plan. Do not use this to complete the plan; use close_plan."
            ),
            operation="update_plan_progress",
            schema=_snapshot_schema(
                require_request=False,
                require_goal=False,
                allow_status=False,
                require_proposal_details=False,
            ),
        ),
        _tool_definition(
            CLOSE_PLAN_TOOL,
            label="Close plan",
            description=(
                "Build mode only. Final Task Plan closeout before the final answer. Set status "
                "to completed when the task is done, or active with remaining steps when it is "
                "clearly unfinished."
            ),
            operation="close_plan",
            schema=_snapshot_schema(
                require_request=False,
                require_goal=False,
                allow_status=True,
                require_proposal_details=False,
            ),
        ),
    ]
    return [tool for tool in definitions if allow(tool.name)]


def _tool_definition(
    name: str,
    *,
    label: str,
    description: str,
    operation: _Operation,
    schema: dict[str, Any],
) -> ToolDefinition:
    metadata = get_builtin_tool_metadata(name)
    if metadata is None:
        raise ValueError(f"Missing builtin metadata for {name}")
    return ToolDefinition(
        name=name,
        label=label,
        description=description,
        parameters=schema,
        metadata=metadata,
        execute=_execute_plan_tool(operation),
    )


def _execute_plan_tool(operation: _Operation):
    async def execute(
        request: ToolCallRequest,
        signal: Any = None,
        on_update: Any = None,
    ) -> ToolResult:
        _ = signal, on_update
        try:
            _validate_mode(operation, request.current_mode)
            snapshot = _validate_snapshot(
                request.arguments,
                operation=operation,
            )
        except ValueError as exc:
            return ToolResult(
                content=[TextContent(text=str(exc))],
                status="error",
                is_error=True,
                error_code="invalid_plan_snapshot",
            )
        return ToolResult(
            content=[TextContent(text="Task Plan snapshot submitted.")],
            metadata={"plan_operation": operation, "plan_snapshot": snapshot},
        )

    return execute


def _validate_mode(operation: _Operation, current_mode: str) -> None:
    mode = str(current_mode).strip()
    if operation == "propose_plan" and mode != "plan":
        raise ValueError("propose_plan is only available in plan mode")
    if operation != "propose_plan" and mode != "build":
        raise ValueError(f"{operation} is only available in build mode")


def _snapshot_schema(
    *,
    require_request: bool,
    require_goal: bool,
    allow_status: bool,
    require_proposal_details: bool,
) -> dict[str, Any]:
    item_properties: dict[str, Any] = {
        "step": {"type": "string"},
        "details": {"type": "string"},
        "verification": {"type": "string"},
    }
    item_required = ["step", "details", "verification"]
    if not require_proposal_details:
        item_properties = {
            "id": {"type": "string", "description": "Required when updating an existing active step."},
            **item_properties,
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
        }
        item_required.append("status")
    item = {
        "type": "object",
        "properties": item_properties,
        "required": item_required,
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {
        "raw_user_request": {
            "type": "string",
            "description": "The user's original request text or a faithful concise quote of it.",
        },
        "interpreted_goal": {
            "type": "string",
            "description": "The software outcome Build should achieve, excluding control phrases like 'write a plan'.",
        },
        "summary": {"type": "string"},
        "task_understanding": {
            "type": "string",
            "description": "Plan mode: how the agent understands the user's software task.",
        },
        "current_implementation": {
            "type": "string",
            "description": "Plan mode: concrete repository facts, files, symbols, tests, or call paths observed.",
        },
        "target_design": {
            "type": "string",
            "description": "Plan mode: intended design or behavior after Build executes the plan.",
        },
        "impact_scope": {
            "type": "string",
            "description": "Plan mode: modules, APIs, tests, data, or compatibility surfaces affected.",
        },
        "risks_and_open_questions": {
            "type": "array",
            "minItems": 0,
            "maxItems": 8,
            "items": {"type": "string"},
        },
        "verification_plan": {
            "type": "string",
            "description": "Plan mode: Build-time tests, checks, or manual verification needed.",
        },
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
        "change_reason": {
            "type": "string",
            "enum": ["user_request", "repeated_execution_failure"],
        },
        "explanation": {"type": "string"},
    }
    required = ["summary", "completion_criteria", "items"]
    if require_request:
        required.insert(0, "raw_user_request")
    if require_goal:
        required.insert(1 if require_request else 0, "interpreted_goal")
    if require_proposal_details:
        required.extend(
            [
                "task_understanding",
                "current_implementation",
                "target_design",
                "impact_scope",
                "risks_and_open_questions",
                "verification_plan",
            ]
        )
    if allow_status:
        properties["status"] = {
            "type": "string",
            "enum": ["active", "completed"],
            "description": "Use completed only after checking the completion criteria; use active if clearly unfinished.",
        }
        required.append("status")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _validate_snapshot(
    params: dict[str, Any],
    *,
    operation: _Operation,
) -> dict[str, Any]:
    allowed = {
        "raw_user_request",
        "interpreted_goal",
        "summary",
        "task_understanding",
        "current_implementation",
        "target_design",
        "impact_scope",
        "risks_and_open_questions",
        "verification_plan",
        "completion_criteria",
        "items",
        "status",
        "change_reason",
        "explanation",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError("unknown plan snapshot fields: " + ", ".join(unknown))

    require_creation_fields = operation in {"propose_plan", "create_build_plan"}
    raw_user_request = _optional_text(params.get("raw_user_request"))
    interpreted_goal = _optional_text(params.get("interpreted_goal"))
    if require_creation_fields and raw_user_request is None:
        raise ValueError("raw_user_request is required")
    if require_creation_fields and interpreted_goal is None:
        raise ValueError("interpreted_goal is required")

    proposal_details = _validate_proposal_details(params, operation=operation)

    summary = _required_text(params.get("summary"), "summary")
    criteria = _text_list(
        params.get("completion_criteria"),
        "completion_criteria",
        min_items=1,
        max_items=5,
        allow_string=operation == "propose_plan",
    )

    raw_items = params.get("items")
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= PLAN_ITEM_LIMIT:
        raise ValueError(f"items must contain between 1 and {PLAN_ITEM_LIMIT} items")
    items: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"items[{index}] must be an object")
        allowed_item_fields = (
            {"step", "details", "verification"}
            if operation == "propose_plan"
            else {"id", "step", "details", "verification", "status"}
        )
        unknown_item = sorted(set(raw_item) - allowed_item_fields)
        if unknown_item:
            raise ValueError(f"items[{index}] has unknown fields: " + ", ".join(unknown_item))
        item = {
            "step": _required_text(raw_item.get("step"), f"items[{index}].step"),
            "details": _required_text(raw_item.get("details"), f"items[{index}].details"),
            "verification": _required_text(raw_item.get("verification"), f"items[{index}].verification"),
            "status": "pending" if operation == "propose_plan" else str(raw_item.get("status") or "").strip(),
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
    if operation != "close_plan" and status is not None:
        raise ValueError(f"{operation} cannot set status")
    if operation == "close_plan" and status not in _PLAN_STATUSES:
        raise ValueError("close_plan status must be active or completed")
    change_reason = params.get("change_reason")
    if change_reason is not None and change_reason not in _CHANGE_REASONS:
        raise ValueError("snapshot change_reason is invalid")

    result: dict[str, Any] = {
        "summary": summary,
        "completion_criteria": criteria,
        "items": items,
        **proposal_details,
        "explanation": _optional_text(params.get("explanation")),
    }
    if raw_user_request is not None:
        result["raw_user_request"] = raw_user_request
    if interpreted_goal is not None:
        result["interpreted_goal"] = interpreted_goal
    if status is not None:
        result["status"] = status
    if change_reason is not None:
        result["change_reason"] = change_reason
    return result


def _validate_proposal_details(
    params: dict[str, Any],
    *,
    operation: _Operation,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if operation != "propose_plan":
        for field in _PROPOSAL_DETAIL_FIELDS:
            if field in params:
                if field == "risks_and_open_questions":
                    values[field] = _text_list(
                        params.get(field),
                        field,
                        min_items=0,
                        max_items=8,
                    )
                else:
                    values[field] = _required_text(params.get(field), field)
        return values

    for field in [
        "task_understanding",
        "current_implementation",
        "target_design",
        "impact_scope",
        "verification_plan",
    ]:
        values[field] = _required_text(params.get(field), field)
    values["risks_and_open_questions"] = _text_list(
        params.get("risks_and_open_questions"),
        "risks_and_open_questions",
        min_items=0,
        max_items=8,
        allow_string=True,
    )
    return values


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _text_list(
    value: object,
    field_name: str,
    *,
    min_items: int,
    max_items: int,
    allow_string: bool = False,
) -> list[str]:
    if allow_string and isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not min_items <= len(value) <= max_items:
        raise ValueError(
            f"{field_name} must contain between {min_items} and {max_items} items"
        )
    return [_required_text(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


__all__ = ["create_plan_tools"]
