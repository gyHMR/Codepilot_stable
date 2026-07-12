from __future__ import annotations

"""Core-owned canonical registrations for Task Plan operations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from codepilot.protocols import (
    CLOSE_PLAN_TOOL,
    CREATE_BUILD_PLAN_TOOL,
    PLAN_ITEM_LIMIT,
    PROPOSE_PLAN_TOOL,
    UPDATE_PLAN_PROGRESS_TOOL,
)
from codepilot.tools.codecs import JsonObjectCodec
from codepilot.tools.contracts import (
    ToolExecutionRequest,
    ToolHandlerError,
    ToolRegistration,
    ToolSpec,
)
from codepilot.tools.results import TextContent
from codepilot.tools.security import (
    ConcurrencyPolicy,
    OutputLimits,
    OutputTrustPolicy,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolPolicy,
    ToolResource,
)

from ..plan import (
    PlanOperation,
    PlanSnapshot,
    PlanState,
    PlanValidationError,
    apply_plan_snapshot,
    ensure_plan_operation,
    load_plan_state,
)


class PlanService(Protocol):
    """Narrow Core boundary used by Plan handlers to atomically submit one operation."""

    def submit(
        self,
        operation: PlanOperation,
        snapshot: PlanSnapshot,
        request: ToolExecutionRequest,
    ) -> Mapping[str, object]:
        ...


@dataclass
class StoreBackedPlanService:
    load: Callable[[], Mapping[str, object] | None]
    save: Callable[[Mapping[str, object] | PlanState], Mapping[str, object]]
    qualified_failure_count: Callable[[str], int] = lambda _run_id: 0

    def submit(
        self,
        operation: PlanOperation,
        snapshot: PlanSnapshot,
        request: ToolExecutionRequest,
    ) -> Mapping[str, object]:
        current = load_plan_state(self.load())
        state = apply_plan_snapshot(
            current,
            snapshot,
            mode="plan" if request.mode == "plan" else "build",
            run_id=request.run_id,
            operation=operation,
            qualified_failure_count=self.qualified_failure_count(request.run_id),
        )
        return self.save(state)


def create_plan_registrations(
    *,
    service: PlanService,
    allow: Callable[[str], bool] | None = None,
) -> list[ToolRegistration]:
    allowed = allow or (lambda _name: True)
    operations = (
        (PROPOSE_PLAN_TOOL, "plan", _plan_description(PROPOSE_PLAN_TOOL)),
        (CREATE_BUILD_PLAN_TOOL, "execute", _plan_description(CREATE_BUILD_PLAN_TOOL)),
        (UPDATE_PLAN_PROGRESS_TOOL, "execute", _plan_description(UPDATE_PLAN_PROGRESS_TOOL)),
        (CLOSE_PLAN_TOOL, "execute", _plan_description(CLOSE_PLAN_TOOL)),
    )
    return [
        _plan_registration(name, mode=mode, description=description, service=service)
        for name, mode, description in operations
        if allowed(name)
    ]


def _plan_registration(
    name: str,
    *,
    mode: str,
    description: str,
    service: PlanService,
) -> ToolRegistration:
    operation = ensure_plan_operation(name)
    input_schema = _snapshot_schema(operation)
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "plan_operation": {"type": "string", "const": operation},
            "plan_snapshot": {"type": "object"},
            "plan_state": {"type": "object"},
        },
        "required": ["plan_operation", "plan_snapshot", "plan_state"],
        "additionalProperties": False,
    }
    input_codec = JsonObjectCodec(input_schema)
    output_codec = JsonObjectCodec(output_schema)

    class Resolver:
        def resolve(self, input, request):
            resource = ToolResource(f"session://{request.session_id}/task-plan")
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(operation,),
                    resources=(resource,),
                    effects=frozenset({"session_state_write"}),
                    risk="low",
                    reason=f"Submit Task Plan operation {operation}",
                ),
            )

    async def handler(input, context):
        try:
            snapshot = _snapshot_from_input(operation, input)
            state = service.submit(operation, snapshot, _request_from_input_context(input))
        except PlanValidationError as exc:
            raise ToolHandlerError("plan.invalid", str(exc)) from exc
        context.effects.report(
            ToolEffect(
                kind="session_state_write",
                resource=ToolResource(f"session://{_request_from_input_context(input).session_id}/task-plan"),
                operation=operation,
                status="completed",
                certainty="observed",
            )
        )
        return {
            "plan_operation": operation,
            "plan_snapshot": _snapshot_to_dict(snapshot),
            "plan_state": dict(state),
        }

    class BoundResolver:
        def resolve(self, input, request):
            resolved = Resolver().resolve(input, request)
            bound = dict(resolved.input)
            bound["__run_id__"] = request.run_id
            bound["__session_id__"] = request.session_id
            bound["__tool_call_id__"] = request.tool_call_id
            bound["__tool_name__"] = request.tool_name
            bound["__mode__"] = request.mode
            bound["__registration_id__"] = request.registration_id
            return ToolAccessResolution(input=bound, access=resolved.access)

    class Renderer:
        def render(self, data):
            return (TextContent(text=f"Task Plan operation {data['plan_operation']} submitted."),)

    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(name, description, input_schema, output_schema),
        category="plan",
        source="builtin",
        owner="core.plan",
        policy=ToolPolicy(
            allowed_modes=frozenset({mode}),
            declared_effects=frozenset({"session_state_write"}),
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(5_000, 15_000),
            concurrency=ConcurrencyPolicy(mode="serial", group="task_plan"),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=input_codec,
        output_codec=output_codec,
        handler=handler,
        renderer=Renderer(),
        access_resolver=BoundResolver(),
    )


def _request_from_input_context(input: Mapping[str, object]) -> ToolExecutionRequest:
    internal = {
        "__run_id__",
        "__session_id__",
        "__tool_call_id__",
        "__tool_name__",
        "__mode__",
        "__registration_id__",
    }
    try:
        return ToolExecutionRequest(
            run_id=str(input["__run_id__"]),
            session_id=str(input["__session_id__"]),
            tool_call_id=str(input["__tool_call_id__"]),
            tool_name=str(input["__tool_name__"]),
            arguments={key: value for key, value in input.items() if key not in internal},
            mode=str(input["__mode__"]),
            registration_id=str(input["__registration_id__"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanValidationError("Plan request context is missing") from exc


def _snapshot_from_input(operation: PlanOperation, input: Mapping[str, object]) -> PlanSnapshot:
    raw = {key: value for key, value in input.items() if not key.startswith("__")}
    items = raw.get("items")
    if operation == PROPOSE_PLAN_TOOL and isinstance(items, list):
        raw["items"] = [{**item, "status": "pending"} for item in items]
    return PlanSnapshot.from_mapping(raw)


def _snapshot_to_dict(snapshot: PlanSnapshot) -> dict[str, object]:
    result: dict[str, object] = {
        "summary": snapshot.summary,
        "completion_criteria": list(snapshot.completion_criteria),
        "items": [
            {
                **({"id": item.id} if item.id is not None else {}),
                "step": item.step,
                "details": item.details,
                "verification": item.verification,
                "status": item.status,
            }
            for item in snapshot.items
        ],
        "explanation": snapshot.explanation,
    }
    for name in (
        "raw_user_request",
        "interpreted_goal",
        "task_understanding",
        "current_implementation",
        "target_design",
        "impact_scope",
        "verification_plan",
        "status",
        "change_reason",
    ):
        value = getattr(snapshot, name)
        if value is not None:
            result[name] = value
    if snapshot.risks_and_open_questions:
        result["risks_and_open_questions"] = list(snapshot.risks_and_open_questions)
    return result


def _snapshot_schema(operation: PlanOperation) -> dict[str, object]:
    proposal = operation == PROPOSE_PLAN_TOOL
    item_properties: dict[str, object] = {
        "step": {"type": "string", "minLength": 1},
        "details": {"type": "string", "minLength": 1},
        "verification": {"type": "string", "minLength": 1},
    }
    item_required = ["step", "details", "verification"]
    if not proposal:
        item_properties = {
            "id": {"type": "string", "minLength": 1},
            **item_properties,
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
        }
        item_required.append("status")
    properties: dict[str, object] = {
        "raw_user_request": {"type": "string", "minLength": 1},
        "interpreted_goal": {"type": "string", "minLength": 1},
        "task_understanding": {"type": "string", "minLength": 1},
        "current_implementation": {"type": "string", "minLength": 1},
        "target_design": {"type": "string", "minLength": 1},
        "impact_scope": {"type": "string", "minLength": 1},
        "risks_and_open_questions": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1},
        },
        "verification_plan": {"type": "string", "minLength": 1},
        "summary": {"type": "string", "minLength": 1},
        "completion_criteria": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 1},
        },
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": PLAN_ITEM_LIMIT,
            "items": {
                "type": "object",
                "properties": item_properties,
                "required": item_required,
                "additionalProperties": False,
            },
        },
        "change_reason": {
            "type": "string",
            "enum": ["user_request", "repeated_execution_failure"],
        },
        "explanation": {"type": "string"},
    }
    required = ["summary", "completion_criteria", "items"]
    if operation in {PROPOSE_PLAN_TOOL, CREATE_BUILD_PLAN_TOOL}:
        required = ["raw_user_request", "interpreted_goal", *required]
    if proposal:
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
    if operation == CLOSE_PLAN_TOOL:
        properties["status"] = {"type": "string", "enum": ["active", "completed"]}
        required.append("status")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _plan_description(operation: str) -> str:
    return {
        PROPOSE_PLAN_TOOL: (
            "Plan mode only. Submit the complete canonical implementation plan after repository "
            "evidence is sufficient. Success means the proposal was stored; user plan approval is "
            "a separate Core workflow and is not a tool security approval."
        ),
        CREATE_BUILD_PLAN_TOOL: (
            "Build mode only. Create a lightweight active Task Plan for a complex task when no "
            "current plan exists."
        ),
        UPDATE_PLAN_PROGRESS_TOOL: (
            "Build mode only. Atomically update progress or a controlled revision of the active "
            "Task Plan; use close_plan for completion."
        ),
        CLOSE_PLAN_TOOL: (
            "Build mode only. Atomically close the active Task Plan as completed, or keep it active "
            "with explicit remaining work."
        ),
    }[operation]


__all__ = ["PlanService", "StoreBackedPlanService", "create_plan_registrations"]
