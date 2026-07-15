"""将计划控制命令适配为受 Core 状态约束的内部工具。"""

from __future__ import annotations

"""Plan tools translate model input into Core commands only."""

from collections.abc import Callable, Mapping

from codepilot.protocols import (
    CLOSE_PLAN_TOOL,
    CREATE_BUILD_PLAN_TOOL,
    PLAN_ITEM_LIMIT,
    PROPOSE_PLAN_TOOL,
    UPDATE_PLAN_PROGRESS_TOOL,
)
from codepilot.tools.codecs import JsonObjectCodec
from codepilot.tools.contracts import ToolHandlerError, ToolRegistration, ToolSpec
from codepilot.tools.results import TextContent
from codepilot.tools.security import (
    ConcurrencyPolicy,
    OutputLimits,
    OutputTrustPolicy,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolPolicy,
    ToolResource,
)

from ..commands import (
    ProposePlanRevision,
    RequestPlanClose,
    SubmitPlan,
    UpdatePlanProgress,
    core_command_to_dict,
)
from ..plan import (
    PlanDefinition,
    PlanStepDefinition,
    PlanStepUpdate,
    ensure_plan_operation,
    ensure_plan_revision_reason,
)


def create_plan_registrations(
    *,
    allow: Callable[[str], bool] | None = None,
) -> list[ToolRegistration]:
    """创建提交、更新和关闭计划所需的内部工具注册。"""
    allowed = allow or (lambda _name: True)
    operations = (
        (PROPOSE_PLAN_TOOL, "plan"),
        (CREATE_BUILD_PLAN_TOOL, "execute"),
        (UPDATE_PLAN_PROGRESS_TOOL, "execute"),
        (CLOSE_PLAN_TOOL, "execute"),
    )
    return [
        _plan_registration(name, mode=mode)
        for name, mode in operations
        if allowed(name)
    ]


def _plan_registration(name: str, *, mode: str) -> ToolRegistration:
    operation = ensure_plan_operation(name)
    input_schema = _input_schema(operation)
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "plan_operation": {"type": "string", "const": operation},
            "core_command": {"type": "object"},
        },
        "required": ["plan_operation", "core_command"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input, request):
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(operation,),
                    resources=(
                        ToolResource(f"session://{request.session_id}/task-plan"),
                    ),
                    effects=frozenset(),
                    risk="low",
                    reason=f"Submit Core Plan command {operation}",
                ),
            )

    async def handler(input, context):
        try:
            command = _command_from_input(
                operation,
                input,
                command_id=context.request.tool_call_id,
            )
        except (TypeError, ValueError) as exc:
            raise ToolHandlerError("plan.invalid", str(exc)) from exc
        return {
            "plan_operation": operation,
            "core_command": core_command_to_dict(command),
        }

    class Renderer:
        def render(self, data):
            return (
                TextContent(
                    text=f"Task Plan command {data['plan_operation']} submitted."
                ),
            )

    return ToolRegistration(
        version="1.0.0",
        implementation_version="2",
        spec=ToolSpec(
            operation,
            _plan_description(operation),
            input_schema,
            output_schema,
        ),
        category="plan",
        source="builtin",
        owner="core.plan",
        policy=ToolPolicy(
            allowed_modes=frozenset({mode}),
            declared_effects=frozenset(),
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(5_000, 15_000),
            concurrency=ConcurrencyPolicy(mode="serial", group="task_plan"),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=JsonObjectCodec(input_schema),
        output_codec=JsonObjectCodec(output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


def _command_from_input(
    operation: str,
    raw: Mapping[str, object],
    *,
    command_id: str,
):
    if operation in {PROPOSE_PLAN_TOOL, CREATE_BUILD_PLAN_TOOL}:
        return SubmitPlan(
            command_id=command_id,
            definition=_definition_from_input(raw),
            steps=_step_definitions(raw.get("items")),
        )
    if operation == UPDATE_PLAN_PROGRESS_TOOL:
        expected_revision = _non_negative_int(
            raw.get("expected_revision"), "expected_revision"
        )
        revision = raw.get("revision")
        if isinstance(revision, Mapping):
            return ProposePlanRevision(
                command_id=command_id,
                expected_revision=expected_revision,
                reason=ensure_plan_revision_reason(revision.get("reason")),
                definition=_definition_from_input(revision),
                steps=_step_definitions(revision.get("items")),
            )
        return UpdatePlanProgress(
            command_id=command_id,
            expected_revision=expected_revision,
            updates=tuple(
                PlanStepUpdate.from_mapping(item)
                for item in _mapping_list(raw.get("updates"), "updates")
            ),
        )
    if operation == CLOSE_PLAN_TOOL:
        return RequestPlanClose(
            command_id=command_id,
            expected_revision=_non_negative_int(
                raw.get("expected_revision"), "expected_revision"
            ),
            summary=_required_text(raw.get("summary"), "summary"),
            evidence_refs=_string_list(raw.get("evidence_refs"), "evidence_refs"),
        )
    raise ValueError(f"Unknown Plan operation: {operation}")


def _definition_from_input(raw: Mapping[str, object]) -> PlanDefinition:
    return PlanDefinition(
        summary=_required_text(raw.get("summary"), "summary"),
        completion_criteria=_string_list(
            raw.get("completion_criteria"), "completion_criteria"
        ),
        task_understanding=_optional_text(raw.get("task_understanding")) or "",
        current_implementation=_optional_text(raw.get("current_implementation")) or "",
        target_design=_optional_text(raw.get("target_design")) or "",
        impact_scope=_optional_text(raw.get("impact_scope")) or "",
        risks_and_open_questions=_string_list(
            raw.get("risks_and_open_questions", ()),
            "risks_and_open_questions",
        ),
        verification_plan=_optional_text(raw.get("verification_plan")) or "",
        explanation=_optional_text(raw.get("explanation")) or "",
    )


def _step_definitions(value: object) -> tuple[PlanStepDefinition, ...]:
    return tuple(
        PlanStepDefinition.from_mapping(item)
        for item in _mapping_list(value, "items")
    )


def _input_schema(operation: str) -> dict[str, object]:
    if operation in {PROPOSE_PLAN_TOOL, CREATE_BUILD_PLAN_TOOL}:
        return _submission_schema(detailed=operation == PROPOSE_PLAN_TOOL)
    if operation == UPDATE_PLAN_PROGRESS_TOOL:
        return _progress_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "expected_revision": {"type": "integer", "minimum": 0},
            "summary": {"type": "string", "minLength": 1},
            "evidence_refs": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": ["expected_revision", "summary", "evidence_refs"],
        "additionalProperties": False,
    }


def _submission_schema(*, detailed: bool) -> dict[str, object]:
    properties = _definition_properties()
    properties["items"] = _step_definitions_schema()
    required = ["summary", "completion_criteria", "items"]
    if detailed:
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
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _progress_schema() -> dict[str, object]:
    revision_properties = _definition_properties()
    revision_properties.update(
        {
            "reason": {
                "type": "string",
                "enum": [
                    "user_request",
                    "repeated_execution_failure",
                    "new_evidence",
                ],
            },
            "items": _step_definitions_schema(),
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "expected_revision": {"type": "integer", "minimum": 0},
            "updates": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string", "minLength": 1},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                        "completion_note": {"type": "string"},
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                    "required": ["step_id", "status"],
                    "additionalProperties": False,
                },
            },
            "revision": {
                "type": "object",
                "properties": revision_properties,
                "required": [
                    "reason",
                    "summary",
                    "completion_criteria",
                    "items",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["expected_revision"],
        "oneOf": [{"required": ["updates"]}, {"required": ["revision"]}],
        "additionalProperties": False,
    }


def _definition_properties() -> dict[str, object]:
    return {
        "summary": {"type": "string", "minLength": 1},
        "completion_criteria": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 1},
        },
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
        "explanation": {"type": "string"},
    }


def _step_definitions_schema() -> dict[str, object]:
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": PLAN_ITEM_LIMIT,
        "items": {
            "type": "object",
            "properties": {
                "step": {"type": "string", "minLength": 1},
                "details": {"type": "string", "minLength": 1},
                "verification": {"type": "string", "minLength": 1},
            },
            "required": ["step", "details", "verification"],
            "additionalProperties": False,
        },
    }


def _plan_description(operation: str) -> str:
    return {
        PROPOSE_PLAN_TOOL: (
            "Plan mode only. Submit the session's complete canonical proposed Task Plan for user review once repository evidence is sufficient. On user feedback, submit the full revised plan and preserve decisions the feedback did not change; plain text is not a submission. Each item must describe post-approval implementation or verification work, never exploration, writing the plan, replying, or waiting for approval. A successful call means stop planning and wait for Runtime review; it does not authorize implementation."
        ),
        CREATE_BUILD_PLAN_TOOL: (
            "Build mode only. Create one lightweight active Task Plan when the implementation is genuinely multi-step and no canonical plan exists. Items must be concrete implementation or verification work with independently checkable outcomes. Do not call this for a simple task or while an active plan already exists."
        ),
        UPDATE_PLAN_PROGRESS_TOOL: (
            "Build mode only. Update the existing active plan using its current expected_revision. For normal progress, submit only changed step IDs with status, completion note, and evidence; never resend the whole plan. Use revision only when user direction, repeated failure, or new repository evidence invalidates the approved structure, and provide the complete revised definition and items."
        ),
        CLOSE_PLAN_TOOL: (
            "Build mode only. Request closeout of the current active plan using its expected_revision, an accurate summary, and concrete evidence references. Call only after checking every completion criterion and the latest verification state. This submits evidence to Core; Core decides whether the plan completes or remains active."
        ),
    }[operation]


def _mapping_list(
    value: object,
    field_name: str,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f"{field_name} must contain objects")
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(_required_text(item, f"{field_name} item") for item in value)


def _non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = ["create_plan_registrations"]
