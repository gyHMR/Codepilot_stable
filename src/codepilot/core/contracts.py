from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Awaitable, Callable, Literal, Protocol, TypeAlias

from codepilot.llm.ports import ModelDescriptor, ModelPort
from codepilot.protocols import (
    AssistantMessage,
    Message,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from codepilot.tools.contracts import ToolExecutionPort
from codepilot.tools.registry import ToolCatalogSnapshot
from codepilot.tools.results import ToolResult
from .errors import CoreContractError
from .events import CoreDomainEvent
from .plan import RunMode, ensure_run_mode
from .state import CoreState, load_core_state


CoreBoundaryKind = Literal[
    "before_model",
    "after_model",
    "before_tools",
    "after_tools",
    "waiting",
    "before_terminal",
]
CoreWaitKind = Literal[
    "tool_approval",
    "user_input",
    "plan_confirmation",
    "continuation",
]
CoreOutcomeStatus = Literal["completed", "waiting", "failed", "cancelled"]
ModelPurpose = Literal[
    "reasoning",
    "recovery",
    "verification",
    "replan",
    "plan_closeout",
    "final_response",
]
TerminationStatus = Literal["completed", "failed", "cancelled"]
ContextPurpose = Literal["reasoning", "verification", "finalization"]


@dataclass(frozen=True)
class TaskStepView:
    step_id: str
    step: str
    details: str
    verification: str
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_core_text(self.step_id, "step_id"))
        object.__setattr__(self, "step", _required_core_text(self.step, "step"))
        object.__setattr__(self, "details", _required_core_text(self.details, "details"))
        object.__setattr__(
            self,
            "verification",
            _required_core_text(self.verification, "verification"),
        )
        if self.status not in {"pending", "in_progress", "completed"}:
            raise CoreContractError(f"Unknown task step status: {self.status}")


@dataclass(frozen=True)
class TaskPlanView:
    plan_id: str
    origin: str
    status: str
    revision: int
    definition: Mapping[str, object]
    steps: tuple[TaskStepView, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _required_core_text(self.plan_id, "plan_id"))
        object.__setattr__(self, "origin", _required_core_text(self.origin, "plan origin"))
        object.__setattr__(self, "status", _required_core_text(self.status, "plan status"))
        _require_non_negative(self.revision, "plan revision")
        if not isinstance(self.definition, Mapping):
            raise CoreContractError("plan definition must be a mapping")
        object.__setattr__(
            self,
            "definition",
            MappingProxyType(deepcopy(dict(self.definition))),
        )
        steps = tuple(self.steps)
        if any(not isinstance(step, TaskStepView) for step in steps):
            raise CoreContractError("plan steps must contain TaskStepView values")
        object.__setattr__(self, "steps", steps)

    def to_mapping(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "origin": self.origin,
            "status": self.status,
            "revision": self.revision,
            "definition": deepcopy(dict(self.definition)),
            "steps": [
                {
                    "step_id": step.step_id,
                    "step": step.step,
                    "details": step.details,
                    "verification": step.verification,
                    "status": step.status,
                }
                for step in self.steps
            ],
        }


@dataclass(frozen=True)
class VerificationView:
    status: str
    verified_revision: int | None = None
    attempted_checks: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "none",
            "unknown",
            "passed",
            "failed",
            "stale",
            "unavailable",
        }:
            raise CoreContractError(f"Unknown verification status: {self.status}")
        if self.verified_revision is not None:
            _require_non_negative(self.verified_revision, "verified_revision")
        object.__setattr__(
            self,
            "attempted_checks",
            tuple(
                _required_core_text(value, "attempted check")
                for value in self.attempted_checks
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(
                _required_core_text(value, "evidence ref")
                for value in self.evidence_refs
            ),
        )
        object.__setattr__(
            self,
            "unavailable_reason",
            _optional_core_text(self.unavailable_reason),
        )


@dataclass(frozen=True)
class CoreContextView:
    original_request: str
    goal: str
    mode: RunMode
    task_status: str
    plan: TaskPlanView | None
    current_step: TaskStepView | None
    verification: VerificationView
    blocked_reason: str | None = None
    workspace_revision: int = 0
    workspace_changed: bool = False
    affected_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_request",
            _required_core_text(self.original_request, "original_request"),
        )
        object.__setattr__(self, "goal", _required_core_text(self.goal, "goal"))
        object.__setattr__(self, "mode", ensure_run_mode(self.mode))
        object.__setattr__(
            self,
            "task_status",
            _required_core_text(self.task_status, "task_status"),
        )
        if self.plan is not None and not isinstance(self.plan, TaskPlanView):
            raise CoreContractError("plan must be TaskPlanView or None")
        if self.current_step is not None and not isinstance(
            self.current_step, TaskStepView
        ):
            raise CoreContractError("current_step must be TaskStepView or None")
        if not isinstance(self.verification, VerificationView):
            raise CoreContractError("verification must be VerificationView")
        object.__setattr__(
            self,
            "blocked_reason",
            _optional_core_text(self.blocked_reason),
        )
        _require_non_negative(self.workspace_revision, "workspace_revision")
        if not isinstance(self.workspace_changed, bool):
            raise CoreContractError("workspace_changed must be bool")
        object.__setattr__(
            self,
            "affected_paths",
            tuple(
                sorted(
                    {
                        _required_core_text(path, "affected path")
                        for path in self.affected_paths
                    }
                )
            ),
        )

    @classmethod
    def from_state(cls, state: CoreState, mode: RunMode) -> "CoreContextView":
        plan = state.task.plan
        plan_view = None
        current_step = None
        if plan is not None:
            steps = tuple(
                TaskStepView(
                    step_id=step.step_id,
                    step=step.step,
                    details=step.details,
                    verification=step.verification,
                    status=step.status,
                )
                for step in plan.steps
            )
            plan_view = TaskPlanView(
                plan_id=plan.plan_id,
                origin=plan.origin,
                status=plan.status,
                revision=plan.revision,
                definition=plan.definition.to_dict(),
                steps=steps,
            )
            current_step = next(
                (step for step in steps if step.status == "in_progress"),
                next((step for step in steps if step.status == "pending"), None),
            )
        verification = state.facts.verification
        blocked_reason = "; ".join(
            blocker.reason for blocker in state.task.blockers
        ) or None
        if blocked_reason is None and state.facts.failures.latest is not None:
            blocked_reason = state.facts.failures.latest.message
        workspace = state.facts.workspace
        return cls(
            original_request=state.task.original_request,
            goal=state.task.current_goal,
            mode=mode,
            task_status=state.task.status,
            plan=plan_view,
            current_step=current_step,
            verification=VerificationView(
                status=verification.status,
                verified_revision=verification.verified_revision,
                attempted_checks=verification.attempted_checks,
                evidence_refs=verification.evidence_refs,
                unavailable_reason=verification.unavailable_reason,
            ),
            blocked_reason=blocked_reason,
            workspace_revision=workspace.revision,
            workspace_changed=workspace.changed,
            affected_paths=workspace.affected_paths,
        )


@dataclass(frozen=True)
class ContextPrepareRequest:
    session_id: str
    run_id: str
    purpose: ContextPurpose
    directive: str | None
    messages: tuple[Message, ...]
    core_view: CoreContextView
    model: ModelDescriptor
    tool_catalog: ToolCatalogSnapshot | None
    seed: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _required_core_text(self.session_id, "session_id"),
        )
        object.__setattr__(self, "run_id", _required_core_text(self.run_id, "run_id"))
        if self.purpose not in {"reasoning", "verification", "finalization"}:
            raise CoreContractError(f"Unknown context purpose: {self.purpose}")
        object.__setattr__(self, "directive", _optional_core_text(self.directive))
        object.__setattr__(self, "messages", _core_messages(self.messages))
        if not isinstance(self.core_view, CoreContextView):
            raise CoreContractError("core_view must be CoreContextView")
        if not isinstance(self.model, ModelDescriptor):
            raise CoreContractError("context model must be ModelDescriptor")
        if self.tool_catalog is not None and not isinstance(
            self.tool_catalog, ToolCatalogSnapshot
        ):
            raise CoreContractError("tool_catalog must be ToolCatalogSnapshot or None")
        if not isinstance(self.seed, Mapping):
            raise CoreContractError("context seed must be a mapping")
        object.__setattr__(
            self,
            "seed",
            MappingProxyType(deepcopy(dict(self.seed))),
        )


@dataclass(frozen=True)
class PreparedModelContext:
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[Tool, ...]
    projection_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "system_prompt",
            _clean_core_text(self.system_prompt),
        )
        object.__setattr__(self, "messages", _core_messages(self.messages))
        tools = tuple(self.tools)
        if any(not isinstance(tool, Tool) for tool in tools):
            raise CoreContractError("prepared tools must contain Tool values")
        object.__setattr__(self, "tools", tools)
        object.__setattr__(
            self,
            "projection_ref",
            _required_core_text(self.projection_ref, "projection_ref"),
        )


class ContextPreparationPort(Protocol):
    def prepare(
        self,
        request: ContextPrepareRequest,
    ) -> PreparedModelContext | Awaitable[PreparedModelContext]: ...


@dataclass(frozen=True)
class ModelEntry:
    """Start or continue by asking the model to act on the message history."""

    message: AssistantMessage | None = None
    kind: Literal["model"] = field(default="model", init=False)

    def __post_init__(self) -> None:
        if self.message is not None and not isinstance(self.message, AssistantMessage):
            raise CoreContractError("ModelEntry message must be AssistantMessage")


@dataclass(frozen=True)
class ToolResultEntry:
    """Resume Core with final canonical results produced by Runtime/Tools."""

    results: tuple[ToolResult, ...] = ()
    kind: Literal["tool_results"] = field(default="tool_results", init=False)

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not results:
            raise CoreContractError("ToolResultEntry requires at least one result")
        if any(not isinstance(result, ToolResult) for result in results):
            raise CoreContractError(
                "ToolResultEntry requires canonical ToolResult values"
            )
        if any(
            result.status in {"approval_required", "user_input_required"}
            for result in results
        ):
            raise CoreContractError(
                "ToolResultEntry accepts only final ToolResult values"
            )
        object.__setattr__(self, "results", results)


CoreEntry: TypeAlias = ModelEntry | ToolResultEntry


@dataclass(frozen=True)
class CoreLimits:
    max_model_turns: int = 100
    max_tool_iterations: int = 240
    max_tool_calls_per_turn: int | None = 16
    max_tool_calls: int | None = None
    max_recovery_attempts: int = 5
    repeated_tool_call_limit: int = 3

    def __post_init__(self) -> None:
        for name in (
            "max_model_turns",
            "max_tool_iterations",
            "max_recovery_attempts",
            "repeated_tool_call_limit",
        ):
            _require_non_negative(getattr(self, name), name)
        for name in ("max_tool_calls_per_turn", "max_tool_calls"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative(value, name)


@dataclass(frozen=True)
class CoreReason:
    code: str
    message: str = ""
    source: str = "core"
    recoverable: bool = False
    evidence_refs: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required_core_text(self.code, "reason code"))
        object.__setattr__(self, "message", _clean_core_text(self.message).strip())
        object.__setattr__(
            self, "source", _required_core_text(self.source, "reason source")
        )
        if not isinstance(self.recoverable, bool):
            raise TypeError("reason recoverable must be bool")
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(
                dict.fromkeys(
                    _required_core_text(value, "evidence ref")
                    for value in self.evidence_refs
                )
            ),
        )
        if not isinstance(self.details, Mapping):
            raise TypeError("reason details must be a mapping")
        object.__setattr__(
            self,
            "details",
            MappingProxyType(deepcopy(dict(self.details))),
        )


@dataclass(frozen=True)
class CoreDirective:
    code: str
    constraints: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "code", _required_core_text(self.code, "directive code")
        )
        object.__setattr__(
            self,
            "constraints",
            tuple(
                _required_core_text(value, "constraint") for value in self.constraints
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(
                dict.fromkeys(
                    _required_core_text(value, "evidence ref")
                    for value in self.evidence_refs
                )
            ),
        )


@dataclass(frozen=True)
class CoreWait:
    kind: CoreWaitKind
    request_id: str
    reason: CoreReason
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {
            "tool_approval",
            "user_input",
            "plan_confirmation",
            "continuation",
        }:
            raise ValueError(f"Unknown wait kind: {self.kind}")
        object.__setattr__(
            self, "request_id", _required_core_text(self.request_id, "request_id")
        )
        if not isinstance(self.reason, CoreReason):
            raise TypeError("wait reason must be CoreReason")
        if not isinstance(self.payload, Mapping):
            raise TypeError("wait payload must be a mapping")
        object.__setattr__(
            self,
            "payload",
            MappingProxyType(deepcopy(dict(self.payload))),
        )


@dataclass(frozen=True)
class CallModel:
    purpose: ModelPurpose
    directive: CoreDirective
    reason: CoreReason

    def __post_init__(self) -> None:
        if self.purpose not in {
            "reasoning",
            "recovery",
            "verification",
            "replan",
            "plan_closeout",
            "final_response",
        }:
            raise ValueError(f"Unknown model purpose: {self.purpose}")
        if not isinstance(self.directive, CoreDirective):
            raise TypeError("model directive must be CoreDirective")
        if not isinstance(self.reason, CoreReason):
            raise TypeError("model reason must be CoreReason")


@dataclass(frozen=True)
class ExecuteTools:
    calls: tuple[ToolCall, ...]
    reason: CoreReason

    def __post_init__(self) -> None:
        calls = tuple(self.calls)
        if not calls or any(not isinstance(call, ToolCall) for call in calls):
            raise ValueError("ExecuteTools requires ToolCall values")
        object.__setattr__(self, "calls", calls)
        if not isinstance(self.reason, CoreReason):
            raise TypeError("tool decision reason must be CoreReason")


@dataclass(frozen=True)
class Wait:
    wait: CoreWait

    def __post_init__(self) -> None:
        if not isinstance(self.wait, CoreWait):
            raise TypeError("wait decision requires CoreWait")


@dataclass(frozen=True)
class Terminate:
    status: TerminationStatus
    reason_code: str
    message: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"Unknown termination status: {self.status}")
        object.__setattr__(
            self, "reason_code", _required_core_text(self.reason_code, "reason_code")
        )
        object.__setattr__(self, "message", _clean_core_text(self.message).strip())
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(
                dict.fromkeys(
                    _required_core_text(value, "evidence ref")
                    for value in self.evidence_refs
                )
            ),
        )

    @property
    def reason(self) -> CoreReason:
        return CoreReason(
            code=self.reason_code,
            message=self.message,
            evidence_refs=self.evidence_refs,
        )


CoreDecision: TypeAlias = CallModel | ExecuteTools | Wait | Terminate


@dataclass(frozen=True)
class CoreRunInput:
    session_id: str
    run_id: str
    entry: CoreEntry
    messages: tuple[Message, ...]
    state: CoreState | Mapping[str, object]
    mode: RunMode
    model: ModelDescriptor
    limits: CoreLimits = field(default_factory=CoreLimits)
    context_seed: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _required_core_text(self.session_id, "session_id"),
        )
        object.__setattr__(self, "run_id", _required_core_text(self.run_id, "run_id"))
        if not isinstance(self.entry, (ModelEntry, ToolResultEntry)):
            raise CoreContractError("CoreRunInput entry must be a typed Core entry")
        messages = _core_messages(self.messages)
        object.__setattr__(self, "messages", messages)
        request = _first_user_text(messages)
        object.__setattr__(
            self,
            "state",
            load_core_state(self.state, original_request=request),
        )
        object.__setattr__(self, "mode", ensure_run_mode(self.mode))
        if not isinstance(self.model, ModelDescriptor):
            raise CoreContractError("CoreRunInput model must be ModelDescriptor")
        if not isinstance(self.limits, CoreLimits):
            raise CoreContractError("CoreRunInput limits must be CoreLimits")
        if not isinstance(self.context_seed, Mapping):
            raise CoreContractError("CoreRunInput context_seed must be a mapping")
        object.__setattr__(
            self,
            "context_seed",
            MappingProxyType(deepcopy(dict(self.context_seed))),
        )


class BoundaryPort(Protocol):
    def commit(self, boundary: "CoreBoundary") -> None | Awaitable[None]: ...


class CancellationProbe(Protocol):
    def raise_if_cancelled(self) -> None: ...


LiveEventSink = Callable[[Mapping[str, object]], None | Awaitable[None]]


@dataclass(frozen=True)
class CorePorts:
    model: ModelPort
    context: ContextPreparationPort
    boundary: BoundaryPort
    tools: ToolExecutionPort | None = None
    live_events: LiveEventSink | None = None
    cancellation: CancellationProbe | None = None

    def __post_init__(self) -> None:
        if self.model is None or not callable(getattr(self.model, "stream", None)):
            raise CoreContractError("CorePorts requires a model port")
        if self.context is None or not callable(getattr(self.context, "prepare", None)):
            raise CoreContractError("CorePorts requires a context port")
        if self.boundary is None or not callable(
            getattr(self.boundary, "commit", None)
        ):
            raise CoreContractError("CorePorts requires a boundary port")
        if self.tools is not None and any(
            not callable(getattr(self.tools, method, None))
            for method in ("catalog_snapshot", "prepare_batch", "execute_prepared")
        ):
            raise CoreContractError(
                "CorePorts tools must support catalog, preparation, and execution"
            )
        if self.live_events is not None and not callable(self.live_events):
            raise CoreContractError("CorePorts live_events must be callable")
        if self.cancellation is not None and not callable(
            getattr(self.cancellation, "raise_if_cancelled", None)
        ):
            raise CoreContractError(
                "CorePorts cancellation must implement CancellationProbe"
            )


@dataclass(frozen=True)
class CoreBoundary:
    kind: CoreBoundaryKind
    state: CoreState
    new_messages: tuple[Message, ...] = ()
    domain_events: tuple[CoreDomainEvent, ...] = ()
    wait: CoreWait | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "before_model",
            "after_model",
            "before_tools",
            "after_tools",
            "waiting",
            "before_terminal",
        }:
            raise CoreContractError(f"Unknown Core boundary kind: {self.kind}")
        if not isinstance(self.state, CoreState):
            raise CoreContractError("CoreBoundary state must be CoreState")
        object.__setattr__(self, "new_messages", _core_messages(self.new_messages))
        events = tuple(self.domain_events)
        if any(not isinstance(event, CoreDomainEvent) for event in events):
            raise CoreContractError(
                "CoreBoundary domain_events must contain CoreDomainEvent values"
            )
        object.__setattr__(self, "domain_events", events)
        if self.kind == "waiting" and self.wait is None:
            raise CoreContractError("Waiting boundary requires CoreWait")
        if self.kind != "waiting" and self.wait is not None:
            raise CoreContractError("CoreWait is only valid on a waiting boundary")


@dataclass(frozen=True)
class CoreOutcome:
    status: CoreOutcomeStatus
    reason: CoreReason
    state: CoreState
    new_messages: tuple[Message, ...] = ()
    final_message: AssistantMessage | None = None
    wait: CoreWait | None = None
    usage: Usage | None = None
    error: object | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "waiting", "failed", "cancelled"}:
            raise CoreContractError(f"Unknown Core outcome status: {self.status}")
        if not isinstance(self.reason, CoreReason):
            raise CoreContractError("CoreOutcome reason must be CoreReason")
        if not isinstance(self.state, CoreState):
            raise CoreContractError("CoreOutcome state must be CoreState")
        object.__setattr__(self, "new_messages", _core_messages(self.new_messages))
        if self.final_message is not None and not isinstance(
            self.final_message, AssistantMessage
        ):
            raise CoreContractError(
                "CoreOutcome final_message must be AssistantMessage"
            )
        if self.status == "waiting" and self.wait is None:
            raise CoreContractError("A waiting outcome requires CoreWait")
        if self.status != "waiting" and self.wait is not None:
            raise CoreContractError("CoreWait is only valid on a waiting outcome")
        if self.status == "completed" and self.final_message is None:
            raise CoreContractError("A completed outcome requires final_message")
        if self.status in {"completed", "waiting"} and self.error is not None:
            raise CoreContractError(
                "Only failed or cancelled outcomes may include an error"
            )

    @property
    def final_text(self) -> str:
        if self.final_message is None:
            return ""
        return "".join(
            block.text
            for block in self.final_message.content
            if isinstance(block, TextContent)
        )


def _core_messages(value: object) -> tuple[Message, ...]:
    if not isinstance(value, (tuple, list)):
        raise CoreContractError("Core messages must be a sequence")
    messages = tuple(value)
    if any(
        not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage))
        for message in messages
    ):
        raise CoreContractError("Core messages must contain canonical Message values")
    return messages


def _first_user_text(messages: tuple[Message, ...]) -> str | None:
    for message in messages:
        if not isinstance(message, UserMessage):
            continue
        if isinstance(message.content, str):
            text = message.content.strip()
        else:
            text = "".join(
                block.text
                for block in message.content
                if isinstance(block, TextContent)
            ).strip()
        if text:
            return text
    return None


def _require_non_negative(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CoreContractError(f"{field_name} must be a non-negative integer")


def _clean_core_text(value: object) -> str:
    return str(value) if value is not None else ""


def _optional_core_text(value: object) -> str | None:
    text = _clean_core_text(value).strip()
    return text or None


def _required_core_text(value: object, field_name: str) -> str:
    text = _clean_core_text(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


__all__ = [
    "BoundaryPort",
    "CallModel",
    "CancellationProbe",
    "ContextPreparationPort",
    "ContextPrepareRequest",
    "ContextPurpose",
    "CoreContextView",
    "CoreBoundaryKind",
    "CoreBoundary",
    "CoreDecision",
    "CoreDirective",
    "CoreEntry",
    "CoreLimits",
    "CoreOutcome",
    "CoreOutcomeStatus",
    "CorePorts",
    "CoreReason",
    "CoreRunInput",
    "CoreWait",
    "CoreWaitKind",
    "ExecuteTools",
    "LiveEventSink",
    "ModelEntry",
    "ModelPurpose",
    "PreparedModelContext",
    "TaskPlanView",
    "TaskStepView",
    "Terminate",
    "TerminationStatus",
    "ToolResultEntry",
    "VerificationView",
    "Wait",
]
