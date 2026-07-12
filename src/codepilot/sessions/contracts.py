from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional

from codepilot.core.contracts import (
    AgentMessage,
    AgentLoopInput,
    AgentLoopOutcome,
    AgentResumeInput,
    AgentLoopStatus,
    ContextPort,
    PrepareContextFn,
    RunStatePort,
)
from codepilot.core.plan import PlanningBudgetProfile, RunMode
from codepilot.llm.provider_types import ProviderSimpleStreamFn
from codepilot.protocols import AgentEvent, Message
from codepilot.protocols import Model
from codepilot.protocols.commands import LifecycleHook, RegisteredCommand


SESSION_STATE_SCHEMA_VERSION = 1
RUN_STATE_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
MESSAGE_RECORD_SCHEMA_VERSION = 1

SessionKind = Literal["primary", "subagent"]
RunStatus = Literal[
    "created",
    "running",
    "waiting",
    "completed",
    "failed",
    "cancelled",
]
RunPhase = Literal["received", "model", "tools", "finalizing", "finished"]
ResumePoint = Literal[
    "before_model",
    "after_model",
    "before_tools",
    "after_tools",
    "before_finalization",
]
WaitingKind = Literal["tool_approval", "user_input", "plan_confirmation"]
CheckpointOwner = Literal["tools", "context"]
RecoveryStatus = Literal["ready", "needs_validation", "blocked", "not_found"]
WorkspaceRecoveryStatus = Literal["unchanged", "changed", "missing", "unknown"]

_RUN_STATUSES = frozenset({"created", "running", "waiting", "completed", "failed", "cancelled"})
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
_RUN_PHASES = frozenset({"received", "model", "tools", "finalizing", "finished"})
_RESUME_POINTS = frozenset(
    {"before_model", "after_model", "before_tools", "after_tools", "before_finalization"}
)
_WAITING_KINDS = frozenset({"tool_approval", "user_input", "plan_confirmation"})
_CHECKPOINT_OWNERS = frozenset({"tools", "context"})
_RECOVERY_STATUSES = frozenset({"ready", "needs_validation", "blocked", "not_found"})
_WORKSPACE_RECOVERY_STATUSES = frozenset({"unchanged", "changed", "missing", "unknown"})
_PHASE_RESUME_POINTS = {
    "received": frozenset({"before_model"}),
    "model": frozenset({"before_model", "after_model"}),
    "tools": frozenset({"before_tools", "after_tools"}),
    "finalizing": frozenset({"before_finalization"}),
    "finished": frozenset(),
}
_RUN_TRANSITIONS = {
    "created": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"running", "waiting", "completed", "failed", "cancelled"}),
    "waiting": frozenset({"running", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class SessionStateValidationError(ValueError):
    """Raised when a Sessions v2 state object violates an invariant."""


class SessionStateConflictError(RuntimeError):
    """Raised when persisted state changed after a caller loaded it."""


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text(self.provider, "model provider"))
        object.__setattr__(self, "model", _require_text(self.model, "model id"))


@dataclass(frozen=True)
class WorkspaceEffectsSnapshot:
    changed: bool = False
    affected_paths: tuple[str, ...] = ()
    baseline_ref: str | None = None
    final_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "affected_paths", _text_tuple(self.affected_paths, "affected_paths"))
        object.__setattr__(self, "baseline_ref", _optional_text(self.baseline_ref))
        object.__setattr__(self, "final_fingerprint", _optional_text(self.final_fingerprint))


@dataclass(frozen=True)
class MessageRecord:
    message_id: str
    session_id: str
    message: Message
    run_id: str | None = None
    parent_id: str | None = None
    created_at: str = ""
    schema_version: int = MESSAGE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, MESSAGE_RECORD_SCHEMA_VERSION, "message record")
        object.__setattr__(self, "message_id", _require_text(self.message_id, "message_id"))
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "parent_id", _optional_text(self.parent_id))
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))


@dataclass(frozen=True)
class MessageCursor:
    leaf_message_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "leaf_message_id",
            _require_text(self.leaf_message_id, "leaf_message_id"),
        )


@dataclass(frozen=True)
class WaitingState:
    kind: WaitingKind
    request_id: str
    payload: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _WAITING_KINDS:
            raise SessionStateValidationError(f"Unknown waiting kind: {self.kind}")
        object.__setattr__(self, "request_id", _require_text(self.request_id, "request_id"))
        object.__setattr__(self, "payload", _serializable_mapping(self.payload, "waiting payload"))


@dataclass(frozen=True)
class ComponentCheckpoint:
    owner: CheckpointOwner
    schema_version: int
    state: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.owner not in _CHECKPOINT_OWNERS:
            raise SessionStateValidationError(f"Unknown checkpoint owner: {self.owner}")
        _require_positive_int(self.schema_version, "component schema_version")
        object.__setattr__(self, "state", _serializable_mapping(self.state, "component state"))


@dataclass(frozen=True)
class WorkspaceCheckpoint:
    root: str
    git_head: str | None = None
    dirty_paths: tuple[str, ...] = ()
    tracked_path_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _require_text(self.root, "workspace root"))
        object.__setattr__(self, "git_head", _optional_text(self.git_head))
        object.__setattr__(self, "dirty_paths", _text_tuple(self.dirty_paths, "dirty_paths"))
        hashes = {
            _require_text(path, "tracked path"): _require_text(value, "tracked path hash")
            for path, value in self.tracked_path_hashes.items()
        }
        object.__setattr__(self, "tracked_path_hashes", hashes)


@dataclass(frozen=True)
class RunCheckpoint:
    checkpoint_id: str
    resume_point: ResumePoint
    message_cursor: MessageCursor
    waiting: WaitingState | None = None
    components: tuple[ComponentCheckpoint, ...] = ()
    workspace: WorkspaceCheckpoint | None = None
    created_at: str = ""
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, CHECKPOINT_SCHEMA_VERSION, "run checkpoint")
        object.__setattr__(self, "checkpoint_id", _require_text(self.checkpoint_id, "checkpoint_id"))
        if self.resume_point not in _RESUME_POINTS:
            raise SessionStateValidationError(f"Unknown resume point: {self.resume_point}")
        object.__setattr__(self, "components", tuple(self.components))
        owners = [component.owner for component in self.components]
        if len(owners) != len(set(owners)):
            raise SessionStateValidationError("Checkpoint component owners must be unique")
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))


@dataclass(frozen=True)
class SessionState:
    session_id: str
    workspace_root: str
    model: ModelRef
    current_run_id: str | None = None
    last_run_id: str | None = None
    leaf_message_id: str | None = None
    parent_session_id: str | None = None
    parent_run_id: str | None = None
    session_kind: SessionKind = "primary"
    current_mode: str = "build"
    system_prompt_hash: str = ""
    created_at: str = ""
    updated_at: str = ""
    revision: int = 1
    schema_version: int = SESSION_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, SESSION_STATE_SCHEMA_VERSION, "session state")
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "workspace_root", _require_text(self.workspace_root, "workspace_root"))
        object.__setattr__(self, "current_run_id", _optional_text(self.current_run_id))
        object.__setattr__(self, "last_run_id", _optional_text(self.last_run_id))
        object.__setattr__(self, "leaf_message_id", _optional_text(self.leaf_message_id))
        object.__setattr__(self, "parent_session_id", _optional_text(self.parent_session_id))
        object.__setattr__(self, "parent_run_id", _optional_text(self.parent_run_id))
        if self.session_kind not in {"primary", "subagent"}:
            raise SessionStateValidationError(f"Unknown session kind: {self.session_kind}")
        if self.session_kind == "primary" and self.parent_run_id is not None:
            raise SessionStateValidationError("Primary sessions cannot have parent_run_id")
        if self.session_kind == "subagent" and self.parent_session_id is None:
            raise SessionStateValidationError("Subagent sessions require parent_session_id")
        object.__setattr__(self, "current_mode", _require_text(self.current_mode, "current_mode"))
        object.__setattr__(
            self,
            "system_prompt_hash",
            _require_text(self.system_prompt_hash, "system_prompt_hash"),
        )
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _require_text(self.updated_at, "updated_at"))
        _require_positive_int(self.revision, "session revision")


@dataclass(frozen=True)
class RunState:
    run_id: str
    session_id: str
    status: RunStatus
    phase: RunPhase
    input_message_id: str
    core_state: dict[str, object]
    workspace_effects: WorkspaceEffectsSnapshot = field(default_factory=WorkspaceEffectsSnapshot)
    stop_reason: str | None = None
    latest_message_id: str | None = None
    checkpoint: RunCheckpoint | None = None
    result_ref: str | None = None
    resume_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    revision: int = 1
    schema_version: int = RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, RUN_STATE_SCHEMA_VERSION, "run state")
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        if self.status not in _RUN_STATUSES:
            raise SessionStateValidationError(f"Unknown run status: {self.status}")
        if self.phase not in _RUN_PHASES:
            raise SessionStateValidationError(f"Unknown run phase: {self.phase}")
        object.__setattr__(
            self,
            "input_message_id",
            _require_text(self.input_message_id, "input_message_id"),
        )
        object.__setattr__(self, "core_state", _serializable_mapping(self.core_state, "core_state"))
        object.__setattr__(self, "stop_reason", _optional_text(self.stop_reason))
        object.__setattr__(self, "latest_message_id", _optional_text(self.latest_message_id))
        object.__setattr__(self, "result_ref", _optional_text(self.result_ref))
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _require_text(self.updated_at, "updated_at"))
        object.__setattr__(self, "started_at", _optional_text(self.started_at))
        object.__setattr__(self, "ended_at", _optional_text(self.ended_at))
        _require_non_negative_int(self.resume_count, "resume_count")
        _require_positive_int(self.revision, "run revision")
        validate_run_state(self)


@dataclass(frozen=True)
class RecoveryRequest:
    session_id: str
    run_id: str | None = None
    expected_checkpoint_id: str | None = None
    expected_waiting_kind: WaitingKind | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(
            self,
            "expected_checkpoint_id",
            _optional_text(self.expected_checkpoint_id),
        )
        if self.expected_waiting_kind is not None and self.expected_waiting_kind not in _WAITING_KINDS:
            raise SessionStateValidationError(
                f"Unknown expected waiting kind: {self.expected_waiting_kind}"
            )


@dataclass(frozen=True)
class RecoveryIssue:
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_text(self.code, "recovery issue code"))
        object.__setattr__(self, "message", _require_text(self.message, "recovery issue message"))


@dataclass(frozen=True)
class WorkspaceRecoveryState:
    status: WorkspaceRecoveryStatus
    changed_paths: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _WORKSPACE_RECOVERY_STATUSES:
            raise SessionStateValidationError(f"Unknown workspace recovery status: {self.status}")
        object.__setattr__(self, "changed_paths", _text_tuple(self.changed_paths, "changed_paths"))
        object.__setattr__(self, "missing_paths", _text_tuple(self.missing_paths, "missing_paths"))


@dataclass(frozen=True)
class RecoveryBundle:
    session: SessionState
    run: RunState
    messages: tuple[MessageRecord, ...]
    workspace_status: WorkspaceRecoveryState

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        if self.run.session_id != self.session.session_id:
            raise SessionStateValidationError("Recovery run does not belong to recovery session")


@dataclass(frozen=True)
class RecoveryResult:
    status: RecoveryStatus
    bundle: RecoveryBundle | None = None
    issues: tuple[RecoveryIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _RECOVERY_STATUSES:
            raise SessionStateValidationError(f"Unknown recovery status: {self.status}")
        object.__setattr__(self, "issues", tuple(self.issues))
        if self.status in {"ready", "needs_validation"} and self.bundle is None:
            raise SessionStateValidationError(f"Recovery status {self.status} requires a bundle")
        if self.status in {"blocked", "not_found"} and self.bundle is not None:
            raise SessionStateValidationError(f"Recovery status {self.status} cannot include a bundle")


def validate_run_state(state: RunState) -> None:
    """Validate invariants that can be checked from one persisted Run State."""

    terminal = state.status in _TERMINAL_RUN_STATUSES
    if terminal:
        if state.phase != "finished":
            raise SessionStateValidationError("Terminal runs require phase=finished")
        if state.checkpoint is not None:
            raise SessionStateValidationError("Terminal runs cannot retain an active checkpoint")
        if state.ended_at is None:
            raise SessionStateValidationError("Terminal runs require ended_at")
        return

    if state.phase == "finished":
        raise SessionStateValidationError("Non-terminal runs cannot use phase=finished")
    if state.checkpoint is None:
        raise SessionStateValidationError("Non-terminal runs require a checkpoint")
    allowed_resume_points = _PHASE_RESUME_POINTS[state.phase]
    if state.checkpoint.resume_point not in allowed_resume_points:
        raise SessionStateValidationError(
            f"Resume point {state.checkpoint.resume_point} is invalid for phase {state.phase}"
        )
    if state.latest_message_id != state.checkpoint.message_cursor.leaf_message_id:
        raise SessionStateValidationError(
            "latest_message_id must match the checkpoint message cursor"
        )
    if state.status == "waiting" and state.checkpoint.waiting is None:
        raise SessionStateValidationError("Waiting runs require waiting checkpoint state")
    if state.status != "waiting" and state.checkpoint.waiting is not None:
        raise SessionStateValidationError("Only waiting runs can contain waiting checkpoint state")
    if state.status == "created":
        if state.phase != "received" or state.checkpoint.resume_point != "before_model":
            raise SessionStateValidationError(
                "Created runs must start at received/before_model"
            )


def validate_run_transition(previous: RunState, current: RunState) -> None:
    """Validate a revisioned transition between two Run State snapshots."""

    if previous.run_id != current.run_id or previous.session_id != current.session_id:
        raise SessionStateValidationError("Run identity cannot change")
    if current.revision != previous.revision + 1:
        raise SessionStateValidationError("Run revision must increase by exactly one")
    allowed = _RUN_TRANSITIONS[previous.status]
    if current.status not in allowed:
        raise SessionStateValidationError(
            f"Invalid run transition: {previous.status} -> {current.status}"
        )


ConvertToLlmFn = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
SystemPromptBuilder = Callable[[RunMode], str]
SessionContinuationKind = Literal[
    "plan_approved",
    "plan_rejected",
    "plan_feedback",
    "plan_clarification",
    "tool_approved",
    "tool_denied",
    "mode_changed",
    "automatic_continuation",
]
_CONTINUATION_KINDS = {
    "plan_approved",
    "plan_rejected",
    "plan_feedback",
    "plan_clarification",
    "tool_approved",
    "tool_denied",
    "mode_changed",
    "automatic_continuation",
}


@dataclass
class SessionOptions:
    """Runtime-supplied configuration for opening a session controller."""

    model: Model
    workspace_dir: str | Path
    system_prompt: str = ""
    system_prompt_builder: Optional[SystemPromptBuilder] = None
    session_id: Optional[str] = None
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: str = "off"
    max_tool_calls_per_turn: int = 16
    memory_enabled: bool = True
    current_mode: RunMode = "build"
    planning_budget_profile: PlanningBudgetProfile = "balanced"
    convert_to_llm: Optional[ConvertToLlmFn] = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    stream_fn: ProviderSimpleStreamFn | None = None
    prepare_context: PrepareContextFn | None = None


@dataclass(frozen=True)
class SessionRunIntent:
    text: str
    images: tuple[str, ...] = ()
    mode_hint: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "run text"))
        object.__setattr__(self, "mode_hint", _optional_text(self.mode_hint))
        object.__setattr__(self, "run_id", _optional_text(self.run_id))


@dataclass(frozen=True)
class SessionResumeIntent:
    approval_id: str
    decision: str
    reason: str = ""
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(self, "decision", _approval_decision(self.decision))
        object.__setattr__(self, "reason", _optional_text(self.reason) or "")
        object.__setattr__(self, "run_id", _optional_text(self.run_id))


@dataclass(frozen=True)
class SessionContinuationIntent:
    kind: SessionContinuationKind
    run_id: str | None = None
    text: str = ""
    approval_id: str = ""
    decision: str = ""
    reason: str = ""
    target_mode: str | None = None

    def __post_init__(self) -> None:
        kind = _require_text(self.kind, "continuation kind")
        if kind not in _CONTINUATION_KINDS:
            raise ValueError(f"Unknown continuation kind: {self.kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "text", _optional_text(self.text) or "")
        object.__setattr__(self, "approval_id", _optional_text(self.approval_id) or "")
        object.__setattr__(self, "decision", _optional_text(self.decision) or "")
        object.__setattr__(self, "reason", _optional_text(self.reason) or "")
        object.__setattr__(self, "target_mode", _optional_text(self.target_mode))
        if kind == "plan_feedback" and not self.text:
            raise ValueError("plan feedback text is required")
        if kind in {"tool_approved", "tool_denied"} and not self.approval_id:
            raise ValueError("tool continuation approval_id is required")


@dataclass(frozen=True)
class SessionCommandIntent:
    text: str
    tool_catalog: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "command text"))
        object.__setattr__(self, "tool_catalog", tuple(self.tool_catalog))


@dataclass(frozen=True)
class CancelRunIntent:
    run_id: str | None = None
    reason: str = "user"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "reason", _optional_text(self.reason) or "user")


SessionIntent = (
    SessionRunIntent
    | SessionResumeIntent
    | SessionContinuationIntent
    | SessionCommandIntent
    | CancelRunIntent
)


@dataclass(frozen=True)
class SessionView:
    session_id: str
    message_count: int = 0
    last_run_id: str | None = None
    current_mode: str = "build"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackBaselineRef:
    session_id: str
    run_id: str
    kind: Literal["rollback_baseline_ref"] = field(
        default="rollback_baseline_ref",
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))


@dataclass(frozen=True)
class PreparedAgentRun:
    run_id: str
    session_id: str
    loop_input: AgentLoopInput
    resume_input: AgentResumeInput | None = None
    context_port: ContextPort | None = None
    state_port: RunStatePort | None = None
    input_messages: list[Message] = field(default_factory=list)
    rollback_baseline: RollbackBaselineRef | None = None
    context_refs: dict[str, Any] = field(default_factory=dict)
    memory_refs: dict[str, Any] = field(default_factory=dict)
    plan_refs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionRunRecord:
    run_id: str
    session_id: str
    status: AgentLoopStatus
    stop_reason: str
    new_messages: list[Message] = field(default_factory=list)
    final_text: str = ""
    events: list[AgentEvent] = field(default_factory=list)
    outcome: AgentLoopOutcome | None = None
    snapshots: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionCommandRecord:
    session_id: str
    command: str
    handled: bool
    output_lines: tuple[str, ...] = ()
    switched_session_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text must be a string or None")
    return value.strip() or None


def _approval_decision(value: object) -> str:
    text = _require_text(value, "approval decision").lower()
    if text not in {"approve", "deny"}:
        raise ValueError(f"Unknown approval decision: {value}")
    return text


def _require_schema(actual: object, expected: int, name: str) -> None:
    if actual != expected:
        raise SessionStateValidationError(
            f"Unsupported {name} schema_version: {actual}; expected {expected}"
        )


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SessionStateValidationError(f"{field_name} must be a positive integer")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SessionStateValidationError(f"{field_name} must be a non-negative integer")


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SessionStateValidationError(f"{field_name} must be a list or tuple")
    return tuple(_require_text(item, field_name) for item in value)


def _serializable_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SessionStateValidationError(f"{field_name} must be a mapping")
    result = dict(value)
    if not all(isinstance(key, str) for key in result):
        raise SessionStateValidationError(f"{field_name} keys must be strings")
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SessionStateValidationError(f"{field_name} must be JSON serializable") from exc
    return result


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CancelRunIntent",
    "CheckpointOwner",
    "ComponentCheckpoint",
    "ConvertToLlmFn",
    "MESSAGE_RECORD_SCHEMA_VERSION",
    "MessageCursor",
    "MessageRecord",
    "ModelRef",
    "PreparedAgentRun",
    "RUN_STATE_SCHEMA_VERSION",
    "RecoveryBundle",
    "RecoveryIssue",
    "RecoveryRequest",
    "RecoveryResult",
    "RecoveryStatus",
    "ResumePoint",
    "RollbackBaselineRef",
    "RunCheckpoint",
    "RunPhase",
    "RunState",
    "RunStatus",
    "SESSION_STATE_SCHEMA_VERSION",
    "SessionCommandIntent",
    "SessionCommandRecord",
    "SessionContinuationIntent",
    "SessionContinuationKind",
    "SessionIntent",
    "SessionKind",
    "SessionResumeIntent",
    "SessionRunIntent",
    "SessionRunRecord",
    "SessionOptions",
    "SessionState",
    "SessionStateConflictError",
    "SessionStateValidationError",
    "SessionView",
    "SystemPromptBuilder",
    "WaitingKind",
    "WaitingState",
    "WorkspaceCheckpoint",
    "WorkspaceEffectsSnapshot",
    "WorkspaceRecoveryState",
    "WorkspaceRecoveryStatus",
    "validate_run_state",
    "validate_run_transition",
]
