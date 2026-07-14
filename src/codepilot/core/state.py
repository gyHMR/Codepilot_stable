from __future__ import annotations

"""Canonical task state and observable run facts owned by Core."""

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, cast

from .errors import CoreContractError
from .plan import PlanState, load_plan_state


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text is not None:
            result.append(text)
    return result


CORE_STATE_SCHEMA_VERSION = 2

TaskStatus = Literal["active", "blocked", "satisfied", "abandoned"]
TaskBlockerKind = Literal[
    "user_input_required",
    "tool_unavailable",
    "verification_failed",
    "plan_incomplete",
    "replan_required",
]
VerificationFactStatus = Literal[
    "none",
    "unknown",
    "passed",
    "failed",
    "stale",
    "unavailable",
]
CoreAssessmentStatus = Literal[
    "active",
    "blocked",
    "needs_verification",
    "needs_replan",
    "ready_to_finish",
    "satisfied",
]


@dataclass(frozen=True)
class TaskBlocker:
    kind: TaskBlockerKind
    reason: str
    evidence_refs: tuple[str, ...] = ()
    recoverable: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {
            "user_input_required",
            "tool_unavailable",
            "verification_failed",
            "plan_incomplete",
            "replan_required",
        }:
            raise ValueError(f"Unknown task blocker: {self.kind}")
        object.__setattr__(
            self, "reason", _required_core_text(self.reason, "blocker reason")
        )
        object.__setattr__(self, "evidence_refs", _text_tuple(self.evidence_refs))
        if not isinstance(self.recoverable, bool):
            raise TypeError("recoverable must be bool")


@dataclass(frozen=True)
class TaskState:
    original_request: str
    current_goal: str
    status: TaskStatus = "active"
    plan: PlanState | None = None
    blockers: tuple[TaskBlocker, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_request",
            _required_core_text(self.original_request, "original_request"),
        )
        object.__setattr__(
            self,
            "current_goal",
            _required_core_text(self.current_goal, "current_goal"),
        )
        if self.status not in {"active", "blocked", "satisfied", "abandoned"}:
            raise ValueError(f"Unknown task status: {self.status}")
        if self.plan is not None and not isinstance(self.plan, PlanState):
            raise TypeError("plan must be PlanState or None")
        blockers = tuple(self.blockers)
        if any(not isinstance(item, TaskBlocker) for item in blockers):
            raise TypeError("blockers must contain TaskBlocker values")
        object.__setattr__(self, "blockers", blockers)


@dataclass(frozen=True)
class CoreCounters:
    model_turns: int = 0
    model_attempts: int | None = None
    tool_iterations: int = 0
    tool_calls: int = 0
    total_recoveries: int = 0

    def __post_init__(self) -> None:
        if self.model_attempts is None:
            object.__setattr__(self, "model_attempts", self.model_turns)
        for name in (
            "model_turns",
            "model_attempts",
            "tool_iterations",
            "tool_calls",
            "total_recoveries",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class WorkspaceFacts:
    revision: int = 0
    changed: bool = False
    affected_paths: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 0
        ):
            raise ValueError("workspace revision must be a non-negative integer")
        if not isinstance(self.changed, bool):
            raise TypeError("workspace changed must be bool")
        object.__setattr__(
            self, "affected_paths", tuple(sorted(set(_text_tuple(self.affected_paths))))
        )
        object.__setattr__(self, "evidence_refs", _text_tuple(self.evidence_refs))


@dataclass(frozen=True)
class VerificationFacts:
    status: VerificationFactStatus = "none"
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
            raise ValueError(f"Unknown verification status: {self.status}")
        if self.verified_revision is not None and (
            not isinstance(self.verified_revision, int)
            or isinstance(self.verified_revision, bool)
            or self.verified_revision < 0
        ):
            raise ValueError("verified_revision must be a non-negative integer or None")
        object.__setattr__(self, "attempted_checks", _text_tuple(self.attempted_checks))
        object.__setattr__(self, "evidence_refs", _text_tuple(self.evidence_refs))
        object.__setattr__(
            self, "unavailable_reason", _optional_text(self.unavailable_reason)
        )
        if self.status == "unavailable" and self.unavailable_reason is None:
            raise ValueError("unavailable verification requires a reason")


@dataclass(frozen=True)
class FailureRecord:
    code: str
    source: str
    message: str
    recoverable: bool
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required_core_text(self.code, "failure code"))
        object.__setattr__(
            self, "source", _required_core_text(self.source, "failure source")
        )
        object.__setattr__(
            self, "message", _required_core_text(self.message, "failure message")
        )
        if not isinstance(self.recoverable, bool):
            raise TypeError("failure recoverable must be bool")
        object.__setattr__(self, "evidence_refs", _text_tuple(self.evidence_refs))


@dataclass(frozen=True)
class FailureCount:
    code: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "code", _required_core_text(self.code, "failure count code")
        )
        if (
            not isinstance(self.count, int)
            or isinstance(self.count, bool)
            or self.count <= 0
        ):
            raise ValueError("failure count must be positive")


@dataclass(frozen=True)
class FailureFacts:
    latest: FailureRecord | None = None
    counts: tuple[FailureCount, ...] = ()

    def __post_init__(self) -> None:
        if self.latest is not None and not isinstance(self.latest, FailureRecord):
            raise TypeError("latest failure must be FailureRecord or None")
        counts = tuple(self.counts)
        if any(not isinstance(item, FailureCount) for item in counts):
            raise TypeError("failure counts must contain FailureCount values")
        if len({item.code for item in counts}) != len(counts):
            raise ValueError("failure count codes must be unique")
        object.__setattr__(
            self, "counts", tuple(sorted(counts, key=lambda item: item.code))
        )

    def count_for(self, code: str) -> int:
        return next((item.count for item in self.counts if item.code == code), 0)


@dataclass(frozen=True)
class LoopGuardFacts:
    last_tool_fingerprint: str | None = None
    repeated_tool_calls: int = 0
    repeated_no_progress: int = 0
    seen_tool_call_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "last_tool_fingerprint", _optional_text(self.last_tool_fingerprint)
        )
        for name in ("repeated_tool_calls", "repeated_no_progress"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(
            self, "seen_tool_call_ids", _text_tuple(self.seen_tool_call_ids)
        )


@dataclass(frozen=True)
class ObservationLedger:
    applied_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied_observation_ids",
            _text_tuple(self.applied_observation_ids),
        )


@dataclass(frozen=True)
class RunFacts:
    counters: CoreCounters = field(default_factory=CoreCounters)
    workspace: WorkspaceFacts = field(default_factory=WorkspaceFacts)
    verification: VerificationFacts = field(default_factory=VerificationFacts)
    failures: FailureFacts = field(default_factory=FailureFacts)
    loop_guards: LoopGuardFacts = field(default_factory=LoopGuardFacts)
    observation_ledger: ObservationLedger = field(default_factory=ObservationLedger)


@dataclass(frozen=True)
class CoreAssessment:
    status: CoreAssessmentStatus
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoreState:
    task: TaskState
    facts: RunFacts = field(default_factory=RunFacts)
    schema_version: int = CORE_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CORE_STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported CoreState schema: {self.schema_version}")
        if not isinstance(self.task, TaskState):
            raise TypeError("task must be TaskState")
        if not isinstance(self.facts, RunFacts):
            raise TypeError("facts must be RunFacts")

    @classmethod
    def new(cls, original_request: str, current_goal: str | None = None) -> "CoreState":
        request = _required_core_text(original_request, "original_request")
        goal = _optional_text(current_goal) or request
        return cls(task=TaskState(original_request=request, current_goal=goal))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task": {
                "original_request": self.task.original_request,
                "current_goal": self.task.current_goal,
                "status": self.task.status,
                "plan": (
                    self.task.plan.to_dict() if self.task.plan is not None else None
                ),
                "blockers": [
                    {
                        "kind": item.kind,
                        "reason": item.reason,
                        "evidence_refs": list(item.evidence_refs),
                        "recoverable": item.recoverable,
                    }
                    for item in self.task.blockers
                ],
            },
            "facts": {
                "counters": asdict(self.facts.counters),
                "workspace": {
                    **asdict(self.facts.workspace),
                    "affected_paths": list(self.facts.workspace.affected_paths),
                    "evidence_refs": list(self.facts.workspace.evidence_refs),
                },
                "verification": {
                    **asdict(self.facts.verification),
                    "attempted_checks": list(self.facts.verification.attempted_checks),
                    "evidence_refs": list(self.facts.verification.evidence_refs),
                },
                "failures": {
                    "latest": (
                        asdict(self.facts.failures.latest)
                        if self.facts.failures.latest
                        else None
                    ),
                    "counts": [asdict(item) for item in self.facts.failures.counts],
                },
                "loop_guards": {
                    **asdict(self.facts.loop_guards),
                    "seen_tool_call_ids": list(
                        self.facts.loop_guards.seen_tool_call_ids
                    ),
                },
                "observation_ledger": {
                    "applied_observation_ids": list(
                        self.facts.observation_ledger.applied_observation_ids
                    )
                },
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "CoreState":
        if not isinstance(raw, Mapping):
            raise TypeError("CoreState must be an object")
        if raw.get("schema_version") != CORE_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported CoreState schema")
        task_raw = _mapping(raw.get("task"), "task")
        facts_raw = _mapping(raw.get("facts"), "facts")
        counters_raw = _mapping(facts_raw.get("counters"), "facts.counters")
        workspace_raw = _mapping(facts_raw.get("workspace"), "facts.workspace")
        verification_raw = _mapping(facts_raw.get("verification"), "facts.verification")
        failures_raw = _mapping(facts_raw.get("failures"), "facts.failures")
        loop_raw = _mapping(facts_raw.get("loop_guards"), "facts.loop_guards")
        ledger_raw = _mapping(
            facts_raw.get("observation_ledger"), "facts.observation_ledger"
        )
        blocker_values = _mapping_list(task_raw.get("blockers"), "task.blockers")
        count_values = _mapping_list(
            failures_raw.get("counts"), "facts.failures.counts"
        )
        latest_raw = failures_raw.get("latest")
        latest = (
            _failure_record_from_mapping(_mapping(latest_raw, "facts.failures.latest"))
            if latest_raw is not None
            else None
        )
        return cls(
            task=TaskState(
                original_request=_required_core_text(
                    task_raw.get("original_request"), "original_request"
                ),
                current_goal=_required_core_text(
                    task_raw.get("current_goal"), "current_goal"
                ),
                status=cast(TaskStatus, task_raw.get("status")),
                plan=load_plan_state(task_raw.get("plan")),
                blockers=tuple(
                    TaskBlocker(
                        kind=cast(TaskBlockerKind, item.get("kind")),
                        reason=_required_core_text(
                            item.get("reason"), "blocker reason"
                        ),
                        evidence_refs=tuple(_string_list(item.get("evidence_refs"))),
                        recoverable=bool(item.get("recoverable")),
                    )
                    for item in blocker_values
                ),
            ),
            facts=RunFacts(
                counters=CoreCounters(
                    model_turns=_non_negative_int(counters_raw.get("model_turns")),
                    model_attempts=_non_negative_int(
                        counters_raw.get(
                            "model_attempts",
                            counters_raw.get("model_turns"),
                        )
                    ),
                    tool_iterations=_non_negative_int(
                        counters_raw.get("tool_iterations")
                    ),
                    tool_calls=_non_negative_int(counters_raw.get("tool_calls")),
                    total_recoveries=_non_negative_int(
                        counters_raw.get("total_recoveries")
                    ),
                ),
                workspace=WorkspaceFacts(
                    revision=_non_negative_int(workspace_raw.get("revision")),
                    changed=bool(workspace_raw.get("changed")),
                    affected_paths=tuple(
                        _string_list(workspace_raw.get("affected_paths"))
                    ),
                    evidence_refs=tuple(
                        _string_list(workspace_raw.get("evidence_refs"))
                    ),
                ),
                verification=VerificationFacts(
                    status=cast(VerificationFactStatus, verification_raw.get("status")),
                    verified_revision=_optional_int(
                        verification_raw.get("verified_revision")
                    ),
                    attempted_checks=tuple(
                        _string_list(verification_raw.get("attempted_checks"))
                    ),
                    evidence_refs=tuple(
                        _string_list(verification_raw.get("evidence_refs"))
                    ),
                    unavailable_reason=_optional_text(
                        verification_raw.get("unavailable_reason")
                    ),
                ),
                failures=FailureFacts(
                    latest=latest,
                    counts=tuple(
                        FailureCount(
                            code=_required_core_text(
                                item.get("code"), "failure count code"
                            ),
                            count=_positive_int(item.get("count"), "failure count"),
                        )
                        for item in count_values
                    ),
                ),
                loop_guards=LoopGuardFacts(
                    last_tool_fingerprint=_optional_text(
                        loop_raw.get("last_tool_fingerprint")
                    ),
                    repeated_tool_calls=_non_negative_int(
                        loop_raw.get("repeated_tool_calls")
                    ),
                    repeated_no_progress=_non_negative_int(
                        loop_raw.get("repeated_no_progress")
                    ),
                    seen_tool_call_ids=tuple(
                        _string_list(loop_raw.get("seen_tool_call_ids"))
                    ),
                ),
                observation_ledger=ObservationLedger(
                    applied_observation_ids=tuple(
                        _string_list(ledger_raw.get("applied_observation_ids"))
                    )
                ),
            ),
        )


def load_core_state(
    raw: CoreState | Mapping[str, object],
    *,
    original_request: str | None = None,
    current_goal: str | None = None,
) -> CoreState:
    """Load the target schema or upgrade the current pre-refactor payload."""

    if isinstance(raw, CoreState):
        return raw
    if not isinstance(raw, Mapping):
        raise CoreContractError("Core state must be an object")
    schema_version = raw.get("schema_version")
    if schema_version is not None:
        if schema_version != CORE_STATE_SCHEMA_VERSION:
            raise CoreContractError(f"Unsupported CoreState schema: {schema_version}")
        try:
            return CoreState.from_mapping(raw)
        except (TypeError, ValueError) as exc:
            raise CoreContractError(f"Invalid CoreState schema: {exc}") from exc
    try:
        return _upgrade_current_sessions_v2_state(
            raw,
            original_request=original_request,
            current_goal=current_goal,
        )
    except CoreContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise CoreContractError(
            f"Invalid current Sessions v2 Core payload: {exc}"
        ) from exc


def _upgrade_current_sessions_v2_state(
    raw: Mapping[str, object],
    *,
    original_request: str | None,
    current_goal: str | None,
) -> CoreState:
    legacy_plan_raw = raw.get("plan_state")
    legacy_plan = legacy_plan_raw if isinstance(legacy_plan_raw, Mapping) else {}
    plan = load_plan_state(legacy_plan_raw)
    request = _optional_text(original_request)
    if request is None:
        request = _optional_text(legacy_plan.get("raw_user_request"))
    if request is None:
        raise CoreContractError(
            "Legacy Core state requires original_request for schema upgrade"
        )
    goal = _optional_text(current_goal)
    if goal is None:
        goal = _optional_text(legacy_plan.get("interpreted_goal"))
    goal = goal or request

    counters_raw = raw.get("counters")
    counters = counters_raw if isinstance(counters_raw, Mapping) else {}
    workspace_changed = bool(raw.get("workspace_changed"))
    workspace_revision = 1 if workspace_changed else 0
    verification_values = _legacy_verification_values(raw.get("verification"))
    verification_status = _legacy_verification_status(raw.get("verification_status"))
    attempted_checks = tuple(
        dict.fromkeys(
            command
            for item in verification_values
            if (command := _optional_text(item.get("command"))) is not None
        )
    )
    verification_refs = tuple(
        dict.fromkeys(
            call_id
            for item in verification_values
            if (call_id := _optional_text(item.get("tool_call_id"))) is not None
        )
    )

    last_error_raw = raw.get("last_error")
    latest_failure = None
    if isinstance(last_error_raw, Mapping):
        failure_code = (
            _optional_text(last_error_raw.get("error_code"))
            or _optional_text(last_error_raw.get("code"))
            or "tool.execution_failed"
        )
        latest_failure = FailureRecord(
            code=failure_code,
            source="tools",
            message=_optional_text(last_error_raw.get("message")) or failure_code,
            recoverable=True,
            evidence_refs=tuple(
                item
                for item in (_optional_text(last_error_raw.get("tool_call_id")),)
                if item is not None
            ),
        )
    if bool(raw.get("cancelled")):
        latest_failure = FailureRecord(
            code="run.cancelled",
            source="runtime",
            message="Run was cancelled",
            recoverable=False,
        )

    task_status: TaskStatus = "abandoned" if bool(raw.get("cancelled")) else "active"
    recovery_raw = raw.get("recovery_outcome")
    if isinstance(recovery_raw, Mapping):
        recovered_status = _optional_text(recovery_raw.get("status"))
        if recovered_status == "completed":
            task_status = "satisfied"
        elif recovered_status == "cancelled":
            task_status = "abandoned"
            latest_failure = FailureRecord(
                code="run.cancelled",
                source="runtime",
                message="Recovered run was cancelled",
                recoverable=False,
            )
        elif recovered_status == "failed":
            task_status = "blocked"
            failure_code = (
                _optional_text(recovery_raw.get("stop_reason"))
                or "runtime.recovered_failure"
            )
            latest_failure = FailureRecord(
                code=failure_code,
                source="runtime",
                message=failure_code,
                recoverable=False,
            )

    blockers: list[TaskBlocker] = []
    if bool(raw.get("tool_unavailable")):
        blockers.append(
            TaskBlocker(
                "tool_unavailable",
                "Requested tool is unavailable",
                recoverable=True,
            )
        )
    if verification_status == "failed":
        blockers.append(
            TaskBlocker(
                "verification_failed",
                "Verification failed",
                evidence_refs=verification_refs,
                recoverable=True,
            )
        )

    return CoreState(
        task=TaskState(
            original_request=request,
            current_goal=goal,
            status=task_status,
            plan=plan,
            blockers=tuple(blockers),
        ),
        facts=RunFacts(
            counters=CoreCounters(
                model_turns=_non_negative_int(
                    raw.get("model_turns", counters.get("model_attempts"))
                ),
                model_attempts=_non_negative_int(
                    counters.get(
                        "model_attempts",
                        raw.get("model_turns"),
                    )
                ),
                tool_iterations=_non_negative_int(counters.get("tool_iterations")),
                tool_calls=_non_negative_int(counters.get("tool_calls")),
            ),
            workspace=WorkspaceFacts(
                revision=workspace_revision,
                changed=workspace_changed,
                affected_paths=tuple(_string_list(raw.get("affected_paths"))),
                evidence_refs=tuple(
                    f"tool_call:{call_id}"
                    for call_id in _string_list(raw.get("seen_tool_call_ids"))
                ),
            ),
            verification=VerificationFacts(
                status=verification_status,
                verified_revision=(
                    workspace_revision
                    if verification_status in {"passed", "failed"}
                    else None
                ),
                attempted_checks=attempted_checks,
                evidence_refs=verification_refs,
            ),
            failures=FailureFacts(
                latest=latest_failure,
                counts=(
                    (FailureCount(latest_failure.code, 1),)
                    if latest_failure is not None
                    else ()
                ),
            ),
            loop_guards=LoopGuardFacts(
                last_tool_fingerprint=_optional_text(raw.get("last_tool_fingerprint")),
                repeated_tool_calls=_non_negative_int(raw.get("repeated_tool_calls")),
                seen_tool_call_ids=tuple(_string_list(raw.get("seen_tool_call_ids"))),
            ),
        ),
    )


def _legacy_verification_values(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _legacy_verification_status(value: object) -> VerificationFactStatus:
    status = _optional_text(value)
    if status in {"passed", "failed", "stale"}:
        return cast(VerificationFactStatus, status)
    return "unknown"


def assess_core_state(state: CoreState) -> CoreAssessment:
    if state.task.status == "satisfied":
        return CoreAssessment("satisfied")
    if state.task.status in {"blocked", "abandoned"}:
        return CoreAssessment("blocked", (f"task_{state.task.status}",))
    if any(item.kind == "replan_required" for item in state.task.blockers):
        return CoreAssessment("needs_replan", ("replan_required",))
    workspace = state.facts.workspace
    verification = state.facts.verification
    if workspace.changed and not (
        verification.status in {"passed", "unavailable"}
        and verification.verified_revision == workspace.revision
    ):
        return CoreAssessment("needs_verification", ("verification_not_fresh",))
    plan = state.task.plan
    if (
        plan is not None
        and plan.status == "active"
        and plan.close_request is not None
        and all(item.status == "completed" for item in plan.steps)
    ):
        return CoreAssessment("ready_to_finish")
    return CoreAssessment("active")


def _required_core_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _text_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise TypeError("expected a sequence of text values")
    return tuple(
        dict.fromkeys(_required_core_text(value, "text value") for value in values)
    )


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _mapping_list(value: object, field_name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{field_name} must be a list of objects")
    return list(value)


def _failure_record_from_mapping(raw: Mapping[str, object]) -> FailureRecord:
    return FailureRecord(
        code=_required_core_text(raw.get("code"), "failure code"),
        source=_required_core_text(raw.get("source"), "failure source"),
        message=_required_core_text(raw.get("message"), "failure message"),
        recoverable=bool(raw.get("recoverable")),
        evidence_refs=tuple(_string_list(raw.get("evidence_refs"))),
    )


def _positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


__all__ = [
    "CORE_STATE_SCHEMA_VERSION",
    "CoreAssessment",
    "CoreAssessmentStatus",
    "CoreCounters",
    "CoreState",
    "FailureCount",
    "FailureFacts",
    "FailureRecord",
    "LoopGuardFacts",
    "ObservationLedger",
    "RunFacts",
    "TaskBlocker",
    "TaskBlockerKind",
    "TaskState",
    "TaskStatus",
    "VerificationFactStatus",
    "VerificationFacts",
    "WorkspaceFacts",
    "assess_core_state",
    "load_core_state",
]
