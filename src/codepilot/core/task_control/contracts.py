from __future__ import annotations

"""Stable task planning control contracts."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, cast


TaskMode = Literal["read", "edit", "plan"]
PlanningBudgetProfile = Literal["conservative", "balanced", "wide"]
PlanningPhase = Literal["none", "discovery", "synthesis", "execution", "recovered"]
PlanningStatus = Literal["skipped", "completed", "failed", "budget_exhausted"]
PlanSource = Literal["default", "llm", "llm_with_discovery", "fallback", "recovered"]

_TASK_MODES = frozenset({"read", "edit", "plan"})
_PLANNING_BUDGET_PROFILES = frozenset({"conservative", "balanced", "wide"})
_PLANNING_PHASES = frozenset({"none", "discovery", "synthesis", "execution", "recovered"})
_PLANNING_STATUSES = frozenset({"skipped", "completed", "failed", "budget_exhausted"})
_PLAN_SOURCES = frozenset({"default", "llm", "llm_with_discovery", "fallback", "recovered"})


@dataclass(frozen=True)
class PlanningBudget:
    profile: PlanningBudgetProfile
    max_model_rounds: int
    max_tool_calls: int
    max_estimated_tokens: int
    max_wall_seconds: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", ensure_planning_budget_profile(self.profile))
        object.__setattr__(self, "max_model_rounds", _positive_int(self.max_model_rounds, "max_model_rounds"))
        object.__setattr__(self, "max_tool_calls", _positive_int(self.max_tool_calls, "max_tool_calls"))
        object.__setattr__(self, "max_estimated_tokens", _positive_int(self.max_estimated_tokens, "max_estimated_tokens"))
        object.__setattr__(self, "max_wall_seconds", _positive_int(self.max_wall_seconds, "max_wall_seconds"))

    def to_signal(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningBudgetUsage:
    model_rounds: int = 0
    tool_calls: int = 0
    estimated_tokens: int = 0
    stop_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_rounds", _non_negative_int(self.model_rounds, "model_rounds"))
        object.__setattr__(self, "tool_calls", _non_negative_int(self.tool_calls, "tool_calls"))
        object.__setattr__(self, "estimated_tokens", _non_negative_int(self.estimated_tokens, "estimated_tokens"))
        object.__setattr__(self, "stop_reason", _clean_text(self.stop_reason, limit=120))

    def to_signal(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningDiscoveryReport:
    status: PlanningStatus
    facts: tuple[str, ...] = ()
    relevant_files: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    verification_hints: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    budget: PlanningBudgetUsage = field(default_factory=PlanningBudgetUsage)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ensure_planning_status(self.status))
        object.__setattr__(self, "facts", _clean_tuple(self.facts, limit=300))
        object.__setattr__(self, "relevant_files", _clean_tuple(self.relevant_files, limit=240))
        object.__setattr__(self, "risks", _clean_tuple(self.risks, limit=240))
        object.__setattr__(self, "verification_hints", _clean_tuple(self.verification_hints, limit=240))
        object.__setattr__(self, "open_questions", _clean_tuple(self.open_questions, limit=240))
        object.__setattr__(self, "evidence_refs", _dedupe_tuple(self.evidence_refs, limit=120))
        if not isinstance(self.budget, PlanningBudgetUsage):
            object.__setattr__(self, "budget", planning_budget_usage_from_mapping(self.budget))

    def to_signal(self) -> dict[str, object]:
        return {
            "status": self.status,
            "facts": list(self.facts),
            "relevant_files": list(self.relevant_files),
            "risks": list(self.risks),
            "verification_hints": list(self.verification_hints),
            "open_questions": list(self.open_questions),
            "evidence_refs": list(self.evidence_refs),
            "budget": self.budget.to_signal(),
        }


@dataclass(frozen=True)
class TaskPlanningState:
    phase: PlanningPhase = "none"
    source: PlanSource = "default"
    budget: PlanningBudget | None = None
    discovery: PlanningDiscoveryReport | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", ensure_planning_phase(self.phase))
        object.__setattr__(self, "source", ensure_plan_source(self.source))
        if self.budget is not None and not isinstance(self.budget, PlanningBudget):
            object.__setattr__(self, "budget", planning_budget_from_mapping(self.budget))
        if self.discovery is not None and not isinstance(self.discovery, PlanningDiscoveryReport):
            object.__setattr__(self, "discovery", planning_discovery_report_from_mapping(self.discovery))
        object.__setattr__(
            self,
            "fallback_reason",
            _clean_text(self.fallback_reason, limit=240) or None,
        )

    def to_signal(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "source": self.source,
            "budget": self.budget.to_signal() if self.budget else None,
            "discovery": self.discovery.to_signal() if self.discovery else None,
            "fallback_reason": self.fallback_reason,
        }


def ensure_task_mode(value: object) -> TaskMode:
    text = value.strip() if isinstance(value, str) else ""
    if text not in _TASK_MODES:
        raise ValueError(f"Unknown task mode: {value}")
    return cast(TaskMode, text)


def ensure_planning_budget_profile(value: object) -> PlanningBudgetProfile:
    text = value.strip() if isinstance(value, str) else ""
    if text not in _PLANNING_BUDGET_PROFILES:
        raise ValueError(f"Unknown planning budget profile: {value}")
    return cast(PlanningBudgetProfile, text)


def ensure_planning_phase(value: object) -> PlanningPhase:
    text = value.strip() if isinstance(value, str) else ""
    if text not in _PLANNING_PHASES:
        raise ValueError(f"Unknown planning phase: {value}")
    return cast(PlanningPhase, text)


def ensure_planning_status(value: object) -> PlanningStatus:
    text = value.strip() if isinstance(value, str) else ""
    if text not in _PLANNING_STATUSES:
        raise ValueError(f"Unknown planning status: {value}")
    return cast(PlanningStatus, text)


def ensure_plan_source(value: object) -> PlanSource:
    text = value.strip() if isinstance(value, str) else ""
    if text not in _PLAN_SOURCES:
        raise ValueError(f"Unknown plan source: {value}")
    return cast(PlanSource, text)


def budget_for_profile(profile: PlanningBudgetProfile | str = "balanced") -> PlanningBudget:
    normalized = ensure_planning_budget_profile(profile)
    if normalized == "conservative":
        return PlanningBudget(
            profile="conservative",
            max_model_rounds=2,
            max_tool_calls=6,
            max_estimated_tokens=6000,
            max_wall_seconds=30,
        )
    if normalized == "wide":
        return PlanningBudget(
            profile="wide",
            max_model_rounds=6,
            max_tool_calls=20,
            max_estimated_tokens=20000,
            max_wall_seconds=120,
        )
    return PlanningBudget(
        profile="balanced",
        max_model_rounds=4,
        max_tool_calls=12,
        max_estimated_tokens=12000,
        max_wall_seconds=60,
    )


def planning_budget_from_mapping(value: object) -> PlanningBudget:
    data = value if isinstance(value, Mapping) else {}
    return PlanningBudget(
        profile=ensure_planning_budget_profile(data.get("profile") or "balanced"),
        max_model_rounds=int(data.get("max_model_rounds") or 4),
        max_tool_calls=int(data.get("max_tool_calls") or 12),
        max_estimated_tokens=int(data.get("max_estimated_tokens") or 12000),
        max_wall_seconds=int(data.get("max_wall_seconds") or 60),
    )


def planning_budget_usage_from_mapping(value: object) -> PlanningBudgetUsage:
    data = value if isinstance(value, Mapping) else {}
    return PlanningBudgetUsage(
        model_rounds=int(data.get("model_rounds") or 0),
        tool_calls=int(data.get("tool_calls") or 0),
        estimated_tokens=int(data.get("estimated_tokens") or 0),
        stop_reason=str(data.get("stop_reason") or ""),
    )


def planning_discovery_report_from_mapping(value: object) -> PlanningDiscoveryReport:
    data = value if isinstance(value, Mapping) else {}
    return PlanningDiscoveryReport(
        status=ensure_planning_status(data.get("status") or "failed"),
        facts=tuple(_list(data.get("facts"))),
        relevant_files=tuple(_list(data.get("relevant_files"))),
        risks=tuple(_list(data.get("risks"))),
        verification_hints=tuple(_list(data.get("verification_hints"))),
        open_questions=tuple(_list(data.get("open_questions"))),
        evidence_refs=tuple(_list(data.get("evidence_refs"))),
        budget=planning_budget_usage_from_mapping(data.get("budget")),
    )


def task_planning_state_from_mapping(value: object) -> TaskPlanningState:
    data = value if isinstance(value, Mapping) else {}
    return TaskPlanningState(
        phase=ensure_planning_phase(data.get("phase") or "none"),
        source=ensure_plan_source(data.get("source") or "default"),
        budget=(
            planning_budget_from_mapping(data.get("budget"))
            if isinstance(data.get("budget"), Mapping)
            else None
        ),
        discovery=(
            planning_discovery_report_from_mapping(data.get("discovery"))
            if isinstance(data.get("discovery"), Mapping)
            else None
        ),
        fallback_reason=str(data.get("fallback_reason") or "") or None,
    )


def _list(value: object) -> list[object]:
    if isinstance(value, list | tuple):
        return list(value)
    return []


def _clean_tuple(items: object, *, limit: int) -> tuple[str, ...]:
    return _dedupe_tuple(items, limit=limit)


def _dedupe_tuple(items: object, *, limit: int) -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for item in _list(items):
        text = _clean_text(item, limit=limit)
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    return tuple(values)


def _clean_text(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).strip().split())
    return text[:limit]


def _positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


__all__ = [
    "TaskMode",
    "PlanningBudgetProfile",
    "PlanningPhase",
    "PlanningStatus",
    "PlanSource",
    "PlanningBudget",
    "PlanningBudgetUsage",
    "PlanningDiscoveryReport",
    "TaskPlanningState",
    "ensure_task_mode",
    "ensure_planning_budget_profile",
    "ensure_planning_phase",
    "ensure_planning_status",
    "ensure_plan_source",
    "budget_for_profile",
    "planning_budget_from_mapping",
    "planning_budget_usage_from_mapping",
    "planning_discovery_report_from_mapping",
    "task_planning_state_from_mapping",
]
