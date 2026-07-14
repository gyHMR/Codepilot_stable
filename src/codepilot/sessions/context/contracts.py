from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from codepilot.protocols import AssistantMessage, Message, ToolResultMessage, UserMessage


ContextLayer = Literal["l0", "l1", "l2", "l3", "l4"]
RetentionClass = Literal["required", "protected", "budgeted", "discard_first"]
ContextPressureLevel = Literal["normal", "tight", "critical", "overflow"]
ProjectionAction = Literal[
    "keep_full",
    "keep_projected",
    "replace_with_artifact_ref",
    "covered_by_compact_summary",
    "discard_orphan",
]
EvidenceAction = Literal[
    "show_status_only",
    "show_projected_status",
    "show_stale_warning",
    "discard",
]

_LAYERS = frozenset({"l0", "l1", "l2", "l3", "l4"})
_RETENTION = frozenset({"required", "protected", "budgeted", "discard_first"})
_PRESSURE = frozenset({"normal", "tight", "critical", "overflow"})


@dataclass(frozen=True)
class ContextSourceRef:
    kind: str
    ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _required_text(self.kind, "source kind"))
        object.__setattr__(self, "ref", _required_text(self.ref, "source ref"))


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    layer: ContextLayer
    retention: RetentionClass
    content: str
    source: ContextSourceRef
    estimated_tokens: int
    relevance: int = 0
    freshness: str = "unknown"
    recency: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _required_text(self.item_id, "item id"))
        if self.layer not in _LAYERS:
            raise ValueError(f"unknown context layer: {self.layer}")
        if self.retention not in _RETENTION:
            raise ValueError(f"unknown retention class: {self.retention}")
        object.__setattr__(self, "content", _required_text(self.content, "item content"))
        if not isinstance(self.source, ContextSourceRef):
            raise TypeError("context item source must be ContextSourceRef")
        if not isinstance(self.estimated_tokens, int) or self.estimated_tokens < 0:
            raise ValueError("estimated_tokens must be a non-negative integer")
        if not isinstance(self.relevance, int):
            raise TypeError("relevance must be int")
        object.__setattr__(self, "freshness", _required_text(self.freshness, "freshness"))
        if not isinstance(self.recency, int):
            raise TypeError("recency must be int")


@dataclass(frozen=True)
class ContextBudget:
    effective_input_tokens: int
    output_reserve_tokens: int
    safety_margin_tokens: int
    layer_weights: Mapping[ContextLayer, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("effective_input_tokens", self.effective_input_tokens),
            ("output_reserve_tokens", self.output_reserve_tokens),
            ("safety_margin_tokens", self.safety_margin_tokens),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(
            self,
            "layer_weights",
            MappingProxyType(dict(self.layer_weights)),
        )


@dataclass(frozen=True)
class ContextPressure:
    level: ContextPressureLevel
    raw_tokens: int
    effective_budget: int
    conversation_tokens: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.level not in _PRESSURE:
            raise ValueError(f"unknown context pressure: {self.level}")
        for name, value in (
            ("raw_tokens", self.raw_tokens),
            ("effective_budget", self.effective_budget),
            ("conversation_tokens", self.conversation_tokens),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(
            self,
            "reasons",
            tuple(_required_text(reason, "pressure reason") for reason in self.reasons),
        )


@dataclass(frozen=True)
class ProjectedMessage:
    source_ref: str
    message: Message
    action: ProjectionAction
    group_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ref", _required_text(self.source_ref, "source ref"))
        if not isinstance(self.message, (UserMessage, AssistantMessage, ToolResultMessage)):
            raise TypeError("projected message must contain a canonical Message")
        object.__setattr__(self, "group_id", _required_text(self.group_id, "group id"))


@dataclass(frozen=True)
class ProjectedEvidence:
    source_ref: str
    content: str
    action: EvidenceAction
    freshness: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ref", _required_text(self.source_ref, "source ref"))
        object.__setattr__(self, "content", _required_text(self.content, "evidence content"))
        object.__setattr__(self, "freshness", _required_text(self.freshness, "freshness"))


@dataclass(frozen=True)
class ProjectionPlan:
    messages: tuple[ProjectedMessage, ...] = ()
    evidence: tuple[ProjectedEvidence, ...] = ()

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        evidence = tuple(self.evidence)
        if any(not isinstance(item, ProjectedMessage) for item in messages):
            raise TypeError("projection messages must contain ProjectedMessage")
        if any(not isinstance(item, ProjectedEvidence) for item in evidence):
            raise TypeError("projection evidence must contain ProjectedEvidence")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "evidence", evidence)

    @property
    def model_messages(self) -> tuple[Message, ...]:
        return tuple(
            item.message
            for item in self.messages
            if item.action not in {"covered_by_compact_summary", "discard_orphan"}
        )


@dataclass(frozen=True)
class CompactSummary:
    original_goal: str
    user_constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    completed_work: tuple[str, ...] = ()
    files_and_symbols: tuple[str, ...] = ()
    important_evidence: tuple[str, ...] = ()
    errors_and_resolutions: tuple[str, ...] = ()
    verification_state: str = ""
    open_questions: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_goal", _required_text(self.original_goal, "original goal"))
        for name in (
            "user_constraints",
            "decisions",
            "completed_work",
            "files_and_symbols",
            "important_evidence",
            "errors_and_resolutions",
            "open_questions",
            "next_actions",
            "source_refs",
        ):
            object.__setattr__(
                self,
                name,
                tuple(_required_text(item, name) for item in getattr(self, name)),
            )
        if not self.source_refs:
            raise ValueError("compact summary requires source refs")
        object.__setattr__(self, "verification_state", str(self.verification_state or "").strip())

    def to_mapping(self) -> dict[str, object]:
        return {
            "original_goal": self.original_goal,
            "user_constraints": list(self.user_constraints),
            "decisions": list(self.decisions),
            "completed_work": list(self.completed_work),
            "files_and_symbols": list(self.files_and_symbols),
            "important_evidence": list(self.important_evidence),
            "errors_and_resolutions": list(self.errors_and_resolutions),
            "verification_state": self.verification_state,
            "open_questions": list(self.open_questions),
            "next_actions": list(self.next_actions),
            "source_refs": list(self.source_refs),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "CompactSummary":
        allowed = {
            "original_goal",
            "user_constraints",
            "decisions",
            "completed_work",
            "files_and_symbols",
            "important_evidence",
            "errors_and_resolutions",
            "verification_state",
            "open_questions",
            "next_actions",
            "source_refs",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError("unknown compact summary fields: " + ", ".join(unknown))
        return cls(
            original_goal=_required_text(raw.get("original_goal"), "original goal"),
            user_constraints=_text_tuple(raw.get("user_constraints")),
            decisions=_text_tuple(raw.get("decisions")),
            completed_work=_text_tuple(raw.get("completed_work")),
            files_and_symbols=_text_tuple(raw.get("files_and_symbols")),
            important_evidence=_text_tuple(raw.get("important_evidence")),
            errors_and_resolutions=_text_tuple(raw.get("errors_and_resolutions")),
            verification_state=str(raw.get("verification_state") or "").strip(),
            open_questions=_text_tuple(raw.get("open_questions")),
            next_actions=_text_tuple(raw.get("next_actions")),
            source_refs=_text_tuple(raw.get("source_refs")),
        )

    def render(self) -> str:
        sections = [f"Original goal: {self.original_goal}"]
        for title, values in (
            ("User constraints", self.user_constraints),
            ("Decisions", self.decisions),
            ("Completed work", self.completed_work),
            ("Files and symbols", self.files_and_symbols),
            ("Important evidence", self.important_evidence),
            ("Errors and resolutions", self.errors_and_resolutions),
            ("Open questions", self.open_questions),
            ("Next actions", self.next_actions),
            ("Source refs", self.source_refs),
        ):
            if values:
                sections.append(f"{title}: " + "; ".join(values))
        if self.verification_state:
            sections.append(f"Verification state: {self.verification_state}")
        return "\n".join(sections)


@dataclass(frozen=True)
class CompactSnapshotRef:
    compact_id: str
    path: str
    compacted_until_message_id: str
    source_digest: str
    estimated_tokens_before: int
    estimated_tokens_after: int

    def __post_init__(self) -> None:
        for name in ("compact_id", "path", "compacted_until_message_id", "source_digest"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.estimated_tokens_before < 0 or self.estimated_tokens_after < 0:
            raise ValueError("snapshot token counts must be non-negative")


@dataclass(frozen=True)
class ContextCheckpointState:
    compact_snapshot_ref: str | None = None
    compacted_until_message_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "compact_snapshot_ref", _optional_text(self.compact_snapshot_ref))
        object.__setattr__(
            self,
            "compacted_until_message_id",
            _optional_text(self.compacted_until_message_id),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "compact_snapshot_ref": self.compact_snapshot_ref,
            "compacted_until_message_id": self.compacted_until_message_id,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "ContextCheckpointState":
        if not isinstance(raw, Mapping):
            raise TypeError("context checkpoint must be a mapping")
        allowed = {"compact_snapshot_ref", "compacted_until_message_id"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError("unknown context checkpoint fields: " + ", ".join(unknown))
        return cls(
            compact_snapshot_ref=_optional_text(raw.get("compact_snapshot_ref")),
            compacted_until_message_id=_optional_text(raw.get("compacted_until_message_id")),
        )


@dataclass(frozen=True)
class ContextSummaryRequest:
    messages: tuple[Message, ...]
    original_goal: str
    previous_summary: CompactSummary | None = None

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if any(
            not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage))
            for message in messages
        ):
            raise TypeError("summary messages must contain canonical Message values")
        if not messages:
            raise ValueError("summary messages cannot be empty")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "original_goal", _required_text(self.original_goal, "original goal"))
        if self.previous_summary is not None and not isinstance(
            self.previous_summary, CompactSummary
        ):
            raise TypeError("previous_summary must be CompactSummary or None")


@dataclass(frozen=True)
class ContextSummaryResult:
    summary: CompactSummary
    compacted_until_message_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.summary, CompactSummary):
            raise TypeError("summary must be CompactSummary")
        object.__setattr__(
            self,
            "compacted_until_message_id",
            _required_text(self.compacted_until_message_id, "compacted_until_message_id"),
        )


class ContextSummarizerPort(Protocol):
    def summarize(
        self,
        request: ContextSummaryRequest,
    ) -> ContextSummaryResult | Awaitable[ContextSummaryResult]: ...


class ContextCheckpointPort(Protocol):
    def checkpoint_state(self) -> Mapping[str, object] | None: ...

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None: ...


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("compact summary list fields must be lists or tuples")
    return tuple(_required_text(item, "summary item") for item in value)


__all__ = [
    "CompactSnapshotRef",
    "CompactSummary",
    "ContextBudget",
    "ContextCheckpointPort",
    "ContextCheckpointState",
    "ContextItem",
    "ContextLayer",
    "ContextPressure",
    "ContextPressureLevel",
    "ContextSourceRef",
    "ContextSummarizerPort",
    "ContextSummaryRequest",
    "ContextSummaryResult",
    "EvidenceAction",
    "ProjectedEvidence",
    "ProjectedMessage",
    "ProjectionAction",
    "ProjectionPlan",
    "RetentionClass",
]
