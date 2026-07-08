from __future__ import annotations

"""Executable tool contracts owned by the tools layer."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Literal, Protocol, TypeAlias, cast

from codepilot.protocols import (
    AssistantMessage,
    ContentBlock,
    RunVerification,
    TextContent,
    Tool,
    ToolCall,
    ToolHookContextSnapshot,
)
from codepilot.protocols.tools import ToolResult, ToolResultStatus, ToolRiskLevel

ToolScope: TypeAlias = Literal["read", "plan", "build", "memory", "extension"]
ToolObservationStatus: TypeAlias = Literal[
    "success",
    "error",
    "denied",
    "approval_required",
    "cancelled",
]
ToolInvocationSource: TypeAlias = Literal["agent", "approval_resume"]
ToolResumeDecisionValue: TypeAlias = Literal["approve", "deny"]

_OBSERVATION_STATUSES = frozenset(
    {"success", "error", "denied", "approval_required", "cancelled"}
)
_RESUME_DECISIONS = frozenset({"approve", "deny"})
_RISK_LEVELS = frozenset({"low", "medium", "high"})
_SCOPES = frozenset({"read", "plan", "build", "memory", "extension"})

ToolUpdateCallback: TypeAlias = Callable[[ToolResult], None]


class ToolExecuteFn(Protocol):
    def __call__(
        self,
        request: "ToolCallRequest",
        signal: Any | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> Awaitable[ToolResult] | ToolResult:
        ...


@dataclass(frozen=True)
class ToolMetadata:
    """Static runtime metadata used for exposure, scheduling, and safety."""

    name: str
    category: str
    read_only: bool
    concurrency_safe: bool
    exclusive: bool
    requires_approval: bool
    risk_level: ToolRiskLevel
    scopes: tuple[str, ...]
    network_access: bool = False
    credential_required: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "metadata.name"))
        object.__setattr__(
            self,
            "category",
            _require_text(self.category, "metadata.category"),
        )
        for field_name in (
            "read_only",
            "concurrency_safe",
            "exclusive",
            "requires_approval",
            "network_access",
            "credential_required",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"ToolMetadata {field_name} must be bool")
        risk_level = _clean_text(self.risk_level)
        if risk_level not in _RISK_LEVELS:
            raise ValueError(f"Unknown tool risk level: {self.risk_level}")
        object.__setattr__(self, "risk_level", cast(ToolRiskLevel, risk_level))
        object.__setattr__(self, "scopes", tuple(_clean_scopes(self.scopes)))
        if not isinstance(self.extra, dict):
            raise TypeError("ToolMetadata extra must be a dict")
        object.__setattr__(self, "extra", deepcopy(self.extra))

    def visible_in(self, current_mode: str) -> bool:
        mode = _clean_text(current_mode)
        return mode in self.scopes or "extension" in self.scopes


@dataclass
class ToolDefinition:
    """A callable tool plus its model-visible schema and runtime metadata."""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    metadata: ToolMetadata
    execute: ToolExecuteFn

    def __post_init__(self) -> None:
        self.name = _require_text(self.name, "tool.name")
        self.label = _require_text(self.label, "tool.label")
        self.description = _require_text(self.description, "tool.description")
        if not isinstance(self.parameters, dict):
            raise TypeError("ToolDefinition parameters must be a dict")
        if not isinstance(self.metadata, ToolMetadata):
            raise TypeError("ToolDefinition metadata must be ToolMetadata")
        if self.metadata.name != self.name:
            raise ValueError(
                f"Tool metadata name must match tool name: {self.metadata.name} != {self.name}"
            )
        if not callable(self.execute):
            raise TypeError("ToolDefinition execute must be callable")
        self.parameters = deepcopy(self.parameters)

    def to_spec(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass(frozen=True)
class ToolCatalogItem:
    spec: Tool
    metadata: ToolMetadata


@dataclass(frozen=True)
class ToolCatalogView:
    items: tuple[ToolCatalogItem, ...] = field(default_factory=tuple)

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(item.spec for item in self.items)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class ToolPolicyContext:
    session_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _optional_text(self.session_id))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )


@dataclass(frozen=True)
class ToolCallRequest:
    run_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: ToolMetadata | None = None
    current_mode: str = "build"
    source: ToolInvocationSource = "agent"
    policy_context: ToolPolicyContext = field(default_factory=ToolPolicyContext)
    assistant_message: AssistantMessage | None = None
    context: ToolHookContextSnapshot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "tool_call_id",
            _require_text(self.tool_call_id, "tool_call_id"),
        )
        object.__setattr__(self, "name", _require_text(self.name, "tool.name"))
        if not isinstance(self.arguments, dict):
            raise TypeError("ToolCallRequest arguments must be a dict")
        object.__setattr__(self, "arguments", deepcopy(self.arguments))
        object.__setattr__(self, "current_mode", _require_text(self.current_mode, "current_mode"))
        source = _clean_text(self.source)
        if source not in {"agent", "approval_resume"}:
            raise ValueError(f"Unknown tool invocation source: {self.source}")
        object.__setattr__(self, "source", cast(ToolInvocationSource, source))


@dataclass(frozen=True)
class PreparedToolCall:
    definition: ToolDefinition
    request: ToolCallRequest

    @property
    def metadata(self) -> ToolMetadata:
        return self.definition.metadata


@dataclass(frozen=True)
class PreparedToolCallResult:
    call: PreparedToolCall | None = None
    error_code: str | None = None
    message: str = ""
    recovery_hint: str = ""

    @property
    def valid(self) -> bool:
        return self.call is not None and self.error_code is None


@dataclass(frozen=True)
class ToolRiskView:
    level: str
    summary: str = ""


@dataclass(frozen=True)
class ToolInterruption:
    approval_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object] = field(default_factory=dict)
    reason: str = ""
    risk: ToolRiskView = field(default_factory=lambda: ToolRiskView(level="unknown"))


@dataclass(frozen=True)
class ToolInvocation:
    run_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, object] = field(default_factory=dict)
    current_mode: str = "build"
    source: ToolInvocationSource = "agent"
    policy_context: ToolPolicyContext = field(default_factory=ToolPolicyContext)
    assistant_message: AssistantMessage | None = None
    context: ToolHookContextSnapshot | None = None


@dataclass(frozen=True)
class ToolObservation:
    tool_call_id: str
    name: str
    status: ToolObservationStatus
    content: tuple[ContentBlock, ...] = field(default_factory=tuple)
    affected_paths: tuple[str, ...] = field(default_factory=tuple)
    workspace_changed: bool = False
    verification: tuple[RunVerification, ...] = field(default_factory=tuple)
    interruption: ToolInterruption | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = _clean_text(self.status)
        if status not in _OBSERVATION_STATUSES:
            raise ValueError(f"Unknown tool observation status: {self.status}")
        object.__setattr__(self, "status", cast(ToolObservationStatus, status))


@dataclass(frozen=True)
class ToolResumeDecision:
    approval_id: str
    decision: ToolResumeDecisionValue
    reason: str = ""

    def __post_init__(self) -> None:
        decision = _clean_text(self.decision)
        if decision not in _RESUME_DECISIONS:
            raise ValueError(f"Unknown tool resume decision: {self.decision}")
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(self, "decision", cast(ToolResumeDecisionValue, decision))
        object.__setattr__(self, "reason", _clean_text(self.reason))


class ToolPort(Protocol):
    def catalog(self, current_mode: str = "build") -> ToolCatalogView:
        ...

    async def execute(self, invocation: ToolInvocation) -> ToolObservation:
        ...

    async def resume(self, decision: ToolResumeDecision) -> ToolObservation:
        ...


def tool_call_from_invocation(invocation: ToolInvocation) -> ToolCall:
    return ToolCall(
        id=invocation.tool_call_id,
        name=invocation.name,
        arguments=dict(invocation.arguments),
    )


def error_result(message: str, *, status: ToolResultStatus = "error", error_code: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(text=message)],
        status=status,
        is_error=status != "success",
        error_code=error_code,
    )


def _clean_scopes(values: tuple[str, ...]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text not in _SCOPES:
            raise ValueError(f"Unknown tool scope: {value}")
        if text not in seen:
            cleaned.append(text)
            seen.add(text)
    if not cleaned:
        raise ValueError("ToolMetadata scopes cannot be empty")
    return cleaned


def _require_text(value: object, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object) -> str | None:
    text = _clean_text(value)
    return text or None


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


__all__ = [
    "PreparedToolCall",
    "PreparedToolCallResult",
    "ToolCallRequest",
    "ToolCatalogItem",
    "ToolCatalogView",
    "ToolDefinition",
    "ToolExecuteFn",
    "ToolInterruption",
    "ToolInvocation",
    "ToolInvocationSource",
    "ToolMetadata",
    "ToolObservation",
    "ToolObservationStatus",
    "ToolPolicyContext",
    "ToolPort",
    "ToolResumeDecision",
    "ToolResumeDecisionValue",
    "ToolResult",
    "ToolRiskView",
    "ToolScope",
    "ToolUpdateCallback",
    "error_result",
    "tool_call_from_invocation",
]
