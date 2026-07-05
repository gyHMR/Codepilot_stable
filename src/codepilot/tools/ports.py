from __future__ import annotations

# 新手导读：ports.py 只定义 core 可消费的工具能力端口和执行观察 DTO。
# 关注点：不要在这里 import ToolRuntime、权限策略或内置工具实现；具体适配放 adapters.py。

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from codepilot.protocols import (
    AssistantMessage,
    ContentBlock,
    RunVerification,
    Tool,
    ToolHookContextSnapshot,
)


ToolObservationStatus = Literal["success", "error", "denied", "approval_required", "cancelled"]
ToolInvocationSource = Literal["agent", "task_control", "approval_resume"]
ToolResumeDecisionValue = Literal["approve", "deny"]


@dataclass(frozen=True)
class ToolCatalogView:
    tools: tuple[Tool, ...] = field(default_factory=tuple)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)


@dataclass(frozen=True)
class ToolPolicyContext:
    session_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        session_id = str(self.session_id).strip() if self.session_id is not None else None
        object.__setattr__(self, "session_id", session_id or None)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )


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


@dataclass(frozen=True)
class ToolResumeDecision:
    approval_id: str
    decision: ToolResumeDecisionValue
    reason: str = ""

    def __post_init__(self) -> None:
        decision = self.decision.strip().lower()
        if decision not in {"approve", "deny"}:
            raise ValueError(f"Unknown tool resume decision: {self.decision}")
        object.__setattr__(self, "decision", decision)


class ToolPort(Protocol):
    def catalog(self) -> ToolCatalogView:
        ...

    async def execute(self, invocation: ToolInvocation) -> ToolObservation:
        ...

    async def resume(self, decision: ToolResumeDecision) -> ToolObservation:
        ...


__all__ = [
    "ToolCatalogView",
    "ToolInterruption",
    "ToolInvocation",
    "ToolInvocationSource",
    "ToolObservation",
    "ToolObservationStatus",
    "ToolPolicyContext",
    "ToolPort",
    "ToolResumeDecision",
    "ToolResumeDecisionValue",
    "ToolRiskView",
]
