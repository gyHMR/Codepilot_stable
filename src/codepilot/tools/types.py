from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from codepilot.protocols.tools import (
    Tool,
    ToolMetadata,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
)


AgentToolResult = ToolResult
AgentToolUpdateCallback = Callable[[AgentToolResult], None]


class ToolExecuteFn(Protocol):
    def __call__(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> Awaitable[AgentToolResult] | AgentToolResult:
        ...


@dataclass
class AgentTool:
    """Executable tool definition owned by the tools layer."""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    execute: ToolExecuteFn
    runtime_managed: bool = False

    def to_spec(self) -> Tool:
        """Return the provider-facing tool description without the executor."""

        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass(frozen=True)
class ToolRuntimeRequest:
    tool_call_id: str
    name: str
    params: dict[str, Any]
    source: str = "agent"


@dataclass(frozen=True)
class ToolRuntimeResult:
    result: AgentToolResult
    status: ToolResultStatus = "success"
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "ToolExecuteFn",
    "ToolMetadata",
    "ToolResultStatus",
    "ToolRiskLevel",
    "ToolRuntimeRequest",
    "ToolRuntimeResult",
]
