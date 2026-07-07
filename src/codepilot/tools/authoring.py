from __future__ import annotations

"""Executable tool types owned by the tools layer."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, TYPE_CHECKING

from codepilot.protocols.tools import (
    Tool,
    ToolResult,
    ToolResultStatus,
    ensure_tool_result_status,
)

if TYPE_CHECKING:
    from codepilot.protocols.tools import ToolMetadata


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
    """A callable tool plus the model-visible tool specification."""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    execute: ToolExecuteFn
    runtime_managed: bool = False
    metadata: ToolMetadata | None = None

    def __post_init__(self) -> None:
        self.name = _require_text(self.name, owner="AgentTool", field_name="tool name")
        self.label = _require_text(self.label, owner="AgentTool", field_name="label")
        self.description = _require_text(
            self.description,
            owner="AgentTool",
            field_name="description",
        )
        if not isinstance(self.parameters, dict):
            raise TypeError("AgentTool parameters must be a dict")
        if not callable(self.execute):
            raise TypeError("AgentTool execute must be callable")
        if not isinstance(self.runtime_managed, bool):
            raise TypeError("AgentTool runtime_managed must be bool")
        self.parameters = deepcopy(self.parameters)

    def to_spec(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass(frozen=True)
class ToolRuntimeRequest:
    """One tool call entering the runtime safety pipeline."""

    tool_call_id: str
    name: str
    params: dict[str, Any]
    source: str = "agent"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_call_id",
            _require_text(
                self.tool_call_id,
                owner="ToolRuntimeRequest",
                field_name="tool_call_id",
            ),
        )
        object.__setattr__(
            self,
            "name",
            _require_text(
                self.name,
                owner="ToolRuntimeRequest",
                field_name="tool name",
            ),
        )
        if not isinstance(self.params, dict):
            raise TypeError("ToolRuntimeRequest params must be a dict")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(
            self,
            "source",
            _require_text(
                self.source,
                owner="ToolRuntimeRequest",
                field_name="source",
            ),
        )


@dataclass(frozen=True)
class ToolRuntimeResult:
    """The normalized result returned by the tool runtime pipeline."""

    result: AgentToolResult
    status: ToolResultStatus = "success"
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, ToolResult):
            raise TypeError("ToolRuntimeResult result must be AgentToolResult")
        status = ensure_tool_result_status(self.status)
        is_error = bool(self.is_error)
        if is_error and status == "success":
            status = "error"
        elif status != "success":
            is_error = True
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "is_error", is_error)
        if not isinstance(self.approved, bool):
            raise TypeError("ToolRuntimeResult approved must be bool")
        if self.approval_id is not None:
            object.__setattr__(
                self,
                "approval_id",
                _require_text(
                    self.approval_id,
                    owner="ToolRuntimeResult",
                    field_name="approval_id",
                ),
            )


def _require_text(value: object, *, owner: str, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{owner} {field_name} cannot be empty")
    return text


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "ToolExecuteFn",
    "ToolRuntimeRequest",
    "ToolRuntimeResult",
]
