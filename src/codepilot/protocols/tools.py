from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from .content import ImageContent, TextContent


ToolRiskLevel = Literal["low", "medium", "high"]
ToolResultStatus = Literal["success", "error", "denied", "approval_required", "cancelled"]


@dataclass
class Tool:
    """Model-visible tool specification."""

    name: str
    description: str
    parameters: dict[str, Any]


ToolSpec = Tool


@dataclass
class ToolCall:
    """Normalized tool call requested by a model."""

    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    index: int | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ToolResultBlock = Union[TextContent, ImageContent]


@dataclass
class ToolResult:
    """Normalized tool execution result."""

    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultBlock] = field(default_factory=list)
    status: ToolResultStatus = "success"
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None
    details: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.is_error and self.status == "success":
            self.status = "error"
        elif self.status != "success":
            self.is_error = True


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    category: str
    read_only: bool
    concurrency_safe: bool
    exclusive: bool
    requires_approval: bool
    risk_level: ToolRiskLevel
    resource_scope: tuple[str, ...]
    network_access: bool = False
    credential_required: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Tool",
    "ToolCall",
    "ToolMetadata",
    "ToolResult",
    "ToolResultBlock",
    "ToolResultStatus",
    "ToolRiskLevel",
    "ToolSpec",
]
