from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ApprovalDecision = Literal["approve", "deny"]
WebEventKind = Literal[
    "agent_event",
    "session_created",
    "session_state",
    "tool_approval_required",
    "error",
]


@dataclass(frozen=True)
class WebSessionRef:
    workspace_dir: str
    session_id: str | None = None


@dataclass(frozen=True)
class WebCreateSessionRequest:
    workspace_dir: str
    provider: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None
    session_id: str | None = None
    read_only_mode: bool | None = None
    load_workspace_resources: bool = True


@dataclass(frozen=True)
class WebSessionSummary:
    session_id: str


@dataclass(frozen=True)
class WebPromptRequest:
    session: WebSessionRef
    text: str
    images: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WebToolApproval:
    session_id: str
    tool_call_id: str
    decision: ApprovalDecision
    reason: str = ""


@dataclass(frozen=True)
class WebEventEnvelope:
    type: WebEventKind
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WebRouteSpec:
    method: str
    path: str
    description: str
