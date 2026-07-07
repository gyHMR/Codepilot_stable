from __future__ import annotations

"""Approval request objects and the default deferred approval provider."""

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from codepilot.protocols.tools import ToolMetadata

from .authoring import ToolRuntimeRequest
from .policy import ToolDecision


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reason: str = ""
    approval_id: str | None = None
    deferred: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise TypeError("ApprovalDecision approved must be bool")
        if not isinstance(self.deferred, bool):
            raise TypeError("ApprovalDecision deferred must be bool")
        object.__setattr__(self, "reason", _clean_text(self.reason))
        if self.approval_id is not None:
            object.__setattr__(
                self,
                "approval_id",
                _require_text(self.approval_id, "approval_id"),
            )


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    tool_call_id: str
    tool_name: str
    params_preview: dict[str, object]
    reason: str
    risk_level: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "approval_id",
            _require_text(self.approval_id, "approval_id"),
        )
        object.__setattr__(
            self,
            "tool_call_id",
            _require_text(self.tool_call_id, "tool_call_id"),
        )
        object.__setattr__(
            self,
            "tool_name",
            _require_text(self.tool_name, "tool_name"),
        )
        if not isinstance(self.params_preview, dict):
            raise TypeError("ApprovalRequest params_preview must be a dict")
        object.__setattr__(self, "params_preview", dict(self.params_preview))
        object.__setattr__(self, "reason", _clean_text(self.reason))
        object.__setattr__(
            self,
            "risk_level",
            _require_text(self.risk_level, "risk_level"),
        )
        object.__setattr__(
            self,
            "capabilities",
            tuple(_clean_unique_items(self.capabilities)),
        )


class ApprovalProvider(Protocol):
    async def request_approval(
        self,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
    ) -> ApprovalDecision:
        ...


class DeferredApprovalProvider:
    """Default provider: pause execution and let the interface resume it later."""

    async def request_approval(
        self,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
    ) -> ApprovalDecision:
        approval = build_approval_request(request, metadata, decision)
        return ApprovalDecision(
            approved=False,
            reason=decision.reason,
            approval_id=approval.approval_id,
            deferred=True,
        )


def build_approval_request(
    request: ToolRuntimeRequest,
    metadata: ToolMetadata | None,
    decision: ToolDecision,
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=f"approval_{uuid.uuid4().hex[:12]}",
        tool_call_id=request.tool_call_id,
        tool_name=request.name,
        params_preview=_params_preview(request.name, request.params),
        reason=decision.reason,
        risk_level=_risk_level(metadata, decision),
        capabilities=tuple(
            str(item)
            for item in decision.details.get("capabilities", [])
            if isinstance(item, str)
        ),
    )


def _risk_level(metadata: ToolMetadata | None, decision: ToolDecision) -> str:
    if decision.details.get("risk_level"):
        return str(decision.details["risk_level"])
    if metadata is not None:
        return metadata.risk_level
    return "medium"


def _params_preview(name: str, params: dict[str, Any]) -> dict[str, object]:
    if name == "write":
        return {
            "path": str(params.get("path", ""))[:300],
            "content_chars": len(str(params.get("content", ""))),
            "overwrite": bool(params.get("overwrite", True)),
        }
    if name == "edit":
        return {
            "path": str(params.get("path", ""))[:300],
            "old_text": str(params.get("old_text", ""))[:160],
            "new_text": str(params.get("new_text", ""))[:160],
            "replace_all": bool(params.get("replace_all", False)),
        }
    if name == "bash":
        return {
            "command": str(params.get("command", ""))[:2000],
            "cwd": str(params.get("cwd", "."))[:300],
            "timeout_seconds": params.get("timeout_seconds", 30),
        }
    return {
        str(key): (
            "[REDACTED]"
            if any(
                marker in str(key).upper()
                for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL", "COOKIE")
            )
            else _safe_preview(value)
        )
        for key, value in list(params.items())[:12]
    }


def _safe_preview(value: object) -> object:
    if isinstance(value, str):
        return value[:300]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return f"<{type(value).__name__}>"


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_text(value: object, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"Approval {field_name} cannot be empty")
    return text


def _clean_unique_items(values: tuple[str, ...]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


__all__ = [
    "ApprovalDecision",
    "ApprovalProvider",
    "ApprovalRequest",
    "DeferredApprovalProvider",
    "build_approval_request",
]
