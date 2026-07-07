from __future__ import annotations

"""Permission decisions for tool calls."""

import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from codepilot.protocols.tools import ToolMetadata

from .registry import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES
from .workspace import classify_shell_command


ToolDecisionKind = Literal["allow", "deny", "approval_required"]
ToolPermissionMode = Literal["read-only", "workspace-write", "ask"]
_TOOL_DECISION_KINDS = frozenset({"allow", "deny", "approval_required"})
_TOOL_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})
_FORBIDDEN_MODEL_AUTH_KEYS = {
    "allow_dangerous",
    "bypass_approval",
    "ignore_workspace_boundary",
    "trusted",
}


@dataclass(frozen=True)
class ToolRequest:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"
    metadata: ToolMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "tool name"))
        if not isinstance(self.params, dict):
            raise TypeError("ToolRequest params must be a dict")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "source", _require_text(self.source, "source"))


@dataclass(frozen=True)
class ToolDecision:
    kind: ToolDecisionKind
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _ensure_decision_kind(self.kind))
        object.__setattr__(self, "reason", _clean_text(self.reason))
        if not isinstance(self.details, dict):
            raise TypeError("ToolDecision details must be a dict")
        object.__setattr__(self, "details", dict(self.details))

    @property
    def allowed(self) -> bool:
        return self.kind == "allow"

    @property
    def denied(self) -> bool:
        return self.kind == "deny"

    @property
    def requires_approval(self) -> bool:
        return self.kind == "approval_required"


@dataclass(frozen=True)
class PermissionPolicy:
    """Decide whether a tool call may execute, must ask, or is denied."""

    mode: ToolPermissionMode = "workspace-write"
    read_only: bool = False
    block_dangerous_bash: bool = True
    bash_allow_patterns: list[str] | None = None
    bash_block_patterns: list[str] | None = None
    require_approval_for_mutations: bool = False
    require_approval_for_high_risk: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _ensure_permission_mode(self.mode))
        validate_patterns(self.bash_allow_patterns)
        validate_patterns(self.bash_block_patterns)
        if self.read_only and self.mode != "read-only":
            object.__setattr__(self, "mode", "read-only")

    def decide(self, request: ToolRequest) -> ToolDecision:
        details = _decision_details(request.metadata, mode=self.mode)

        attempted_auth = sorted(_FORBIDDEN_MODEL_AUTH_KEYS.intersection(request.params))
        if attempted_auth:
            return ToolDecision(
                "deny",
                "model_authorization_forbidden",
                {**details, "forbidden_params": attempted_auth},
            )

        read_only = _is_read_only(request)
        mutating = _is_mutating(request, read_only=read_only)
        if self.mode == "read-only" and mutating:
            return ToolDecision("deny", "read_only_mode", details)

        if request.name == "bash":
            return self._decide_shell(request, details)

        if request.metadata and request.metadata.requires_approval:
            return ToolDecision(
                "approval_required",
                "tool_metadata_requires_approval",
                details,
            )
        if (
            request.metadata
            and self.require_approval_for_high_risk
            and request.metadata.risk_level == "high"
        ):
            return ToolDecision(
                "approval_required",
                "high_risk_tool_requires_approval",
                details,
            )
        if self.mode == "ask" and mutating:
            return ToolDecision("approval_required", "ask_mode", details)
        if self.require_approval_for_mutations and mutating:
            return ToolDecision("approval_required", "mutation_requires_approval", details)
        return ToolDecision("allow", "policy_allow", details)

    def _decide_shell(
        self,
        request: ToolRequest,
        details: dict[str, Any],
    ) -> ToolDecision:
        command = str(request.params.get("command", "")).strip()
        classification = classify_shell_command(command)
        shell_details = {
            **details,
            "command": command,
            "shell_class": classification,
            "capabilities": ["process.execute"],
        }

        if matches_any_pattern(command, self.bash_block_patterns):
            return ToolDecision("deny", "block_pattern", shell_details)
        if classification == "high_risk" and self.block_dangerous_bash:
            return ToolDecision("deny", "dangerous_command", shell_details)
        if matches_any_pattern(command, self.bash_allow_patterns):
            return ToolDecision("allow", "allow_pattern", shell_details)
        if classification == "verification" and self.mode in {"workspace-write", "ask"}:
            return ToolDecision("allow", "verification_command", shell_details)
        if classification == "read_only" and self.mode in {"workspace-write", "ask"}:
            return ToolDecision("allow", "safe_read_only_command", shell_details)
        if classification == "mutation" and self.mode == "workspace-write":
            return ToolDecision("allow", "workspace_mutation_command", shell_details)
        if self.mode == "ask":
            return ToolDecision("approval_required", "ask_mode", shell_details)
        if classification == "unknown":
            return ToolDecision(
                "approval_required",
                "unknown_shell_command",
                shell_details,
            )
        return ToolDecision("allow", "policy_allow", shell_details)


def matches_any_pattern(text: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    return any(re.search(pattern, text) is not None for pattern in patterns)


def validate_patterns(patterns: list[str] | None) -> None:
    for pattern in patterns or []:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid permission regex {pattern!r}: {exc}") from exc


def is_dangerous_bash_command(command: str) -> bool:
    return classify_shell_command(command) == "high_risk"


def _is_read_only(request: ToolRequest) -> bool:
    if request.metadata is not None:
        return request.metadata.read_only
    return request.name in READ_ONLY_TOOL_NAMES


def _is_mutating(request: ToolRequest, *, read_only: bool) -> bool:
    if request.metadata is not None:
        return not read_only
    return request.name in MUTATING_TOOL_NAMES


def _decision_details(
    metadata: ToolMetadata | None,
    *,
    mode: ToolPermissionMode,
) -> dict[str, Any]:
    if metadata is None:
        return {"policy_mode": mode, "capabilities": []}
    capabilities = metadata.extra.get("capabilities", [])
    return {
        "tool": metadata.name,
        "category": metadata.category,
        "risk_level": metadata.risk_level,
        "resource_scope": list(metadata.resource_scope),
        "network_access": metadata.network_access,
        "credential_required": metadata.credential_required,
        "policy_mode": mode,
        "capabilities": list(capabilities) if isinstance(capabilities, (list, tuple)) else [],
    }


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_text(value: object, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"Tool policy {field_name} cannot be empty")
    return text


def _ensure_decision_kind(value: object) -> ToolDecisionKind:
    text = _clean_text(value)
    if text not in _TOOL_DECISION_KINDS:
        raise ValueError(f"Unknown tool decision kind: {value}")
    return cast(ToolDecisionKind, text)


def _ensure_permission_mode(value: object) -> ToolPermissionMode:
    text = _clean_text(value)
    if text not in _TOOL_PERMISSION_MODES:
        raise ValueError(f"Unknown tool permission mode: {value}")
    return cast(ToolPermissionMode, text)


__all__ = [
    "PermissionPolicy",
    "ToolDecision",
    "ToolDecisionKind",
    "ToolPermissionMode",
    "ToolRequest",
    "is_dangerous_bash_command",
    "matches_any_pattern",
    "validate_patterns",
]
