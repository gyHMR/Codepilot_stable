from __future__ import annotations

"""Permission decisions for prepared tool calls."""

import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .contracts import ToolCallRequest
from .sandbox import classify_shell_command, command_mentions_internal_state

ToolDecisionKind = Literal["allow", "deny", "approval_required"]
ToolPermissionMode = Literal["read-only", "workspace-write", "ask"]

_DECISIONS = frozenset({"allow", "deny", "approval_required"})
_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})
_FORBIDDEN_MODEL_AUTH_KEYS = {
    "allow_dangerous",
    "bypass_approval",
    "ignore_workspace_boundary",
    "trusted",
    "sudo",
    "force_without_approval",
}


@dataclass(frozen=True)
class ToolDecision:
    kind: ToolDecisionKind
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if kind not in _DECISIONS:
            raise ValueError(f"Unknown tool decision: {self.kind}")
        object.__setattr__(self, "kind", cast(ToolDecisionKind, kind))
        object.__setattr__(self, "reason", str(self.reason).strip())
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
    """Decide allow / deny / approval_required for a prepared tool request."""

    permission_mode: ToolPermissionMode = "workspace-write"
    block_dangerous_bash: bool = True
    bash_allow_patterns: list[str] | None = None
    bash_block_patterns: list[str] | None = None
    require_approval_for_high_risk: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "permission_mode",
            _ensure_permission_mode(self.permission_mode),
        )
        _validate_patterns(self.bash_allow_patterns)
        _validate_patterns(self.bash_block_patterns)

    def decide(self, request: ToolCallRequest) -> ToolDecision:
        metadata = request.metadata
        details = _decision_details(request, permission_mode=self.permission_mode)
        attempted_auth = sorted(_FORBIDDEN_MODEL_AUTH_KEYS.intersection(request.arguments))
        if attempted_auth:
            return ToolDecision(
                "deny",
                "model_authorization_forbidden",
                {**details, "forbidden_params": attempted_auth},
            )
        if metadata is None:
            return ToolDecision("deny", "tool_metadata_missing", details)
        if not metadata.visible_in(request.current_mode):
            return ToolDecision("deny", "mode_scope_denied", details)
        if self.permission_mode == "read-only" and not metadata.read_only:
            return ToolDecision("deny", "read_only_permission_mode", details)
        if request.current_mode in {"read", "plan"} and not metadata.read_only:
            return ToolDecision("deny", "mode_scope_denied", details)
        if request.name == "bash":
            return self._decide_shell(request, details)
        if request.source == "approval_resume":
            return ToolDecision("allow", "approved_by_user", details)
        if metadata.requires_approval:
            return ToolDecision("approval_required", "tool_metadata_requires_approval", details)
        if self.require_approval_for_high_risk and metadata.risk_level == "high":
            return ToolDecision("approval_required", "high_risk_tool_requires_approval", details)
        if self.permission_mode == "ask" and not metadata.read_only:
            return ToolDecision("approval_required", "ask_mode", details)
        return ToolDecision("allow", "policy_allow", details)

    def _decide_shell(
        self,
        request: ToolCallRequest,
        details: dict[str, Any],
    ) -> ToolDecision:
        command = str(request.arguments.get("command", "")).strip()
        classification = classify_shell_command(command)
        shell_details = {**details, "command": command, "shell_class": classification}
        if _matches_any(command, self.bash_block_patterns):
            return ToolDecision("deny", "block_pattern", shell_details)
        if command_mentions_internal_state(command):
            return ToolDecision("deny", "internal_state_mutation_forbidden", shell_details)
        if classification == "high_risk" and self.block_dangerous_bash:
            return ToolDecision("deny", "dangerous_command", shell_details)
        if request.source == "approval_resume":
            return ToolDecision("allow", "approved_by_user", shell_details)
        if _matches_any(command, self.bash_allow_patterns):
            return ToolDecision("allow", "allow_pattern", shell_details)
        if classification in {"verification", "read_only"}:
            return ToolDecision("allow", f"{classification}_command", shell_details)
        if classification == "mutation":
            if self.permission_mode == "workspace-write":
                return ToolDecision("allow", "workspace_mutation_command", shell_details)
            return ToolDecision("approval_required", "ask_mode", shell_details)
        return ToolDecision("approval_required", "unknown_shell_command", shell_details)


def _decision_details(
    request: ToolCallRequest,
    *,
    permission_mode: str,
) -> dict[str, Any]:
    metadata = request.metadata
    if metadata is None:
        return {
            "tool": request.name,
            "current_mode": request.current_mode,
            "permission_mode": permission_mode,
        }
    capabilities = metadata.extra.get("capabilities", [])
    return {
        "tool": metadata.name,
        "category": metadata.category,
        "risk_level": metadata.risk_level,
        "scopes": list(metadata.scopes),
        "network_access": metadata.network_access,
        "credential_required": metadata.credential_required,
        "current_mode": request.current_mode,
        "permission_mode": permission_mode,
        "capabilities": list(capabilities) if isinstance(capabilities, (list, tuple)) else [],
    }


def _ensure_permission_mode(value: object) -> ToolPermissionMode:
    text = str(value).strip()
    if text not in _PERMISSION_MODES:
        raise ValueError(f"Unknown permission mode: {value}")
    return cast(ToolPermissionMode, text)


def _matches_any(text: str, patterns: list[str] | None) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns or [])


def _validate_patterns(patterns: list[str] | None) -> None:
    for pattern in patterns or []:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid permission regex {pattern!r}: {exc}") from exc


__all__ = [
    "PermissionPolicy",
    "ToolDecision",
    "ToolDecisionKind",
    "ToolPermissionMode",
]
