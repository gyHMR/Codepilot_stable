from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .shell_policy import classify_shell_command
from .types import ToolMetadata

READ_ONLY_TOOL_NAMES = {"read", "grep", "find", "ls", "workspace_status"}
MUTATING_TOOL_NAMES = {"write", "edit", "bash"}

ToolDecisionKind = Literal["allow", "deny", "approval_required"]
ToolPermissionMode = Literal["read-only", "workspace-write", "ask"]


@dataclass(frozen=True)
class ToolRequest:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"
    metadata: ToolMetadata | None = None


@dataclass(frozen=True)
class ToolDecision:
    kind: ToolDecisionKind
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.kind == "allow"

    @property
    def denied(self) -> bool:
        return self.kind == "deny"

    @property
    def requires_approval(self) -> bool:
        return self.kind == "approval_required"


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


@dataclass(frozen=True)
class PermissionPolicy:
    mode: ToolPermissionMode = "workspace-write"
    read_only: bool = False
    block_dangerous_bash: bool = True
    bash_allow_patterns: list[str] | None = None
    bash_block_patterns: list[str] | None = None
    require_approval_for_mutations: bool = False
    require_approval_for_high_risk: bool = True

    def __post_init__(self) -> None:
        validate_patterns(self.bash_allow_patterns)
        validate_patterns(self.bash_block_patterns)
        if self.read_only and self.mode != "read-only":
            object.__setattr__(self, "mode", "read-only")

    def decide(self, request: ToolRequest) -> ToolDecision:
        name = request.name
        metadata = request.metadata
        read_only = metadata.read_only if metadata is not None else name in READ_ONLY_TOOL_NAMES
        mutating = not read_only if metadata is not None else name in MUTATING_TOOL_NAMES
        details = _decision_details(metadata, mode=self.mode)

        forbidden_keys = {
            "allow_dangerous",
            "bypass_approval",
            "ignore_workspace_boundary",
            "trusted",
        }
        attempted = sorted(forbidden_keys.intersection(request.params))
        if attempted:
            return ToolDecision(
                "deny",
                "model_authorization_forbidden",
                {**details, "forbidden_params": attempted},
            )

        if self.mode == "read-only" and not read_only:
            return ToolDecision("deny", "read_only_mode", details)

        if name == "bash":
            command = str(request.params.get("command", "")).strip()
            classification = classify_shell_command(command)
            details = {
                **details,
                "command": command,
                "shell_class": classification,
                "capabilities": ["process.execute"],
            }
            blocklisted = matches_any_pattern(command, self.bash_block_patterns)
            if blocklisted:
                return ToolDecision("deny", "block_pattern", details)
            if classification == "high_risk" and self.block_dangerous_bash:
                return ToolDecision("deny", "dangerous_command", details)
            if matches_any_pattern(command, self.bash_allow_patterns):
                return ToolDecision("allow", "allow_pattern", details)
            if self.mode == "read-only":
                return ToolDecision("deny", "read_only_mode", details)
            if classification == "verification" and self.mode == "workspace-write":
                return ToolDecision("allow", "verification_command", details)
            if self.mode == "ask":
                return ToolDecision("approval_required", "ask_mode", details)
            if classification in {"unknown", "mutation"}:
                return ToolDecision(
                    "approval_required",
                    f"{classification}_shell_command",
                    details,
                )

        if metadata and metadata.requires_approval:
            return ToolDecision(
                "approval_required",
                "tool_metadata_requires_approval",
                details,
            )
        if metadata and self.require_approval_for_high_risk and metadata.risk_level == "high":
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


def is_dangerous_bash_command(command: str) -> bool:
    return classify_shell_command(command) == "high_risk"


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


__all__ = [
    "MUTATING_TOOL_NAMES",
    "PermissionPolicy",
    "READ_ONLY_TOOL_NAMES",
    "ToolDecision",
    "ToolDecisionKind",
    "ToolPermissionMode",
    "ToolRequest",
    "is_dangerous_bash_command",
    "matches_any_pattern",
    "validate_patterns",
]
