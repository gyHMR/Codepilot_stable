from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .types import ToolMetadata

READ_ONLY_TOOL_NAMES = {"read", "grep", "find", "ls"}
MUTATING_TOOL_NAMES = {"write", "edit", "bash"}

ToolDecisionKind = Literal["allow", "deny", "approval_required"]


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
    for pattern in patterns:
        try:
            if re.search(pattern, text):
                return True
        except re.error:
            continue
    return False


@dataclass(frozen=True)
class PermissionPolicy:
    read_only: bool = False
    block_dangerous_bash: bool = True
    bash_allow_patterns: list[str] | None = None
    bash_block_patterns: list[str] | None = None
    require_approval_for_mutations: bool = False
    require_approval_for_high_risk: bool = True

    def decide(self, request: ToolRequest) -> ToolDecision:
        name = request.name
        metadata = request.metadata
        read_only = metadata.read_only if metadata is not None else name in READ_ONLY_TOOL_NAMES
        mutating = not read_only if metadata is not None else name in MUTATING_TOOL_NAMES
        if self.read_only and not read_only:
            return ToolDecision(
                "deny",
                "read_only_mode",
                {
                    "tool": name,
                    "category": metadata.category if metadata else "",
                    "resource_scope": list(metadata.resource_scope) if metadata else [],
                },
            )

        if name == "bash":
            command = str(request.params.get("command", "")).strip()
            allowlisted = matches_any_pattern(command, self.bash_allow_patterns)
            blocklisted = matches_any_pattern(command, self.bash_block_patterns)
            if blocklisted and not allowlisted:
                return ToolDecision("deny", "block_pattern", {"command": command})
            if self.block_dangerous_bash and is_dangerous_bash_command(command) and not allowlisted:
                if request.params.get("allow_dangerous"):
                    return ToolDecision("allow", "dangerous_command_allowed", {"command": command})
                return ToolDecision("deny", "dangerous_command", {"command": command})

        if metadata and metadata.requires_approval:
            return ToolDecision(
                "approval_required",
                "tool_metadata_requires_approval",
                _metadata_details(metadata),
            )

        if metadata and self.require_approval_for_high_risk and metadata.risk_level == "high":
            return ToolDecision(
                "approval_required",
                "high_risk_tool_requires_approval",
                _metadata_details(metadata),
            )

        if self.require_approval_for_mutations and mutating:
            details = _metadata_details(metadata) if metadata else {"tool": name}
            return ToolDecision("approval_required", "mutation_requires_approval", details)

        return ToolDecision("allow")


def is_dangerous_bash_command(command: str) -> bool:
    text = command.lower()
    patterns = [
        "rm -rf",
        "rm -r ",
        "rm -fr",
        "del /f",
        "rmdir /s",
        "format ",
        "mkfs",
        "shutdown",
        "reboot",
        "remove-item -recurse",
    ]
    return any(pattern in text for pattern in patterns)


def _metadata_details(metadata: ToolMetadata | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    return {
        "tool": metadata.name,
        "category": metadata.category,
        "risk_level": metadata.risk_level,
        "resource_scope": list(metadata.resource_scope),
        "network_access": metadata.network_access,
        "credential_required": metadata.credential_required,
    }
