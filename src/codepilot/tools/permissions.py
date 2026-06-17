from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

READ_ONLY_TOOL_NAMES = {"read", "read_file", "grep", "find", "ls", "list_dir"}
MUTATING_TOOL_NAMES = {"write", "write_file", "edit", "bash"}

ToolDecisionKind = Literal["allow", "deny", "approval_required"]


@dataclass(frozen=True)
class ToolRequest:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"


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

    def decide(self, request: ToolRequest) -> ToolDecision:
        name = request.name
        if self.read_only and name not in READ_ONLY_TOOL_NAMES:
            return ToolDecision("deny", "read_only_mode", {"tool": name})

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

        if self.require_approval_for_mutations and name in MUTATING_TOOL_NAMES:
            return ToolDecision("approval_required", "mutation_requires_approval", {"tool": name})

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
