from __future__ import annotations

# 新手导读：PermissionPolicy 在工具执行前做硬权限决策，防止模型用参数给自己授权。
# 关注点：重点看 read-only、ask、bash 分类和 forbidden_keys 的处理顺序。

import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .contracts import ToolMetadata
from .metadata import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES
from .shell_safety import classify_shell_command

# 工具决策类型：允许/拒绝/需要审批
ToolDecisionKind = Literal["allow", "deny", "approval_required"]
# 工具权限模式：只读/工作区写入/询问
ToolPermissionMode = Literal["read-only", "workspace-write", "ask"]
_TOOL_DECISION_KINDS = frozenset({"allow", "deny", "approval_required"})
_TOOL_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})


@dataclass(frozen=True)
class ToolRequest:
    """工具权限检查请求。"""
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"
    metadata: ToolMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_policy_text(self.name, field_name="tool name"),
        )
        if not isinstance(self.params, dict):
            raise TypeError("ToolRequest params must be a dict")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(
            self,
            "source",
            _require_policy_text(self.source, field_name="source"),
        )


@dataclass(frozen=True)
class ToolDecision:
    """工具权限决策结果。"""
    kind: ToolDecisionKind
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _ensure_decision_kind(self.kind))
        object.__setattr__(self, "reason", _clean_policy_text(self.reason))
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
    """工具权限策略：根据模式和规则决定工具调用的允许/拒绝/审批。"""
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
        """根据权限策略对工具请求做出决策。"""
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


def _clean_policy_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_policy_text(value: object, *, field_name: str) -> str:
    text = _clean_policy_text(value)
    if not text:
        raise ValueError(f"Tool policy {field_name} cannot be empty")
    return text


def _ensure_decision_kind(value: object) -> ToolDecisionKind:
    text = _clean_policy_text(value)
    if text not in _TOOL_DECISION_KINDS:
        raise ValueError(f"Unknown tool decision kind: {value}")
    return cast(ToolDecisionKind, text)


def _ensure_permission_mode(value: object) -> ToolPermissionMode:
    text = _clean_policy_text(value)
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
