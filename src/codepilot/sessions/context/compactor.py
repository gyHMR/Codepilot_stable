from __future__ import annotations

# 新手导读：这里定义 critical 压力下的 LLM compact 端口数据。
# 关注点：真正的模型调用由注入的 compactor 实现，本模块只放小型 DTO。

"""Context compaction port contracts."""

from dataclasses import dataclass, field
from typing import Any

from codepilot.protocols import ContextArtifactRef, ContextPressure


@dataclass(frozen=True)
class ContextCompactRequest:
    session_id: str
    goal: str
    pressure: ContextPressure
    task_state_lines: list[str] = field(default_factory=list)
    working_set_lines: list[str] = field(default_factory=list)
    memory_lines: list[str] = field(default_factory=list)
    conversation_lines: list[str] = field(default_factory=list)
    artifact_refs: list[ContextArtifactRef] = field(default_factory=list)
    token_budget: int = 0


@dataclass(frozen=True)
class ContextCompactResult:
    recovery_summary: str = ""
    task_state_lines: list[str] = field(default_factory=list)
    working_set_lines: list[str] = field(default_factory=list)
    memory_lines: list[str] = field(default_factory=list)
    conversation_lines: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "recovery_summary",
            _clean_text(self.recovery_summary, limit=4000),
        )
        for field_name in (
            "task_state_lines",
            "working_set_lines",
            "memory_lines",
            "conversation_lines",
            "evidence_refs",
        ):
            object.__setattr__(
                self,
                field_name,
                _clean_list(getattr(self, field_name), limit=1000),
            )
        if not isinstance(self.raw, dict):
            object.__setattr__(self, "raw", {})


def _clean_text(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


def _clean_list(values: object, *, limit: int) -> list[str]:
    if not isinstance(values, list | tuple):
        return []
    return [
        text
        for item in values
        if (text := _clean_text(item, limit=limit))
    ]


__all__ = ["ContextCompactRequest", "ContextCompactResult"]
