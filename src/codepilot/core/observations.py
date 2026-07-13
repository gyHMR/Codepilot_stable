from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from codepilot.protocols import AssistantMessage, ToolCall
from codepilot.tools.results import ToolResult

from .commands import CoreCommand, is_core_command
from .state import FailureRecord


ModelObservationStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class ModelObservation:
    observation_id: str
    message: AssistantMessage | None = None
    status: ModelObservationStatus = "completed"
    error: FailureRecord | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _required_text(self.observation_id, "observation_id"),
        )
        if self.status not in {"completed", "failed"}:
            raise ValueError(f"Unknown model observation status: {self.status}")
        if self.message is not None and not isinstance(self.message, AssistantMessage):
            raise TypeError("message must be AssistantMessage or None")
        if self.error is not None and not isinstance(self.error, FailureRecord):
            raise TypeError("error must be FailureRecord or None")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed model observation requires an error")


@dataclass(frozen=True)
class ToolBatchObservation:
    observation_id: str
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()
    commands: tuple[CoreCommand, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _required_text(self.observation_id, "observation_id"),
        )
        calls = tuple(self.calls)
        results = tuple(self.results)
        commands = tuple(self.commands)
        if any(not isinstance(item, ToolCall) for item in calls):
            raise TypeError("calls must contain ToolCall values")
        if any(not isinstance(item, ToolResult) for item in results):
            raise TypeError("results must contain ToolResult values")
        if any(not is_core_command(item) for item in commands):
            raise TypeError("commands must contain CoreCommand values")
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "commands", commands)


@dataclass(frozen=True)
class UserInputObservation:
    observation_id: str
    text: str
    current_goal: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _required_text(self.observation_id, "observation_id"),
        )
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        goal = str(self.current_goal).strip() if self.current_goal is not None else None
        object.__setattr__(self, "current_goal", goal or None)


@dataclass(frozen=True)
class CoreCommandObservation:
    observation_id: str
    command: CoreCommand

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _required_text(self.observation_id, "observation_id"),
        )
        if not is_core_command(self.command):
            raise TypeError("command must be a CoreCommand")


@dataclass(frozen=True)
class CancellationObservation:
    observation_id: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _required_text(self.observation_id, "observation_id"),
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))


CoreObservation: TypeAlias = (
    ModelObservation
    | ToolBatchObservation
    | UserInputObservation
    | CoreCommandObservation
    | CancellationObservation
)


def _required_text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


__all__ = [
    "CancellationObservation",
    "CoreCommandObservation",
    "CoreObservation",
    "ModelObservation",
    "ModelObservationStatus",
    "ToolBatchObservation",
    "UserInputObservation",
]
