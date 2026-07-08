from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from codepilot.sessions.contracts import SessionCommandRecord, SessionRunRecord
from codepilot.tools.contracts import ToolInterruption


ApprovalDecisionValue = Literal["approve", "deny"]


@dataclass(frozen=True)
class PromptSubmitted:
    text: str
    images: tuple[str, ...] | list[str] | None = None
    mode_hint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "prompt text"))
        object.__setattr__(self, "images", _clean_images(self.images))
        object.__setattr__(self, "mode_hint", _optional_text(self.mode_hint))


@dataclass(frozen=True)
class CommandSubmitted:
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "command text"))


@dataclass(frozen=True)
class ApprovalDecided:
    approval_id: str
    decision: ApprovalDecisionValue
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(self, "decision", _approval_decision(self.decision))
        object.__setattr__(self, "reason", _optional_text(self.reason) or "")


@dataclass(frozen=True)
class RunCancelled:
    reason: str = "user"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _optional_text(self.reason) or "user")


UserAction = PromptSubmitted | CommandSubmitted | ApprovalDecided | RunCancelled


@dataclass(frozen=True)
class ProgressFrame:
    event: dict[str, Any]
    kind: Literal["progress"] = field(default="progress", init=False)


@dataclass(frozen=True)
class ApprovalRequiredFrame:
    approval: ToolInterruption
    kind: Literal["approval_required"] = field(default="approval_required", init=False)


@dataclass(frozen=True)
class RunFinishedFrame:
    record: SessionRunRecord
    kind: Literal["run_finished"] = field(default="run_finished", init=False)


@dataclass(frozen=True)
class CommandFinishedFrame:
    record: SessionCommandRecord
    kind: Literal["command_finished"] = field(default="command_finished", init=False)


@dataclass(frozen=True)
class CancelledFrame:
    session_id: str
    cancelled: bool = False
    reason: str = "user"
    kind: Literal["cancelled"] = field(default="cancelled", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))


@dataclass(frozen=True)
class FailedFrame:
    error: Any
    kind: Literal["failed"] = field(default="failed", init=False)


RuntimeFrame = (
    ProgressFrame
    | ApprovalRequiredFrame
    | RunFinishedFrame
    | CommandFinishedFrame
    | CancelledFrame
    | FailedFrame
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text must be a string or None")
    return value.strip() or None


def _clean_images(value: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("images must be a sequence of strings")
    images: list[str] = []
    for item in value:
        images.append(_require_text(item, "image"))
    return tuple(images)


def _approval_decision(value: object) -> ApprovalDecisionValue:
    text = _require_text(value, "approval decision").lower()
    if text not in {"approve", "deny"}:
        raise ValueError(f"Unknown approval decision: {value}")
    return text  # type: ignore[return-value]


__all__ = [
    "ApprovalDecided",
    "ApprovalDecisionValue",
    "ApprovalRequiredFrame",
    "CancelledFrame",
    "CommandFinishedFrame",
    "CommandSubmitted",
    "FailedFrame",
    "ProgressFrame",
    "PromptSubmitted",
    "RunCancelled",
    "RunFinishedFrame",
    "RuntimeFrame",
    "UserAction",
]
