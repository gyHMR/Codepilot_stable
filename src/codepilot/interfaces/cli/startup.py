from __future__ import annotations

"""CLI startup view model.

The runtime owns live session state.  The renderer owns terminal formatting.
This module is the small adapter between them: it converts ``SessionStatus``
into the exact, stable fields the CLI startup banner and toolbar need.
"""

from dataclasses import dataclass, field

from codepilot.runtime.types import SessionStatus


_CLI_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})
_CLI_TASK_MODES = frozenset({"read", "edit", "plan"})


@dataclass(frozen=True)
class CliStartupState:
    """CLI 启动时需要展示的状态信息。"""

    version: str
    model_id: str
    workspace: str
    session_id: str
    permission_mode: str = "workspace-write"
    task_mode: str = "edit"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "version",
            _require_cli_startup_text(self.version, field_name="version"),
        )
        object.__setattr__(
            self,
            "model_id",
            _require_cli_startup_text(self.model_id, field_name="model_id"),
        )
        object.__setattr__(
            self,
            "workspace",
            _require_cli_startup_text(self.workspace, field_name="workspace"),
        )
        object.__setattr__(
            self,
            "session_id",
            _require_cli_startup_text(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "permission_mode",
            _ensure_cli_permission_mode(self.permission_mode),
        )
        object.__setattr__(
            self,
            "task_mode",
            _ensure_cli_task_mode(self.task_mode),
        )
        object.__setattr__(
            self,
            "warnings",
            _normalize_warning_texts(self.warnings),
        )


def build_startup_state(
    status: SessionStatus,
    warnings: list[str] | None = None,
) -> CliStartupState:
    """Build the CLI startup view model from runtime session status."""

    return CliStartupState(
        version="0.3",
        model_id=status.model_id,
        workspace=status.workspace,
        session_id=status.session_id,
        permission_mode=status.permission_mode,
        task_mode=status.task_mode,
        warnings=tuple(warnings) if warnings is not None else tuple(status.warnings or ()),
    )


def _require_cli_startup_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"CliStartupState.{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"CliStartupState.{field_name} cannot be empty")
    return text


def _ensure_cli_permission_mode(value: object) -> str:
    text = _require_cli_startup_text(value, field_name="permission_mode")
    if text not in _CLI_PERMISSION_MODES:
        raise ValueError(f"Unknown CLI permission_mode: {value}")
    return text


def _ensure_cli_task_mode(value: object) -> str:
    text = _require_cli_startup_text(value, field_name="task_mode")
    if text not in _CLI_TASK_MODES:
        raise ValueError(f"Unknown CLI task_mode: {value}")
    return text


def _normalize_warning_texts(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("CliStartupState.warnings must be a sequence of strings")
    warnings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("CliStartupState.warnings must contain strings")
        text = item.strip()
        if text:
            warnings.append(text)
    return tuple(warnings)


__all__ = ["CliStartupState", "build_startup_state"]
