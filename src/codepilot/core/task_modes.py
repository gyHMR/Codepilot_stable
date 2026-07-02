from __future__ import annotations

"""Task mode policy for Codepilot's user-facing run behavior."""

from dataclasses import asdict, dataclass
from typing import Literal, cast


TaskMode = Literal["read", "edit", "plan"]
_TASK_MODES = frozenset({"read", "edit", "plan"})


@dataclass(frozen=True)
class TaskModePolicy:
    """Small, serializable policy derived from the selected task mode."""

    mode: TaskMode
    planner_required: bool = False
    read_only: bool = False
    default_step_title: str = "完成当前请求"
    guidance: str = ""

    def to_signal(self) -> dict[str, object]:
        return asdict(self)


def ensure_task_mode(value: object) -> TaskMode:
    text = value.strip() if isinstance(value, str) else ""
    if text not in _TASK_MODES:
        raise ValueError(f"Unknown task mode: {value}")
    return cast(TaskMode, text)


def policy_for_mode(mode: TaskMode | str) -> TaskModePolicy:
    normalized = ensure_task_mode(mode)
    if normalized == "read":
        return TaskModePolicy(
            mode="read",
            read_only=True,
            default_step_title="只读分析并回答当前请求",
            guidance=(
                "Read mode: inspect and explain only. Do not modify files or run "
                "state-changing commands."
            ),
        )
    if normalized == "plan":
        return TaskModePolicy(
            mode="plan",
            planner_required=True,
            default_step_title="按计划完成当前请求",
            guidance=(
                "Plan mode: follow the generated plan step by step, verify changes, "
                "and replan when evidence shows the current path is failing."
            ),
        )
    return TaskModePolicy(
        mode="edit",
        default_step_title="完成当前请求",
        guidance=(
            "Edit mode: make the smallest useful change, collect evidence, and "
            "verify when the workspace changes."
        ),
    )


__all__ = ["TaskMode", "TaskModePolicy", "ensure_task_mode", "policy_for_mode"]
