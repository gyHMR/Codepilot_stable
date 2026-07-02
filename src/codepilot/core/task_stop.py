from __future__ import annotations

"""Completion gate helpers for task control."""

from codepilot.protocols import TextContent, UserMessage

from .run_state import RunState
from .task_state import CompletionCheck, TaskState


def build_completion_check(task: TaskState, run: RunState) -> CompletionCheck:
    if task.blocked_step_titles():
        return CompletionCheck(
            satisfied=False,
            reason="blocked_steps",
            missing=["unblocked_steps"],
            can_continue=False,
        )

    if run.workspace_changed and not run.fresh_verification_passed:
        can_continue = task.completion_prompt_count == 0
        return CompletionCheck(
            satisfied=False,
            reason="modified_without_fresh_verification",
            missing=["fresh_verification"],
            can_continue=can_continue,
            unverified=not can_continue,
        )

    incomplete = task.pending_step_titles()
    if incomplete:
        return CompletionCheck(
            satisfied=False,
            reason="incomplete_steps",
            missing=incomplete,
            can_continue=False,
        )

    return CompletionCheck(satisfied=True, reason="all_steps_completed")


def completion_steering_message(check: CompletionCheck) -> UserMessage:
    text = (
        "工作区已经发生修改，但当前没有与最新工作区状态一致的成功验证。\n"
        "请运行最相关的测试或检查；如果环境无法验证，请明确记录原因和剩余风险。"
    )
    return UserMessage(
        content=[TextContent(text=text)],
        metadata={
            "task_completion_gate": {
                "reason": check.reason,
                "missing": list(check.missing),
            }
        },
    )


__all__ = ["build_completion_check", "completion_steering_message"]
