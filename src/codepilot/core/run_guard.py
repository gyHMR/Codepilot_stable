from __future__ import annotations

"""Final-answer run guard.

RunGuard checks observable process risks after the assistant has produced a
final answer. It does not decide whether the user's semantic goal is complete.
"""

from dataclasses import dataclass
from typing import Literal

from codepilot.protocols import AssistantMessage, RunSignalsSummary, TextContent

from .plan import RunMode


RunGuardAction = Literal["completed", "continue_with_instruction", "waiting_user", "stopped"]


@dataclass(frozen=True)
class RunGuardDecision:
    action: RunGuardAction
    reason: str = "final_answer"
    instruction: str = ""


class RunGuard:
    """Lightweight gate for final assistant answers."""

    def check(
        self,
        *,
        assistant: AssistantMessage,
        signals: RunSignalsSummary,
        mode: RunMode,
    ) -> RunGuardDecision:
        if signals.cancelled:
            return RunGuardDecision(action="stopped", reason="cancelled")
        if signals.tool_unavailable:
            return RunGuardDecision(action="stopped", reason="tool_unavailable")
        if signals.approval_required:
            return RunGuardDecision(action="waiting_user", reason="approval_required")
        if not final_answer_text(assistant).strip():
            return RunGuardDecision(
                action="continue_with_instruction",
                reason="empty_final_answer",
                instruction="你刚才没有给出用户可见的最终答复。请根据已有上下文给出清晰答复。",
            )
        if mode == "read" and signals.workspace_changed:
            return RunGuardDecision(
                action="stopped",
                reason="read_mode_workspace_changed",
            )
        if signals.verification_status == "failed":
            return RunGuardDecision(
                action="continue_with_instruction",
                reason="verification_failed",
                instruction=(
                    "刚才的验证结果明确失败。请只总结已经完成的工作、失败证据和剩余任务，"
                    "不要继续调用工具、不要扩大任务范围，也不要声明任务已经完成。"
                ),
            )
        return RunGuardDecision(action="completed", reason="final_answer")


def final_answer_text(message: AssistantMessage) -> str:
    chunks: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            chunks.append(block.text)
    return "".join(chunks)


__all__ = ["RunGuard", "RunGuardAction", "RunGuardDecision", "final_answer_text"]
