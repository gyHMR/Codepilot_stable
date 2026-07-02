from __future__ import annotations

"""Small replan decision helpers for task control."""

from codepilot.protocols import ToolResultMessage

from .verifier import verification_failure_detail


def repair_next_action(results: list[ToolResultMessage]) -> str:
    detail = verification_failure_detail(results)
    if detail:
        return (
            f"根据验证失败证据修复实现：{detail}。"
            "先读取失败断言和相关调用链，完成最小修复后重新运行同一验证。"
        )
    return "根据最新验证失败证据定位根因，完成最小修复后重新运行相关验证"


def should_propose_revert_after_repeated_failure(
    *,
    failure_count: int,
    has_change_sets: bool,
) -> bool:
    return failure_count >= 2 and has_change_sets


__all__ = ["repair_next_action", "should_propose_revert_after_repeated_failure"]
