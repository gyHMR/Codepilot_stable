from __future__ import annotations

"""Helpers for turning context freshness checks into Agent steering messages."""

from codepilot.protocols import TextContent, UserMessage
from codepilot.sessions.persistence import FreshnessResult


def build_context_freshness_notice(
    freshness: FreshnessResult,
) -> UserMessage | None:
    """Build a steering message when previous run evidence may be stale.

    ``RunStore.evaluate_freshness()`` decides whether tracked files still match
    the workspace.  This helper owns the user/agent-facing wording, keeping the
    Session method focused on when to evaluate freshness and how to persist the
    audit event.
    """

    if not freshness.requires_steering():
        return None
    payload = freshness.to_event_payload()
    lines = [
        "[Context Freshness]",
        f"status={freshness.status}",
    ]
    if freshness.changed_paths:
        lines.append("changed_files=" + ", ".join(freshness.changed_paths))
    if freshness.missing_paths:
        lines.append("missing_files=" + ", ".join(freshness.missing_paths))
    lines.append("旧工具结果可能已过期；依赖这些文件前请重新读取。")
    return UserMessage(
        content=[TextContent(text="\n".join(lines))],
        metadata={"context_freshness": payload},
    )


__all__ = ["build_context_freshness_notice"]
