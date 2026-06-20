from __future__ import annotations

"""将记忆记录渲染为人类可读的文本格式。"""

from .records import MemoryRecord


def render_memory(record: MemoryRecord) -> str:
    """将记忆记录渲染为单行人类可读文本。"""
    content = record.content
    if record.kind == "task":
        parts = [f"Task goal: {content.get('goal', '')}"]
        for key, label in [
            ("constraints", "Constraints"),
            ("confirmed_findings", "Confirmed"),
            ("blocked_on", "Blocked"),
        ]:
            values = content.get(key)
            if isinstance(values, list) and values:
                parts.append(f"{label}: {'; '.join(str(item) for item in values[:5])}")
        if content.get("next_action"):
            parts.append(f"Next: {content['next_action']}")
        progress = content.get("task_progress")
        if isinstance(progress, dict):
            pending = progress.get("pending_steps")
            if isinstance(pending, list) and pending:
                parts.append(f"Pending: {'; '.join(str(item) for item in pending[:5])}")
        return " | ".join(parts)
    if record.kind == "file":
        return (
            f"File {content.get('path', '')}: {content.get('summary', '')} "
            f"(hash={next(iter(record.source_hashes.values()), '')[:12]})"
        ).strip()
    if record.kind == "failure":
        return (
            f"Failure lesson: {content.get('action', '')} -> "
            f"{content.get('failure_signature', '')}; "
            f"cause={content.get('cause') or 'unknown'}; "
            f"resolution={content.get('resolution') or 'not confirmed'}"
        )
    if record.kind == "decision":
        return f"Decision: {content.get('decision', '')}; rationale={content.get('rationale', '')}"
    return f"Project knowledge: {content.get('knowledge', '')}"


__all__ = ["render_memory"]
