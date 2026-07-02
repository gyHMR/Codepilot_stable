from __future__ import annotations

"""Render Memory v2 records for prompt and CLI surfaces."""

from .records import MemoryRecord


def render_memory(record: MemoryRecord) -> str:
    label = {
        "correction": "Correction",
        "constraint": "Constraint",
        "decision": "Decision",
        "experience": "Experience",
    }[record.kind]
    return f"{label}: {record.text}"


__all__ = ["render_memory"]
