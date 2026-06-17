from __future__ import annotations

"""Diff event placeholders for future file-change tracking."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileDiff:
    path: Path
    before: str
    after: str


class DiffRecorder:
    """No-op recorder used until file tools emit before/after snapshots."""

    def record(self, diff: FileDiff) -> None:
        _ = diff

    def flush(self) -> list[FileDiff]:
        return []
