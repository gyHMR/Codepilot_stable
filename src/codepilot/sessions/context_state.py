from __future__ import annotations

"""Session-scoped working context state.

This is intentionally not a second chat history. It only stores compact,
source-bound facts that help the runtime compile the next model context.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codepilot.protocols import RepositorySnapshot, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path


@dataclass
class ActiveFile:
    path: str
    role: str
    reason: str
    source_hash: str | None = None
    access_count: int = 1
    last_accessed_at: float = field(default_factory=time.time)


@dataclass
class FileSummary:
    path: str
    summary: str
    source_hash: str
    relevant_symbols: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    freshness: str = "fresh"


@dataclass
class ContextEvidence:
    kind: str
    content: str
    trust: str
    source: str
    source_hash: str | None = None
    workspace_fingerprint: str | None = None
    freshness: str = "unknown"
    path: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionContextState:
    workspace_dir: Path
    active_files: dict[str, ActiveFile] = field(default_factory=dict)
    file_summaries: dict[str, FileSummary] = field(default_factory=dict)
    evidence: list[ContextEvidence] = field(default_factory=list)
    last_repository_snapshot: RepositorySnapshot | None = None
    observed_tool_call_ids: set[str] = field(default_factory=set)

    def observe_tool_result(
        self,
        message: ToolResultMessage,
        *,
        repository_fingerprint: str | None = None,
    ) -> None:
        if message.tool_call_id and message.tool_call_id in self.observed_tool_call_ids:
            return
        if message.tool_call_id:
            self.observed_tool_call_ids.add(message.tool_call_id)
        details = message.details if isinstance(message.details, dict) else {}
        state = details.get("file_state")
        if not isinstance(state, dict):
            state = message.metadata.get("file_state")
        path = state.get("path") if isinstance(state, dict) else None
        source_hash = state.get("sha256") if isinstance(state, dict) else None

        paths = [str(item) for item in message.affected_paths]
        if isinstance(path, str) and path not in paths:
            paths.append(path)

        role = "target" if message.workspace_changed else "reference"
        for item in paths:
            self.touch_file(
                item,
                role=role,
                reason=f"{message.tool_name} tool result",
                source_hash=source_hash if item == path else None,
            )

        if message.workspace_changed:
            self.invalidate_paths(paths)
            self.invalidate_verification()

        text = _tool_result_text(message)
        if text:
            self.evidence.append(
                ContextEvidence(
                    kind="tool_result",
                    content=text,
                    trust="observed",
                    source=message.tool_name,
                    source_hash=source_hash if isinstance(source_hash, str) else None,
                    workspace_fingerprint=repository_fingerprint,
                    freshness="fresh",
                    path=path if isinstance(path, str) else None,
                )
            )
            self.evidence = self.evidence[-80:]

        if message.verification:
            self.evidence.append(
                ContextEvidence(
                    kind="verification",
                    content=str(message.verification),
                    trust="observed",
                    source=message.tool_name,
                    workspace_fingerprint=repository_fingerprint,
                    freshness="fresh",
                )
            )

    def touch_file(
        self,
        path: str,
        *,
        role: str,
        reason: str,
        source_hash: str | None = None,
    ) -> None:
        normalized = Path(path).as_posix()
        current = self.active_files.get(normalized)
        if current is None:
            self.active_files[normalized] = ActiveFile(
                path=normalized,
                role=role,
                reason=reason,
                source_hash=source_hash,
            )
            return
        current.access_count += 1
        current.last_accessed_at = time.time()
        current.reason = reason
        if role == "target" or current.role == "reference":
            current.role = role
        if source_hash:
            current.source_hash = source_hash

    def invalidate_paths(self, paths: list[str]) -> None:
        for path in paths:
            normalized = Path(path).as_posix()
            summary = self.file_summaries.get(normalized)
            if summary is not None:
                summary.freshness = "stale"
            for evidence in self.evidence:
                if evidence.path == normalized:
                    evidence.freshness = "stale"

    def invalidate_verification(self) -> None:
        for evidence in self.evidence:
            if evidence.kind == "verification":
                evidence.freshness = "stale"

    def validate_sources(self, repository_fingerprint: str) -> list[str]:
        stale: list[str] = []
        for path, summary in list(self.file_summaries.items()):
            state = file_state_for_path(self.workspace_dir, path)
            if not state.get("exists"):
                summary.freshness = "missing"
            elif state.get("sha256") != summary.source_hash:
                summary.freshness = "stale"
            else:
                summary.freshness = "fresh"
            if summary.freshness != "fresh":
                stale.append(f"file_summary:{path}:{summary.freshness}")

        for evidence in self.evidence:
            if (
                evidence.kind == "verification"
                and evidence.workspace_fingerprint
                and evidence.workspace_fingerprint != repository_fingerprint
            ):
                evidence.freshness = "stale"
            if evidence.freshness in {"stale", "missing"}:
                stale.append(f"evidence:{evidence.source}:{evidence.freshness}")
        return stale


def _tool_result_text(message: ToolResultMessage, *, limit: int = 1200) -> str:
    parts = [getattr(block, "text", "") for block in message.content]
    text = "".join(part for part in parts if part).strip()
    return text[:limit]


__all__ = [
    "ActiveFile",
    "ContextEvidence",
    "FileSummary",
    "SessionContextState",
]
