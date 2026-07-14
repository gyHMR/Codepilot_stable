from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from codepilot.protocols import Message, RepositoryDelta, RepositorySnapshot, ToolResultMessage
from codepilot.sessions.workspace import build_repository_bootstrap, file_state_for_path


@dataclass
class ActiveFile:
    path: str
    role: str
    reason: str
    source_hash: str | None = None
    freshness: str = "unknown"
    access_count: int = 1
    last_accessed_at: float = field(default_factory=time.time)


@dataclass
class ContextEvidence:
    evidence_id: str
    kind: str
    summary: str
    source_ref: str
    source_tool_call_id: str
    affected_paths: tuple[str, ...] = ()
    status: str = "success"
    artifact_ref: str | None = None
    source_hash: str | None = None
    workspace_fingerprint: str | None = None
    freshness: str = "unknown"
    created_at: float = field(default_factory=time.time)


@dataclass
class ContextState:
    workspace_dir: Path
    active_files: dict[str, ActiveFile] = field(default_factory=dict)
    evidence: dict[str, ContextEvidence] = field(default_factory=dict)
    last_repository_snapshot: RepositorySnapshot | None = None
    observed_tool_call_ids: set[str] = field(default_factory=set)
    max_active_files: int = 40

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir)

    def observe_messages(
        self,
        messages: tuple[Message, ...],
        *,
        repository_fingerprint: str | None,
    ) -> None:
        for message in messages:
            if not isinstance(message, ToolResultMessage):
                continue
            self.observe_tool_result(
                message,
                repository_fingerprint=repository_fingerprint,
            )

    def observe_tool_result(
        self,
        message: ToolResultMessage,
        *,
        repository_fingerprint: str | None,
    ) -> None:
        if message.tool_call_id in self.observed_tool_call_ids:
            return
        if message.tool_call_id:
            self.observed_tool_call_ids.add(message.tool_call_id)
        source_ref = _message_source_ref(message)
        text = _tool_text(message)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        paths = tuple(dict.fromkeys(_message_paths(message)))
        if message.workspace_changed:
            self.invalidate_paths(paths)
            self.invalidate_verification()
        for path in paths:
            self.touch_file(
                path,
                role="target" if message.workspace_changed or _is_read(message) else "reference",
                reason=f"{message.tool_name} tool result",
                source_hash=_file_hash_from_message(message, path),
            )
        kind = (
            "error"
            if message.is_error or message.status != "success"
            else "mutation"
            if message.workspace_changed
            else "verification"
            if message.verification
            else "observation"
        )
        summary = _evidence_summary(message, text)
        self.evidence[source_ref] = ContextEvidence(
            evidence_id=f"evidence:{message.tool_call_id or digest or len(self.evidence)}",
            kind=kind,
            summary=summary,
            source_ref=source_ref,
            source_tool_call_id=message.tool_call_id,
            affected_paths=paths,
            status=str(message.status),
            artifact_ref=_optional_text(message.metadata.get("artifact_ref")),
            source_hash=digest,
            workspace_fingerprint=repository_fingerprint,
            freshness="fresh",
        )
        if len(self.evidence) > 80:
            oldest = min(
                self.evidence.values(),
                key=lambda item: (item.created_at, item.evidence_id),
            )
            self.evidence.pop(oldest.source_ref, None)

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
                freshness="fresh" if source_hash else "unknown",
            )
        else:
            current.access_count += 1
            current.last_accessed_at = time.time()
            current.reason = reason
            if role == "target" or current.role == "reference":
                current.role = role
            if source_hash:
                current.source_hash = source_hash
                current.freshness = "fresh"
        self._prune_active_files()

    def invalidate_paths(self, paths: tuple[str, ...] | list[str]) -> None:
        normalized = {Path(path).as_posix() for path in paths}
        for path in normalized:
            active = self.active_files.get(path)
            if active is not None:
                active.freshness = "stale"
        for evidence in self.evidence.values():
            if normalized.intersection(evidence.affected_paths):
                evidence.freshness = "stale"

    def invalidate_verification(self) -> None:
        for evidence in self.evidence.values():
            if evidence.kind == "verification":
                evidence.freshness = "stale"

    def refresh_freshness(self, repository_fingerprint: str) -> tuple[str, ...]:
        stale: list[str] = []
        for path, active in self.active_files.items():
            state = file_state_for_path(self.workspace_dir, path)
            if not state.get("exists"):
                active.freshness = "missing"
            elif active.source_hash and state.get("sha256") != active.source_hash:
                active.freshness = "stale"
            elif active.source_hash:
                active.freshness = "fresh"
            else:
                active.freshness = "unknown"
            if active.freshness in {"stale", "missing"}:
                stale.append(f"active_file:{path}:{active.freshness}")
        for evidence in self.evidence.values():
            if (
                evidence.kind == "verification"
                and evidence.workspace_fingerprint
                and evidence.workspace_fingerprint != repository_fingerprint
            ):
                evidence.freshness = "stale"
            if evidence.freshness in {"stale", "missing"}:
                stale.append(f"{evidence.evidence_id}:{evidence.freshness}")
        return tuple(sorted(stale))

    def _prune_active_files(self) -> None:
        if len(self.active_files) <= self.max_active_files:
            return
        ranked = sorted(
            self.active_files.values(),
            key=lambda item: (
                item.role == "target",
                item.freshness == "fresh",
                item.access_count,
                item.last_accessed_at,
                item.path,
            ),
            reverse=True,
        )
        self.active_files = {item.path: item for item in ranked[: self.max_active_files]}


class RepositoryTracker:
    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()

    def snapshot(self) -> RepositorySnapshot:
        bootstrap = build_repository_bootstrap(self.workspace_dir)
        git_status = [
            line
            for line in _git_lines(self.workspace_dir, ["status", "--porcelain"])
            if len(line) < 4 or not _is_internal_path(line[3:].strip())
        ]
        top_level_entries = [
            path for path in bootstrap.top_level_entries if not _is_internal_path(path)
        ]
        instruction_hashes = {
            path: _file_sha256(self.workspace_dir / path)
            for path in bootstrap.instruction_files
            if (self.workspace_dir / path).is_file()
        }
        dirty_hashes = {
            path: _file_sha256(self.workspace_dir / path)
            for path in _status_paths(git_status)
            if (self.workspace_dir / path).is_file()
        }
        payload = [
            bootstrap.workspace_root,
            bootstrap.project_type or "",
            *bootstrap.manifest_files,
            *top_level_entries,
            *bootstrap.test_directories,
            *(f"{key}:{value}" for key, value in sorted(instruction_hashes.items())),
            *(f"{key}:{value}" for key, value in sorted(dirty_hashes.items())),
            bootstrap.git.branch if bootstrap.git and bootstrap.git.branch else "",
            bootstrap.git.head_sha if bootstrap.git and bootstrap.git.head_sha else "",
            *git_status,
        ]
        fingerprint = hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()
        return RepositorySnapshot(
            workspace_root=bootstrap.workspace_root,
            project_type=bootstrap.project_type,
            manifest_files=list(bootstrap.manifest_files),
            top_level_entries=top_level_entries,
            test_directories=list(bootstrap.test_directories),
            instruction_files=list(bootstrap.instruction_files),
            branch=bootstrap.git.branch if bootstrap.git else None,
            head_sha=bootstrap.git.head_sha if bootstrap.git else None,
            git_status=git_status,
            fingerprint=fingerprint,
            instruction_hashes=instruction_hashes,
            dirty_path_hashes=dirty_hashes,
        )

    def refresh(
        self,
        previous: RepositorySnapshot | None,
    ) -> tuple[RepositorySnapshot, RepositoryDelta]:
        current = self.snapshot()
        return current, compare_snapshots(previous, current)


def compare_snapshots(
    previous: RepositorySnapshot | None,
    current: RepositorySnapshot,
) -> RepositoryDelta:
    if previous is None:
        return RepositoryDelta()
    old_status = _status_map(previous.git_status)
    new_status = _status_map(current.git_status)
    modified = sorted(
        path
        for path, status in new_status.items()
        if status != "??"
        and not _is_internal_path(path)
        and (
            old_status.get(path) != status
            or previous.dirty_path_hashes.get(path) != current.dirty_path_hashes.get(path)
        )
    )
    old_paths = set(previous.top_level_entries)
    new_paths = set(current.top_level_entries)
    return RepositoryDelta(
        added_paths=sorted(
            (new_paths - old_paths)
            | {path for path, status in new_status.items() if status == "??"}
        ),
        modified_paths=modified,
        deleted_paths=sorted(
            {
                *(
                    path
                    for path, status in new_status.items()
                    if "D" in status and not _is_internal_path(path)
                ),
                *(path for path in old_paths - new_paths if not _is_internal_path(path)),
            }
        ),
        branch_changed=previous.branch != current.branch,
        head_changed=previous.head_sha != current.head_sha,
        instructions_changed=previous.instruction_hashes != current.instruction_hashes,
    )


def repository_summary(snapshot: RepositorySnapshot, delta: RepositoryDelta) -> str:
    lines = [
        f"Repository fingerprint: {snapshot.fingerprint[:12]}",
        f"Project type: {snapshot.project_type or 'unknown'}",
        f"Manifests: {', '.join(snapshot.manifest_files) or '(none)'}",
        f"Top-level: {', '.join(snapshot.top_level_entries) or '(empty)'}",
        f"Tests: {', '.join(snapshot.test_directories) or '(none)'}",
        f"Git branch: {snapshot.branch or 'unknown'}",
    ]
    if delta.changed:
        lines.extend(
            [
                f"Added: {', '.join(delta.added_paths) or '(none)'}",
                f"Modified: {', '.join(delta.modified_paths) or '(none)'}",
                f"Deleted: {', '.join(delta.deleted_paths) or '(none)'}",
            ]
        )
    return "\n".join(lines)


def _message_source_ref(message: ToolResultMessage) -> str:
    message_id = _optional_text(message.metadata.get("session_message_id"))
    return f"message:{message_id}" if message_id else f"tool:{message.tool_call_id}"


def _message_paths(message: ToolResultMessage) -> list[str]:
    paths = [str(path) for path in message.affected_paths if str(path).strip()]
    raw = message.metadata.get("read_paths")
    if isinstance(raw, list):
        paths.extend(str(path) for path in raw if str(path).strip())
    return list(dict.fromkeys(Path(path).as_posix() for path in paths))


def _file_hash_from_message(message: ToolResultMessage, path: str) -> str | None:
    raw = message.metadata.get("file_state")
    if not isinstance(raw, dict):
        raw = message.details if isinstance(message.details, dict) else None
    if isinstance(raw, dict) and str(raw.get("path") or "") == path:
        return _optional_text(raw.get("sha256"))
    return None


def _tool_text(message: ToolResultMessage) -> str:
    return "\n".join(
        block.text for block in message.content if hasattr(block, "text") and block.text
    )


def _evidence_summary(message: ToolResultMessage, text: str) -> str:
    preview = " ".join(text.split())[:600]
    parts = [f"{message.tool_name} status={message.status}"]
    if message.exit_code is not None:
        parts.append(f"exit_code={message.exit_code}")
    if message.affected_paths:
        parts.append("paths=" + ", ".join(message.affected_paths[:8]))
    if message.verification:
        parts.append(f"verification={message.verification}")
    if preview:
        parts.append(f"summary={preview}")
    return " ".join(parts)


def _is_read(message: ToolResultMessage) -> bool:
    return message.tool_name.lower() in {"read", "read_file", "view", "cat"}


def _git_lines(root: Path, args: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def _status_map(lines: list[str]) -> dict[str, str]:
    return {
        line[3:].strip().replace("\\", "/"): line[:2]
        for line in lines
        if len(line) >= 4
    }


def _status_paths(lines: list[str]) -> list[str]:
    return [line[3:].strip().replace("\\", "/") for line in lines if len(line) >= 4]


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _is_internal_path(path: str) -> bool:
    return any(part in {".git", ".codepilot", ".pytest_cache", "__pycache__"} for part in Path(path).parts)


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = [
    "ActiveFile",
    "ContextEvidence",
    "ContextState",
    "RepositoryTracker",
    "compare_snapshots",
    "repository_summary",
]
