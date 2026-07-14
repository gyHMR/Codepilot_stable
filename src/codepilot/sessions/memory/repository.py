from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from .contracts import MemoryRecord, MemoryScope


_CURRENT_FIELDS = frozenset(
    {"id", "scope", "type", "key", "content", "source", "status", "updated_at"}
)
_TYPE_MAPPING = {
    "preference": "profile",
    "correction": "feedback",
    "constraint": "project",
    "decision": "project",
    "workflow": "experience",
    "experience": "experience",
    "profile": "profile",
    "feedback": "feedback",
    "project": "project",
    "reference": "reference",
}
_STATUS_MAPPING = {
    "candidate": "candidate",
    "active": "active",
    "disabled": "disabled",
    "superseded": "superseded",
    "deleted": "deleted",
}


class _JsonlMemoryRepository:
    """JSONL persistence for exactly one Memory scope."""

    def __init__(self, path: str | Path, *, scope: MemoryScope) -> None:
        self.path = Path(path)
        self.scope = scope

    def all_records(self) -> list[MemoryRecord]:
        latest: dict[str, MemoryRecord] = {}
        for payload in _read_json_rows(self.path, recover_incomplete_tail=True):
            record = MemoryRecord.from_dict(payload)
            if record.scope != self.scope:
                raise ValueError(
                    f"memory scope {record.scope} is stored in {self.scope} repository"
                )
            latest[record.id] = record
        return sorted(latest.values(), key=lambda record: (record.updated_at, record.id))

    def get(self, memory_id: str) -> MemoryRecord | None:
        return next(
            (record for record in self.all_records() if record.id == memory_id),
            None,
        )

    def save(self, record: MemoryRecord) -> None:
        if record.scope != self.scope:
            raise ValueError(
                f"cannot save {record.scope} memory in {self.scope} repository"
            )
        rows = _read_json_rows(self.path, recover_incomplete_tail=True)
        rows.append(record.to_dict())
        _atomic_write_rows(self.path, rows)

    def records_for_key(self, key: str) -> list[MemoryRecord]:
        return [record for record in self.all_records() if record.key == key]

    def purge(self, memory_id: str) -> bool:
        rows = _read_json_rows(self.path, recover_incomplete_tail=True)
        kept = [row for row in rows if str(row.get("id") or "") != memory_id]
        if len(kept) == len(rows):
            return False
        _atomic_write_rows(self.path, kept)
        return True

    def replace_all(self, records: Iterable[MemoryRecord]) -> None:
        rows: list[dict[str, object]] = []
        for record in records:
            if record.scope != self.scope:
                raise ValueError(
                    f"cannot save {record.scope} memory in {self.scope} repository"
                )
            rows.append(record.to_dict())
        _atomic_write_rows(self.path, rows)


class UserMemoryRepository(_JsonlMemoryRepository):
    def __init__(self, home_dir: str | Path | None = None) -> None:
        root = Path(home_dir).expanduser() if home_dir is not None else Path.home()
        super().__init__(
            root / ".codepilot" / "memory" / "memories.jsonl",
            scope="user",
        )


class ProjectMemoryRepository(_JsonlMemoryRepository):
    def __init__(self, workspace_dir: str | Path) -> None:
        root = Path(workspace_dir)
        super().__init__(
            root / ".codepilot" / "memory" / "memories.jsonl",
            scope="project",
        )


@dataclass(frozen=True)
class MemoryMigrationReport:
    required: bool
    valid: bool
    converted: int = 0
    skipped: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class _MigrationPlan:
    report: MemoryMigrationReport
    user_records: tuple[MemoryRecord, ...]
    project_records: tuple[MemoryRecord, ...]
    source_paths: tuple[Path, ...]


class LegacyMemoryMigrator:
    """One-way upgrader isolated from strict MemoryRecord parsing."""

    def __init__(
        self,
        user_repository: UserMemoryRepository,
        project_repository: ProjectMemoryRepository,
    ) -> None:
        self.user_repository = user_repository
        self.project_repository = project_repository

    def dry_run(self) -> MemoryMigrationReport:
        return self._plan().report

    def migrate(self) -> MemoryMigrationReport:
        plan = self._plan()
        if not plan.report.valid:
            raise ValueError(
                "conflicting active legacy memories: " + "; ".join(plan.report.conflicts)
            )
        if not plan.report.required:
            return plan.report

        original_bytes = {
            path: path.read_bytes() if path.exists() else None for path in plan.source_paths
        }
        user_temp = _stage_rows(
            self.user_repository.path,
            [record.to_dict() for record in plan.user_records],
        )
        project_temp = _stage_rows(
            self.project_repository.path,
            [record.to_dict() for record in plan.project_records],
        )
        try:
            for path, content in original_bytes.items():
                if content is None:
                    continue
                backup = _legacy_backup_path(path)
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
            os.replace(user_temp, self.user_repository.path)
            os.replace(project_temp, self.project_repository.path)
            self.user_repository.all_records()
            self.project_repository.all_records()
        except Exception:
            for path, content in original_bytes.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_write_bytes(path, content)
            raise
        finally:
            user_temp.unlink(missing_ok=True)
            project_temp.unlink(missing_ok=True)
        return plan.report

    def _plan(self) -> _MigrationPlan:
        source_paths = tuple(
            dict.fromkeys((self.user_repository.path, self.project_repository.path))
        )
        required = False
        converted = 0
        skipped: list[str] = []
        collected: list[MemoryRecord] = []

        for path in source_paths:
            rows = _read_json_rows(path, recover_incomplete_tail=False)
            expected_scope = (
                self.user_repository.scope
                if path == self.user_repository.path
                else self.project_repository.scope
            )
            for payload in rows:
                if set(payload) == _CURRENT_FIELDS:
                    record = MemoryRecord.from_dict(payload)
                    if record.scope != expected_scope:
                        required = True
                    collected.append(record)
                    continue
                required = True
                record = _convert_legacy_record(payload, skipped)
                if record is not None:
                    converted += 1
                    collected.append(record)

        latest = _latest_records(collected)
        latest, normalized = _normalize_current_versions(latest)
        required = required or normalized
        conflicts = _current_conflicts(latest)
        report = MemoryMigrationReport(
            required=required,
            valid=not conflicts,
            converted=converted,
            skipped=tuple(skipped),
            conflicts=tuple(conflicts),
        )
        return _MigrationPlan(
            report=report,
            user_records=tuple(record for record in latest if record.scope == "user"),
            project_records=tuple(record for record in latest if record.scope == "project"),
            source_paths=source_paths,
        )


def _read_json_rows(
    path: Path,
    *,
    recover_incomplete_tail: bool,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            is_incomplete_tail = index == len(lines) - 1 and not text.endswith(("\n", "\r"))
            if recover_incomplete_tail and is_incomplete_tail:
                _atomic_write_rows(path, rows)
                return rows
            raise ValueError(f"invalid memory JSON on line {index + 1}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid memory record on line {index + 1}")
        rows.append(payload)
    return rows


def _atomic_write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_write_bytes(path, content)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temp = _stage_bytes(path, content)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _stage_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> Path:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return _stage_bytes(path, content)


def _stage_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return temp


def _convert_legacy_record(
    payload: Mapping[str, object],
    skipped: list[str],
) -> MemoryRecord | None:
    memory_id = _required_legacy_text(payload.get("id"), "id")
    old_scope = str(payload.get("scope") or "").strip().lower()
    if old_scope == "session":
        skipped.append(f"{memory_id}:session_scope")
        return None
    scope: MemoryScope
    if old_scope in {"global", "user"}:
        scope = "user"
    elif old_scope in {"workspace", "project"}:
        scope = "project"
    else:
        raise ValueError(f"unknown legacy memory scope: {old_scope}")

    old_type = str(payload.get("type") or payload.get("kind") or "").strip().lower()
    try:
        memory_type = _TYPE_MAPPING[old_type]
    except KeyError as exc:
        raise ValueError(f"unknown legacy memory type: {old_type}") from exc
    content = _required_legacy_text(
        payload.get("content") or payload.get("value"),
        "content",
    )
    key = _legacy_key(payload, old_type, content)
    old_status = str(payload.get("status") or "candidate").strip().lower()
    try:
        status = _STATUS_MAPPING[old_status]
    except KeyError as exc:
        raise ValueError(f"unknown legacy memory status: {old_status}") from exc
    if key.startswith("legacy.") and status == "active":
        status = "candidate"
    source = _legacy_source(payload.get("source"))
    updated_at = _legacy_datetime(payload.get("updated_at") or payload.get("created_at"))
    return MemoryRecord(
        id=memory_id,
        scope=scope,
        type=memory_type,  # type: ignore[arg-type]
        key=key,
        content=content,
        source=source,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        updated_at=updated_at,
    )


def _legacy_key(
    payload: Mapping[str, object],
    old_type: str,
    content: str,
) -> str:
    subject = str(payload.get("subject") or "").strip().lower()
    predicate = str(payload.get("predicate") or "is").strip().lower()
    slug_parts = re.findall(r"[a-z0-9]+", subject)
    if predicate and predicate != "is":
        slug_parts.extend(re.findall(r"[a-z0-9]+", predicate))
    slug = "_".join(slug_parts)
    prefix = {
        "preference": "profile",
        "correction": "feedback",
        "constraint": "project.constraint",
        "decision": "project.decision",
        "workflow": "experience",
        "experience": "experience",
        "profile": "profile",
        "feedback": "feedback",
        "project": "project.legacy",
        "reference": "reference",
    }[old_type]
    if slug:
        return f"{prefix}.{slug}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"legacy.{digest}"


def _legacy_source(value: object) -> str:
    source = str(value or "").strip().lower()
    if source in {"user_explicit", "user_approved", "manual_edit"}:
        return "user_explicit"
    if source == "user_correction":
        return "user_feedback"
    if source == "verified_run" or "verified" in source:
        return "verified_run"
    return "agent_extracted"


def _legacy_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("invalid legacy memory timestamp") from exc
    else:
        parsed = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _latest_records(records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
    latest: dict[str, MemoryRecord] = {}
    for record in records:
        previous = latest.get(record.id)
        if previous is None or record.updated_at >= previous.updated_at:
            latest[record.id] = record
    return sorted(latest.values(), key=lambda record: (record.updated_at, record.id))


def _normalize_current_versions(
    records: list[MemoryRecord],
) -> tuple[list[MemoryRecord], bool]:
    grouped: dict[tuple[str, str, str], list[MemoryRecord]] = {}
    for record in records:
        group = "current" if record.status in {"active", "disabled"} else record.status
        if group not in {"current", "candidate"}:
            continue
        grouped.setdefault((record.scope, record.key, group), []).append(record)
    replacements: dict[str, MemoryRecord] = {}
    changed = False
    for items in grouped.values():
        if len(items) <= 1 or len({record.content for record in items}) > 1:
            continue
        winner = max(items, key=lambda record: (record.updated_at, record.id))
        for record in items:
            if record.id == winner.id:
                continue
            replacements[record.id] = replace(record, status="superseded")
            changed = True
    return [replacements.get(record.id, record) for record in records], changed


def _current_conflicts(records: Iterable[MemoryRecord]) -> list[str]:
    grouped: dict[tuple[str, str, str], list[MemoryRecord]] = {}
    for record in records:
        group = "current" if record.status in {"active", "disabled"} else record.status
        if group not in {"current", "candidate"}:
            continue
        grouped.setdefault((record.scope, record.key, group), []).append(record)
    conflicts: list[str] = []
    for (scope, key, group), items in grouped.items():
        if len(items) > 1:
            conflicts.append(f"{scope}:{key}:{group}")
    return sorted(conflicts)


def _required_legacy_text(value: object, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"legacy memory {field_name} is required")
    return text


def _legacy_backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".legacy.bak")


__all__ = [
    "LegacyMemoryMigrator",
    "MemoryMigrationReport",
    "ProjectMemoryRepository",
    "UserMemoryRepository",
]
