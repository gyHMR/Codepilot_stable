from __future__ import annotations

import json
from pathlib import Path

import pytest

from codepilot.sessions.memory.repository import (
    LegacyMemoryMigrator,
    ProjectMemoryRepository,
    UserMemoryRepository,
)


def _legacy(memory_id: str, *, scope: str, type: str, subject: str, content: str) -> dict[str, object]:
    return {
        "schema_version": 4,
        "id": memory_id,
        "scope": scope,
        "type": type,
        "subject": subject,
        "predicate": "is",
        "value": content,
        "content": content,
        "status": "active",
        "source": "user_explicit",
        "updated_at": "2026-07-13T00:00:00+00:00",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


def test_legacy_migration_routes_scopes_backs_up_and_is_idempotent(tmp_path: Path) -> None:
    user = UserMemoryRepository(tmp_path / "home")
    project = ProjectMemoryRepository(tmp_path / "workspace")
    original = _write_jsonl(
        project.path,
        [
            _legacy(
                "mem_user",
                scope="global",
                type="preference",
                subject="response_language",
                content="Use Chinese.",
            ),
            _legacy(
                "mem_project",
                scope="project",
                type="constraint",
                subject="api_compatibility",
                content="Keep public APIs compatible.",
            ),
        ],
    )
    migrator = LegacyMemoryMigrator(user, project)

    report = migrator.dry_run()
    assert report.required is True
    assert report.valid is True
    assert report.converted == 2
    migrated = migrator.migrate()

    assert migrated.converted == 2
    assert project.path.with_suffix(project.path.suffix + ".legacy.bak").read_text(
        encoding="utf-8"
    ) == original
    assert [record.key for record in user.all_records()] == ["profile.response_language"]
    assert [record.key for record in project.all_records()] == [
        "project.constraint.api_compatibility"
    ]
    assert migrator.migrate().required is False


def test_conflicting_active_legacy_records_fail_without_modifying_source(tmp_path: Path) -> None:
    user = UserMemoryRepository(tmp_path / "home")
    project = ProjectMemoryRepository(tmp_path / "workspace")
    original = _write_jsonl(
        project.path,
        [
            _legacy(
                "mem_1",
                scope="project",
                type="constraint",
                subject="test_command",
                content="Use pytest.",
            ),
            _legacy(
                "mem_2",
                scope="project",
                type="constraint",
                subject="test_command",
                content="Use unittest.",
            ),
        ],
    )
    migrator = LegacyMemoryMigrator(user, project)

    report = migrator.dry_run()
    assert report.valid is False
    assert report.conflicts
    with pytest.raises(ValueError, match="conflicting active"):
        migrator.migrate()
    assert project.path.read_text(encoding="utf-8") == original
    assert not project.path.with_suffix(project.path.suffix + ".legacy.bak").exists()
