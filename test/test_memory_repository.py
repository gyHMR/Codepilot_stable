from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from codepilot.sessions.memory.contracts import MemoryRecord
from codepilot.sessions.memory.repository import (
    ProjectMemoryRepository,
    UserMemoryRepository,
)


def _record(memory_id: str, *, scope: str = "project", content: str = "Use pytest.") -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        scope=scope,  # type: ignore[arg-type]
        type="project",
        key="project.verification.command",
        content=content,
        source="user_explicit",
        status="active",
        updated_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
    )


def test_user_and_project_repositories_use_separate_files(tmp_path: Path) -> None:
    user = UserMemoryRepository(tmp_path / "home")
    project = ProjectMemoryRepository(tmp_path / "workspace")

    user.save(_record("mem_user", scope="user"))
    project.save(_record("mem_project"))

    assert user.path == tmp_path / "home" / ".codepilot" / "memory" / "memories.jsonl"
    assert project.path == tmp_path / "workspace" / ".codepilot" / "memory" / "memories.jsonl"
    assert [record.id for record in user.all_records()] == ["mem_user"]
    assert [record.id for record in project.all_records()] == ["mem_project"]


def test_repository_loads_latest_snapshot_and_purge_removes_all_snapshots(tmp_path: Path) -> None:
    repository = ProjectMemoryRepository(tmp_path)
    original = _record("mem_1")
    updated = MemoryRecord.from_dict(
        {
            **original.to_dict(),
            "content": "Use uv run pytest.",
            "updated_at": "2026-07-13T01:00:00+00:00",
        }
    )

    repository.save(original)
    repository.save(updated)

    assert repository.get("mem_1") == updated
    assert repository.all_records() == [updated]
    repository.purge("mem_1")
    assert repository.get("mem_1") is None
    assert repository.path.read_text(encoding="utf-8") == ""


def test_repository_recovers_only_an_incomplete_tail_line(tmp_path: Path) -> None:
    repository = ProjectMemoryRepository(tmp_path)
    repository.save(_record("mem_1"))
    with repository.path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"id":"partial"')

    assert [record.id for record in repository.all_records()] == ["mem_1"]
    assert "partial" not in repository.path.read_text(encoding="utf-8")
