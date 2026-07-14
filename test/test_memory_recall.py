from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from codepilot.sessions.memory.contracts import MemoryQuery, MemoryRecord
from codepilot.sessions.memory.recall import MemoryRecallEngine
from codepilot.sessions.memory.repository import (
    ProjectMemoryRepository,
    UserMemoryRepository,
)


def _record(
    memory_id: str,
    *,
    scope: str,
    key: str,
    content: str,
    type: str = "project",
    status: str = "active",
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        scope=scope,  # type: ignore[arg-type]
        type=type,  # type: ignore[arg-type]
        key=key,
        content=content,
        source="user_explicit",
        status=status,  # type: ignore[arg-type]
        updated_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
    )


def test_project_memory_shadows_user_memory_with_the_same_key(tmp_path: Path) -> None:
    user = UserMemoryRepository(tmp_path / "home")
    project = ProjectMemoryRepository(tmp_path / "workspace")
    user.save(
        _record(
            "mem_user",
            scope="user",
            key="project.verification.command",
            content="Use pytest.",
        )
    )
    project.save(
        _record(
            "mem_project",
            scope="project",
            key="project.verification.command",
            content="Use uv run pytest.",
        )
    )

    result = MemoryRecallEngine(user, project).recall(
        MemoryQuery(user_request="运行测试", task_goal="verify with pytest")
    )

    assert [item.memory_id for item in result.retrieved] == ["mem_project"]
    assert result.dropped["mem_user"] == "shadowed_by_project"


def test_recall_matches_chinese_bigrams_and_excludes_non_active_records(tmp_path: Path) -> None:
    user = UserMemoryRepository(tmp_path / "home")
    project = ProjectMemoryRepository(tmp_path / "workspace")
    project.save(
        _record(
            "mem_chinese",
            scope="project",
            key="project.constraint.encoding",
            content="处理中文文件时必须显式使用 UTF-8 编码。",
        )
    )
    project.save(
        _record(
            "mem_disabled",
            scope="project",
            key="project.constraint.shell",
            content="使用 PowerShell。",
            status="disabled",
        )
    )

    result = MemoryRecallEngine(user, project).recall(
        MemoryQuery(user_request="修复中文乱码和编码问题", task_goal="读取设计文档")
    )

    assert [item.memory_id for item in result.retrieved] == ["mem_chinese"]
    assert any(reason.startswith("term:") for reason in result.retrieved[0].rank_reasons)
    assert result.dropped["mem_disabled"] == "status:disabled"
