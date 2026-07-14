from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from codepilot.sessions.memory import (
    AddMemory,
    ApproveMemory,
    DeleteMemory,
    DisableMemory,
    EditMemory,
    EnableMemory,
    HistoryMemory,
    MemoryActor,
    MemoryProposal,
    MemoryProposalBatch,
    MemoryService,
    PurgeMemory,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 13, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(
        workspace_dir=tmp_path / "workspace",
        user_home=tmp_path / "home",
        clock=_Clock(),
    )


def test_candidate_approval_supersedes_the_current_active_record(tmp_path: Path) -> None:
    service = _service(tmp_path)
    actor = MemoryActor("user")
    active = service.execute(
        AddMemory(
            scope="project",
            type="project",
            key="project.verification.command",
            content="Use pytest.",
        ),
        actor,
    ).records[0]
    candidate = service.submit_proposals(
        MemoryProposalBatch(
            session_id="session_1",
            run_id="run_1",
            origin="agent_finalization",
            verification_passed=True,
            proposals=(
                MemoryProposal(
                    scope="project",
                    type="project",
                    key="project.verification.command",
                    content="Use uv run pytest.",
                ),
            ),
        )
    ).records[0]

    approved = service.execute(ApproveMemory(candidate.id), actor).records[0]
    history = service.execute(
        HistoryMemory("project", "project.verification.command"), actor
    ).records

    assert approved.id == candidate.id
    assert approved.status == "active"
    assert {record.id: record.status for record in history} == {
        active.id: "superseded",
        candidate.id: "active",
    }


def test_disable_enable_edit_delete_and_purge_follow_five_state_lifecycle(tmp_path: Path) -> None:
    service = _service(tmp_path)
    actor = MemoryActor("user")
    original = service.execute(
        AddMemory(
            scope="user",
            type="profile",
            key="profile.response.language",
            content="Use Chinese.",
        ),
        actor,
    ).records[0]

    assert service.execute(DisableMemory(original.id), actor).records[0].status == "disabled"
    assert service.execute(EnableMemory(original.id), actor).records[0].status == "active"
    edited = service.execute(EditMemory(original.id, "Use concise Chinese."), actor).records[0]
    assert edited.id != original.id
    assert service.execute(DeleteMemory(edited.id), actor).records[0].status == "deleted"
    service.execute(PurgeMemory(edited.id), actor)

    history = service.execute(
        HistoryMemory("user", "profile.response.language"), actor
    ).records
    assert [(record.id, record.status) for record in history] == [
        (original.id, "superseded")
    ]


def test_duplicate_is_noop_and_conflicting_explicit_add_requires_edit(tmp_path: Path) -> None:
    service = _service(tmp_path)
    actor = MemoryActor("user")
    command = AddMemory(
        scope="project",
        type="project",
        key="project.verification.command",
        content="Use pytest.",
    )
    original = service.execute(command, actor).records[0]
    duplicate = service.execute(command, actor).records[0]

    assert duplicate == original
    assert len(service.project_repository.all_records()) == 1
    with pytest.raises(ValueError, match="conflict"):
        service.execute(
            AddMemory(
                scope="project",
                type="project",
                key="project.verification.command",
                content="Use unittest.",
            ),
            actor,
        )
