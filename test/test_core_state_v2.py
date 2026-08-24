from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from codepilot.core.state import (
    CoreState,
    RunFacts,
    TaskBlocker,
    TaskState,
    VerificationFacts,
    WorkspaceFacts,
    assess_core_state,
)


def test_core_state_round_trips_without_runtime_or_context_state() -> None:
    state = CoreState(
        task=TaskState(
            original_request="fix the failing test",
            current_goal="make the focused test pass",
            blockers=(
                TaskBlocker(
                    kind="verification_failed",
                    reason="focused test failed",
                    evidence_refs=("tool_1",),
                    recoverable=True,
                ),
            ),
        ),
        facts=RunFacts(
            workspace=WorkspaceFacts(
                revision=2,
                changed=True,
                affected_paths=("src/example.py",),
                evidence_refs=("tool_1",),
            ),
            verification=VerificationFacts(
                status="failed",
                verified_revision=2,
                attempted_checks=("pytest test_example.py",),
                evidence_refs=("tool_2",),
            ),
        ),
    )

    restored = CoreState.from_mapping(state.to_dict())

    assert restored == state
    assert "messages" not in state.to_dict()
    assert "context" not in state.to_dict()
    assert "deadline" not in state.to_dict()


def test_core_state_is_immutable() -> None:
    state = CoreState.new("inspect the repository")

    with pytest.raises(FrozenInstanceError):
        state.task = TaskState(  # type: ignore[misc]
            original_request="changed",
            current_goal="changed",
        )


def test_core_assessment_marks_old_verification_as_stale() -> None:
    state = CoreState(
        task=TaskState(
            original_request="update code",
            current_goal="update code",
        ),
        facts=RunFacts(
            workspace=WorkspaceFacts(revision=2, changed=True),
            verification=VerificationFacts(
                status="passed",
                verified_revision=1,
            ),
        ),
    )

    assessment = assess_core_state(state)

    assert assessment.status == "needs_verification"
    assert assessment.reasons == ("verification_not_fresh",)
