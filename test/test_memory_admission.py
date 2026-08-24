from __future__ import annotations

from datetime import datetime, timezone

from codepilot.sessions.memory.admission import AdmissionPolicy
from codepilot.sessions.memory.contracts import MemoryProposal, MemoryRecord


def _record(content: str) -> MemoryRecord:
    return MemoryRecord(
        id="mem_existing",
        scope="project",
        type="project",
        key="project.verification.command",
        content=content,
        source="user_explicit",
        status="active",
        updated_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
    )


def test_admission_normalizes_content_and_assigns_server_owned_fields() -> None:
    policy = AdmissionPolicy()
    decision = policy.evaluate(
        MemoryProposal(
            scope="project",
            type="project",
            key="project.verification.command",
            content="  Use   python -m pytest.  ",
        ),
        existing=(),
        origin="agent_finalization",
        verification_passed=True,
    )

    assert decision.disposition == "accept"
    assert decision.proposal.content == "Use python -m pytest."
    assert decision.source == "agent_extracted"
    assert decision.status == "candidate"


def test_admission_rejects_secrets_before_persistence() -> None:
    decision = AdmissionPolicy().evaluate(
        MemoryProposal(
            scope="project",
            type="reference",
            key="reference.api.token",
            content="api_key = sk-abcdefghijklmnopqrstuvwxyz123456",
        ),
        existing=(),
        origin="user_explicit",
        verification_passed=True,
    )

    assert decision.disposition == "reject"
    assert decision.reason == "sensitive_content"


def test_admission_distinguishes_exact_duplicate_and_same_key_conflict() -> None:
    policy = AdmissionPolicy()
    duplicate = policy.evaluate(
        MemoryProposal(
            scope="project",
            type="project",
            key="project.verification.command",
            content="Use pytest.",
        ),
        existing=(_record("Use pytest."),),
        origin="user_explicit",
        verification_passed=True,
    )
    conflict = policy.evaluate(
        MemoryProposal(
            scope="project",
            type="project",
            key="project.verification.command",
            content="Use unittest.",
        ),
        existing=(_record("Use pytest."),),
        origin="user_explicit",
        verification_passed=True,
    )

    assert duplicate.disposition == "duplicate"
    assert duplicate.existing_id == "mem_existing"
    assert conflict.disposition == "conflict"


def test_unverified_automatic_experience_is_rejected() -> None:
    decision = AdmissionPolicy().evaluate(
        MemoryProposal(
            scope="project",
            type="experience",
            key="experience.parser.snapshot_update",
            content="Run snapshot tests after grammar changes.",
        ),
        existing=(),
        origin="agent_finalization",
        verification_passed=False,
    )

    assert decision.disposition == "reject"
    assert decision.reason == "experience_requires_verification"
