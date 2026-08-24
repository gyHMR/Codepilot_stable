from __future__ import annotations

from datetime import datetime, timezone

import pytest

from codepilot.core.contracts import (
    ContextPrepareRequest,
    CoreContextView,
    PreparedModelContext,
)
from codepilot.core.state import CoreState
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import UserMessage
from codepilot.sessions.context.contracts import ContextCheckpointState
from codepilot.sessions.memory.contracts import (
    AddMemory,
    MemoryProposal,
    MemoryProposalBatch,
    MemoryRecord,
)
from codepilot.tools.registry import ToolRegistry
from codepilot.tools.runtime import ToolRuntime
from codepilot.tools.security import ApprovalResponse


def test_memory_record_uses_exact_eight_field_schema() -> None:
    record = MemoryRecord(
        id="mem_1",
        scope="project",
        type="project",
        key="project.constraint.api_compatibility",
        content="Public APIs remain backward compatible.",
        source="user_explicit",
        status="active",
        updated_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
    )

    payload = record.to_dict()

    assert set(payload) == {
        "id",
        "scope",
        "type",
        "key",
        "content",
        "source",
        "status",
        "updated_at",
    }
    assert MemoryRecord.from_dict(payload) == record
    with pytest.raises(ValueError, match="unknown memory fields"):
        MemoryRecord.from_dict({**payload, "confidence": "explicit"})


def test_memory_proposals_are_bounded_and_do_not_choose_source_or_status() -> None:
    proposal = MemoryProposal(
        scope="project",
        type="experience",
        key="experience.parser.snapshot_update",
        content="Run snapshot tests after parser grammar changes.",
    )
    batch = MemoryProposalBatch(
        session_id="session_1",
        run_id="run_1",
        origin="agent_finalization",
        verification_passed=True,
        proposals=(proposal,),
    )

    assert batch.proposals == (proposal,)
    assert not hasattr(proposal, "source")
    assert not hasattr(proposal, "status")
    with pytest.raises(ValueError, match="at most three"):
        MemoryProposalBatch(
            session_id="session_1",
            run_id="run_1",
            origin="agent_finalization",
            verification_passed=True,
            proposals=(proposal, proposal, proposal, proposal),
        )


def test_memory_management_commands_validate_the_same_key_contract() -> None:
    command = AddMemory(
        scope="user",
        type="profile",
        key="profile.response_language",
        content="Use Chinese by default.",
    )

    assert command.key == "profile.response_language"
    with pytest.raises(ValueError, match="dot notation"):
        AddMemory(
            scope="user",
            type="profile",
            key="Response Language",
            content="Use Chinese by default.",
        )


def test_core_context_request_is_typed_and_freezes_seed() -> None:
    state = CoreState.new("Inspect the project")
    request = ContextPrepareRequest(
        session_id="session_1",
        run_id="run_1",
        purpose="reasoning",
        directive="core.reasoning",
        messages=(UserMessage(content="Inspect the project"),),
        core_view=CoreContextView.from_state(state, "read"),
        model=ModelDescriptor(provider="unit", model_id="unit"),
        tool_catalog=None,
        seed={"system_prompt": "system"},
    )
    prepared = PreparedModelContext(
        system_prompt="system",
        messages=request.messages,
        tools=(),
        projection_ref="context:run_1",
    )

    assert request.core_view.goal == "Inspect the project"
    assert prepared.messages == request.messages
    with pytest.raises(TypeError):
        request.seed["system_prompt"] = "changed"  # type: ignore[index]


def test_context_checkpoint_contract_rejects_unknown_state() -> None:
    state = ContextCheckpointState(
        compact_snapshot_ref="artifact:compact_1",
        compacted_until_message_id="message_10",
    )

    assert ContextCheckpointState.from_mapping(state.to_mapping()) == state
    with pytest.raises(ValueError, match="unknown context checkpoint fields"):
        ContextCheckpointState.from_mapping({"compact_summary": "legacy"})


def test_tool_runtime_can_stage_resume_without_executing_it() -> None:
    runtime = ToolRuntime(ToolRegistry())
    preparation = runtime.prepare_resume(
        ApprovalResponse(
            approval_id="approval_1",
            request_fingerprint="fingerprint_1",
            decision="approve",
        )
    )

    assert preparation.resume_id
    assert runtime.pending_challenges() == ()
    runtime.restore_checkpoint_state(preparation.checkpoint_state)
    assert runtime.pending_prepared_resume() is not None
    reopened = ToolRuntime(ToolRegistry())
    reopened.restore_checkpoint_state(preparation.checkpoint_state)
    restored = reopened.pending_prepared_resume()
    assert restored is not None and restored.resume_id == preparation.resume_id
