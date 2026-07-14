from __future__ import annotations

import asyncio
from pathlib import Path

from codepilot.core.contracts import ContextPrepareRequest, CoreContextView
from codepilot.core.state import CoreState
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import UserMessage
from codepilot.sessions.context import ContextBudgetConfig, ContextService
from codepilot.sessions.memory import MemoryRecallResult, RecalledMemory


class _MemoryRecall:
    def recall(self, _query) -> MemoryRecallResult:
        return MemoryRecallResult(
            retrieved=(
                RecalledMemory(
                    memory_id="mem_1",
                    scope="project",
                    type="project",
                    key="project.verification.command",
                    content="Use focused pytest commands.",
                    source="user_explicit",
                    rank_reasons=("term:pytest",),
                ),
            )
        )


class _FailingMemoryRecall:
    def recall(self, _query):
        raise OSError("memory store unavailable")


def _request() -> ContextPrepareRequest:
    state = CoreState.new("Refactor context governance")
    message = UserMessage(
        content="Continue the context refactor.",
        metadata={"session_message_id": "msg_current"},
    )
    return ContextPrepareRequest(
        session_id="session_1",
        run_id="run_1",
        purpose="reasoning",
        directive="core.reasoning",
        messages=(message,),
        core_view=CoreContextView.from_state(state, "build"),
        model=ModelDescriptor(provider="unit", model_id="unit"),
        tool_catalog=None,
        seed={
            "system_prompt": "L0 immutable rules.",
            "permission_mode": "default",
            "checkpoint_phase": "running",
        },
    )


def test_service_materializes_five_layers_without_putting_dynamic_state_in_l0(
    tmp_path: Path,
) -> None:
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        memory_recall=_MemoryRecall(),
        budget_config=ContextBudgetConfig(
            context_window=8000,
            max_output_tokens=500,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(_request()))

    assert prepared.system_prompt == "L0 immutable rules."
    assert prepared.messages[0].metadata["context_attachment"] is True
    attachment = str(prepared.messages[0].content)
    assert "L1 Runtime And Task State" in attachment
    assert "Refactor context governance" in attachment
    assert "L2 Working Set And Evidence" in attachment
    assert "L3 Recalled Memory" in attachment
    assert "project.verification.command" in attachment
    assert prepared.messages[-1].metadata["session_message_id"] == "msg_current"
    assert service.latest_report["layers"]["l0"] == ["L0 immutable rules."]


def test_memory_recall_failure_degrades_to_empty_l3(tmp_path: Path) -> None:
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        memory_recall=_FailingMemoryRecall(),
        budget_config=ContextBudgetConfig(
            context_window=8000,
            max_output_tokens=500,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(_request()))

    assert "## L3 Recalled Memory\n- (none)" in str(prepared.messages[0].content)
    assert service.latest_report["memory_error"] == "memory store unavailable"
