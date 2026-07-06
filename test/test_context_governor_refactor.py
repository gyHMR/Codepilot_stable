from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def test_pressure_policy_uses_effective_budget_and_three_levels() -> None:
    from codepilot.core.contracts import ContextPreparationRequest
    from codepilot.sessions.context.policy import ContextPressurePolicy

    policy = ContextPressurePolicy(
        safety_margin_tokens=100,
        tight_ratio=0.70,
        critical_ratio=0.90,
    )
    request = ContextPreparationRequest(
        session_id="session_1",
        model_context_window=1000,
        model_max_output_tokens=100,
    )

    normal = policy.evaluate(
        request,
        estimated_tokens=500,
        tool_output_tokens=50,
        history_tokens=200,
    )
    tight = policy.evaluate(
        request,
        estimated_tokens=610,
        tool_output_tokens=260,
        history_tokens=300,
    )
    critical = policy.evaluate(
        request,
        estimated_tokens=730,
        tool_output_tokens=260,
        history_tokens=430,
    )

    assert normal.effective_budget == 800
    assert normal.level == "normal"
    assert tight.level == "tight"
    assert "tool_output_pressure" in tight.reasons
    assert critical.level == "critical"
    assert "critical_budget_pressure" in critical.reasons


def test_context_protocols_describe_view_checkpoint_and_artifacts() -> None:
    import pytest
    from codepilot.protocols import (
        ContextArtifactRef,
        ContextCheckpoint,
        ContextPressure,
        ContextReport,
        ContextView,
    )

    pressure = ContextPressure(
        level="tight",
        effective_budget=800,
        estimated_tokens=720,
        reasons=["tool_output_pressure"],
    )
    artifact = ContextArtifactRef(
        kind="tool_output",
        path=".codepilot/sessions/s1/artifacts/tool.txt",
        source_hash="abc123",
        summary="pytest failed with one assertion",
        original_tokens=1200,
        visible_tokens=40,
    )
    checkpoint = ContextCheckpoint(
        goal="fix failing tests",
        active_files=["src/app.py"],
        changed_files=["src/app.py"],
        key_evidence=["pytest failed before fix"],
        verification_state="stale",
        open_questions=[],
        next_actions=["rerun pytest"],
        source_refs=[artifact.path],
    )
    view = ContextView(
        stable_rules=["AGENTS.md: keep UTF-8"],
        task_state=["goal: fix failing tests"],
        recalled_memory=["previous pytest failure required cwd setup"],
        working_set=["pytest failed before fix"],
        conversation=["user: fix tests"],
        tools=["read", "shell"],
    )
    report = ContextReport(
        context_id="ctx_1",
        repository_fingerprint="repo",
        total_budget_tokens=800,
        estimated_tokens_before=1500,
        estimated_tokens_after=620,
        pressure=pressure,
        context_view=view,
        checkpoint_created=checkpoint,
        artifact_refs=[artifact],
        tokens_by_layer={"system": 20, "working_set": 40},
        prefix_hash="prefix",
        dynamic_hash="dynamic",
    )

    payload = report.to_dict()

    assert payload["pressure"]["level"] == "tight"
    assert payload["context_view"]["stable_rules"] == ["AGENTS.md: keep UTF-8"]
    assert payload["checkpoint_created"]["goal"] == "fix failing tests"
    assert payload["artifact_refs"][0]["visible_tokens"] == 40
    assert payload["prefix_hash"] == "prefix"

    with pytest.raises(ValueError, match="Unknown context pressure level"):
        ContextPressure(level="panic", effective_budget=1, estimated_tokens=2)


def test_context_ledger_normalizer_maps_legacy_view_aliases() -> None:
    from codepilot.sessions.context.ledger import normalize_context_view_payload

    payload = normalize_context_view_payload(
        {
            "stable_rules": ["AGENTS.md"],
            "working_state": ["goal: old"],
            "evidence": ["pytest failed"],
            "recent_messages": ["user: fix"],
            "recalled_memory": ["memory"],
            "tools": ["read"],
        }
    )

    assert payload == {
        "stable_rules": ["AGENTS.md"],
        "task_state": ["goal: old"],
        "working_set": ["pytest failed"],
        "recalled_memory": ["memory"],
        "conversation": ["user: fix"],
        "tools": ["read"],
    }


def test_context_view_rejects_legacy_alias_fields() -> None:
    from codepilot.protocols import ContextView

    with pytest.raises(TypeError):
        ContextView(working_state=["legacy"])  # type: ignore[call-arg]


def test_tool_artifact_ledger_persists_large_outputs_and_projects_light_messages(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import TextContent, ToolResultMessage
    from codepilot.sessions.context.ledger import ToolArtifactLedger

    ledger = ToolArtifactLedger(
        workspace_dir=tmp_path,
        session_id="session_1",
    )
    large_output = "failure line\n" * 500
    message = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="shell",
        content=[TextContent(text=large_output)],
        status="error",
        affected_paths=["src/app.py"],
        verification={"status": "failed"},
        metadata={"file_state": {"path": "src/app.py", "sha256": "abc"}},
    )

    entry = ledger.record_tool_result(run_id="run_1", message=message)
    projected = ledger.project_tool_result(message, preserve_full=False)

    artifact_path = tmp_path / entry.artifact.path
    assert artifact_path.is_file()
    assert artifact_path.read_text(encoding="utf-8") == large_output
    assert entry.artifact.original_tokens > entry.artifact.visible_tokens
    assert entry.affected_paths == ["src/app.py"]
    assert "failure line" not in projected.content[0].text * 20
    assert entry.artifact.path in projected.content[0].text
    assert ledger.load_entries()[0].tool_call_id == "call_1"
    session_dir = tmp_path / ".codepilot" / "sessions" / "session_1"
    assert (session_dir / "context_ledger.jsonl").exists()
    assert not (session_dir / "tool_ledger.jsonl").exists()


def test_context_governor_projects_decision_view_with_checkpoint_and_memory(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.policy import ContextPressurePolicy
    from codepilot.sessions.context.state import SessionContextState
    from codepilot.sessions.memory.records import MemoryRecall, MemoryRecord, RetrievedMemory

    class FakeMemoryRetriever:
        def validate_freshness(self) -> list[object]:
            return []

        def recall(self, _query) -> MemoryRecall:
            return MemoryRecall(
                pinned_text="Pinned: use UTF-8 and LF.",
                selected=[
                    RetrievedMemory(
                        record=MemoryRecord(
                            id="mem_1",
                            type="experience",
                            scope="session",
                            subject="experience:verification:cwd_setup",
                            predicate="is",
                            value="Previous pytest failure required cwd setup.",
                            content="Previous pytest failure required cwd setup.",
                            keywords=["error:verification_failed", "intent:debug_failure"],
                            source="run",
                        ),
                        score=90,
                        reasons=["recent_error"],
                    )
                ],
            )

    state = SessionContextState(workspace_dir=tmp_path)
    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_1",
        state=state,
        memory_retriever=FakeMemoryRetriever(),
        pressure_policy=ContextPressurePolicy(
            safety_margin_tokens=50,
            tight_ratio=0.45,
            critical_ratio=0.60,
        ),
    )
    tool_call = ToolCall(id="call_1", name="shell", arguments={"command": "pytest -q"})
    large_output = "long pytest failure output\n" * 500
    context = AgentContext(
        system_prompt="System rules.\n\nAGENTS.md: project files use UTF-8.",
        messages=[
            UserMessage(content="Please fix the failing tests."),
            AssistantMessage(content=[tool_call], stop_reason="toolUse"),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="shell",
                content=[TextContent(text=large_output)],
                status="error",
                affected_paths=["test/test_app.py"],
                verification={"status": "failed"},
            ),
            UserMessage(content="Continue from the failure."),
        ],
        current_task="Goal: fix failing tests.",
        task_signal={
            "phase": "acting",
            "action_intent": "debug_failure",
            "recent_error_code": "verification_failed",
        },
    )
    request = ContextPreparationRequest(
        session_id="session_1",
        model_context_window=900,
        model_max_output_tokens=200,
    )

    prepared = asyncio.run(governor.prepare(context, request))
    rendered = prepared.system_prompt + "\n".join(
        getattr(block, "text", "")
        for message in prepared.messages
        if isinstance(message, ToolResultMessage)
        for block in message.content
    )

    assert "AGENTS.md: project files use UTF-8." in prepared.system_prompt
    assert "Previous pytest failure required cwd setup." in prepared.system_prompt
    assert "long pytest failure output" not in rendered
    assert prepared.report.pressure.level == "critical"
    assert prepared.report.checkpoint_created is not None
    assert prepared.report.artifact_refs
    assert prepared.report.context_view is not None
    assert prepared.report.context_view.recalled_memory
    assert any(
        item.get("path") == "test/test_app.py"
        for item in prepared.report.selected_items
    )
    assert any(
        item.get("kind") == "memory" and item.get("id") == "mem_1"
        for item in prepared.report.selected_items
    )
    assert governor.checkpoints.load_latest() is not None
    session_dir = tmp_path / ".codepilot" / "sessions" / "session_1"
    assert (session_dir / "context_ledger.jsonl").exists()
    assert not (session_dir / "context_views.jsonl").exists()
    assert not (session_dir / "checkpoints.jsonl").exists()


def test_context_governor_counts_tool_schemas_in_budget_estimates(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import Tool, UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.policy import ContextPressurePolicy
    from codepilot.sessions.context.state import SessionContextState

    tools = [
        Tool(
            name=f"tool_{index}",
            description="A model-visible tool with schema budget.",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        )
        for index in range(4)
    ]
    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_tool_budget",
        state=SessionContextState(workspace_dir=tmp_path),
        pressure_policy=ContextPressurePolicy(
            safety_margin_tokens=0,
            tight_ratio=0.72,
            critical_ratio=0.90,
        ),
    )

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[UserMessage(content="Use the available tools.")],
                tools=tools,
            ),
            ContextPreparationRequest(
                session_id="session_tool_budget",
                model_context_window=1100,
                model_max_output_tokens=100,
            ),
        )
    )

    assert prepared.report.estimated_tokens_before >= 800
    assert prepared.report.estimated_tokens_after >= 800
    assert prepared.report.pressure.level == "tight"
    assert "tight_budget_pressure" in prepared.report.pressure.reasons


def test_context_governor_rechecks_projected_context_and_archives_tool_output(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.policy import ContextPressurePolicy
    from codepilot.sessions.context.state import SessionContextState
    from codepilot.sessions.memory.records import MemoryRecall

    class LargeMemoryRetriever:
        def recall(self, _query) -> MemoryRecall:
            return MemoryRecall(pinned_text="memory pressure " * 120)

    raw_output = "RAW_TOOL_LINE_DO_NOT_INLINE\n" * 100
    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_projection_budget",
        state=SessionContextState(workspace_dir=tmp_path),
        memory_retriever=LargeMemoryRetriever(),
        pressure_policy=ContextPressurePolicy(
            safety_margin_tokens=0,
            tight_ratio=0.72,
            critical_ratio=0.90,
        ),
    )

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[
                    UserMessage(content="Review the recent shell output."),
                    ToolResultMessage(
                        tool_call_id="call_projection",
                        tool_name="shell",
                        content=[TextContent(text=raw_output)],
                        status="success",
                    ),
                ],
            ),
            ContextPreparationRequest(
                session_id="session_projection_budget",
                model_context_window=1000,
                model_max_output_tokens=0,
            ),
        )
    )
    rendered_tool_results = "\n".join(
        getattr(block, "text", "")
        for message in prepared.messages
        if isinstance(message, ToolResultMessage)
        for block in message.content
    )

    assert prepared.report.pressure.level == "normal"
    assert prepared.report.estimated_tokens_after < prepared.report.pressure.effective_budget
    assert "[Tool output archived]" in rendered_tool_results
    assert "RAW_TOOL_LINE_DO_NOT_INLINE" not in rendered_tool_results


def test_critical_checkpoint_prefers_task_goal_over_latest_continue_prompt(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.policy import ContextPressurePolicy
    from codepilot.sessions.context.state import SessionContextState

    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_checkpoint_goal",
        state=SessionContextState(workspace_dir=tmp_path),
        pressure_policy=ContextPressurePolicy(
            safety_margin_tokens=0,
            tight_ratio=0.50,
            critical_ratio=0.60,
        ),
    )

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[
                    UserMessage(content="请修复上下文链路。"),
                    ToolResultMessage(
                        tool_call_id="call_critical",
                        tool_name="shell",
                        content=[TextContent(text="critical pressure\n" * 200)],
                        status="success",
                    ),
                    UserMessage(content="继续"),
                ],
                current_task="Goal: repair the context governor budget chain.",
            ),
            ContextPreparationRequest(
                session_id="session_checkpoint_goal",
                model_context_window=500,
                model_max_output_tokens=0,
            ),
        )
    )

    assert prepared.report.pressure.level == "critical"
    assert prepared.report.checkpoint_created is not None
    assert (
        prepared.report.checkpoint_created.goal
        == "Goal: repair the context governor budget chain."
    )


def test_critical_context_uses_llm_compactor_and_writes_compact_summary(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage
    from codepilot.sessions.context.compactor import ContextCompactResult
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.policy import ContextPressurePolicy
    from codepilot.sessions.context.state import SessionContextState

    class FakeCompactor:
        def __init__(self) -> None:
            self.calls = []

        async def compact(self, request):
            self.calls.append(request)
            return ContextCompactResult(
                recovery_summary="LLM compact: keep the failing assertion and next pytest command.",
                task_state_lines=["Compacted task: fix context budget loop."],
                working_set_lines=["Compacted evidence: pytest failed in context governor."],
                conversation_lines=["Compacted conversation: user asked to continue."],
                evidence_refs=["tool:call_critical"],
            )

    compactor = FakeCompactor()
    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_llm_compact",
        state=SessionContextState(workspace_dir=tmp_path),
        pressure_policy=ContextPressurePolicy(
            safety_margin_tokens=0,
            tight_ratio=0.50,
            critical_ratio=0.60,
        ),
        context_compactor=compactor,
    )

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[
                    UserMessage(content="请修复上下文压缩。"),
                    ToolResultMessage(
                        tool_call_id="call_critical",
                        tool_name="shell",
                        content=[TextContent(text="critical raw output\n" * 400)],
                        status="error",
                        verification={"status": "failed"},
                    ),
                    UserMessage(content="继续"),
                ],
                current_task="Goal: implement LLM compact.",
            ),
            ContextPreparationRequest(
                session_id="session_llm_compact",
                model_context_window=900,
                model_max_output_tokens=0,
            ),
        )
    )

    assert len(compactor.calls) == 1
    assert prepared.report.pressure.level != "critical"
    assert "llm_compact" in prepared.report.pressure.reasons
    assert "LLM compact: keep the failing assertion" in prepared.system_prompt
    task_state = json.loads(
        (
            tmp_path
            / ".codepilot"
            / "sessions"
            / "session_llm_compact"
            / "task_state.json"
        ).read_text(encoding="utf-8")
    )
    assert task_state["recovery_summary"].startswith("LLM compact:")
    ledger_line = (
        tmp_path
        / ".codepilot"
        / "sessions"
        / "session_llm_compact"
        / "context_ledger.jsonl"
    ).read_text(encoding="utf-8").splitlines()[-1]
    ledger = json.loads(ledger_line)
    assert ledger["compact_summary"].startswith("LLM compact:")


def test_critical_context_emergency_trims_when_compact_still_critical(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage
    from codepilot.sessions.context.compactor import ContextCompactResult
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.policy import ContextPressurePolicy
    from codepilot.sessions.context.state import SessionContextState

    class VerboseCompactor:
        async def compact(self, _request):
            return ContextCompactResult(
                recovery_summary="verbose compact summary " * 500,
                task_state_lines=["verbose task line " * 200],
                working_set_lines=["verbose evidence line " * 200],
                conversation_lines=["verbose conversation line " * 200],
            )

    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_emergency_trim",
        state=SessionContextState(workspace_dir=tmp_path),
        pressure_policy=ContextPressurePolicy(
            safety_margin_tokens=0,
            tight_ratio=0.50,
            critical_ratio=0.60,
        ),
        context_compactor=VerboseCompactor(),
    )

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[
                    UserMessage(content="请修复上下文压缩。"),
                    ToolResultMessage(
                        tool_call_id="call_critical",
                        tool_name="shell",
                        content=[TextContent(text="critical raw output\n" * 400)],
                        status="error",
                    ),
                    UserMessage(content="继续"),
                ],
                current_task="Goal: implement emergency trim.",
            ),
            ContextPreparationRequest(
                session_id="session_emergency_trim",
                model_context_window=1000,
                model_max_output_tokens=0,
            ),
        )
    )

    assert prepared.report.pressure.level != "critical"
    assert "emergency_context_trim" in prepared.report.pressure.reasons
    assert "verbose compact summary " * 20 not in prepared.system_prompt
    assert len(prepared.messages) <= 2


def test_critical_context_fails_closed_when_emergency_still_exceeds_budget(
    tmp_path: Path,
) -> None:
    import pytest
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import Tool, UserMessage
    from codepilot.sessions.context.compactor import ContextCompactResult
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.policy import ContextPressurePolicy
    from codepilot.sessions.context.state import SessionContextState

    class SmallCompactor:
        async def compact(self, _request):
            return ContextCompactResult(recovery_summary="small compact summary")

    tools = [
        Tool(
            name=f"tool_{index}",
            description="tool schema pressure",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        )
        for index in range(8)
    ]
    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_fail_closed",
        state=SessionContextState(workspace_dir=tmp_path),
        pressure_policy=ContextPressurePolicy(
            safety_margin_tokens=0,
            tight_ratio=0.50,
            critical_ratio=0.60,
        ),
        context_compactor=SmallCompactor(),
    )

    with pytest.raises(RuntimeError, match="context remains critical"):
        asyncio.run(
            governor.prepare(
                AgentContext(
                    system_prompt="System rules.",
                    messages=[UserMessage(content="Use tools.")],
                    tools=tools,
                ),
                ContextPreparationRequest(
                    session_id="session_fail_closed",
                    model_context_window=1000,
                    model_max_output_tokens=0,
                ),
            )
        )


def test_context_compiler_is_not_public_sessions_api() -> None:
    import codepilot.sessions as sessions
    import codepilot.sessions.context as context

    assert not hasattr(sessions, "ContextCompiler")
    assert not hasattr(sessions, "ContextPolicy")
    assert not hasattr(context, "ContextCompiler")
    assert not hasattr(context, "ContextPolicy")
