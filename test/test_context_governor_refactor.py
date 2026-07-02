from __future__ import annotations

import asyncio
from pathlib import Path


def test_pressure_policy_uses_effective_budget_and_three_levels() -> None:
    from codepilot.core import ContextPreparationRequest
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
        working_state=["goal: fix failing tests"],
        recalled_memory=["previous pytest failure required cwd setup"],
        evidence=["pytest failed before fix"],
        recent_messages=["user: fix tests"],
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
        tokens_by_layer={"stable_rules": 20, "evidence": 40},
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
    from codepilot.core import AgentContext, ContextPreparationRequest
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.policy import ContextPressurePolicy
    from codepilot.sessions.context.state import SessionContextState
    from codepilot.sessions.memory.records import MemoryRecord, RetrievedMemory

    class FakeMemoryRetriever:
        def validate_freshness(self) -> list[object]:
            return []

        def pinned_memory(self) -> str:
            return "Pinned: use UTF-8 and LF."

        def retrieve(self, _query) -> list[RetrievedMemory]:
            return [
                RetrievedMemory(
                    record=MemoryRecord(
                        id="mem_1",
                        kind="experience",
                        scope="session",
                        content={"lesson": "Previous pytest failure required cwd setup."},
                        source="run_summary",
                        trust="verified",
                    ),
                    score=90,
                    reasons=["recent_error"],
                )
            ]

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
    assert governor.checkpoints.load_latest() is not None
    session_dir = tmp_path / ".codepilot" / "sessions" / "session_1"
    assert (session_dir / "context_ledger.jsonl").exists()
    assert not (session_dir / "context_views.jsonl").exists()
    assert not (session_dir / "checkpoints.jsonl").exists()


def test_context_compiler_is_not_public_sessions_api() -> None:
    import codepilot.sessions as sessions
    import codepilot.sessions.context as context

    assert not hasattr(sessions, "ContextCompiler")
    assert not hasattr(sessions, "ContextPolicy")
    assert not hasattr(context, "ContextCompiler")
    assert not hasattr(context, "ContextPolicy")
