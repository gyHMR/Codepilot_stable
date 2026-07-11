from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_context_governor_prepares_linear_context_with_memory_and_artifacts(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage
    from codepilot.sessions.context import ContextGovernor
    from codepilot.sessions.context import SessionContextState
    from codepilot.sessions.memory import MemoryRecall, MemoryRecord, RetrievedMemory

    class FakeMemoryRetriever:
        def recall(self, _query) -> MemoryRecall:
            return MemoryRecall(
                retrieved=[
                    RetrievedMemory(
                        record=MemoryRecord(
                            id="mem_1",
                            type="experience",
                            scope="project",
                            subject="experience:pytest",
                            predicate="is",
                            value="Run pytest from the repo root.",
                            content="Run pytest from the repo root.",
                            keywords=["pytest"],
                            status="active",
                            source="task_experience",
                            confidence="observed",
                            created_by_session_id="session_1",
                            evidence_refs=["session:session_1"],
                        ),
                        score=90,
                        reasons=["test"],
                    )
                ]
            )

    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_1",
        state=SessionContextState(workspace_dir=tmp_path),
        memory_retriever=FakeMemoryRetriever(),
    )

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[
                    UserMessage(content="Fix failing tests."),
                    ToolResultMessage(
                        tool_call_id="call_1",
                        tool_name="shell",
                        content=[TextContent(text="pytest failed\n" * 600)],
                        status="error",
                        affected_paths=["test/test_app.py"],
                        verification={"status": "failed"},
                    ),
                ],
                mode="build",
                plan_state={
                    "schema_version": 6,
                    "plan_id": "plan_1",
                    "owner_run_id": "run_1",
                    "status": "active",
                    "origin_mode": "build",
                    "raw_user_request": "Fix failing tests.",
                    "interpreted_goal": "Fix failing tests.",
                    "task_understanding": "Fix the failing tests with a focused implementation change.",
                    "current_implementation": "pytest currently fails and points at test/test_app.py.",
                    "target_design": "Repair the implementation while keeping the current API.",
                    "impact_scope": "Implementation and focused tests for the failing path.",
                    "risks_and_open_questions": ["No open blocker."],
                    "verification_plan": "Run pytest from the repo root.",
                    "summary": "Fix the failing test suite with a focused change.",
                    "completion_criteria": ["Focused tests pass"],
                    "items": [
                        {
                            "id": "item_1",
                            "step": "Fix failing tests.",
                            "details": "Locate and repair the failing implementation.",
                            "verification": "Run the focused tests.",
                            "status": "in_progress",
                        }
                    ],
                    "revision": 1,
                    "explanation": "",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "completed_at": None,
                    "completion_source": None,
                },
                run_signals={"verification_status": "failed"},
                runtime_state={
                    "run_id": "run_1",
                    "mode": "build",
                    "checkpoint_phase": "running",
                    "mode_policy": "Execute the approved plan.",
                },
            ),
            ContextPreparationRequest(
                session_id="session_1",
                model_context_window=4000,
                model_max_output_tokens=500,
            ),
        )
    )

    assert "## Mode Policy" in prepared.system_prompt
    assert "## Runtime State" in prepared.system_prompt
    assert "## Current User Request" in prepared.system_prompt
    assert "## Task Plan" in prepared.system_prompt
    assert "Approved Execution Contract" in prepared.system_prompt
    assert "Fix failing tests." in prepared.system_prompt
    assert "Run pytest from the repo root." in prepared.system_prompt
    assert prepared.report.context_view is not None
    assert prepared.report.retrieved_memory_ids == ["mem_1"]
    assert any(ref.path.endswith(".txt") for ref in prepared.report.artifact_refs)
    assert (tmp_path / ".codepilot" / "sessions" / "session_1" / "context_ledger.jsonl").exists()


def test_context_governor_filters_archived_plan_from_store(tmp_path: Path) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context import ContextGovernor
    from codepilot.sessions.plan_state import PlanStateStore
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_archived_plan")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    PlanStateStore(session_store).save(
        {
                "schema_version": 6,
            "plan_id": "plan_done",
            "owner_run_id": "run_done",
            "status": "completed",
            "origin_mode": "plan",
                "raw_user_request": "旧任务",
                "interpreted_goal": "旧任务",
            "task_understanding": "旧任务已经完成。",
            "current_implementation": "旧任务完成时的实现证据。",
            "target_design": "旧任务目标设计。",
            "impact_scope": "旧任务影响范围。",
            "risks_and_open_questions": ["旧任务无待确认项。"],
            "verification_plan": "旧任务验证方案。",
            "summary": "旧计划已经完成。",
            "completion_criteria": ["旧任务完成"],
            "items": [
                {
                    "id": "item_1",
                    "step": "旧步骤",
                    "details": "旧步骤详情。",
                    "verification": "旧验证。",
                    "status": "pending",
                }
            ],
            "revision": 1,
            "explanation": "",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T01:00:00+00:00",
            "completed_at": "2026-01-01T01:00:00+00:00",
            "completion_source": "model_closeout",
        }
    )
    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_archived_plan",
        store=session_store,
    )

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[UserMessage(content="开始新任务。")],
                mode="build",
            ),
            ContextPreparationRequest(
                session_id="session_archived_plan",
                model_context_window=4000,
                model_max_output_tokens=500,
            ),
        )
    )

    assert "## Task Plan" not in prepared.system_prompt
    assert prepared.report.context_view is not None
    assert prepared.report.context_view.task_plan == []


def test_context_governor_renders_synthetic_control_separately(tmp_path: Path) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context import ContextGovernor

    governor = ContextGovernor(workspace_dir=tmp_path, session_id="session_synthetic_control")

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[UserMessage(content="回答问题")],
                mode="build",
                runtime_state={
                    "run_id": "run_1",
                    "mode": "build",
                    "checkpoint_phase": "running",
                    "mode_policy": "Execute the task.",
                    "synthetic_control": {
                        "source": "runner",
                        "kind": "empty_final_answer",
                        "scope": "final_answer_only",
                        "instruction": "你刚才没有给出用户可见的最终答复。",
                        "expires_after_turns": 1,
                    },
                },
            ),
            ContextPreparationRequest(
                session_id="session_synthetic_control",
                model_context_window=4000,
                model_max_output_tokens=500,
            ),
        )
    )

    assert "## Synthetic Control" in prepared.system_prompt
    assert "Scope: final_answer_only" in prepared.system_prompt
    assert "This is not a user request" in prepared.system_prompt
    assert "Raw request: 回答问题" in prepared.system_prompt
    assert "Raw request: 你刚才没有给出用户可见的最终答复" not in prepared.system_prompt


def test_context_governor_surfaces_recent_read_paths_in_working_set(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage
    from codepilot.sessions.context import ContextGovernor
    from codepilot.sessions.context import SessionContextState

    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_read_working_set",
        state=SessionContextState(workspace_dir=tmp_path),
    )

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[
                    UserMessage(content="Review src/app.py."),
                    ToolResultMessage(
                        tool_call_id="read_1",
                        tool_name="read",
                        content=[TextContent(text="1\tprint('hello')")],
                        status="success",
                        metadata={
                            "read_paths": ["src/app.py"],
                            "file_state": {"path": "src/app.py", "sha256": "abc"},
                        },
                    ),
                ],
            ),
            ContextPreparationRequest(
                session_id="session_read_working_set",
                model_context_window=4000,
                model_max_output_tokens=500,
            ),
        )
    )

    assert prepared.report.context_view is not None
    assert any(
        "Active file: src/app.py role=target" in line
        for line in prepared.report.context_view.working_set
    )


def test_context_governor_counts_tool_schemas_in_budget_estimates(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import Tool, UserMessage
    from codepilot.sessions.context import ContextGovernor
    from codepilot.sessions.context import ContextPressurePolicy
    from codepilot.sessions.context import SessionContextState

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

    assert prepared.report.tokens_by_layer["tools"] >= 800
    assert "json_schema" in prepared.report.estimation["by_type"]


def test_context_ledger_records_simple_projection(tmp_path: Path) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context import ContextGovernor

    governor = ContextGovernor(workspace_dir=tmp_path, session_id="session_ledger")

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(
                system_prompt="System rules.",
                messages=[UserMessage(content="hello")],
            ),
            ContextPreparationRequest(
                session_id="session_ledger",
                model_context_window=4000,
                model_max_output_tokens=500,
            ),
        )
    )

    ledger_path = tmp_path / ".codepilot" / "sessions" / "session_ledger" / "context_ledger.jsonl"
    payload = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[-1])

    assert payload["type"] == "context_projection"
    assert payload["context_id"] == prepared.report.context_id
    assert "tokens_by_layer" in payload
    assert "task_plan" in payload["tokens_by_layer"]
    assert "runtime" in payload["tokens_by_layer"]
    assert "memory_retrieval_reasons" in payload
    assert "dropped_memory_reasons" in payload
    assert "runner_preflight" in payload


def test_context_governor_compacts_old_conversation_on_critical_pressure(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context import ContextGovernor

    messages = [
        UserMessage(
            content=f"old request {index} " + ("details " * 80),
            metadata={"session_message_id": f"msg_{index:03d}"},
        )
        for index in range(18)
    ]
    governor = ContextGovernor(workspace_dir=tmp_path, session_id="session_compact")

    prepared = asyncio.run(
        governor.prepare(
            AgentContext(system_prompt="System rules.", messages=messages),
            ContextPreparationRequest(
                session_id="session_compact",
                model_context_window=900,
                model_max_output_tokens=100,
            ),
        )
    )

    meta = governor.store.read_meta()
    context_meta = meta["context"]
    ledger_path = tmp_path / ".codepilot" / "sessions" / "session_compact" / "context_ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]

    assert context_meta["compacted_until_message_id"] == "msg_011"
    assert context_meta["last_compact_summary"]
    assert any(row["type"] == "context_compaction" for row in rows)
    assert [message.metadata.get("session_message_id") for message in prepared.messages] == [
        "msg_012",
        "msg_013",
        "msg_014",
        "msg_015",
        "msg_016",
        "msg_017",
    ]


def test_token_estimator_classifies_content_types_and_calibrates_usage(
    tmp_path: Path,
) -> None:
    from codepilot.llm.estimation import (
        ContextUsageCalibrator,
        classify_content_type,
        estimate_text_tokens,
    )

    assert classify_content_type("请按照上下文设计进行完全重构") == "chinese_text"
    assert classify_content_type('{"type":"object","properties":{"x":{"type":"string"}}}') == "json_schema"
    assert classify_content_type("def run():\n    return {'ok': True}\n") == "code_text"
    assert classify_content_type("Plain English sentence for estimation.") == "english_text"

    raw = estimate_text_tokens("hello world", content_type="english_text").total
    adjusted = estimate_text_tokens(
        "hello world",
        content_type="english_text",
        correction_factors={"english_text": 1.8},
    ).total
    assert adjusted >= raw

    calibrator = ContextUsageCalibrator(tmp_path)
    calibrator.update(
        provider="test",
        model="model",
        raw_estimate=100,
        actual_input_tokens=1000,
        breakdown={"english_text": 100},
    )

    payload = json.loads((tmp_path / ".codepilot" / "context_usage.json").read_text(encoding="utf-8"))
    record = payload["test:model:english_text"]
    assert record["correction_factor"] <= 1.8
    assert record["sample_count"] == 1
