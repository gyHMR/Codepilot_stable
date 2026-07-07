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
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.state import SessionContextState
    from codepilot.sessions.memory.records import MemoryRecall, MemoryRecord, RetrievedMemory

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
                        content=[TextContent(text="pytest failed\n" * 50)],
                        status="error",
                        affected_paths=["test/test_app.py"],
                        verification={"status": "failed"},
                    ),
                ],
                task_state={
                    "raw_user_request": "Fix failing tests.",
                    "goal": {"value": "Fix failing tests."},
                    "current_mode": "build",
                },
            ),
            ContextPreparationRequest(
                session_id="session_1",
                model_context_window=4000,
                model_max_output_tokens=500,
            ),
        )
    )

    assert "## Task State" in prepared.system_prompt
    assert "Fix failing tests." in prepared.system_prompt
    assert "Run pytest from the repo root." in prepared.system_prompt
    assert prepared.report.context_view is not None
    assert prepared.report.retrieved_memory_ids == ["mem_1"]
    assert any(ref.path.endswith(".txt") for ref in prepared.report.artifact_refs)
    assert (tmp_path / ".codepilot" / "sessions" / "session_1" / "context_ledger.jsonl").exists()


def test_context_governor_surfaces_recent_read_paths_in_working_set(
    tmp_path: Path,
) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.state import SessionContextState

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

    assert prepared.report.tokens_by_layer["tools"] >= 800
    assert "json_schema" in prepared.report.estimation["by_type"]


def test_context_ledger_records_simple_projection(tmp_path: Path) -> None:
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context.governor import ContextGovernor

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
