from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def _session_store(tmp_path: Path, session_id: str = "session_memory_v2"):
    from codepilot.sessions.persistence.store import SessionStore

    store = SessionStore(tmp_path, session_id)
    store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    return store


def test_memory_record_v2_schema_rejects_legacy_kinds() -> None:
    from codepilot.sessions.memory import MemoryRecord

    record = MemoryRecord(
        id="mem_constraint",
        scope="project",
        kind="constraint",
        status="active",
        key="constraint:project_boundary",
        text="Keep Codepilot explainable and demo-friendly.",
        triggers=["topic:architecture"],
        related_paths=["docs/design/2context-design.md"],
        evidence_refs=["user:prompt"],
        source="user",
    )

    assert record.text == "Keep Codepilot explainable and demo-friendly."
    assert record.is_retrievable
    assert record.to_dict()["schema_version"] == 2

    for legacy_kind in ("task", "file", "failure", "project"):
        with pytest.raises(ValueError, match="Unknown memory kind"):
            MemoryRecord(
                id=f"mem_{legacy_kind}",
                scope="session",
                kind=legacy_kind,
                key=f"legacy:{legacy_kind}",
                text="legacy",
                source="run",
            )


def test_prompt_correction_supersedes_conflicting_project_memory(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryRecord, MemoryStore, MemoryWriter

    store = MemoryStore(_session_store(tmp_path))
    old = store.update(
        MemoryRecord(
            id="mem_old",
            scope="project",
            kind="constraint",
            key="constraint:context_design",
            text="Context should keep every tool output inline.",
            triggers=["topic:context"],
            source="user",
        )
    )
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    correction = writer.admit_prompt_memory(
        "纠正一下：上下文设计不是保留所有工具输出，而是大输出只保留摘要和 artifact 引用。",
        run_id="run_correction",
    )

    assert correction is not None
    assert correction.kind == "correction"
    assert correction.scope == "project"
    assert correction.key == "constraint:context_design"
    assert correction.supersedes == [old.id]
    assert "artifact 引用" in correction.text
    assert store.get(old.id).status == "superseded"


def test_verified_experience_merges_and_promotes_after_repeat(tmp_path: Path) -> None:
    from codepilot.protocols import AgentRunResult, TextContent, ToolResultMessage
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    store = MemoryStore(_session_store(tmp_path))
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    def result(run_id: str) -> AgentRunResult:
        return AgentRunResult(
            run_id=run_id,
            session_id="session_memory_v2",
            status="completed",
            stop_reason="final_answer",
            messages=[
                ToolResultMessage(
                    tool_call_id=f"{run_id}_bad",
                    tool_name="edit",
                    content=[TextContent(text="old_text was not unique")],
                    status="error",
                    is_error=True,
                    error_code="multiple_matches",
                    affected_paths=["src/app.py"],
                ),
                ToolResultMessage(
                    tool_call_id=f"{run_id}_good",
                    tool_name="edit",
                    content=[TextContent(text="edited")],
                    status="success",
                    affected_paths=["src/app.py"],
                    workspace_changed=True,
                ),
                ToolResultMessage(
                    tool_call_id=f"{run_id}_verify",
                    tool_name="shell",
                    content=[TextContent(text="passed")],
                    verification={"status": "passed", "command": "pytest -q"},
                ),
            ],
        )

    first = writer.finalize_run(result("run_1"))
    second = writer.finalize_run(result("run_2"))

    assert len(first) == 1
    assert first[0].scope == "session"
    assert first[0].kind == "experience"
    assert second[0].occurrences == 2
    project = store.load_project()
    assert [record.kind for record in project] == ["experience"]
    assert project[0].key == second[0].key
    assert project[0].source == "promoted"


def test_memory_recall_orders_layers_and_excludes_inactive(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryQuery, MemoryRecord, MemoryRetriever, MemoryStore
    from codepilot.sessions.memory.files import save_global_memory

    store = MemoryStore(_session_store(tmp_path))
    save_global_memory(tmp_path, "Always prefer focused tests before broad test suites.")
    for record in [
        MemoryRecord(
            id="mem_exp",
            scope="session",
            kind="experience",
            key="experience:edit:multiple_matches",
            text="When edit reports multiple_matches, read the target area first.",
            triggers=["intent:edit_file", "error:multiple_matches"],
            related_paths=["src/app.py"],
            evidence_refs=["tool:bad", "tool:good", "verification:pytest"],
            source="run",
        ),
        MemoryRecord(
            id="mem_decision",
            scope="project",
            kind="decision",
            key="decision:memory_contract",
            text="Memory stores durable knowledge only.",
            triggers=["topic:memory"],
            source="user",
        ),
        MemoryRecord(
            id="mem_constraint",
            scope="project",
            kind="constraint",
            key="constraint:project_boundary",
            text="Keep the project learning-oriented.",
            triggers=["always", "topic:architecture"],
            source="user",
        ),
        MemoryRecord(
            id="mem_correction",
            scope="project",
            kind="correction",
            key="constraint:context_design",
            text="Do not inline old large tool outputs; use artifact refs.",
            triggers=["topic:context"],
            source="user",
        ),
        MemoryRecord(
            id="mem_deleted",
            scope="project",
            kind="constraint",
            key="constraint:deleted",
            text="Deleted memory",
            status="deleted",
            source="user",
        ),
    ]:
        store.update(record)

    recall = MemoryRetriever(store=store, workspace_dir=tmp_path).recall(
        MemoryQuery(
            text="修复 context memory edit 问题",
            active_paths=["src/app.py"],
            action_intent="edit_file",
            recent_error="multiple_matches",
            retrieval_mode="repair",
        )
    )

    assert recall.pinned_text == "Always prefer focused tests before broad test suites."
    assert [item.record.id for item in recall.always] == [
        "mem_correction",
        "mem_constraint",
    ]
    assert [item.record.id for item in recall.selected] == [
        "mem_decision",
        "mem_exp",
    ]
    assert "mem_deleted" in recall.dropped


def test_context_governor_uses_memory_recall_layers(tmp_path: Path) -> None:
    asyncio.run(_context_governor_memory_recall_case(tmp_path))


async def _context_governor_memory_recall_case(tmp_path: Path) -> None:
    from codepilot.core import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.state import SessionContextState
    from codepilot.sessions.memory import MemoryRecall, MemoryRecord, RetrievedMemory

    class FakeMemoryRetriever:
        def recall(self, _query):
            correction = MemoryRecord(
                id="mem_correction",
                scope="project",
                kind="correction",
                key="constraint:context_design",
                text="Use artifact refs for old large tool outputs.",
                source="user",
            )
            experience = MemoryRecord(
                id="mem_exp",
                scope="session",
                kind="experience",
                key="experience:edit:multiple_matches",
                text="Read target area before retrying edit.",
                source="run",
            )
            return MemoryRecall(
                pinned_text="Pinned: UTF-8 only.",
                always=[
                    RetrievedMemory(correction, score=1000, reasons=["layer:correction"])
                ],
                selected=[
                    RetrievedMemory(experience, score=90, reasons=["error:multiple_matches"])
                ],
                dropped={},
            )

    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_memory_recall",
        state=SessionContextState(workspace_dir=tmp_path),
        memory_retriever=FakeMemoryRetriever(),
    )

    prepared = await governor.prepare(
        AgentContext(
            system_prompt="rules",
            messages=[UserMessage(content="fix edit multiple matches")],
            task_signal={
                "action_intent": "edit_file",
                "recent_error_code": "multiple_matches",
            },
        ),
        ContextPreparationRequest(
            session_id="session_memory_recall",
            model_context_window=4000,
            model_max_output_tokens=500,
        ),
    )

    recalled = prepared.report.context_view.recalled_memory
    assert recalled == [
        "[Pinned memory] Pinned: UTF-8 only.",
        "[Correction] Use artifact refs for old large tool outputs. [reasons=layer:correction]",
        "[Experience] Read target area before retrying edit. [reasons=error:multiple_matches]",
    ]
    assert prepared.report.retrieved_memory_ids == ["mem_correction", "mem_exp"]
