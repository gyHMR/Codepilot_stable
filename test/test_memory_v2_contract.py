from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def _session_store(tmp_path: Path, session_id: str = "session_memory_v2"):
    from codepilot.sessions.storage import SessionStore

    store = SessionStore(tmp_path, session_id)
    store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    return store


def _record(
    memory_id: str,
    *,
    memory_type: str,
    subject: str,
    content: str,
    scope: str = "project",
    keywords: list[str] | None = None,
    paths: list[str] | None = None,
    status: str = "active",
    source: str = "user",
):
    from codepilot.sessions.memory import MemoryRecord

    return MemoryRecord(
        id=memory_id,
        type=memory_type,
        scope=scope,
        subject=subject,
        predicate="is",
        value=content,
        content=content,
        keywords=keywords or [],
        paths=paths or [],
        status=status,
        source=source,
    )


def test_memory_record_v3_schema_keeps_structured_payload() -> None:
    from codepilot.sessions.memory import MemoryRecord

    record = MemoryRecord(
        id="mem_constraint",
        type="constraint",
        scope="project",
        subject="constraint:project_boundary",
        predicate="is",
        value="Keep Codepilot explainable and demo-friendly.",
        content="Keep Codepilot explainable and demo-friendly.",
        keywords=["topic:architecture"],
        paths=["docs/design/2context-design.md"],
        status="active",
        evidence_refs=["user:prompt"],
        source="user",
    )

    assert record.content == "Keep Codepilot explainable and demo-friendly."
    assert record.is_retrievable
    assert record.to_dict()["schema_version"] == 3

    with pytest.raises(TypeError):
        MemoryRecord(
            id="legacy",
            scope="session",
            kind="task",  # type: ignore[call-arg]
            key="legacy:task",
            text="legacy",
            source="run",
        )

    with pytest.raises(ValueError, match="Unsupported memory schema_version"):
        MemoryRecord.from_dict(
            {
                "schema_version": 2,
                "id": "legacy",
                "scope": "session",
                "kind": "task",
                "key": "legacy:task",
                "text": "legacy",
                "source": "run",
            }
        )


def test_memory_retriever_exposes_recall_as_single_query_entrypoint() -> None:
    from codepilot.sessions.memory import MemoryRetriever

    assert hasattr(MemoryRetriever, "recall")
    assert not hasattr(MemoryRetriever, "retrieve")


def test_memory_store_normalizes_legacy_jsonl_at_read_boundary(tmp_path: Path) -> None:
    import json

    from codepilot.sessions.memory import MemoryStore

    session_store = _session_store(tmp_path)
    memory_file = session_store.layout.project_memory_file
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "id": "legacy",
                "scope": "project",
                "kind": "constraint",
                "key": "constraint:legacy",
                "text": "Legacy memory text.",
                "triggers": ["always"],
                "related_paths": ["src/app.py"],
                "source": "user",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    store = MemoryStore(session_store)
    record = store.load_project()[0]

    assert record.type == "constraint"
    assert record.subject == "constraint:legacy"
    assert record.content == "Legacy memory text."
    assert record.keywords == ["always"]
    assert record.paths == ["src/app.py"]

    store.update(record)
    payloads = [
        json.loads(line)
        for line in memory_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latest = payloads[-1]
    assert "kind" not in latest
    assert "key" not in latest
    assert "text" not in latest


def test_prompt_correction_supersedes_conflicting_project_memory(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    store = MemoryStore(_session_store(tmp_path))
    old = store.update(
        _record(
            "mem_old",
            memory_type="constraint",
            subject="constraint:context_design",
            content="Context should keep every tool output inline.",
            keywords=["topic:context"],
        )
    )
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    correction = writer.admit_prompt_memory(
        "纠正一下：上下文设计不是保留所有工具输出，而是大输出只保留摘要和 artifact 引用。",
        run_id="run_correction",
    )

    assert correction is not None
    assert correction.type == "correction"
    assert correction.scope == "project"
    assert correction.subject == "constraint:context_design"
    assert correction.status == "candidate"
    assert correction.supersedes == []
    assert "artifact 引用" in correction.content
    assert store.get(old.id).status == "active"


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
    assert first[0].type == "experience"
    assert second[0].occurrences == 2
    project = store.load_project()
    assert [record.type for record in project] == ["experience"]
    assert project[0].subject == second[0].subject
    assert project[0].source == "promoted"


def test_memory_recall_orders_layers_and_excludes_inactive(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryQuery, MemoryRetriever, MemoryStore
    from codepilot.sessions.memory.files import save_global_memory

    store = MemoryStore(_session_store(tmp_path))
    save_global_memory(tmp_path, "Always prefer focused tests before broad test suites.")
    for record in [
        _record(
            "mem_exp",
            scope="session",
            memory_type="experience",
            subject="experience:edit:multiple_matches",
            content="When edit reports multiple_matches, read the target area first.",
            keywords=["intent:edit_file", "error:multiple_matches"],
            paths=["src/app.py"],
            source="run",
        ),
        _record(
            "mem_decision",
            memory_type="decision",
            subject="decision:memory_contract",
            content="Memory stores durable knowledge only.",
            keywords=["topic:memory"],
        ),
        _record(
            "mem_constraint",
            memory_type="constraint",
            subject="constraint:project_boundary",
            content="Keep the project learning-oriented.",
            keywords=["always", "topic:architecture"],
        ),
        _record(
            "mem_correction",
            memory_type="correction",
            subject="constraint:context_design",
            content="Do not inline old large tool outputs; use artifact refs.",
            keywords=["topic:context"],
        ),
        _record(
            "mem_deleted",
            memory_type="constraint",
            subject="constraint:deleted",
            content="Deleted memory",
            status="deleted",
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
    from codepilot.core.contracts import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.state import SessionContextState
    from codepilot.sessions.memory import MemoryRecall, MemoryRecord, RetrievedMemory

    class FakeMemoryRetriever:
        def recall(self, _query):
            correction = _record(
                "mem_correction",
                memory_type="correction",
                subject="constraint:context_design",
                content="Use artifact refs for old large tool outputs.",
            )
            experience = _record(
                "mem_exp",
                scope="session",
                memory_type="experience",
                subject="experience:edit:multiple_matches",
                content="Read target area before retrying edit.",
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
