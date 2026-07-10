from __future__ import annotations

import asyncio
import json


def _metadata(name: str, *, read_only: bool, scopes=("read", "plan", "build")):
    from codepilot.tools.contracts import ToolMetadata

    return ToolMetadata(
        name=name,
        category="test",
        read_only=read_only,
        concurrency_safe=True,
        exclusive=False,
        requires_approval=False,
        risk_level="low",
        scopes=tuple(scopes),
    )


def test_restricted_tool_port_only_exposes_and_executes_read_allowlist() -> None:
    async def run_case() -> None:
        from codepilot.protocols import TextContent, Tool
        from codepilot.tools.contracts import (
            ToolCatalogItem,
            ToolCatalogView,
            ToolInvocation,
            ToolObservation,
        )
        from codepilot.tools.restricted import RestrictedToolPort

        class BaseTools:
            def __init__(self) -> None:
                self.executed_modes: list[str] = []

            def catalog(self, current_mode: str = "build"):
                return ToolCatalogView(
                    (
                        ToolCatalogItem(
                            spec=Tool(name="read", description="Read", parameters={}),
                            metadata=_metadata("read", read_only=True),
                        ),
                        ToolCatalogItem(
                            spec=Tool(name="update_plan", description="Plan", parameters={}),
                            metadata=_metadata("update_plan", read_only=True),
                        ),
                        ToolCatalogItem(
                            spec=Tool(name="dispatch_exploration", description="Dispatch", parameters={}),
                            metadata=_metadata("dispatch_exploration", read_only=True, scopes=("plan",)),
                        ),
                        ToolCatalogItem(
                            spec=Tool(name="write", description="Write", parameters={}),
                            metadata=_metadata("write", read_only=False, scopes=("build",)),
                        ),
                    )
                )

            async def execute(self, invocation):
                self.executed_modes.append(invocation.current_mode)
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="success",
                    content=(TextContent(text="ok"),),
                )

        base = BaseTools()
        restricted = RestrictedToolPort(base)

        names = {item.spec.name for item in restricted.catalog("plan").items}
        assert names == {"read"}

        allowed = await restricted.execute(
            ToolInvocation(run_id="run1", tool_call_id="read1", name="read", current_mode="plan")
        )
        denied = await restricted.execute(
            ToolInvocation(run_id="run1", tool_call_id="write1", name="write", current_mode="read")
        )
        denied_plan = await restricted.execute(
            ToolInvocation(run_id="run1", tool_call_id="plan1", name="update_plan", current_mode="read")
        )

        assert allowed.status == "success"
        assert base.executed_modes == ["read"]
        assert denied.status == "denied"
        assert denied.metadata["error_code"] == "restricted_tool_denied"
        assert denied_plan.status == "denied"

    asyncio.run(run_case())


def test_subagent_store_persists_session_scoped_reports_and_marks_stale(tmp_path) -> None:
    from codepilot.sessions.subagents import SubagentStore

    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("print('v1')\n", encoding="utf-8")

    store = SubagentStore(tmp_path, "session_a")
    stored = store.append_report(
        subagent_id="runtime-reader",
        purpose="Read runtime",
        scope_key="scope1",
        focus_paths=["src"],
        report={
            "status": "completed",
            "summary": "found runtime",
            "findings": ["src/app.py exists"],
            "relevant_files": ["src/app.py"],
            "evidence": [{"path": "src/app.py", "note": "entry"}],
            "risks": [],
            "suggested_plan_notes": [],
            "open_questions": [],
            "confidence": 0.8,
        },
        evidence_paths=["src/app.py"],
    )

    agents = store.list_agents()
    assert stored["report_id"]
    assert agents[0]["subagent_id"] == "runtime-reader"
    assert agents[0]["stale"] is False
    assert (tmp_path / ".codepilot" / "sessions" / "session_a" / "subagents").exists()

    source.write_text("print('v2')\n", encoding="utf-8")

    assert store.list_agents()[0]["stale"] is True


def test_exploration_coordinator_returns_partial_results_and_skips_duplicates(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import ModelDescriptor
        from codepilot.runtime.subagents import ExplorationCoordinator
        from codepilot.sessions.subagents import SubagentStore

        class FakeRunner:
            async def run(self, task, *, peer_assignments, previous_report=None):
                if "risk" in task.purpose.lower():
                    return {
                        "status": "failed",
                        "task_id": task.task_id,
                        "subagent_id": task.subagent_id,
                        "purpose": task.purpose,
                        "scope_key": task.scope_key,
                        "summary": "",
                        "findings": [],
                        "relevant_files": [],
                        "evidence": [],
                        "risks": [],
                        "suggested_plan_notes": [],
                        "open_questions": [],
                        "confidence": 0.0,
                        "error": {"code": "fake.failure"},
                    }
                return {
                    "status": "completed",
                    "task_id": task.task_id,
                    "subagent_id": task.subagent_id,
                    "purpose": task.purpose,
                    "scope_key": task.scope_key,
                    "summary": "ok",
                    "findings": ["fact"],
                    "relevant_files": [],
                    "evidence": [],
                    "risks": [],
                    "suggested_plan_notes": [],
                    "open_questions": [],
                    "confidence": 1.0,
                }

        coordinator = ExplorationCoordinator(
            workspace=tmp_path,
            session_id="session_a",
            model=ModelDescriptor(provider="fake", model_id="unit"),
            model_port=object(),
            tool_port=object(),
            store=SubagentStore(tmp_path, "session_a"),
            runner_factory=FakeRunner,
        )

        result = await coordinator.dispatch(
            {
                "tasks": [
                    {
                        "subagent_id": "reader",
                        "purpose": "Read runtime",
                        "instruction": "read",
                        "focus_paths": ["src/codepilot/runtime"],
                    },
                    {
                        "subagent_id": "risk",
                        "purpose": "Risk review",
                        "instruction": "risk",
                        "focus_paths": ["src/codepilot/tools"],
                    },
                    {
                        "subagent_id": "reader",
                        "purpose": "Read runtime duplicate",
                        "instruction": "read again",
                        "focus_paths": ["src/codepilot/runtime"],
                    },
                ]
            }
        )

        statuses = [report["status"] for report in result["reports"]]
        assert result["batch_status"] == "partial_failed"
        assert "completed" in statuses
        assert "failed" in statuses
        assert "skipped_duplicate" in statuses
        assert result["created_subagent_ids"] == ["reader", "risk"]

    asyncio.run(run_case())


def test_plan_mode_dispatch_exploration_feeds_update_plan_and_pauses(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, Model, TextContent, ToolCall
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent
        from codepilot.runtime.actions import PromptSubmitted, RunPausedFrame

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")

        class ModelPort:
            def __init__(self) -> None:
                self.parent_calls = 0
                self.subagent_calls = 0

            async def stream(self, request):
                if request.correlation.run_id.startswith("subrun_"):
                    self.subagent_calls += 1
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                TextContent(
                                    text=json.dumps(
                                        {
                                            "status": "completed",
                                            "summary": "src/app.py is the only source file.",
                                            "findings": ["src/app.py exists"],
                                            "relevant_files": ["src/app.py"],
                                            "evidence": [{"path": "src/app.py", "note": "source"}],
                                            "risks": [],
                                            "suggested_plan_notes": ["edit src/app.py after approval"],
                                            "open_questions": [],
                                            "confidence": 0.9,
                                        }
                                    )
                                )
                            ]
                        )
                    )
                    return
                self.parent_calls += 1
                if self.parent_calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id="explore1",
                                    name="dispatch_exploration",
                                    arguments={
                                        "tasks": [
                                            {
                                                "subagent_id": "src-reader",
                                                "purpose": "Read src",
                                                "instruction": "Find source files",
                                                "focus_paths": ["src"],
                                            }
                                        ]
                                    },
                                )
                            ]
                        )
                    )
                    return
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="plan1",
                                name="update_plan",
                                arguments={
                                    "summary": "Use exploration evidence to edit src/app.py.",
                                    "plan": [
                                        {
                                            "step": "修改 src/app.py",
                                            "details": "批准后基于探索结果修改 src/app.py。",
                                            "verification": "运行相关 Python 检查。",
                                            "status": "pending",
                                        }
                                    ],
                                },
                            )
                        ]
                    )
                )

        model_port = ModelPort()
        gateway = RuntimeGateway(model_port=model_port)
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                current_mode="plan",
                memory_enabled=False,
                model=Model(
                    id="unit",
                    name="Unit",
                    api="unit",
                    provider="unit",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
            )
        )

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="plan edit"))
        ]
        paused = [frame for frame in frames if isinstance(frame, RunPausedFrame)]
        session = gateway._require_session(ref.session_id)._session  # noqa: SLF001
        messages = session.store.load_session_messages()

        assert paused
        assert paused[-1].record.stop_reason == "plan_approval_required"
        assert model_port.subagent_calls == 1
        assert any(
            message.tool_name == "dispatch_exploration"
            for message in messages
            if getattr(message, "role", "") == "toolResult"
        )
        assert not any(
            "src/app.py is the only source file" in str(getattr(message, "content", ""))
            for message in messages
            if getattr(message, "role", "") == "assistant"
        )

    asyncio.run(run_case())
