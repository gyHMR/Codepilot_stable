from __future__ import annotations

import sys
from pathlib import Path
import asyncio

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_web_contract_keeps_browser_as_interface_only() -> None:
    from codepilot.interfaces.web import WebPromptRequest, WebSessionRef, describe_web_contract

    contract = describe_web_contract()
    request = WebPromptRequest(session=WebSessionRef(workspace_dir="."), text="hello")

    assert request.text == "hello"
    assert "codepilot.runtime" in contract["delegates_to"]
    assert "filesystem_mutation" in contract["non_responsibilities"]
    assert any(route["path"] == "/api/sessions/{session_id}/messages" for route in contract["routes"])


def test_web_backend_can_list_created_sessions_without_server(tmp_path) -> None:
    from codepilot.interfaces.web import WebCreateSessionRequest, create_web_app

    backend = create_web_app()
    event = backend.create_session(
        WebCreateSessionRequest(
            workspace_dir=str(tmp_path),
            provider="openai",
            model_id="gpt-4o-mini",
            load_workspace_resources=False,
        )
    )

    assert event.type == "session_created"
    assert backend.list_sessions()[0].session_id == event.session_id
    backend.runtime.close_all()


def test_runtime_service_send_message_yields_events_before_run_finishes() -> None:
    asyncio.run(_run_runtime_service_send_message_streaming_case())


def test_runtime_service_continue_session_yields_events_before_run_finishes() -> None:
    asyncio.run(_run_runtime_service_continue_session_streaming_case())


def test_runtime_service_exposes_run_result_events_and_report(tmp_path: Path) -> None:
    from codepilot.protocols import (
        AgentRunCounters,
        AgentRunResult,
        AssistantMessage,
        TextContent,
    )
    from codepilot.runtime.service import RuntimeService
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, "session_runtime")
    store.ensure_initialized(model_id="m", provider="p", system_prompt="")
    final = AssistantMessage(content=[TextContent(text="done")])
    result = AgentRunResult(
        run_id="run_runtime",
        session_id="session_runtime",
        status="completed",
        stop_reason="final_answer",
        counters=AgentRunCounters(model_attempts=1, tool_calls=0),
        messages=[final],
        final_message=final,
    )
    store.append_event(
        {
            "type": "agent_start",
            "runId": "run_runtime",
            "turnId": 0,
            "eventId": "run_runtime:1",
            "timestamp": 10,
            "sessionId": "session_runtime",
        }
    )
    store.append_run_result(result)

    class FakeSession:
        session_id = "session_runtime"

        def __init__(self) -> None:
            self.store = store

        def close(self) -> None:
            return None

    runtime = RuntimeService()
    runtime._sessions["session_runtime"] = FakeSession()  # type: ignore[assignment]

    assert runtime.list_runs("session_runtime")[0]["run_id"] == "run_runtime"
    assert runtime.get_run_result("session_runtime", "run_runtime")["status"] == "completed"
    assert runtime.get_run_events("session_runtime", "run_runtime")[0]["type"] == "agent_start"
    report = runtime.get_run_report("session_runtime", "run_runtime")
    assert report["summary"]["run_id"] == "run_runtime"
    assert report["summary"]["status"] == "completed"


async def _run_runtime_service_send_message_streaming_case() -> None:
    from codepilot.runtime.service import RuntimeService, UserInput

    class FakeSession:
        session_id = "s1"

        def __init__(self) -> None:
            self.listeners = []
            self.run_can_finish = asyncio.Event()
            self.run_finished = False

        def subscribe(self, listener):
            self.listeners.append(listener)

            def unsubscribe():
                self.listeners.remove(listener)

            return unsubscribe

        async def run(self, text, *, images=None):
            _ = text, images
            for listener in list(self.listeners):
                listener({"type": "message_update", "delta": "hello"})
            await self.run_can_finish.wait()
            self.run_finished = True

        async def continue_run(self):
            for listener in list(self.listeners):
                listener({"type": "turn_start"})
            await self.run_can_finish.wait()
            self.run_finished = True

    runtime = RuntimeService()
    fake = FakeSession()
    runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

    stream = runtime.send_message("s1", UserInput(text="hello"))
    first_event = await asyncio.wait_for(anext(stream), timeout=1)

    assert first_event["type"] == "message_update"
    assert fake.run_finished is False

    fake.run_can_finish.set()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(stream), timeout=1)


async def _run_runtime_service_continue_session_streaming_case() -> None:
    from codepilot.runtime.service import RuntimeService

    class FakeSession:
        session_id = "s1"

        def __init__(self) -> None:
            self.listeners = []
            self.continue_can_finish = asyncio.Event()
            self.continue_finished = False

        def subscribe(self, listener):
            self.listeners.append(listener)

            def unsubscribe():
                self.listeners.remove(listener)

            return unsubscribe

        async def continue_run(self):
            for listener in list(self.listeners):
                listener({"type": "turn_start"})
            await self.continue_can_finish.wait()
            self.continue_finished = True

    runtime = RuntimeService()
    fake = FakeSession()
    runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

    stream = runtime.continue_session("s1")
    first_event = await asyncio.wait_for(anext(stream), timeout=1)

    assert first_event["type"] == "turn_start"
    assert fake.continue_finished is False

    fake.continue_can_finish.set()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(stream), timeout=1)
