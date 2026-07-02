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


def test_web_create_session_defaults_leave_runtime_config_unspecified() -> None:
    from codepilot.interfaces.web import WebCreateSessionRequest

    request = WebCreateSessionRequest(workspace_dir=".")

    assert request.system_prompt is None
    assert request.read_only_mode is None


def test_web_create_session_request_normalizes_public_config_snapshot() -> None:
    from codepilot.interfaces.web import WebCreateSessionRequest

    request = WebCreateSessionRequest(
        workspace_dir="  E:/workspace  ",
        provider=" openai ",
        model_id=" gpt-4o-mini ",
        system_prompt="  Be concise.  ",
        session_id=" session_1 ",
        read_only_mode=True,
        load_workspace_resources=False,
    )
    blank_optional = WebCreateSessionRequest(
        workspace_dir=".",
        provider=" ",
        model_id=" ",
        system_prompt=" ",
        session_id=" ",
    )

    assert request.workspace_dir == "E:/workspace"
    assert request.provider == "openai"
    assert request.model_id == "gpt-4o-mini"
    assert request.system_prompt == "Be concise."
    assert request.session_id == "session_1"
    assert request.read_only_mode is True
    assert request.load_workspace_resources is False
    assert blank_optional.provider is None
    assert blank_optional.model_id is None
    assert blank_optional.system_prompt is None
    assert blank_optional.session_id is None

    with pytest.raises(ValueError, match="workspace"):
        WebCreateSessionRequest(workspace_dir=" ")

    with pytest.raises(TypeError, match="read_only_mode"):
        WebCreateSessionRequest(workspace_dir=".", read_only_mode="yes")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="load_workspace_resources"):
        WebCreateSessionRequest(workspace_dir=".", load_workspace_resources=None)  # type: ignore[arg-type]


def test_web_session_ref_normalizes_workspace_and_optional_session() -> None:
    from codepilot.interfaces.web import WebSessionRef

    ref = WebSessionRef(
        workspace_dir="  E:/workspace  ",
        session_id=" session_1 ",
    )
    anonymous = WebSessionRef(workspace_dir=".", session_id=" ")

    assert ref.workspace_dir == "E:/workspace"
    assert ref.session_id == "session_1"
    assert anonymous.session_id is None

    with pytest.raises(ValueError, match="workspace"):
        WebSessionRef(workspace_dir=" ")

    with pytest.raises(TypeError, match="session_id"):
        WebSessionRef(workspace_dir=".", session_id=123)  # type: ignore[arg-type]


def test_web_schema_rejects_unknown_event_and_approval_types() -> None:
    from codepilot.interfaces.web import WebEventEnvelope, WebToolApproval

    with pytest.raises(ValueError, match="Unknown web event type"):
        WebEventEnvelope(type="progress")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown web approval decision"):
        WebToolApproval(
            session_id="session_1",
            tool_call_id="tool_1",
            decision="maybe",  # type: ignore[arg-type]
        )


def test_web_event_envelope_normalizes_transport_snapshot() -> None:
    from codepilot.interfaces.web import WebEventEnvelope

    payload = {"message": "hello"}
    event = WebEventEnvelope(
        type="agent_event",
        session_id=" session_1 ",
        payload=payload,
    )
    anonymous = WebEventEnvelope(
        type="session_state",
        session_id=" ",
    )
    payload["message"] = "changed"

    assert event.session_id == "session_1"
    assert event.payload == {"message": "hello"}
    assert anonymous.session_id is None

    with pytest.raises(TypeError, match="payload"):
        WebEventEnvelope(type="agent_event", payload=[])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="session_id"):
        WebEventEnvelope(type="agent_event", session_id=123)  # type: ignore[arg-type]


def test_web_readonly_descriptors_normalize_identity_and_routes() -> None:
    from codepilot.interfaces.web import WebRouteSpec, WebSessionSummary

    summary = WebSessionSummary(session_id=" session_1 ")
    route = WebRouteSpec(
        method=" get ",
        path=" /api/sessions ",
        description=" List sessions ",
    )

    assert summary.session_id == "session_1"
    assert route.method == "GET"
    assert route.path == "/api/sessions"
    assert route.description == "List sessions"

    with pytest.raises(ValueError, match="session_id"):
        WebSessionSummary(session_id=" ")

    with pytest.raises(ValueError, match="method"):
        WebRouteSpec(method="PATCH", path="/api/sessions", description="Patch")

    with pytest.raises(ValueError, match="path"):
        WebRouteSpec(method="GET", path="sessions", description="List")

    with pytest.raises(ValueError, match="description"):
        WebRouteSpec(method="GET", path="/api/sessions", description=" ")


def test_web_route_spec_serializes_public_contract_explicitly() -> None:
    from codepilot.interfaces.web import WebRouteSpec, describe_web_contract

    route = WebRouteSpec(
        method=" get ",
        path=" /api/sessions ",
        description=" List sessions ",
    )

    assert route.to_dict() == {
        "method": "GET",
        "path": "/api/sessions",
        "description": "List sessions",
    }
    assert all(
        set(item) == {"method", "path", "description"}
        for item in describe_web_contract()["routes"]
    )


def test_web_tool_approval_normalizes_recovery_identity() -> None:
    from codepilot.interfaces.web import WebToolApproval

    approval = WebToolApproval(
        session_id=" session_1 ",
        tool_call_id=" tool_1 ",
        approval_id=" approval_1 ",
        decision="approve",
        reason=" user confirmed ",
    )
    fallback = WebToolApproval(
        session_id="session_1",
        tool_call_id="tool_1",
        approval_id=" ",
        decision="deny",
    )

    assert approval.session_id == "session_1"
    assert approval.tool_call_id == "tool_1"
    assert approval.approval_id == "approval_1"
    assert approval.reason == "user confirmed"
    assert fallback.approval_id is None

    with pytest.raises(ValueError, match="session_id"):
        WebToolApproval(session_id=" ", tool_call_id="tool_1", decision="approve")

    with pytest.raises(ValueError, match="tool_call_id"):
        WebToolApproval(session_id="session_1", tool_call_id=" ", decision="approve")

    with pytest.raises(TypeError, match="approval_id"):
        WebToolApproval(
            session_id="session_1",
            tool_call_id="tool_1",
            approval_id=123,  # type: ignore[arg-type]
            decision="approve",
        )


def test_web_prompt_request_normalizes_public_input_snapshot() -> None:
    from codepilot.interfaces.web import WebPromptRequest, WebSessionRef

    images = [" screen.png "]
    request = WebPromptRequest(
        session=WebSessionRef(workspace_dir="."),
        text="  hello  ",
        images=images,
    )
    images.append("late.png")

    assert request.text == "hello"
    assert request.images == ("screen.png",)

    with pytest.raises(ValueError, match="text"):
        WebPromptRequest(session=WebSessionRef(workspace_dir="."), text=" ")

    with pytest.raises(ValueError, match="image"):
        WebPromptRequest(
            session=WebSessionRef(workspace_dir="."),
            text="hello",
            images=["ok.png", " "],
        )

    with pytest.raises(TypeError, match="session"):
        WebPromptRequest(session="session_1", text="hello")  # type: ignore[arg-type]


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
    from codepilot.sessions.persistence.store import SessionStore

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
    assert runtime.get_run_events("session_runtime", "run_runtime")[0]["type"] == "run_started"
    report = runtime.get_run_report("session_runtime", "run_runtime")
    assert report["summary"]["run_id"] == "run_runtime"
    assert report["summary"]["status"] == "completed"


def test_web_backend_submits_tool_approval_by_approval_id() -> None:
    from codepilot.interfaces.web.api import WebConsoleBackend
    from codepilot.interfaces.web.schemas import WebToolApproval

    class FakeRuntime:
        def __init__(self) -> None:
            self.received = None

        async def approve_tool_call(self, approval_id, decision, *, session_id=None):
            self.received = (approval_id, decision, session_id)

    runtime = FakeRuntime()
    backend = WebConsoleBackend(runtime=runtime)  # type: ignore[arg-type]

    event = asyncio.run(
        backend.approve_tool(
            WebToolApproval(
                session_id="s1",
                tool_call_id="tool_1",
                approval_id="approval_1",
                decision="approve",
            )
        )
    )

    assert runtime.received == ("approval_1", "approve", "s1")
    assert event.payload["approval_id"] == "approval_1"


def test_web_event_adapter_exposes_tool_approval_as_dedicated_event() -> None:
    from codepilot.interfaces.web.event_adapter import agent_event_to_web

    envelope = agent_event_to_web(
        {
            "type": "tool_approval_required",
            "sessionId": "s1",
            "approvalId": "approval_1",
            "toolCallId": "tool_1",
            "toolName": "write",
            "riskLevel": "high",
        }
    )

    assert envelope.type == "tool_approval_required"
    assert envelope.session_id == "s1"
    assert envelope.payload["approvalId"] == "approval_1"


def test_web_error_event_requires_non_empty_code_and_message() -> None:
    from codepilot.interfaces.web.event_adapter import error_to_web

    event = error_to_web(
        "  Session missing  ",
        session_id="s1",
        code=" runtime.session_not_found ",
    )

    assert event.type == "error"
    assert event.session_id == "s1"
    assert event.payload == {
        "code": "runtime.session_not_found",
        "message": "Session missing",
    }

    with pytest.raises(ValueError, match="error code"):
        error_to_web("Session missing", code=" ")

    with pytest.raises(ValueError, match="error message"):
        error_to_web("")


def test_websocket_stream_normalizes_session_identity() -> None:
    from codepilot.interfaces.web import WebSocketSessionStream

    class FakeRuntime:
        def __init__(self) -> None:
            self.received_session_id = None

        async def continue_session(self, session_id):
            self.received_session_id = session_id
            yield {"type": "turn_start", "sessionId": session_id}

    runtime = FakeRuntime()
    stream = WebSocketSessionStream(runtime=runtime, session_id=" session_1 ")  # type: ignore[arg-type]
    events = asyncio.run(_collect_async(stream.continue_events()))

    assert runtime.received_session_id == "session_1"
    assert events[0].session_id == "session_1"

    with pytest.raises(ValueError, match="session_id"):
        WebSocketSessionStream(runtime=runtime, session_id=" ")  # type: ignore[arg-type]


async def _run_runtime_service_send_message_streaming_case() -> None:
    from codepilot.runtime.service import RuntimeService
    from codepilot.runtime.contracts import UserInput

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

        async def run(self, text, *, images=None, run_id=None):
            _ = text, images, run_id
            for listener in list(self.listeners):
                listener({"type": "message_update", "delta": "hello"})
            await self.run_can_finish.wait()
            self.run_finished = True

        async def continue_run(self, *, run_id=None):
            _ = run_id
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


async def _collect_async(iterator):
    return [item async for item in iterator]


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

        async def continue_run(self, *, run_id=None):
            _ = run_id
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
