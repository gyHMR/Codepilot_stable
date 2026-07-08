from __future__ import annotations

import asyncio
import inspect


def test_cli_command_delegates_to_runtime_application_command() -> None:
    asyncio.run(_run_cli_command_delegation_case())


async def _run_cli_command_delegation_case() -> None:
    from codepilot.interfaces.cli.interactive import dispatch_command
    from codepilot.runtime.actions import CommandFinishedFrame, CommandSubmitted

    class FakeRuntime:
        def __init__(self) -> None:
            self.received = None

        def get_session(self, _session_id: str):  # pragma: no cover - must not be used
            raise AssertionError("CLI command layer must not request AgentSession")

        async def dispatch(self, session_id, action):
            assert isinstance(action, CommandSubmitted)
            self.received = (session_id, action.text)
            yield CommandFinishedFrame(
                record=type(
                    "Record",
                    (),
                    {
                        "handled": True,
                        "output_lines": ["ok"],
                        "switched_session_id": "session_2",
                    },
                )()
            )

    runtime = FakeRuntime()
    result = await dispatch_command(runtime, "session_1", "/status")

    assert runtime.received == ("session_1", "/status")
    assert result.handled is True
    assert result.output_lines == ["ok"]
    assert result.switched_session_id == "session_2"


def test_render_dispatch_consumes_runtime_stream_result() -> None:
    from codepilot.interfaces.cli.interactive import render_dispatch
    from codepilot.protocols import AssistantMessage, TextContent
    from codepilot.runtime.actions import ProgressFrame, RunFinishedFrame

    final = AssistantMessage(content=[TextContent(text="done")])

    class FakeRuntime:
        def __init__(self) -> None:
            self.sent = None

        async def dispatch(self, session_id, action):
            self.sent = (session_id, action)
            yield ProgressFrame(event={"type": "message_update"})
            yield RunFinishedFrame(
                record=type(
                    "Record",
                    (),
                    {
                        "outcome": type(
                            "Outcome",
                            (),
                            {"final_message": final},
                        )()
                    },
                )()
            )

        def get_latest_assistant_message(self, _session_id):  # pragma: no cover - must not be used
            raise AssertionError("CLI must render the result carried by the runtime stream")

    class FakeRenderer:
        def __init__(self) -> None:
            self.events = []
            self.final = None

        def render_progress_event(self, event):
            self.events.append(event)

        def render_final(self, record):
            self.final = record.outcome.final_message

    runtime = FakeRuntime()
    renderer = FakeRenderer()

    asyncio.run(render_dispatch(runtime.dispatch("session_1", object()), renderer))

    assert runtime.sent[0] == "session_1"
    assert renderer.events == [{"type": "message_update"}]
    assert renderer.final is final


def test_runtime_gateway_accepts_approval_as_user_action() -> None:
    from codepilot.runtime.actions import ApprovalDecided
    from codepilot.runtime.gateway import RuntimeGateway

    signature = inspect.signature(RuntimeGateway.dispatch)
    assert list(signature.parameters) == ["self", "session_id", "action"]

    action = ApprovalDecided(approval_id="approval_1", decision="approve")
    assert action.approval_id == "approval_1"
    assert action.decision == "approve"


def test_runtime_package_does_not_export_legacy_runtime_service() -> None:
    import codepilot.runtime as runtime

    assert not hasattr(runtime, "RuntimeService")
    assert not hasattr(runtime, "assemble_runtime")
    assert not hasattr(runtime, "create_session_controller")
    assert not hasattr(runtime, "explain_runtime_config")


def test_runtime_gateway_public_surface_is_v2_spine_only() -> None:
    from codepilot.runtime.gateway import RuntimeGateway

    public = {
        name
        for name in dir(RuntimeGateway)
        if not name.startswith("_") and callable(getattr(RuntimeGateway, name))
    }

    assert public == {"open_session", "dispatch", "describe", "close", "close_all"}
    assert {
        "submit_turn",
        "execute_command",
        "approve",
        "cancel",
        "get_session",
        "get_assembly",
        "get_session_state",
        "get_session_status",
        "list_runs",
        "get_run_result",
        "get_run_events",
        "list_commands",
        "list_pending_approvals",
        "set_mode",
        "fork_session",
        "switch_entry",
    }.isdisjoint(public)
