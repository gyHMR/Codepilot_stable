from __future__ import annotations

from typing import Any

from codepilot.runtime.service import RuntimeService, UserInput
from codepilot.runtime.types import CreateAgentSessionOptions

from .event_adapter import agent_event_to_web
from .schemas import (
    WebCreateSessionRequest,
    WebEventEnvelope,
    WebPromptRequest,
    WebRouteSpec,
    WebSessionSummary,
    WebToolApproval,
)


def describe_web_contract() -> dict[str, Any]:
    """Return the planned Web Console contract without starting a server."""

    return {
        "transport": ["http", "websocket"],
        "entrypoint": "codepilot.interfaces.web",
        "delegates_to": ["codepilot.runtime", "codepilot.sessions", "codepilot.tools"],
        "routes": [route.__dict__ for route in web_route_specs()],
        "responsibilities": [
            "display_chat_messages",
            "display_tool_events",
            "display_file_tree",
            "display_diffs",
            "display_command_output",
            "submit_tool_approval",
        ],
        "non_responsibilities": [
            "llm_provider_calls",
            "agent_loop",
            "filesystem_mutation",
            "shell_execution",
            "session_persistence",
        ],
    }


def web_route_specs() -> list[WebRouteSpec]:
    return [
        WebRouteSpec("GET", "/api/sessions", "List runtime sessions"),
        WebRouteSpec("POST", "/api/sessions", "Create or resume a session"),
        WebRouteSpec("GET", "/api/sessions/{session_id}", "Get one session summary"),
        WebRouteSpec("GET", "/api/sessions/{session_id}/runs", "List run results for one session"),
        WebRouteSpec("GET", "/api/sessions/{session_id}/runs/{run_id}/report", "Get one run report"),
        WebRouteSpec("POST", "/api/sessions/{session_id}/messages", "Send a user message"),
        WebRouteSpec("POST", "/api/tool-approvals/{approval_id}", "Approve or deny a tool call"),
        WebRouteSpec("WS", "/ws/sessions/{session_id}", "Stream RuntimeEvent envelopes"),
    ]


class WebConsoleBackend:
    """Dependency-light backend skeleton for local Web Console APIs."""

    def __init__(self, runtime: RuntimeService | None = None) -> None:
        self.runtime = runtime or RuntimeService()

    def create_session(self, request: WebCreateSessionRequest) -> WebEventEnvelope:
        handle = self.runtime.create_session(
            CreateAgentSessionOptions(
                workspace_dir=request.workspace_dir,
                provider=request.provider,
                model_id=request.model_id,
                system_prompt=request.system_prompt,
                session_id=request.session_id,
                read_only_mode=request.read_only_mode,
                load_workspace_resources=request.load_workspace_resources,
            )
        )
        return WebEventEnvelope(
            type="session_created",
            session_id=handle.session_id,
            payload={"session_id": handle.session_id},
        )

    def list_sessions(self) -> list[WebSessionSummary]:
        return [WebSessionSummary(session_id=item["session_id"]) for item in self.runtime.list_sessions()]

    def get_session(self, session_id: str) -> WebSessionSummary:
        session = self.runtime.get_session(session_id)
        return WebSessionSummary(session_id=session.session_id)

    def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        return self.runtime.list_runs(session_id)

    def get_run_report(self, session_id: str, run_id: str) -> dict[str, Any]:
        return self.runtime.get_run_report(session_id, run_id)

    async def send_message(self, request: WebPromptRequest) -> list[WebEventEnvelope]:
        session_id = request.session.session_id
        if not session_id:
            created = self.create_session(
                WebCreateSessionRequest(workspace_dir=request.session.workspace_dir)
            )
            session_id = str(created.session_id)

        events: list[WebEventEnvelope] = []
        async for event in self.runtime.send_message(
            session_id,
            UserInput(text=request.text, images=request.images),
        ):
            events.append(agent_event_to_web(event))
        return events

    async def approve_tool(self, approval: WebToolApproval) -> WebEventEnvelope:
        await self.runtime.approve_tool_call(approval.tool_call_id, approval.decision)
        return WebEventEnvelope(
            type="session_state",
            session_id=approval.session_id,
            payload={
                "tool_call_id": approval.tool_call_id,
                "decision": approval.decision,
                "reason": approval.reason,
            },
        )
