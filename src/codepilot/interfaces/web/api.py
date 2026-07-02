from __future__ import annotations

"""Web Console 后端 API 骨架。

提供 WebConsoleBackend 类，将 Web 请求适配到 RuntimeService，
以及路由规格和契约描述函数。
"""

from typing import Any

from codepilot.runtime.service import RuntimeService
from codepilot.runtime.contracts import CreateAgentSessionOptions, UserInput

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
    """返回 Web Console 的契约描述（无需启动服务器）。"""

    return {
        "transport": ["http", "websocket"],
        "entrypoint": "codepilot.interfaces.web",
        "delegates_to": ["codepilot.runtime", "codepilot.sessions", "codepilot.tools"],
        "routes": [route.to_dict() for route in web_route_specs()],
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
    """返回 Web Console 的路由规格列表。"""
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
    """Web Console 后端骨架：轻量依赖，将 Web 请求委托给 RuntimeService。"""

    def __init__(self, runtime: RuntimeService | None = None) -> None:
        self.runtime = runtime or RuntimeService()

    def create_session(self, request: WebCreateSessionRequest) -> WebEventEnvelope:
        """创建新会话：将 Web 请求转换为 CreateAgentSessionOptions 后调用 RuntimeService。"""
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
        """列出所有活跃会话。"""
        return [WebSessionSummary(session_id=item["session_id"]) for item in self.runtime.list_sessions()]

    def get_session(self, session_id: str) -> WebSessionSummary:
        """获取单个会话摘要。"""
        session = self.runtime.get_session(session_id)
        return WebSessionSummary(session_id=session.session_id)

    def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        """列出指定会话的所有 Run 结果。"""
        return self.runtime.list_runs(session_id)

    def get_run_report(self, session_id: str, run_id: str) -> dict[str, Any]:
        """获取指定 Run 的详细报告。"""
        return self.runtime.get_run_report(session_id, run_id)

    async def send_message(self, request: WebPromptRequest) -> list[WebEventEnvelope]:
        """发送用户消息：通过 RuntimeService 发送并收集事件信封列表。"""
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
        """审批工具调用：将审批决策传递给 RuntimeService。"""
        approval_id = approval.approval_id or approval.tool_call_id
        await self.runtime.approve_tool_call(
            approval_id,
            approval.decision,
            session_id=approval.session_id,
        )
        return WebEventEnvelope(
            type="session_state",
            session_id=approval.session_id,
            payload={
                "approval_id": approval_id,
                "tool_call_id": approval.tool_call_id,
                "decision": approval.decision,
                "reason": approval.reason,
            },
        )
