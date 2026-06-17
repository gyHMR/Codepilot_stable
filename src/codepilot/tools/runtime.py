from __future__ import annotations

"""Unified tool execution runtime."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from codepilot.core import AgentTool, AgentToolResult
from codepilot.core.types import AgentToolUpdateCallback
from codepilot.llm.types import TextContent

from .approval import ApprovalProvider, DenyApprovalProvider
from .diff import DiffRecorder
from .permissions import PermissionPolicy, ToolDecision, ToolRequest
from .registry import ToolRegistry
from .types import ToolRuntimeRequest, ToolRuntimeResult


def _error_result(message: str, *, details: dict[str, Any] | None = None) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=message)],
        details=details or {},
        is_error=True,
    )


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


@dataclass
class ToolRuntime:
    registry: ToolRegistry
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    approval_provider: ApprovalProvider = field(default_factory=DenyApprovalProvider)
    diff_recorder: DiffRecorder = field(default_factory=DiffRecorder)

    def as_agent_tools(self) -> list[AgentTool]:
        adapters: list[AgentTool] = []
        for tool in self.registry.list():
            adapters.append(
                AgentTool(
                    name=tool.name,
                    label=tool.label,
                    description=tool.description,
                    parameters=tool.parameters,
                    execute=self._make_execute_adapter(tool.name),
                )
            )
        return adapters

    def _make_execute_adapter(self, name: str):
        async def _execute(
            tool_call_id: str,
            params: dict[str, Any],
            signal: Any | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            runtime_result = await self.execute(
                ToolRuntimeRequest(
                    tool_call_id=tool_call_id,
                    name=name,
                    params=params,
                ),
                signal=signal,
                on_update=on_update,
            )
            result = runtime_result.result
            result.is_error = runtime_result.is_error
            result.approved = runtime_result.approved
            result.approval_id = runtime_result.approval_id
            return result

        return _execute

    async def execute(
        self,
        request: ToolRuntimeRequest,
        *,
        signal: Any | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> ToolRuntimeResult:
        tool = self.registry.get(request.name)
        if tool is None:
            result = _error_result(
                f"Tool {request.name} not found",
                details={"tool": request.name, "reason": "tool_not_found"},
            )
            result.approved = False
            return ToolRuntimeResult(
                result=result,
                is_error=True,
                approved=False,
            )

        metadata = self.registry.metadata_for(request.name)
        decision = self.permission_policy.decide(
            ToolRequest(
                name=request.name,
                params=request.params,
                source=request.source,
                metadata=metadata,
            )
        )
        if decision.denied:
            return self._blocked_result(request, decision)

        if decision.requires_approval:
            approval = await self.approval_provider.request_approval(
                request,
                metadata,
                decision,
            )
            if not approval.approved:
                result = _error_result(
                    approval.reason or "Tool execution requires approval",
                    details={
                        "tool": request.name,
                        "reason": decision.reason,
                        "approval_id": approval.approval_id,
                    },
                )
                result.approved = False
                result.approval_id = approval.approval_id
                return ToolRuntimeResult(
                    result=result,
                    is_error=True,
                    approved=False,
                    approval_id=approval.approval_id,
                )

        try:
            value = tool.execute(request.tool_call_id, request.params, signal, on_update)
            result = await _maybe_await(value)
            return ToolRuntimeResult(
                result=result,
                is_error=bool(result.is_error),
                approved=result.approved,
                approval_id=result.approval_id,
            )
        except Exception as exc:
            return ToolRuntimeResult(
                result=_error_result(
                    str(exc),
                    details={"tool": request.name, "reason": "tool_exception"},
                ),
                is_error=True,
            )

    @staticmethod
    def _blocked_result(request: ToolRuntimeRequest, decision: ToolDecision) -> ToolRuntimeResult:
        result = _error_result(
            decision.reason or "Tool execution was blocked",
            details={"tool": request.name, "reason": decision.reason, **decision.details},
        )
        result.approved = False
        return ToolRuntimeResult(
            result=result,
            is_error=True,
            approved=False,
        )
