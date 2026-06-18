from __future__ import annotations

"""Unified tool execution runtime."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from codepilot.protocols import TextContent

from .approval import ApprovalProvider, DenyApprovalProvider
from .permissions import PermissionPolicy, ToolDecision, ToolRequest
from .registry import ToolRegistry
from .types import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    ToolResultStatus,
    ToolRuntimeRequest,
    ToolRuntimeResult,
)


def _tool_result(
    message: str,
    *,
    status: ToolResultStatus,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
    is_error: bool = True,
) -> AgentToolResult:
    merged_details = {"status": status, **(details or {})}
    return AgentToolResult(
        content=[TextContent(text=message)],
        details=merged_details,
        is_error=is_error,
        status=status,
        error_code=error_code,
    )


def _sync_result_status(
    request: ToolRuntimeRequest,
    result: AgentToolResult,
    status: ToolResultStatus,
    *,
    approved: bool,
    approval_id: str | None = None,
) -> AgentToolResult:
    result.tool_call_id = request.tool_call_id
    result.tool_name = request.name
    result.status = status
    result.approved = approved
    result.approval_id = approval_id
    if result.details is None:
        result.details = {}
    if isinstance(result.details, dict):
        result.details["status"] = status
        if approval_id is not None:
            result.details["approval_id"] = approval_id
    return result


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


@dataclass
class ToolRuntime:
    registry: ToolRegistry
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    approval_provider: ApprovalProvider = field(default_factory=DenyApprovalProvider)

    def as_agent_tools(self) -> list[AgentTool]:
        adapters: list[AgentTool] = []
        for tool in self.registry.list():
            adapter = AgentTool(
                name=tool.name,
                label=tool.label,
                description=tool.description,
                parameters=tool.parameters,
                execute=self._make_execute_adapter(tool.name),
                runtime_managed=True,
            )
            adapters.append(adapter)
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
            result.status = runtime_result.status
            result.approved = runtime_result.approved
            result.approval_id = runtime_result.approval_id
            if isinstance(result.details, dict):
                result.details.setdefault("status", runtime_result.status)
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
            result = _tool_result(
                f"Tool {request.name} not found",
                status="error",
                error_code="tool_not_found",
                details={"tool": request.name, "reason": "tool_not_found"},
            )
            result.tool_call_id = request.tool_call_id
            result.tool_name = request.name
            result.approved = False
            return ToolRuntimeResult(
                result=result,
                status="error",
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
                result = _tool_result(
                    approval.reason or "Tool execution requires approval",
                    status="approval_required",
                    error_code="approval_required",
                    details={
                        "tool": request.name,
                        "reason": decision.reason,
                        "approval_id": approval.approval_id,
                        "policy_reason": decision.reason,
                    },
                )
                result.tool_call_id = request.tool_call_id
                result.tool_name = request.name
                result.approved = False
                result.approval_id = approval.approval_id
                return ToolRuntimeResult(
                    result=result,
                    status="approval_required",
                    is_error=True,
                    approved=False,
                    approval_id=approval.approval_id,
                )

        try:
            value = tool.execute(request.tool_call_id, request.params, signal, on_update)
            result = await _maybe_await(value)
            status: ToolResultStatus = result.status
            if status == "success" and result.is_error:
                status = "error"
            _sync_result_status(
                request,
                result,
                status,
                approved=result.approved,
                approval_id=result.approval_id,
            )
            return ToolRuntimeResult(
                result=result,
                status=status,
                is_error=bool(result.is_error),
                approved=result.approved,
                approval_id=result.approval_id,
            )
        except Exception as exc:
            result = _tool_result(
                str(exc),
                status="error",
                error_code="tool_exception",
                details={"tool": request.name, "reason": "tool_exception", "error_kind": type(exc).__name__},
            )
            result.tool_call_id = request.tool_call_id
            result.tool_name = request.name
            return ToolRuntimeResult(
                result=result,
                status="error",
                is_error=True,
            )

    @staticmethod
    def _blocked_result(request: ToolRuntimeRequest, decision: ToolDecision) -> ToolRuntimeResult:
        result = _tool_result(
            decision.reason or "Tool execution was blocked",
            status="denied",
            error_code=decision.reason or "tool_denied",
            details={
                "tool": request.name,
                "reason": decision.reason,
                "policy_reason": decision.reason,
                **decision.details,
            },
        )
        result.tool_call_id = request.tool_call_id
        result.tool_name = request.name
        result.approved = False
        return ToolRuntimeResult(
            result=result,
            status="denied",
            is_error=True,
            approved=False,
        )
