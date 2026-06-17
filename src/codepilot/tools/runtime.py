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
from .types import ToolResultStatus, ToolRuntimeRequest, ToolRuntimeResult


def _tool_result(
    message: str,
    *,
    status: ToolResultStatus,
    details: dict[str, Any] | None = None,
    is_error: bool = True,
) -> AgentToolResult:
    merged_details = {"status": status, **(details or {})}
    return AgentToolResult(
        content=[TextContent(text=message)],
        details=merged_details,
        is_error=is_error,
        status=status,
    )


def _sync_result_status(
    result: AgentToolResult,
    status: ToolResultStatus,
    *,
    approved: bool,
    approval_id: str | None = None,
) -> AgentToolResult:
    result.status = status
    result.approved = approved
    result.approval_id = approval_id
    if result.details is None:
        result.details = {}
    if isinstance(result.details, dict):
        result.details.setdefault("status", status)
        if approval_id is not None:
            result.details.setdefault("approval_id", approval_id)
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
    diff_recorder: DiffRecorder = field(default_factory=DiffRecorder)

    def as_agent_tools(self) -> list[AgentTool]:
        adapters: list[AgentTool] = []
        for tool in self.registry.list():
            adapter = AgentTool(
                name=tool.name,
                label=tool.label,
                description=tool.description,
                parameters=tool.parameters,
                execute=self._make_execute_adapter(tool.name),
            )
            setattr(adapter, "runtime_managed", True)
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
                details={"tool": request.name, "reason": "tool_not_found"},
            )
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
                    details={
                        "tool": request.name,
                        "reason": decision.reason,
                        "approval_id": approval.approval_id,
                        "policy_reason": decision.reason,
                    },
                )
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
            status: ToolResultStatus = "error" if result.is_error else "success"
            _sync_result_status(
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
                details={"tool": request.name, "reason": "tool_exception", "error_kind": type(exc).__name__},
            )
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
            details={
                "tool": request.name,
                "reason": decision.reason,
                "policy_reason": decision.reason,
                **decision.details,
            },
        )
        result.approved = False
        return ToolRuntimeResult(
            result=result,
            status="denied",
            is_error=True,
            approved=False,
        )
