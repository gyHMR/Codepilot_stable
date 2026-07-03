from __future__ import annotations

# 新手导读：ToolRuntime 是所有工具调用的统一安全管线：查找、权限、schema、审批、执行、结果防护。
# 关注点：理解工具安全策略时，优先从 execute()/execute_approved() 顺序读。

"""统一工具执行运行时：集成权限检查、审批流程和工具执行。"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from codepilot.protocols import TextContent

from .argument_schema import SchemaValidator
from .approval import ApprovalProvider, DeferredApprovalProvider
from .contracts import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    ToolResultStatus,
    ToolRuntimeRequest,
    ToolRuntimeResult,
)
from .policy import PermissionPolicy, ToolDecision, ToolRequest
from .registry import ToolRegistry
from .result_safety import ToolResultGuard


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


def _permission_record(
    decision: ToolDecision,
    *,
    granted_approval_id: str | None,
) -> dict[str, Any]:
    if granted_approval_id is None:
        return {
            "decision": decision.kind,
            "reason": decision.reason,
            **decision.details,
        }
    return {
        "decision": "allow",
        "reason": "approved_by_user",
        "approval_id": granted_approval_id,
        "policy_decision": decision.kind,
        "policy_reason": decision.reason,
        **decision.details,
    }


@dataclass
class ToolRuntime:
    """工具运行时：将注册表、权限策略和审批提供者组合为统一的执行引擎。"""
    registry: ToolRegistry                                      # 工具注册表
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)  # 权限策略
    approval_provider: ApprovalProvider = field(default_factory=DeferredApprovalProvider)  # 审批提供者
    schema_validator: SchemaValidator = field(default_factory=SchemaValidator)  # 参数校验器
    result_guard: ToolResultGuard = field(default_factory=ToolResultGuard)  # 结果防护器

    def as_agent_tools(self) -> list[AgentTool]:
        """将注册表中的工具转换为 Agent 可用的工具列表（包装权限检查逻辑）。"""
        adapters: list[AgentTool] = []
        for tool in self.registry.list():
            adapter = AgentTool(
                name=tool.name,
                label=tool.label,
                description=tool.description,
                parameters=tool.parameters,
                execute=self._make_execute_adapter(tool.name),
                runtime_managed=True,
                metadata=self.registry.metadata_for(tool.name),
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
        """执行工具调用：权限检查 → 参数校验 → 审批 → 执行 → 结果防护。"""
        return await self._execute(
            request,
            signal=signal,
            on_update=on_update,
            granted_approval_id=None,
        )

    async def execute_approved(
        self,
        request: ToolRuntimeRequest,
        *,
        approval_id: str,
        signal: Any | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> ToolRuntimeResult:
        """Execute a user-approved pending tool call through the normal guard path."""
        return await self._execute(
            request,
            signal=signal,
            on_update=on_update,
            granted_approval_id=approval_id,
        )

    async def _execute(
        self,
        request: ToolRuntimeRequest,
        *,
        signal: Any | None,
        on_update: AgentToolUpdateCallback | None,
        granted_approval_id: str | None,
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

        permission_record = _permission_record(
            decision,
            granted_approval_id=granted_approval_id,
        )

        validation = self.schema_validator.validate(tool.parameters, request.params)
        if not validation.valid:
            result = _tool_result(
                "Tool arguments failed schema validation: "
                + "; ".join(validation.errors),
                status="error",
                error_code="invalid_tool_arguments",
                details={
                    "tool": request.name,
                    "reason": "schema_validation_failed",
                    "errors": list(validation.errors),
                },
            )
            result.tool_call_id = request.tool_call_id
            result.tool_name = request.name
            result.approved = False
            result.metadata["permission_decision"] = permission_record
            result.metadata["schema_validation"] = {
                "valid": False,
                "errors": list(validation.errors),
            }
            return ToolRuntimeResult(
                result=result,
                status="error",
                is_error=True,
                approved=False,
            )

        approval_id: str | None = None
        if decision.requires_approval:
            if granted_approval_id is not None:
                approval_id = granted_approval_id
            else:
                approval = await self.approval_provider.request_approval(
                    request,
                    metadata,
                    decision,
                )
                approval_id = approval.approval_id
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
                            "decision": decision.kind,
                            **decision.details,
                        },
                    )
                    result.tool_call_id = request.tool_call_id
                    result.tool_name = request.name
                    result.approved = False
                    result.approval_id = approval.approval_id
                    result.metadata["permission_decision"] = permission_record
                    return ToolRuntimeResult(
                        result=result,
                        status="approval_required",
                        is_error=True,
                        approved=False,
                        approval_id=approval.approval_id,
                    )

        try:
            started_at = time.monotonic()
            value = tool.execute(request.tool_call_id, request.params, signal, on_update)
            result = await _maybe_await(value)
            if approval_id is not None:
                result.approved = True
                result.approval_id = approval_id
            if not isinstance(result.details, dict):
                result.details = {"tool_details": result.details}
            result.details.setdefault(
                "permission",
                permission_record,
            )
            result.metadata.setdefault(
                "permission_decision",
                permission_record,
            )
            result.metadata.setdefault(
                "duration_ms",
                int((time.monotonic() - started_at) * 1000),
            )
            result = self.result_guard.apply(result, metadata=metadata)
            status: ToolResultStatus = result.status
            if status == "success" and result.is_error:
                status = "error"
            _sync_result_status(
                request,
                result,
                status,
                approved=result.approved,
                approval_id=approval_id,
            )
            return ToolRuntimeResult(
                result=result,
                status=status,
                is_error=bool(result.is_error),
                approved=result.approved,
                approval_id=approval_id,
            )
        except Exception as exc:
            result = _tool_result(
                str(exc),
                status="error",
                error_code="tool_exception",
                details={"tool": request.name, "reason": "tool_exception", "error_kind": type(exc).__name__},
            )
            result.approved = True
            result.approval_id = approval_id
            result.metadata.setdefault(
                "permission_decision",
                permission_record,
            )
            result.metadata.setdefault(
                "duration_ms",
                int((time.monotonic() - started_at) * 1000),
            )
            result = self.result_guard.apply(result, metadata=metadata)
            result.tool_call_id = request.tool_call_id
            result.tool_name = request.name
            return ToolRuntimeResult(
                result=result,
                status="error",
                is_error=True,
                approved=result.approved,
                approval_id=approval_id,
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
        result.metadata["permission_decision"] = {
            "decision": decision.kind,
            "reason": decision.reason,
            **decision.details,
        }
        return ToolRuntimeResult(
            result=result,
            status="denied",
            is_error=True,
            approved=False,
        )
