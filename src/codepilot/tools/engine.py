from __future__ import annotations

"""Readable tool execution pipeline."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from codepilot.protocols import TextContent
from codepilot.protocols.tools import ToolMetadata, ToolResultStatus

from .approval import ApprovalDecision, ApprovalProvider, DeferredApprovalProvider
from .authoring import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    ToolRuntimeRequest,
    ToolRuntimeResult,
)
from .guard import ToolResultGuard
from .policy import PermissionPolicy, ToolDecision, ToolRequest
from .registry import ToolRegistry
from .validation import SchemaValidator


@dataclass
class ToolRuntime:
    """Run one tool call through lookup, validation, policy, approval, and guard."""

    registry: ToolRegistry
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    approval_provider: ApprovalProvider = field(default_factory=DeferredApprovalProvider)
    schema_validator: SchemaValidator = field(default_factory=SchemaValidator)
    result_guard: ToolResultGuard = field(default_factory=ToolResultGuard)

    async def execute(
        self,
        request: ToolRuntimeRequest,
        *,
        approval_id: str | None = None,
        signal: Any | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> ToolRuntimeResult:
        tool = self.registry.get(request.name)
        if tool is None:
            return _tool_not_found(request)

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
            return _policy_denied(request, decision)

        permission = _permission_record(decision, granted_approval_id=approval_id)
        validation = self.schema_validator.validate(tool.parameters, request.params)
        if not validation.valid:
            return _invalid_arguments(request, validation.errors, permission)

        resolved_approval = await self._resolve_approval(
            request,
            metadata,
            decision,
            permission,
            granted_approval_id=approval_id,
        )
        if isinstance(resolved_approval, ToolRuntimeResult):
            return resolved_approval

        return await self._run_tool(
            tool,
            request,
            metadata,
            permission,
            approval_id=resolved_approval,
            signal=signal,
            on_update=on_update,
        )

    async def _resolve_approval(
        self,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        decision: ToolDecision,
        permission: dict[str, Any],
        *,
        granted_approval_id: str | None,
    ) -> str | ToolRuntimeResult | None:
        if not decision.requires_approval:
            return None
        if granted_approval_id is not None:
            return granted_approval_id

        approval = await self.approval_provider.request_approval(
            request,
            metadata,
            decision,
        )
        if approval.approved:
            return approval.approval_id
        if approval.deferred:
            return _approval_required(request, decision, approval, permission)
        return _approval_denied(request, decision, approval, permission)

    async def _run_tool(
        self,
        tool: AgentTool,
        request: ToolRuntimeRequest,
        metadata: ToolMetadata | None,
        permission: dict[str, Any],
        *,
        approval_id: str | None,
        signal: Any | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> ToolRuntimeResult:
        started_at = time.monotonic()
        try:
            result = await _maybe_await(
                tool.execute(request.tool_call_id, request.params, signal, on_update)
            )
        except Exception as exc:
            result = _error_result(
                str(exc),
                status="error",
                error_code="tool_exception",
                details={
                    "tool": request.name,
                    "reason": "tool_exception",
                    "error_kind": type(exc).__name__,
                },
            )
            result.approved = True
            result.approval_id = approval_id
            _attach_runtime_metadata(result, permission, started_at)
            result = self.result_guard.apply(result, metadata=metadata)
            return _runtime_result(
                request,
                result,
                status="error",
                approved=result.approved,
                approval_id=approval_id,
            )

        if approval_id is not None:
            result.approved = True
            result.approval_id = approval_id
        if not isinstance(result.details, dict):
            result.details = {"tool_details": result.details}
        result.details.setdefault("permission", permission)
        _attach_runtime_metadata(result, permission, started_at)
        result = self.result_guard.apply(result, metadata=metadata)
        status = _effective_status(result)
        return _runtime_result(
            request,
            result,
            status=status,
            approved=result.approved,
            approval_id=approval_id,
        )


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


def _attach_runtime_metadata(
    result: AgentToolResult,
    permission: dict[str, Any],
    started_at: float,
) -> None:
    result.metadata.setdefault("permission_decision", permission)
    result.metadata.setdefault("duration_ms", int((time.monotonic() - started_at) * 1000))


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


def _tool_not_found(request: ToolRuntimeRequest) -> ToolRuntimeResult:
    result = _error_result(
        f"Tool {request.name} not found",
        status="error",
        error_code="tool_not_found",
        details={"tool": request.name, "reason": "tool_not_found"},
    )
    return _runtime_result(request, result, status="error", approved=False)


def _policy_denied(
    request: ToolRuntimeRequest,
    decision: ToolDecision,
) -> ToolRuntimeResult:
    permission = {
        "decision": decision.kind,
        "reason": decision.reason,
        **decision.details,
    }
    result = _error_result(
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
    result.metadata["permission_decision"] = permission
    return _runtime_result(request, result, status="denied", approved=False)


def _invalid_arguments(
    request: ToolRuntimeRequest,
    errors: tuple[str, ...],
    permission: dict[str, Any],
) -> ToolRuntimeResult:
    result = _error_result(
        "Tool arguments failed schema validation: " + "; ".join(errors),
        status="error",
        error_code="invalid_tool_arguments",
        details={
            "tool": request.name,
            "reason": "schema_validation_failed",
            "errors": list(errors),
        },
    )
    result.metadata["permission_decision"] = permission
    result.metadata["schema_validation"] = {
        "valid": False,
        "errors": list(errors),
    }
    return _runtime_result(request, result, status="error", approved=False)


def _approval_required(
    request: ToolRuntimeRequest,
    decision: ToolDecision,
    approval: ApprovalDecision,
    permission: dict[str, Any],
) -> ToolRuntimeResult:
    result = _error_result(
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
    result.metadata["permission_decision"] = permission
    return _runtime_result(
        request,
        result,
        status="approval_required",
        approved=False,
        approval_id=approval.approval_id,
    )


def _approval_denied(
    request: ToolRuntimeRequest,
    decision: ToolDecision,
    approval: ApprovalDecision,
    permission: dict[str, Any],
) -> ToolRuntimeResult:
    result = _error_result(
        approval.reason or "Tool execution denied by user",
        status="denied",
        error_code="approval_denied",
        details={
            "tool": request.name,
            "reason": approval.reason or decision.reason,
            "approval_id": approval.approval_id,
            "policy_reason": decision.reason,
            "decision": "deny",
            **decision.details,
        },
    )
    result.metadata["permission_decision"] = permission
    return _runtime_result(
        request,
        result,
        status="denied",
        approved=False,
        approval_id=approval.approval_id,
    )


def _error_result(
    message: str,
    *,
    status: ToolResultStatus,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={"status": status, **(details or {})},
        is_error=True,
        status=status,
        error_code=error_code,
    )


def _runtime_result(
    request: ToolRuntimeRequest,
    result: AgentToolResult,
    *,
    status: ToolResultStatus,
    approved: bool,
    approval_id: str | None = None,
) -> ToolRuntimeResult:
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
    return ToolRuntimeResult(
        result=result,
        status=status,
        is_error=bool(result.is_error),
        approved=approved,
        approval_id=approval_id,
    )


def _effective_status(result: AgentToolResult) -> ToolResultStatus:
    if result.status == "success" and result.is_error:
        return "error"
    return result.status


__all__ = ["ToolRuntime"]
