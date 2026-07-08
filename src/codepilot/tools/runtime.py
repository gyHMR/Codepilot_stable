from __future__ import annotations

"""ToolRuntime: prepare, authorize, approve, execute, and normalize tools."""

import asyncio
import inspect
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable

from codepilot.protocols import (
    AfterToolCallContext,
    AfterToolCallResult,
    AssistantMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    RunVerification,
    TextContent,
    ToolCall,
    ToolHookContextSnapshot,
)

from .approvals import ApprovalProvider, DeferredApprovalProvider
from .contracts import (
    PreparedToolCall,
    ToolCatalogView,
    ToolCallRequest,
    ToolInterruption,
    ToolInvocation,
    ToolObservation,
    ToolObservationStatus,
    ToolPort,
    ToolResumeDecision,
    ToolResult,
    ToolRiskView,
    error_result,
)
from .permissions import PermissionPolicy, ToolDecision
from .registry import ToolRegistry, prepare_error_content
from .results import ToolResultPolicy

BeforeToolHook = Callable[
    [BeforeToolCallContext, Any | None],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
AfterToolHook = Callable[
    [AfterToolCallContext, Any | None],
    AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
]


@dataclass
class ToolRuntime(ToolPort):
    registry: ToolRegistry
    permission_policy: PermissionPolicy
    approval_provider: ApprovalProvider = DeferredApprovalProvider()
    result_policy: ToolResultPolicy = ToolResultPolicy()
    before_tool_call: BeforeToolHook | None = None
    after_tool_call: AfterToolHook | None = None

    def __post_init__(self) -> None:
        self._pending: dict[str, PreparedToolCall] = {}

    def catalog(self, current_mode: str = "build") -> ToolCatalogView:
        return self.registry.catalog(current_mode=current_mode)

    async def execute(self, invocation: ToolInvocation) -> ToolObservation:
        prepared = self.registry.prepare_call(
            run_id=invocation.run_id,
            tool_call_id=invocation.tool_call_id,
            name=invocation.name,
            arguments=dict(invocation.arguments),
            current_mode=invocation.current_mode,
            source=invocation.source,
            policy_context=invocation.policy_context,
            assistant_message=invocation.assistant_message,
            context=invocation.context,
        )
        if not prepared.valid or prepared.call is None:
            return ToolObservation(
                tool_call_id=invocation.tool_call_id,
                name=invocation.name,
                status="error",
                content=prepare_error_content(prepared),
                metadata={
                    "error_code": prepared.error_code or "tool_prepare_failed",
                    "recovery_hint": prepared.recovery_hint,
                },
            )
        before = await self._run_before_hook(invocation)
        if before is not None and before.block:
            return ToolObservation(
                tool_call_id=invocation.tool_call_id,
                name=invocation.name,
                status="denied",
                content=(TextContent(text=before.reason or "Tool call blocked by hook"),),
                metadata={"error_code": "tool_hook_blocked"},
            )
        decision = self.permission_policy.decide(prepared.call.request)
        if decision.denied:
            return _blocked_observation(prepared.call, decision, status="denied")
        if decision.requires_approval:
            approval = await self.approval_provider.request_approval(prepared.call, decision)
            if approval.approved:
                return await self._execute_prepared(
                    prepared.call,
                    decision,
                    approval_id=approval.approval_id,
                )
            if approval.deferred:
                if approval.approval_id:
                    self._pending[approval.approval_id] = prepared.call
                return _approval_observation(prepared.call, decision, approval.approval_id)
            return _blocked_observation(
                prepared.call,
                decision,
                status="denied",
                error_code="approval_denied",
                message=approval.reason or "Tool execution denied by user",
                approval_id=approval.approval_id,
            )
        return await self._execute_prepared(prepared.call, decision)

    async def resume(self, decision: ToolResumeDecision) -> ToolObservation:
        prepared = self._pending.get(decision.approval_id)
        if prepared is None:
            return ToolObservation(
                tool_call_id="",
                name="",
                status="error",
                content=(TextContent(text=f"Approval not found: {decision.approval_id}"),),
                metadata={
                    "approval_id": decision.approval_id,
                    "error_code": "approval_not_found",
                },
            )
        if decision.decision == "deny":
            self._pending.pop(decision.approval_id, None)
            return ToolObservation(
                tool_call_id=prepared.request.tool_call_id,
                name=prepared.request.name,
                status="denied",
                content=(TextContent(text=decision.reason or "Tool execution denied by user"),),
                metadata={
                    "approval_id": decision.approval_id,
                    "approved": False,
                    "error_code": "approval_denied",
                },
            )
        self._pending.pop(decision.approval_id, None)
        request = replace(prepared.request, source="approval_resume")
        prepared = replace(prepared, request=request)
        approval_decision = ToolDecision(
            "allow",
            "approved_by_user",
            {"approval_id": decision.approval_id},
        )
        return await self._execute_prepared(
            prepared,
            approval_decision,
            approval_id=decision.approval_id,
        )

    async def _execute_prepared(
        self,
        call: PreparedToolCall,
        decision: ToolDecision,
        *,
        approval_id: str | None = None,
    ) -> ToolObservation:
        started_at = time.monotonic()
        try:
            result = await _maybe_await(call.definition.execute(call.request))
        except asyncio.CancelledError:
            result = error_result(
                "Tool execution cancelled",
                status="cancelled",
                error_code="tool_cancelled",
            )
        except Exception as exc:
            result = error_result(
                f"Tool execution failed: {exc}",
                status="error",
                error_code="tool_exception",
            )
            result.details = {
                "reason": "tool_exception",
                "error_kind": type(exc).__name__,
            }
        normalized = self.result_policy.normalize(
            result,
            tool_call_id=call.request.tool_call_id,
            tool_name=call.request.name,
            metadata=call.metadata,
            approval_id=approval_id,
            permission_decision={"decision": decision.kind, "reason": decision.reason, **decision.details},
            started_at=started_at,
        )
        if normalized.status != "approval_required":
            normalized = await self._run_after_hook(call.request, normalized)
        return _observation_from_result(call.request, normalized)

    async def _run_before_hook(
        self,
        invocation: ToolInvocation,
    ) -> BeforeToolCallResult | None:
        if self.before_tool_call is None:
            return None
        value = self.before_tool_call(_before_context(invocation), None)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _run_after_hook(
        self,
        request: ToolCallRequest,
        result: ToolResult,
    ) -> ToolResult:
        if self.after_tool_call is None:
            return result
        ctx = AfterToolCallContext(
            assistant_message=request.assistant_message or AssistantMessage(content=[]),
            tool_call=ToolCall(
                id=request.tool_call_id,
                name=request.name,
                arguments=dict(request.arguments),
            ),
            args=dict(request.arguments),
            result=result,
            is_error=bool(result.is_error),
            context=request.context or ToolHookContextSnapshot(run_id=request.run_id),
        )
        value = self.after_tool_call(ctx, None)
        if inspect.isawaitable(value):
            value = await value
        if value is not None:
            _apply_after_patch(ctx, value)
        return ctx.result


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


def _before_context(invocation: ToolInvocation) -> BeforeToolCallContext:
    return BeforeToolCallContext(
        assistant_message=invocation.assistant_message or AssistantMessage(content=[]),
        tool_call=ToolCall(
            id=invocation.tool_call_id,
            name=invocation.name,
            arguments=dict(invocation.arguments),
        ),
        args=dict(invocation.arguments),
        context=invocation.context or ToolHookContextSnapshot(run_id=invocation.run_id),
    )


def _apply_after_patch(ctx: AfterToolCallContext, patch: AfterToolCallResult) -> None:
    if patch.content is not None:
        ctx.result.content = list(patch.content)
    if patch.details is not None:
        ctx.result.details = patch.details
    if patch.is_error is not None:
        ctx.is_error = bool(patch.is_error)
        ctx.result.is_error = ctx.is_error
        if ctx.result.is_error and ctx.result.status == "success":
            ctx.result.status = "error"
        elif not ctx.result.is_error and ctx.result.status == "error":
            ctx.result.status = "success"


def _blocked_observation(
    call: PreparedToolCall,
    decision: ToolDecision,
    *,
    status: ToolObservationStatus,
    error_code: str | None = None,
    message: str | None = None,
    approval_id: str | None = None,
) -> ToolObservation:
    metadata = {
        "error_code": error_code or decision.reason or "tool_denied",
        "permission_decision": {"decision": decision.kind, "reason": decision.reason, **decision.details},
    }
    if approval_id:
        metadata["approval_id"] = approval_id
    return ToolObservation(
        tool_call_id=call.request.tool_call_id,
        name=call.request.name,
        status=status,
        content=(TextContent(text=message or decision.reason or "Tool execution was blocked"),),
        metadata=metadata,
    )


def _approval_observation(
    call: PreparedToolCall,
    decision: ToolDecision,
    approval_id: str | None,
) -> ToolObservation:
    approval = approval_id or ""
    reason = decision.reason or "Tool execution requires approval"
    return ToolObservation(
        tool_call_id=call.request.tool_call_id,
        name=call.request.name,
        status="approval_required",
        content=(TextContent(text=reason),),
        interruption=ToolInterruption(
            approval_id=approval,
            run_id=call.request.run_id,
            tool_call_id=call.request.tool_call_id,
            tool_name=call.request.name,
            arguments=dict(call.request.arguments),
            reason=reason,
            risk=ToolRiskView(level=call.metadata.risk_level, summary=reason),
        ),
        metadata={
            "approval_id": approval,
            "error_code": "approval_required",
            "permission_decision": {
                "decision": decision.kind,
                "reason": decision.reason,
                **decision.details,
            },
        },
    )


def _observation_from_result(request: ToolCallRequest, result: ToolResult) -> ToolObservation:
    metadata = dict(result.metadata)
    if result.error_code:
        metadata["error_code"] = result.error_code
    if result.details is not None:
        metadata["details"] = result.details
    if result.approval_id is not None:
        metadata["approval_id"] = result.approval_id
    return ToolObservation(
        tool_call_id=request.tool_call_id,
        name=request.name,
        status=_observation_status(result.status, is_error=result.is_error),
        content=tuple(result.content),
        affected_paths=tuple(result.affected_paths),
        workspace_changed=bool(result.workspace_changed),
        verification=_verification_items(
            result.verification,
            tool_call_id=request.tool_call_id,
            tool_name=request.name,
        ),
        metadata=metadata,
    )


def _observation_status(status: object, *, is_error: bool) -> ToolObservationStatus:
    text = str(status or "").strip()
    if text in {"approval_required", "denied", "cancelled"}:
        return text  # type: ignore[return-value]
    if text == "success" and not is_error:
        return "success"
    return "error"


def _verification_items(
    value: object,
    *,
    tool_call_id: str,
    tool_name: str,
) -> tuple[RunVerification, ...]:
    if isinstance(value, RunVerification):
        return (value,)
    if isinstance(value, dict):
        try:
            return (
                RunVerification(
                    tool_call_id=str(value.get("tool_call_id") or tool_call_id),
                    tool_name=str(value.get("tool_name") or tool_name),
                    status=value.get("status", "unknown"),  # type: ignore[arg-type]
                    command=value.get("command"),
                    exit_code=value.get("exit_code"),
                    summary=str(value.get("summary") or ""),
                ),
            )
        except (TypeError, ValueError):
            return ()
    if isinstance(value, list):
        items: list[RunVerification] = []
        for item in value:
            items.extend(
                _verification_items(
                    item,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )
            )
        return tuple(items)
    return ()


__all__ = ["AfterToolHook", "BeforeToolHook", "ToolRuntime"]
