from __future__ import annotations

# 新手导读：adapters.py 把 ToolRuntime 安全执行管线适配成 core 可消费的 ToolPort。
# 关注点：端口契约在 ports.py；这里才允许接触 ToolRuntime、ToolRuntimeRequest/Result。

import inspect
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

from .authoring import ToolRuntimeRequest, ToolRuntimeResult
from .engine import ToolRuntime
from .ports import (
    ToolCatalogView,
    ToolInvocation,
    ToolInterruption,
    ToolObservation,
    ToolObservationStatus,
    ToolPort,
    ToolResumeDecision,
    ToolRiskView,
)


BeforeToolHook = Callable[
    [BeforeToolCallContext, Any | None],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
AfterToolHook = Callable[
    [AfterToolCallContext, Any | None],
    AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
]


class ToolRuntimePort(ToolPort):
    """V2 ToolPort adapter over the existing ToolRuntime safety pipeline."""

    def __init__(
        self,
        runtime: ToolRuntime,
        *,
        before_tool_call: BeforeToolHook | None = None,
        after_tool_call: AfterToolHook | None = None,
    ) -> None:
        self._runtime = runtime
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._pending: dict[str, _PendingToolExecution] = {}

    def catalog(self) -> ToolCatalogView:
        return ToolCatalogView(
            tuple(tool.to_spec() for tool in self._runtime.registry.list())
        )

    async def execute(self, invocation: ToolInvocation) -> ToolObservation:
        before = await self._run_before_hook(invocation)
        if before is not None and before.block:
            return _blocked_by_hook(invocation, before)

        request = ToolRuntimeRequest(
            tool_call_id=invocation.tool_call_id,
            name=invocation.name,
            params=invocation.arguments,
            source=invocation.source,
        )
        result = await self._runtime.execute(request)
        if result.status != "approval_required":
            result = await self._run_after_hook(invocation, result)
        observation = _observation_from_runtime_result(
            invocation.run_id,
            request,
            result,
        )
        if observation.status == "approval_required" and observation.interruption is not None:
            self._pending[observation.interruption.approval_id] = _PendingToolExecution(
                request=request,
                invocation=invocation,
            )
        return observation

    async def resume(self, decision: ToolResumeDecision) -> ToolObservation:
        pending = self._pending.get(decision.approval_id)
        if pending is None:
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
        request = pending.request
        if decision.decision == "deny":
            self._pending.pop(decision.approval_id, None)
            return ToolObservation(
                tool_call_id=request.tool_call_id,
                name=request.name,
                status="denied",
                content=(TextContent(text=decision.reason or "Tool execution denied by user"),),
                metadata={
                    "approval_id": decision.approval_id,
                    "approved": False,
                    "error_code": "approval_denied",
                },
            )
        result = await self._runtime.execute(
            request,
            approval_id=decision.approval_id,
        )
        result = await self._run_after_hook(pending.invocation, result)
        observation = _observation_from_runtime_result(
            pending.invocation.run_id,
            request,
            result,
            approval_id=decision.approval_id,
        )
        self._pending.pop(decision.approval_id, None)
        return observation

    async def _run_before_hook(
        self,
        invocation: ToolInvocation,
    ) -> BeforeToolCallResult | None:
        if self._before_tool_call is None:
            return None
        value = self._before_tool_call(_before_context(invocation), None)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _run_after_hook(
        self,
        invocation: ToolInvocation,
        result: ToolRuntimeResult,
    ) -> ToolRuntimeResult:
        if self._after_tool_call is None:
            return result
        ctx = _after_context(invocation, result)
        value = self._after_tool_call(ctx, None)
        if inspect.isawaitable(value):
            value = await value
        if value is not None:
            _apply_after_patch(ctx, value)
        return _sync_after_hook_result(result, ctx)


@dataclass(frozen=True)
class _PendingToolExecution:
    request: ToolRuntimeRequest
    invocation: ToolInvocation


def _before_context(invocation: ToolInvocation) -> BeforeToolCallContext:
    return BeforeToolCallContext(
        assistant_message=_assistant_message(invocation),
        tool_call=_tool_call(invocation),
        args=dict(invocation.arguments),
        context=_hook_snapshot(invocation),
    )


def _after_context(
    invocation: ToolInvocation,
    result: ToolRuntimeResult,
) -> AfterToolCallContext:
    return AfterToolCallContext(
        assistant_message=_assistant_message(invocation),
        tool_call=_tool_call(invocation),
        args=dict(invocation.arguments),
        result=result.result,
        is_error=bool(result.is_error),
        context=_hook_snapshot(invocation),
    )


def _assistant_message(invocation: ToolInvocation) -> AssistantMessage:
    return invocation.assistant_message or AssistantMessage(content=[])


def _tool_call(invocation: ToolInvocation) -> ToolCall:
    return ToolCall(
        id=invocation.tool_call_id,
        name=invocation.name,
        arguments=dict(invocation.arguments),
    )


def _hook_snapshot(invocation: ToolInvocation) -> ToolHookContextSnapshot:
    if invocation.context is not None:
        return invocation.context
    return ToolHookContextSnapshot(run_id=invocation.run_id)


def _blocked_by_hook(
    invocation: ToolInvocation,
    result: BeforeToolCallResult,
) -> ToolObservation:
    reason = result.reason or "Tool call blocked by hook"
    return ToolObservation(
        tool_call_id=invocation.tool_call_id,
        name=invocation.name,
        status="denied",
        content=(TextContent(text=reason),),
        metadata={
            "error_code": "tool_hook_blocked",
            "details": {
                "reason": reason,
                "hook": "before_tool_call",
            },
        },
    )


def _apply_after_patch(
    ctx: AfterToolCallContext,
    patch: AfterToolCallResult,
) -> None:
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


def _sync_after_hook_result(
    result: ToolRuntimeResult,
    ctx: AfterToolCallContext,
) -> ToolRuntimeResult:
    is_error = bool(ctx.is_error or ctx.result.is_error)
    status = ctx.result.status
    if is_error and status == "success":
        status = "error"
        ctx.result.status = status
    elif not is_error and status == "error":
        status = "success"
        ctx.result.status = status
    ctx.result.is_error = is_error
    return replace(result, status=status, is_error=is_error)


def _observation_from_runtime_result(
    run_id: str,
    request: ToolRuntimeRequest,
    result: ToolRuntimeResult,
    *,
    approval_id: str | None = None,
) -> ToolObservation:
    tool_result = result.result
    metadata = dict(tool_result.metadata)
    if tool_result.error_code:
        metadata["error_code"] = tool_result.error_code
    if tool_result.details is not None:
        metadata["details"] = tool_result.details
    resolved_approval_id = approval_id or result.approval_id or tool_result.approval_id
    if resolved_approval_id:
        metadata["approval_id"] = resolved_approval_id

    if result.status == "approval_required":
        interruption = ToolInterruption(
            approval_id=resolved_approval_id or "",
            run_id=run_id,
            tool_call_id=request.tool_call_id,
            tool_name=request.name,
            arguments=dict(request.params),
            reason=_approval_reason(tool_result.details),
            risk=ToolRiskView(
                level=_risk_level(tool_result.details),
                summary=_approval_reason(tool_result.details),
            ),
        )
        return ToolObservation(
            tool_call_id=request.tool_call_id,
            name=request.name,
            status="approval_required",
            content=tuple(tool_result.content),
            affected_paths=tuple(tool_result.affected_paths),
            workspace_changed=bool(tool_result.workspace_changed),
            interruption=interruption,
            metadata=metadata,
        )

    status = _observation_status(result.status, is_error=result.is_error)
    return ToolObservation(
        tool_call_id=request.tool_call_id,
        name=request.name,
        status=status,
        content=tuple(tool_result.content),
        affected_paths=tuple(tool_result.affected_paths),
        workspace_changed=bool(tool_result.workspace_changed),
        verification=_verification_items(
            tool_result.verification,
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
    tool_call_id: str = "",
    tool_name: str = "",
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


def _approval_reason(details: Any) -> str:
    if isinstance(details, dict):
        return str(
            details.get("policy_reason")
            or details.get("reason")
            or details.get("status")
            or ""
        )
    return ""


def _risk_level(details: Any) -> str:
    if isinstance(details, dict):
        value = details.get("risk_level")
        if value:
            return str(value)
    return "unknown"


__all__ = [
    "AfterToolHook",
    "BeforeToolHook",
    "ToolRuntimePort",
]
