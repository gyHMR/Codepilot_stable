from __future__ import annotations

"""
工具运行时（ToolRuntime）模块 —— 统一的工具执行管线。

ToolRuntime 是工具层的核心调度引擎，实现了完整的工具执行安全管线:
    1. prepare（准备）  → 查找工具定义，校验参数
    2. before hook（前置钩子） → 执行调用方/扩展注册的拦截逻辑
    3. permission（权限） → PermissionPolicy 判定 allow/deny/approval_required
    4. approval（审批） → 如需审批，通过 ApprovalProvider 请求用户确认
    5. execute（执行）   → 调用工具的实际 execute 函数
    6. after hook（后置钩子） → 执行结果后处理（脱敏、格式化等）
    7. normalize（规范化） → ToolResultPolicy 统一结果格式、脱敏、标记
    8. observe（观测）   → 返回 ToolObservation 供上层（Agent）使用

ToolRuntime 实现了 ToolPort 协议，是 Agent 与工具之间的唯一接口。
Agent 只知道 ToolPort.execute()，不关心内部管线如何运作。
"""

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

# ── 钩子函数类型别名 ──────────────────────────────────────────────────
# 工具执行前的拦截钩子: 接收 BeforeToolCallContext，返回 BeforeToolCallResult
BeforeToolHook = Callable[
    [BeforeToolCallContext, Any | None],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
# 工具执行后的处理钩子: 接收 AfterToolCallContext，返回 AfterToolCallResult
AfterToolHook = Callable[
    [AfterToolCallContext, Any | None],
    AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
]


@dataclass
class ToolRuntime(ToolPort):
    """
    工具运行时 —— Agent 与工具之间的唯一接口。

    实现了完整的安全执行管线，将工具注册、权限判定、审批流程、
    结果规范化组装为统一的执行引擎。

    属性:
        registry: 工具注册表，按名称查找工具定义。
        permission_policy: 权限策略，决定 allow/deny/approval_required。
        approval_provider: 审批提供者，默认使用 DeferredApprovalProvider（暂停等用户确认）。
        result_policy: 结果规范化策略，统一格式 + 脱敏。
        before_tool_call: 工具调用前钩子（可选），可短路拦截。
        after_tool_call: 工具调用后钩子（可选），可修改结果。
        _pending: 待审批的工具调用字典 {approval_id: PreparedToolCall}。
    """

    registry: ToolRegistry
    permission_policy: PermissionPolicy
    approval_provider: ApprovalProvider = DeferredApprovalProvider()
    result_policy: ToolResultPolicy = ToolResultPolicy()
    before_tool_call: BeforeToolHook | None = None
    after_tool_call: AfterToolHook | None = None

    def __post_init__(self) -> None:
        """初始化待审批调用的内存字典。"""
        self._pending: dict[str, PreparedToolCall] = {}

    # ── 工具目录暴露 ────────────────────────────────────────────────

    def catalog(self, current_mode: str = "build") -> ToolCatalogView:
        """
        返回当前模式下可用的工具目录视图。

        这是"模型看到的工具列表"的来源——catalog 返回的是 Tool.to_spec()
        的结果，只有 name/description/parameters，不包含 execute 函数。
        """
        return self.registry.catalog(current_mode=current_mode)

    # ── 主执行入口 ──────────────────────────────────────────────────

    async def execute(self, invocation: ToolInvocation) -> ToolObservation:
        """
        执行一次工具调用，走完整的 8 步安全管线。

        参数:
            invocation: 工具调用描述（名称、参数、来源、上下文等）。

        返回:
            ToolObservation: 结构化的执行结果，包含状态、内容、影响路径、
            验证信息、中断信息等。

        管线步骤:
            1. 通过 registry.prepare_call() 查找工具并校验参数
            2. 如果 prepared 失败 → 返回 error 观测
            3. 执行 before hook，如果 hook 返回 block=True → 返回 denied 观测
            4. 权限判定: permission_policy.decide()
            5. 如果 denied → 返回 denied 观测
            6. 如果 requires_approval:
               a. 调用 approval_provider.request_approval()
               b. 如果 approved → 继续执行
               c. 如果 deferred → 存入 _pending 等待 resume()，返回 approval_required
               d. 如果 denied → 返回 denied 观测
            7. 执行工具: _execute_prepared()
            8. 规范化结果并返回 ToolObservation
        """
        # 步骤 1: 查找工具并校验参数
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
        # 步骤 2: 准备失败（工具不存在/参数不合法）
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
        # 步骤 3: 执行前置钩子
        before = await self._run_before_hook(invocation)
        if before is not None and before.block:
            return ToolObservation(
                tool_call_id=invocation.tool_call_id,
                name=invocation.name,
                status="denied",
                content=(TextContent(text=before.reason or "Tool call blocked by hook"),),
                metadata={"error_code": "tool_hook_blocked"},
            )
        # 步骤 4: 权限判定
        decision = self.permission_policy.decide(prepared.call.request)
        # 步骤 5: 权限拒绝
        if decision.denied:
            return _blocked_observation(prepared.call, decision, status="denied")
        # 步骤 6: 需要审批
        if decision.requires_approval:
            approval = await self.approval_provider.request_approval(prepared.call, decision)
            if approval.approved:
                # 用户批准 → 继续执行
                return await self._execute_prepared(
                    prepared.call,
                    decision,
                    approval_id=approval.approval_id,
                )
            if approval.deferred:
                # 延迟审批 → 暂停，存入 _pending 等待 resume()
                if approval.approval_id:
                    self._pending[approval.approval_id] = prepared.call
                return _approval_observation(prepared.call, decision, approval.approval_id)
            # 用户拒绝
            return _blocked_observation(
                prepared.call,
                decision,
                status="denied",
                error_code="approval_denied",
                message=approval.reason or "Tool execution denied by user",
                approval_id=approval.approval_id,
            )
        # 步骤 7-8: 执行工具 + 规范化结果
        return await self._execute_prepared(prepared.call, decision)

    # ── 审批恢复 ────────────────────────────────────────────────────

    async def resume(self, decision: ToolResumeDecision) -> ToolObservation:
        """
        恢复之前被暂停的审批调用。

        用户通过 CLI/Web 做出审批决定后，Runtime 调用此方法继续或拒绝执行。

        参数:
            decision: 包含 approval_id 和决定（approve/deny）。

        返回:
            ToolObservation: 执行结果（如果批准）或 denied 观测（如果拒绝）。

        异常情况:
            - approval_id 不在 _pending 中 → 返回 error 观测
        """
        # 从 _pending 字典中查找对应的待审批调用
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
        # 用户拒绝
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
        # 用户批准 → 清理 _pending，标记来源为 approval_resume，执行
        self._pending.pop(decision.approval_id, None)
        # 将请求的来源标记为审批恢复，权限策略会因此放行
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

    # ── 内部执行方法 ─────────────────────────────────────────────────

    async def _execute_prepared(
        self,
        call: PreparedToolCall,
        decision: ToolDecision,
        *,
        approval_id: str | None = None,
    ) -> ToolObservation:
        """
        执行已准备好的工具调用（权限已通过）。

        流程:
            1. 记录开始时间
            2. 调用 tool.execute() 执行工具
            3. 捕获 CancelledError / 其他异常
            4. 通过 result_policy.normalize() 规范化结果
            5. 执行 after hook（如果结果不是 approval_required）
            6. 将 ToolResult 转换为 ToolObservation

        参数:
            call: 已准备好的工具调用。
            decision: 权限决策（用于记录在结果 metadata 中）。
            approval_id: 审批 ID（如果有审批流程）。

        返回:
            ToolObservation: 结构化的执行观测结果。
        """
        started_at = time.monotonic()
        try:
            # 调用工具的实际 execute 函数（可能是同步或异步）
            result = await _maybe_await(call.definition.execute(call.request))
        except asyncio.CancelledError:
            # 用户取消了执行
            result = error_result(
                "Tool execution cancelled",
                status="cancelled",
                error_code="tool_cancelled",
            )
        except Exception as exc:
            # 工具内部异常
            result = error_result(
                f"Tool execution failed: {exc}",
                status="error",
                error_code="tool_exception",
            )
            result.details = {
                "reason": "tool_exception",
                "error_kind": type(exc).__name__,
            }

        # 规范化结果: 统一格式、脱敏、标记输出质量、计算耗时
        normalized = self.result_policy.normalize(
            result,
            tool_call_id=call.request.tool_call_id,
            tool_name=call.request.name,
            metadata=call.metadata,
            approval_id=approval_id,
            permission_decision={"decision": decision.kind, "reason": decision.reason, **decision.details},
            started_at=started_at,
        )
        # 如果不是等待审批的状态，执行后置钩子
        if normalized.status != "approval_required":
            normalized = await self._run_after_hook(call.request, normalized)
        # 将最终的 ToolResult 转换为 ToolObservation
        return _observation_from_result(call.request, normalized)

    # ── 钩子执行 ────────────────────────────────────────────────────

    async def _run_before_hook(
        self,
        invocation: ToolInvocation,
    ) -> BeforeToolCallResult | None:
        """
        执行工具调用前的拦截钩子。

        如果注册了 before_tool_call 钩子，构建 BeforeToolCallContext
        并调用它。钩子可以返回 BeforeToolCallResult(block=True) 来
        阻止工具执行。

        返回:
            BeforeToolCallResult 如果钩子有返回值，None 表示不拦截。
        """
        if self.before_tool_call is None:
            return None
        value = self.before_tool_call(_before_context(invocation), None)
        # 钩子可能是同步或异步函数，统一处理
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _run_after_hook(
        self,
        request: ToolCallRequest,
        result: ToolResult,
    ) -> ToolResult:
        """
        执行工具调用后的处理钩子。

        如果注册了 after_tool_call 钩子，构建 AfterToolCallContext
        并调用它。钩子可以修改结果的 content、details 和 is_error 标志。

        参数:
            request: 工具调用请求。
            result: 工具执行结果（可被钩子修改）。

        返回:
            可能被钩子修改后的 ToolResult。
        """
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


# ── 辅助函数 ──────────────────────────────────────────────────────────


async def _maybe_await(value: Any) -> Any:
    """
    统一处理同步/异步调用结果。

    如果 value 是 Future 或 Coroutine，await 它；
    否则直接返回。这样工具 execute 函数可以是同步也可以是异步的。
    """
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


def _before_context(invocation: ToolInvocation) -> BeforeToolCallContext:
    """
    从 ToolInvocation 构建 BeforeToolCallContext。

    将工具层的 ToolInvocation 翻译为 protocols 层的 BeforeToolCallContext，
    供钩子函数使用。
    """
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
    """
    将 after hook 的修改应用到上下文中。

    AfterToolCallResult 的三个字段如果非 None，会覆盖上下文中的对应值:
        - content: 替换结果内容
        - details: 替换详细信息
        - is_error: 修改错误标志（同时修正 status）
    """
    if patch.content is not None:
        ctx.result.content = list(patch.content)
    if patch.details is not None:
        ctx.result.details = patch.details
    if patch.is_error is not None:
        ctx.is_error = bool(patch.is_error)
        ctx.result.is_error = ctx.is_error
        # 自动修正 status 与 is_error 的一致性
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
    """
    构建"被阻止"的观测结果。

    当权限判定为 deny 或用户拒绝审批时调用。
    生成包含拒绝原因和权限决策记录的 ToolObservation。
    """
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
    """
    构建"等待审批"的观测结果。

    当审批提供者返回 deferred=True 时调用。
    生成包含 ToolInterruption 的 ToolObservation，上层可以通过
    interruption.approval_id 关联到待审批调用。
    """
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
    """
    将执行完成的 ToolResult 转换为 ToolObservation。

    这是工具层内部数据（ToolResult）到 Agent 可见数据（ToolObservation）
    的转换边界。提取所有关键字段：状态、内容、影响路径、工作区变更、
    验证信息、元数据等。
    """
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
    """
    将工具结果的状态映射为观测状态。

    映射规则:
        - approval_required/denied/cancelled → 原样
        - success + 非错误 → success
        - 其他情况 → error
    """
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
    """
    从工具结果中提取验证信息。

    支持三种输入格式:
        - RunVerification 实例 → 直接包装为元组
        - dict → 尝试解析为 RunVerification
        - list → 递归提取
        - 其他 → 返回空元组
    """
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
