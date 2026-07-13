"""规范的 ToolRuntime —— 唯一允许启动工具处理器的边界。

ToolRuntime 是整个工具子系统的"运行时层"，它接收来自 Core 的
ToolExecutionRequest，执行完整的处理流程：

1. prepare（准备阶段）：
   - 创建 ToolAttemptRecord（状态记录）
   - 验证注册有效性（materialize）
   - 解码输入参数（input_codec.decode）
   - 解析访问权限（access_resolver.resolve）
   - 执行权限决策（PermissionEngine.decide）
   - 如果需要审批，构建 ApprovalChallenge 并挂起

2. run_handler（执行阶段）：
   - 通过 ExecutionController 调度执行（并发控制 + 超时）
   - 调用工具处理器（handler.__call__）
   - 检查副作用是否超过授权范围
   - 编码输出（output_codec.encode）
   - 渲染内容（renderer.render）
   - 返回 ToolResult

3. resume（恢复阶段）：
   - 处理审批响应（ApprovalResponse）
   - 处理交互响应（InteractionResponse）
   - 恢复被挂起的工具执行

4. batch（批量执行）：
   - 并行/串行执行多个工具
   - 当遇到审批/交互/拒绝时，中断后续未执行的工具
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from .contracts import ToolExecutionContext, ToolExecutionRequest, ToolHandlerError, ToolPort
from .execution import (
    ExecutionController,
    ToolExecutionCancelledError,
    ToolExecutionTimeoutError,
    ToolProgressEvent,
    ToolQueueFullError,
    ToolQueueTimeoutError,
)
from .registry import (
    StaleToolRegistrationError,
    ToolCatalogSnapshot,
    ToolRegistrationNotFoundError,
    ToolRegistry,
)
from .results import (
    ArtifactContent,
    ImageContent,
    TextContent,
    ToolError,
    ToolResult,
    ToolTiming,
)
from .security import (
    ApprovalResponse,
    PermissionEngine,
    ToolEffect,
    approval_fingerprint,
    build_approval_challenge,
    issue_approval_grant,
)
from .state import (
    InMemoryToolStateStore,
    InteractionRequest,
    InteractionResponse,
    ToolAttemptRecord,
    ToolStateConflictError,
    ToolStateStore,
    attempt_id_for,
    transition,
)


@dataclass(frozen=True)
class _Prepared:
    """内部数据结构 —— 准备阶段完成后的结果。

    在 prepare 阶段，如果所有检查通过且不需要审批，
    将 request、registration、resolution 打包为 _Prepared
    传递给 _run_handler 执行。

    参数:
        request: 原始执行请求
        registration: 物化后的工具注册
        resolution: 访问解析结果（包含解码后的输入和权限请求）
        attempt_id: 工具尝试 ID
        started_at_ms: 开始时间戳
    """

    request: ToolExecutionRequest
    registration: object
    resolution: object
    attempt_id: str
    started_at_ms: int


@dataclass
class ToolRuntime(ToolPort):
    """工具运行时 —— 工具端口的标准实现。

    这是 ToolPort 协议的唯一实现，是工具子系统的核心编排者。

    职责：
    1. 接收和执行 ToolExecutionRequest
    2. 与 ToolRegistry 协作查询注册
    3. 与 PermissionEngine 协作决策权限
    4. 与 ExecutionController 协作调度执行
    5. 管理工具状态（通过 ToolStateStore）
    6. 处理审批流程和交互流程

    参数:
        registry: 工具注册中心
        permission_engine: 权限引擎
        state_store: 状态存储（默认 InMemoryToolStateStore）
        execution_controller: 执行控制器
        progress_callback: 进度事件回调（转发到上层）
    """

    registry: ToolRegistry
    permission_engine: PermissionEngine = field(default_factory=PermissionEngine)
    state_store: ToolStateStore = field(default_factory=InMemoryToolStateStore)
    execution_controller: ExecutionController = field(default_factory=ExecutionController)
    progress_callback: Callable[[ToolProgressEvent], Awaitable[None] | None] | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.state_store, "compare_and_set", None)):
            raise TypeError("state_store must implement ToolStateStore")

    def for_session(self, session_id: str) -> "ToolRuntime":
        """Create an equivalent runtime with isolated session attempt state."""

        from .state_store import CheckpointToolStateStore

        grant_store = getattr(self.state_store, "_grant_store", None)
        execution_controller = ExecutionController(self.execution_controller.limits)
        return ToolRuntime(
            registry=self.registry,
            permission_engine=self.permission_engine,
            state_store=CheckpointToolStateStore(
                session_id=session_id,
                grant_store=grant_store,
            ),
            execution_controller=execution_controller,
            progress_callback=self.progress_callback,
        )

    # ── ToolPort 接口实现 ──────────────────────────────────────────────────────

    def catalog_snapshot(self, *, mode=None) -> ToolCatalogSnapshot:
        """获取工具目录快照（透传给 Registry）。"""
        return self.registry.catalog_snapshot(mode=mode)

    async def execute(self, request: ToolExecutionRequest) -> ToolResult:
        """执行一次工具调用（单次执行）。

        参数:
            request: 工具执行请求

        返回:
            ToolResult（可能包含审批挑战、用户输入请求等挂起状态）
        """
        if not isinstance(request, ToolExecutionRequest):
            raise TypeError("ToolRuntime.execute expects ToolExecutionRequest")
        prepared = self._prepare(request)
        if isinstance(prepared, ToolResult):
            return prepared
        return await self._run_handler(prepared)

    async def execute_batch(
        self,
        requests: tuple[ToolExecutionRequest, ...] | list[ToolExecutionRequest],
    ) -> list[ToolResult]:
        """批量执行多个工具调用。

        执行策略：
        1. 对每个请求执行 prepare（准备阶段）
        2. 如果遇到审批/交互/拒绝的"屏障"，中断后续工具
        3. 对 parallel 模式的工具并行执行
        4. 对 serial 模式的工具串行执行

        参数:
            requests: 工具执行请求的列表/元组

        返回:
            ToolResult 列表（与输入顺序对应）
        """
        items = tuple(requests)
        if any(not isinstance(item, ToolExecutionRequest) for item in items):
            raise TypeError("ToolRuntime.execute_batch expects ToolExecutionRequest values")
        admitted: list[_Prepared | ToolResult] = []
        for position, request in enumerate(items):
            item = self._prepare(request)
            admitted.append(item)
            if isinstance(item, ToolResult) or item.registration.category == "interaction":
                admitted.extend(
                    self._interrupt_unstarted(
                        pending,
                        barrier_tool_call_id=request.tool_call_id,
                    )
                    for pending in items[position + 1 :]
                )
                break

        results: list[ToolResult] = []
        index = 0
        while index < len(admitted):
            item = admitted[index]
            if isinstance(item, ToolResult):
                # 遇到准备阶段返回的 ToolResult（错误/审批/交互）
                results.append(item)
                index += 1
                results.extend(self._interrupt_tail(admitted[index:], item.tool_call_id))
                break
            if item.registration.policy.concurrency.mode == "parallel":
                # 收集所有连续的 parallel 工具一起执行
                batch: list[_Prepared] = []
                while index < len(admitted):
                    candidate = admitted[index]
                    if isinstance(candidate, ToolResult):
                        break
                    if candidate.registration.policy.concurrency.mode != "parallel":
                        break
                    batch.append(candidate)
                    index += 1
                batch_results = await asyncio.gather(*(self._run_handler(value) for value in batch))
                results.extend(batch_results)
                # 如果并行批中有工具挂起/拒绝，中断后续
                if any(value.status in {"approval_required", "user_input_required", "denied"} for value in batch_results):
                    barrier = next(
                        value for value in batch_results
                        if value.status in {"approval_required", "user_input_required", "denied"}
                    )
                    results.extend(self._interrupt_tail(admitted[index:], barrier.tool_call_id))
                    break
                continue
            # serial 模式：逐个执行
            result = await self._run_handler(item)
            results.append(result)
            index += 1
            if result.status in {"approval_required", "user_input_required", "denied"}:
                results.extend(self._interrupt_tail(admitted[index:], result.tool_call_id))
                break
        return results

    def _interrupt_tail(
        self,
        items: list[_Prepared | ToolResult],
        barrier_tool_call_id: str,
    ) -> list[ToolResult]:
        """中断列表中剩余的未执行工具（因为前驱工具挂起了）。

        参数:
            items: 剩余的项目列表
            barrier_tool_call_id: 导致挂起的工具调用 ID

        返回:
            interrupted 状态的 ToolResult 列表
        """
        results: list[ToolResult] = []
        for item in items:
            if isinstance(item, ToolResult):
                results.append(item)
                continue
            result = _failure(
                item.request,
                "tool.batch.interrupted",
                "interrupted",
                f"Tool call was not started because '{barrier_tool_call_id}' paused the batch",
                item.started_at_ms,
                status="interrupted",
                details={"barrier_tool_call_id": barrier_tool_call_id},
            )
            results.append(self._settle(item.attempt_id, "interrupted", result))
        return results

    def _interrupt_unstarted(
        self,
        request: ToolExecutionRequest,
        *,
        barrier_tool_call_id: str,
    ) -> ToolResult:
        """标记一个从未开始的工具为 interrupted。

        适用于那些在 prepare 阶段之前就被中断的工具。
        """
        started = _now_ms()
        result = _failure(
            request,
            "tool.batch.interrupted",
            "interrupted",
            f"Tool call was not started because '{barrier_tool_call_id}' paused the batch",
            started,
            status="interrupted",
            details={"barrier_tool_call_id": barrier_tool_call_id},
        )
        try:
            self.state_store.create(
                ToolAttemptRecord(
                    attempt_id=attempt_id_for(request),
                    request=request,
                    state="interrupted",
                    result=result,
                )
            )
        except Exception as exc:
            return _failure(
                request,
                "tool.state.conflict",
                "internal",
                str(exc),
                started,
            )
        return result

    async def cancel(self, attempt_id: str) -> bool:
        """取消正在执行的工具。

        参数:
            attempt_id: 工具尝试的 ID

        返回:
            True 表示成功取消
        """
        return await self.execution_controller.cancel(attempt_id)

    def approval_challenge(self, approval_id: str):
        """获取审批挑战详情。"""
        record = self.state_store.find_by_approval_id(approval_id)
        return record.challenge if record is not None else None

    def pending_challenges(self):
        """获取所有待审批的挑战列表。"""
        return self.state_store.pending_challenges()

    def checkpoint_state(
        self,
        *,
        intent: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        """Return Tool-owned state for an opaque Sessions component checkpoint."""

        snapshot = getattr(self.state_store, "checkpoint_state", None)
        if callable(snapshot):
            return snapshot(intent=intent)
        return dict(intent) if intent else None

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        """Restore pending attempts before Core resumes the owning Run."""

        restore = getattr(self.state_store, "restore_checkpoint_state", None)
        if not callable(restore):
            raise RuntimeError("Configured ToolStateStore cannot restore checkpoints")
        restore(state)

    async def resume(self, response: ApprovalResponse | InteractionResponse) -> ToolResult:
        """恢复被挂起的工具执行。

        根据响应类型分发到不同的恢复流程：
        - ApprovalResponse → _resume_approval（审批响应）
        - InteractionResponse → _resume_interaction（用户输入响应）

        参数:
            response: 审批响应或交互响应

        返回:
            恢复执行后的 ToolResult
        """
        if isinstance(response, InteractionResponse):
            return await self._resume_interaction(response)
        if isinstance(response, ApprovalResponse):
            return await self._resume_approval(response)
        raise TypeError("ToolRuntime.resume expects ApprovalResponse or InteractionResponse")

    # ── 准备阶段 ──────────────────────────────────────────────────────────────

    def _prepare(self, request: ToolExecutionRequest) -> _Prepared | ToolResult:
        """准备阶段 —— 执行所有前置检查。

        流程：
        1. 创建状态记录（received）
        2. 验证参数 JSON 是否有效
        3. 物化注册（检查 name + registration_id 是否有效）
        4. 解码输入参数
        5. 解析访问权限
        6. 执行权限决策
        7. 如果需要审批 → 构建 ApprovalChallenge 并返回 approval_required
        8. 如果通过 → 返回 _Prepared 供执行阶段使用

        参数:
            request: 工具执行请求

        返回:
            _Prepared（可执行）或 ToolResult（错误/审批挂起）
        """
        started = _now_ms()
        attempt_id = attempt_id_for(request)
        try:
            self.state_store.create(ToolAttemptRecord(attempt_id=attempt_id, request=request))
        except Exception as exc:
            return _failure(request, "tool.state.conflict", "internal", str(exc), started)
        self._transition(attempt_id, "validating")
        # 检查参数解析错误
        if request.argument_parse_error is not None:
            return self._settle(
                attempt_id,
                "failed",
                _failure(
                    request,
                    "tool.arguments.invalid_json",
                    "validation",
                    request.argument_parse_error,
                    started,
                    details={"raw_arguments": request.raw_arguments or ""},
                ),
            )
        # 物化注册
        try:
            materialized = self.registry.materialize(request.tool_name, request.registration_id)
        except ToolRegistrationNotFoundError as exc:
            return self._settle(attempt_id, "failed", _failure(request, "tool.registration.not_found", "registration", str(exc), started))
        except StaleToolRegistrationError as exc:
            return self._settle(attempt_id, "failed", _failure(request, "tool.registration.stale", "stale_registration", str(exc), started))
        registration = materialized.registration
        # 解码输入参数
        try:
            decoded = registration.input_codec.decode(request.arguments)
        except Exception as exc:
            return self._settle(attempt_id, "failed", _failure(request, "tool.input.invalid", "validation", str(exc) or "Tool input validation failed", started))
        self._transition(attempt_id, "resolving_access")
        # 解析访问权限
        try:
            resolution = registration.access_resolver.resolve(decoded, request)
        except Exception as exc:
            return self._settle(attempt_id, "denied", _failure(request, "tool.access.invalid", "validation", str(exc) or "Tool access resolution failed", started, status="denied"))
        # 权限引擎决策
        permission = self.permission_engine.decide(request, registration.policy, resolution.access)
        if permission.denied:
            return self._settle(attempt_id, "denied", _failure(request, "tool.permission.denied", "permission", permission.reason, started, status="denied"))
        if permission.requires_approval:
            # 查找可复用的授权
            grant = self.state_store.find_reusable_grant(request, resolution.access)
            if grant is None:
                # 需要新的审批
                challenge = build_approval_challenge(request, resolution.access, reason=permission.reason)
                self._transition(attempt_id, "awaiting_approval", challenge=challenge)
                return _approval_result(request, challenge, started)
            # 找到可复用的授权
            self._transition(attempt_id, "queued", grant=grant, grant_consumed=False)
        else:
            self._transition(attempt_id, "queued")
        return _Prepared(request, registration, resolution, attempt_id, started)

    # ── 执行阶段 ──────────────────────────────────────────────────────────────

    async def _run_handler(self, prepared: _Prepared) -> ToolResult:
        """执行阶段 —— 实际运行工具处理器。

        流程：
        1. 更新状态为 running
        2. 通过 ExecutionController 调度 handler 执行
        3. 检查执行过程中的各种异常
        4. 检查 handler 是否返回了 InteractionRequest（用户输入请求）
        5. 验证工具副作用的实际范围（不超过授权范围）
        6. 编码输出并验证限制
        7. 返回 ToolResult（success 或对应错误）

        参数:
            prepared: 准备阶段的结果

        返回:
            完整的 ToolResult
        """
        request = prepared.request
        registration = prepared.registration
        effects = _EffectReporter()

        def settle(state: str, result: ToolResult) -> ToolResult:
            return self._settle(
                prepared.attempt_id,
                state,
                result,
                cleanup_errors=self.execution_controller.take_cleanup_errors(prepared.attempt_id),
            )

        async def operation(cancellation, progress, cleanup):
            self._transition(prepared.attempt_id, "running")
            return await registration.handler(
                prepared.resolution.input,
                ToolExecutionContext(
                    request=request,
                    cancellation=cancellation,
                    deadline_at_ms=request.deadline_at_ms,
                    progress=progress,
                    effects=effects,
                    cleanup=cleanup,
                ),
            )

        try:
            output = await self.execution_controller.run(
                attempt_id=prepared.attempt_id,
                session_id=request.session_id,
                timeout=registration.policy.timeout,
                requested_execution_ms=prepared.resolution.execution_timeout_ms,
                concurrency=registration.policy.concurrency,
                request_deadline_at_ms=request.deadline_at_ms,
                progress_callback=self.progress_callback,
                operation=operation,
            )
        except ToolQueueFullError:
            return settle("failed", _failure(request, "tool.queue.full", "queue_timeout", "Tool queue is full", prepared.started_at_ms, effects=effects.items))
        except ToolQueueTimeoutError:
            return settle("timed_out", _failure(request, "tool.queue.timeout", "queue_timeout", "Tool queue wait timed out", prepared.started_at_ms, status="timed_out", effects=effects.items))
        except ToolExecutionTimeoutError:
            return settle("timed_out", _failure(request, "tool.execution.timeout", "execution_timeout", "Tool execution timed out", prepared.started_at_ms, status="timed_out", effects=effects.items))
        except ToolExecutionCancelledError:
            return settle("cancelled", _failure(request, "tool.execution.cancelled", "cancelled", "Tool execution cancelled", prepared.started_at_ms, status="cancelled", effects=effects.items))
        except ToolHandlerError as exc:
            return settle("failed", _failure(request, exc.code, "execution", exc.message, prepared.started_at_ms, effects=effects.items, retryable=exc.retryable, details=exc.details))
        except Exception as exc:
            return settle("failed", _failure(request, "tool.execution.handler_error", "execution", f"Tool execution failed: {type(exc).__name__}", prepared.started_at_ms, effects=effects.items))

        # 检查是否返回了 InteractionRequest（用户输入请求）
        if isinstance(output, InteractionRequest):
            if registration.category != "interaction" or effects.items:
                return settle("failed", _failure(request, "tool.interaction.invalid_handler", "interaction", "Invalid interaction suspension", prepared.started_at_ms, effects=effects.items))
            self._transition(
                prepared.attempt_id,
                "awaiting_input",
                interaction=output,
                cleanup_errors=self.execution_controller.take_cleanup_errors(prepared.attempt_id),
            )
            return ToolResult(
                request.tool_call_id,
                request.tool_name,
                "user_input_required",
                interaction=output.to_dict(),
                timing=_timing(prepared.started_at_ms),
                registration_id=request.registration_id,
            )

        # 验证副作用范围（实际效果不超过授权范围）
        actual = frozenset(item.kind for item in effects.items)
        if not actual <= prepared.resolution.access.effects:
            return settle("failed", _failure(request, "tool.effect.policy_violation", "policy_violation", "Observed effects exceed authorized access", prepared.started_at_ms, effects=effects.items))
        if not _effect_resources_authorized(
            effects.items,
            prepared.resolution.access.resources,
        ):
            return settle(
                "failed",
                _failure(
                    request,
                    "tool.effect.resource_violation",
                    "policy_violation",
                    "Observed effect resource exceeds authorized access",
                    prepared.started_at_ms,
                    effects=effects.items,
                ),
            )

        # 编码输出
        try:
            encoded = registration.output_codec.encode(output)
            if not isinstance(encoded, dict):
                raise TypeError("Canonical tool output must encode to an object")
            if (
                registration.output_codec.json_schema is None
                and not registration.policy.output_trust.allow_structurally_validated
            ):
                raise ValueError("Structurally validated output is not allowed by tool policy")
            content = tuple(registration.renderer.render(encoded))
            _guard_output(encoded, content, registration.policy.output_limits)
        except Exception as exc:
            return settle("failed", _failure(request, "tool.output.invalid", "output_validation", str(exc) or "Tool output validation failed", prepared.started_at_ms, effects=effects.items))

        # 构建成功结果
        result = ToolResult(
            request.tool_call_id,
            request.tool_name,
            "success",
            content=content,
            data=encoded,
            effects=effects.items,
            artifacts=tuple(item.artifact for item in content if isinstance(item, ArtifactContent)),
            timing=_timing(prepared.started_at_ms),
            registration_id=request.registration_id,
            output_validation="schema_validated" if registration.output_codec.json_schema is not None else "structurally_validated",
            content_trust=registration.policy.output_trust.default_content_trust,
        )
        return settle("succeeded", result)

    # ── 恢复阶段 ──────────────────────────────────────────────────────────────

    async def _resume_approval(self, response: ApprovalResponse) -> ToolResult:
        """恢复审批 —— 用户对审批挑战做出响应后继续执行。

        流程：
        1. 查找审批挑战对应的 attempt 记录
        2. 验证状态（必须是 awaiting_approval）
        3. 验证指纹（确保响应匹配请求）
        4. 验证范围（allowed_scopes）
        5. 检查过期
        6. 如果拒绝 → 返回 denied
        7. 如果批准 → 重新验证注册、重新解码参数、重新解析权限
        8. 确认指纹未变 → 发放授权 → 执行 handler

        参数:
            response: 用户的审批响应

        返回:
            恢复执行后的 ToolResult
        """
        record = self.state_store.find_by_approval_id(response.approval_id)
        if record is None:
            return _unknown_response("approval", response.approval_id)
        request = record.request
        started = _now_ms()
        if record.state != "awaiting_approval" or record.grant_consumed:
            return _approval_consumed(request, started)
        challenge = record.challenge
        if challenge is None or response.request_fingerprint != challenge.request_fingerprint:
            return self._settle_approval_response(
                record,
                "denied",
                _failure(request, "tool.approval.fingerprint_mismatch", "permission", "Approval fingerprint mismatch", started, status="denied"),
                started,
            )
        if response.scope not in challenge.allowed_scopes:
            return self._settle_approval_response(
                record,
                "denied",
                _failure(request, "tool.approval.scope_denied", "permission", "Approval scope is not allowed", started, status="denied"),
                started,
            )
        if challenge.expires_at_ms is not None and _now_ms() >= challenge.expires_at_ms:
            return self._settle_approval_response(
                record,
                "denied",
                _failure(request, "tool.approval.expired", "permission", "Approval challenge expired", started, status="denied"),
                started,
            )
        if response.decision == "deny":
            return self._settle_approval_response(
                record,
                "denied",
                _failure(request, "tool.approval.denied", "permission", response.reason or "Tool execution denied", started, status="denied"),
                started,
            )
        # 批准：重新验证
        try:
            self.state_store.compare_and_set(
                record.attempt_id,
                "awaiting_approval",
                transition(record, "resolving_access"),
            )
        except ToolStateConflictError:
            return _approval_consumed(request, started)
        try:
            materialized = self.registry.materialize(request.tool_name, request.registration_id)
            registration = materialized.registration
            decoded = registration.input_codec.decode(request.arguments)
            resolution = registration.access_resolver.resolve(decoded, request)
        except Exception as exc:
            return self._settle(record.attempt_id, "denied", _failure(request, "tool.approval.revalidation_failed", "permission", str(exc), started, status="denied"))
        # 确认指纹未变（安全验证）
        if approval_fingerprint(request, resolution.access) != challenge.request_fingerprint:
            return self._settle(record.attempt_id, "denied", _failure(request, "tool.approval.fingerprint_mismatch", "permission", "Resolved approval fingerprint changed", started, status="denied"))
        grant = issue_approval_grant(challenge, response)
        self._transition(record.attempt_id, "queued", grant=grant, grant_consumed=response.scope == "once")
        return await self._run_handler(_Prepared(request, registration, resolution, record.attempt_id, started))

    def _settle_approval_response(
        self,
        record: ToolAttemptRecord,
        state: str,
        result: ToolResult,
        started: int,
    ) -> ToolResult:
        """结算审批响应 —— 设置审批响应的最终状态。"""
        try:
            self.state_store.compare_and_set(
                record.attempt_id,
                "awaiting_approval",
                transition(record, state, result=result),
            )
        except ToolStateConflictError:
            return _approval_consumed(record.request, started)
        return result

    async def _resume_interaction(self, response: InteractionResponse) -> ToolResult:
        """恢复交互 —— 用户提供输入后继续执行。

        流程：
        1. 查找交互对应的 attempt 记录
        2. 验证状态（awaiting_input 且未消费）
        3. 验证指纹、session_id、tool_call_id 等匹配
        4. 如果有选项限制，验证答案在选项中
        5. 编码答案并渲染
        6. 返回成功结果

        参数:
            response: 用户的交互响应

        返回:
            恢复执行后的 ToolResult
        """
        record = self.state_store.find_by_interaction_id(response.interaction_id)
        if record is None:
            return _unknown_response("interaction", response.interaction_id, response.tool_call_id, response.tool_name, response.registration_id)
        interaction = record.interaction
        if record.state != "awaiting_input" or record.interaction_consumed or interaction is None:
            return _interaction_error(response, "tool.interaction.already_consumed", "Interaction response has already been consumed")
        expected = (interaction.request_fingerprint, interaction.session_id, interaction.tool_call_id, interaction.tool_name, interaction.registration_id)
        received = (response.request_fingerprint, response.session_id, response.tool_call_id, response.tool_name, response.registration_id)
        if expected != received:
            return _interaction_error(response, "tool.interaction.fingerprint_mismatch", "Interaction response does not match request")
        if interaction.options and not interaction.allow_free_text and response.answers.get("answer") not in interaction.options:
            return _interaction_error(response, "tool.interaction.invalid_answer", "Answer must be one of the allowed options")
        request = record.request
        started = _now_ms()
        try:
            registration = self.registry.materialize(request.tool_name, request.registration_id).registration
            encoded = registration.output_codec.encode({"answers": dict(response.answers)})
            content = tuple(registration.renderer.render(encoded))
            _guard_output(encoded, content, registration.policy.output_limits)
        except Exception as exc:
            return self._settle(record.attempt_id, "failed", _failure(request, "tool.interaction.output_invalid", "output_validation", str(exc), started))
        result = ToolResult(
            request.tool_call_id,
            request.tool_name,
            "success",
            content=content,
            data=encoded,
            timing=_timing(started),
            registration_id=request.registration_id,
            content_trust=registration.policy.output_trust.default_content_trust,
        )
        try:
            self.state_store.compare_and_set(record.attempt_id, "awaiting_input", transition(record, "succeeded", result=result, interaction_consumed=True))
        except ToolStateConflictError:
            return _interaction_error(response, "tool.interaction.already_consumed", "Interaction response has already been consumed")
        return result

    # ── 状态管理 ──────────────────────────────────────────────────────────────

    def _transition(self, attempt_id: str, state, **changes) -> None:
        """状态转换 —— CAS 更新 attempt 状态。

        参数:
            attempt_id: 工具尝试 ID
            state: 新状态
            changes: 其他要更新的字段
        """
        record = self.state_store.get(attempt_id)
        if record is None:
            raise RuntimeError(f"Tool attempt not found: {attempt_id}")
        self.state_store.compare_and_set(attempt_id, record.state, transition(record, state, **changes))

    def _settle(self, attempt_id: str, state, result: ToolResult, *, cleanup_errors: tuple[str, ...] = ()) -> ToolResult:
        """结算 —— 设置最终状态并返回结果。

        参数:
            attempt_id: 工具尝试 ID
            state: 最终状态
            result: 工具执行结果
            cleanup_errors: 清理错误列表

        返回:
            ToolResult（与输入相同，方便链式调用）
        """
        self._transition(attempt_id, state, result=result, cleanup_errors=cleanup_errors)
        return result


# ── 内部辅助类 ────────────────────────────────────────────────────────────────


@dataclass
class _EffectReporter:
    """内部效果报告器 —— 收集工具处理器报告的副作用。

    在工具执行期间收集所有 ToolEffect，
    执行完成后由 ToolRuntime 验证副作用范围。
    """
    _items: list[ToolEffect] = field(default_factory=list)

    @property
    def items(self) -> tuple[ToolEffect, ...]:
        return tuple(self._items)

    def report(self, effect: object) -> None:
        if not isinstance(effect, ToolEffect):
            raise TypeError("effect reporter expects ToolEffect")
        self._items.append(effect)


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────


def _failure(request, code, kind, message, started, *, status="error", effects=(), retryable=False, details=None) -> ToolResult:
    """构建失败结果。"""
    return ToolResult(
        request.tool_call_id,
        request.tool_name,
        status,
        content=(TextContent(text=message),),
        error=ToolError(code, kind, message, retryable=retryable, details=details or {}),
        effects=tuple(effects),
        timing=_timing(started),
        registration_id=request.registration_id,
    )


def _approval_result(request, challenge, started) -> ToolResult:
    """构建审批挂起结果。"""
    return ToolResult(
        request.tool_call_id,
        request.tool_name,
        "approval_required",
        content=(TextContent(text=challenge.reason),),
        approval=challenge,
        timing=_timing(started),
        registration_id=request.registration_id,
    )


def _approval_consumed(request, started) -> ToolResult:
    """构建审批已消费错误结果。"""
    return _failure(
        request,
        "tool.approval.consumed",
        "approval",
        "Approval has already been consumed",
        started,
    )


def _interaction_error(response, code, message) -> ToolResult:
    """构建交互错误结果。"""
    return ToolResult(
        response.tool_call_id,
        response.tool_name,
        "error",
        content=(TextContent(text=message),),
        error=ToolError(code, "interaction", message),
        registration_id=response.registration_id,
    )


def _unknown_response(kind, identifier, tool_call_id=None, tool_name=None, registration_id=None) -> ToolResult:
    """构建未知响应错误结果。"""
    message = f"{kind.title()} request was not found"
    return ToolResult(
        tool_call_id or identifier,
        tool_name or "unknown",
        "error",
        content=(TextContent(text=message),),
        error=ToolError(f"tool.{kind}.not_found", kind, message),
        registration_id=registration_id or "unknown",
    )


def _timing(started: int) -> ToolTiming:
    """构建执行时间元数据。"""
    finished = _now_ms()
    return ToolTiming(started_at_ms=started, finished_at_ms=finished, duration_ms=max(0, finished - started))


def _guard_output(data, content, limits) -> None:
    """检查输出是否在限制范围内。

    检查 data 字节数、content 字节数、artifacts 数量和大小。
    """
    data_bytes = len(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))
    if data_bytes > limits.max_data_bytes:
        raise ValueError("Tool data exceeds output limit")
    content_bytes = 0
    artifacts = 0
    artifact_bytes = 0
    for item in content:
        if isinstance(item, TextContent):
            content_bytes += len(item.text.encode("utf-8"))
        elif isinstance(item, ImageContent):
            content_bytes += len(item.data.encode("utf-8"))
        elif isinstance(item, ArtifactContent):
            artifacts += 1
            artifact_bytes += item.artifact.size_bytes or 0
        else:
            raise TypeError("Renderer returned unsupported content")
    if content_bytes > limits.max_content_bytes:
        raise ValueError("Tool content exceeds output limit")
    if artifacts > limits.max_artifacts or artifact_bytes > limits.max_artifact_bytes:
        raise ValueError("Tool artifacts exceed output limit")


def _effect_resources_authorized(effects, resources) -> bool:
    """检查工具副作用的资源是否在授权范围内。

    每个副作用的 resource.uri 必须匹配（或以 / 前缀匹配）某个授权资源。
    """
    authorized = tuple(resource.uri.rstrip("/") for resource in resources)
    if not effects:
        return True
    if not authorized:
        return False
    return all(
        any(
            effect.resource.uri.rstrip("/") == base
            or effect.resource.uri.startswith(base + "/")
            for base in authorized
        )
        for effect in effects
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


__all__ = ["ToolRuntime"]
