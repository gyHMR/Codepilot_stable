from __future__ import annotations

"""Canonical ToolRuntime: the only boundary allowed to start tool handlers."""

import asyncio
import json
import time
from dataclasses import dataclass, field

from .contracts import ToolExecutionContext, ToolExecutionRequest, ToolHandlerError, ToolPort
from .execution import (
    ExecutionController,
    ToolExecutionCancelledError,
    ToolExecutionTimeoutError,
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
    request: ToolExecutionRequest
    registration: object
    resolution: object
    attempt_id: str
    started_at_ms: int


@dataclass
class ToolRuntime(ToolPort):
    registry: ToolRegistry
    permission_engine: PermissionEngine = field(default_factory=PermissionEngine)
    state_store: ToolStateStore = field(default_factory=InMemoryToolStateStore)
    execution_controller: ExecutionController = field(default_factory=ExecutionController)

    def __post_init__(self) -> None:
        if not callable(getattr(self.state_store, "compare_and_set", None)):
            raise TypeError("state_store must implement ToolStateStore")

    def catalog_snapshot(self, *, mode=None) -> ToolCatalogSnapshot:
        return self.registry.catalog_snapshot(mode=mode)

    async def execute(self, request: ToolExecutionRequest) -> ToolResult:
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
        items = tuple(requests)
        if any(not isinstance(item, ToolExecutionRequest) for item in items):
            raise TypeError("ToolRuntime.execute_batch expects ToolExecutionRequest values")
        admitted: list[_Prepared | ToolResult] = []
        for request in items:
            item = self._prepare(request)
            admitted.append(item)
            if isinstance(item, ToolResult) or item.registration.category == "interaction":
                break

        results: list[ToolResult] = []
        index = 0
        while index < len(admitted):
            item = admitted[index]
            if isinstance(item, ToolResult):
                results.append(item)
                break
            if item.registration.policy.concurrency.mode == "parallel":
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
                if any(value.status in {"approval_required", "user_input_required", "denied"} for value in batch_results):
                    break
                continue
            result = await self._run_handler(item)
            results.append(result)
            index += 1
            if result.status in {"approval_required", "user_input_required", "denied"}:
                break
        return results

    async def cancel(self, attempt_id: str) -> bool:
        return await self.execution_controller.cancel(attempt_id)

    def approval_challenge(self, approval_id: str):
        record = self.state_store.find_by_approval_id(approval_id)
        return record.challenge if record is not None else None

    def pending_challenges(self):
        return self.state_store.pending_challenges()

    async def resume(self, response: ApprovalResponse | InteractionResponse) -> ToolResult:
        if isinstance(response, InteractionResponse):
            return await self._resume_interaction(response)
        if isinstance(response, ApprovalResponse):
            return await self._resume_approval(response)
        raise TypeError("ToolRuntime.resume expects ApprovalResponse or InteractionResponse")

    def _prepare(self, request: ToolExecutionRequest) -> _Prepared | ToolResult:
        started = _now_ms()
        attempt_id = attempt_id_for(request)
        try:
            self.state_store.create(ToolAttemptRecord(attempt_id=attempt_id, request=request))
        except Exception as exc:
            return _failure(request, "tool.state.conflict", "internal", str(exc), started)
        self._transition(attempt_id, "validating")
        try:
            materialized = self.registry.materialize(request.tool_name, request.registration_id)
        except ToolRegistrationNotFoundError as exc:
            return self._settle(attempt_id, "failed", _failure(request, "tool.registration.not_found", "registration", str(exc), started))
        except StaleToolRegistrationError as exc:
            return self._settle(attempt_id, "failed", _failure(request, "tool.registration.stale", "stale_registration", str(exc), started))
        registration = materialized.registration
        try:
            decoded = registration.input_codec.decode(request.arguments)
        except Exception as exc:
            return self._settle(attempt_id, "failed", _failure(request, "tool.input.invalid", "validation", str(exc) or "Tool input validation failed", started))
        self._transition(attempt_id, "resolving_access")
        try:
            resolution = registration.access_resolver.resolve(decoded, request)
        except Exception as exc:
            return self._settle(attempt_id, "denied", _failure(request, "tool.access.invalid", "validation", str(exc) or "Tool access resolution failed", started, status="denied"))
        permission = self.permission_engine.decide(request, registration.policy, resolution.access)
        if permission.denied:
            return self._settle(attempt_id, "denied", _failure(request, "tool.permission.denied", "permission", permission.reason, started, status="denied"))
        if permission.requires_approval:
            grant = self.state_store.find_reusable_grant(request, resolution.access)
            if grant is None:
                challenge = build_approval_challenge(request, resolution.access, reason=permission.reason)
                self._transition(attempt_id, "awaiting_approval", challenge=challenge)
                return _approval_result(request, challenge, started)
            self._transition(attempt_id, "queued", grant=grant, grant_consumed=False)
        else:
            self._transition(attempt_id, "queued")
        return _Prepared(request, registration, resolution, attempt_id, started)

    async def _run_handler(self, prepared: _Prepared) -> ToolResult:
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
                concurrency=registration.policy.concurrency,
                request_deadline_at_ms=request.deadline_at_ms,
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

        actual = frozenset(item.kind for item in effects.items)
        if not actual <= prepared.resolution.access.effects:
            return settle("failed", _failure(request, "tool.effect.policy_violation", "policy_violation", "Observed effects exceed authorized access", prepared.started_at_ms, effects=effects.items))
        try:
            encoded = registration.output_codec.encode(output)
            if not isinstance(encoded, dict):
                raise TypeError("Canonical tool output must encode to an object")
            content = tuple(registration.renderer.render(encoded))
            _guard_output(encoded, content, registration.policy.output_limits)
        except Exception as exc:
            return settle("failed", _failure(request, "tool.output.invalid", "output_validation", str(exc) or "Tool output validation failed", prepared.started_at_ms, effects=effects.items))
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

    async def _resume_approval(self, response: ApprovalResponse) -> ToolResult:
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

    def _transition(self, attempt_id: str, state, **changes) -> None:
        record = self.state_store.get(attempt_id)
        if record is None:
            raise RuntimeError(f"Tool attempt not found: {attempt_id}")
        self.state_store.compare_and_set(attempt_id, record.state, transition(record, state, **changes))

    def _settle(self, attempt_id: str, state, result: ToolResult, *, cleanup_errors: tuple[str, ...] = ()) -> ToolResult:
        self._transition(attempt_id, state, result=result, cleanup_errors=cleanup_errors)
        return result


@dataclass
class _EffectReporter:
    _items: list[ToolEffect] = field(default_factory=list)

    @property
    def items(self) -> tuple[ToolEffect, ...]:
        return tuple(self._items)

    def report(self, effect: object) -> None:
        if not isinstance(effect, ToolEffect):
            raise TypeError("effect reporter expects ToolEffect")
        self._items.append(effect)


def _failure(request, code, kind, message, started, *, status="error", effects=(), retryable=False, details=None) -> ToolResult:
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
    return _failure(
        request,
        "tool.approval.consumed",
        "approval",
        "Approval has already been consumed",
        started,
    )


def _interaction_error(response, code, message) -> ToolResult:
    return ToolResult(
        response.tool_call_id,
        response.tool_name,
        "error",
        content=(TextContent(text=message),),
        error=ToolError(code, "interaction", message),
        registration_id=response.registration_id,
    )


def _unknown_response(kind, identifier, tool_call_id=None, tool_name=None, registration_id=None) -> ToolResult:
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
    finished = _now_ms()
    return ToolTiming(started_at_ms=started, finished_at_ms=finished, duration_ms=max(0, finished - started))


def _guard_output(data, content, limits) -> None:
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


def _now_ms() -> int:
    return int(time.time() * 1000)


__all__ = ["ToolRuntime"]
