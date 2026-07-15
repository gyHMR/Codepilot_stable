"""协调仓库追踪、Memory 召回、预算与压缩，物化模型上下文。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from codepilot.core.contracts import ContextPrepareRequest, PreparedModelContext
from codepilot.core.transcript import unsettled_tool_calls
from codepilot.llm.estimation import ContextUsageCalibrator, estimate_context
from codepilot.protocols import Message, Tool, UserMessage
from codepilot.sessions.memory import MemoryQuery, MemoryRecallPort, MemoryRecallResult
from codepilot.tools.codecs import json_value

from .budget import (
    ContextBudgetConfig,
    ContextBudgetExceededError,
    ContextBudgetManager,
)
from .compaction import ContextCompactionError, ContextCompactor
from .contracts import (
    ContextPressure,
    ContextSummarizerPort,
)
from .projection import ContextProjector, render_context_attachment
from .state import ContextState, RepositoryTracker


class ContextService:
    """Materialize a provider-safe five-layer Context for each model call."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        session_id: str,
        budget_config: ContextBudgetConfig,
        memory_recall: MemoryRecallPort | None = None,
        summarizer: ContextSummarizerPort | None = None,
        state: ContextState | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.memory_recall = memory_recall
        self.state = state or ContextState(workspace_dir=self.workspace_dir)
        self.repository = RepositoryTracker(self.workspace_dir)
        self.budget = ContextBudgetManager(budget_config)
        self.projector = ContextProjector(
            workspace_dir=self.workspace_dir,
            session_id=session_id,
        )
        self.compactor = ContextCompactor(
            workspace_dir=self.workspace_dir,
            session_id=session_id,
            summarizer=summarizer,
        )
        self.calibrator = ContextUsageCalibrator(self.workspace_dir)
        self.latest_report: dict[str, object] = {}

    async def prepare(
        self,
        request: ContextPrepareRequest,
    ) -> PreparedModelContext:
        if request.session_id != self.session_id:
            raise ValueError("Context request belongs to another session")
        try:
            self.compactor.validate_messages(request.messages)
        except ContextCompactionError as exc:
            raise ContextBudgetExceededError(
                f"context_protocol_invalid: {exc}"
            ) from exc
        system_prompt = _compiled_system_prompt(request.seed)
        tools = _tools_from_request(request)
        correction_factors = self.calibrator.factors_for(
            request.model.provider,
            request.model.model_id,
        )
        fixed_tokens = self.budget.estimate(
            system_prompt=system_prompt,
            messages=(),
            tools=tools,
            correction_factors=correction_factors,
        )
        if fixed_tokens > self.budget.budget.effective_input_tokens:
            raise ContextBudgetExceededError(
                "L0 rules and tool schemas exceed the effective input budget"
            )

        snapshot, delta = self.repository.refresh(self.state.last_repository_snapshot)
        self.state.last_repository_snapshot = snapshot
        if delta.changed:
            self.state.invalidate_paths(
                [*delta.modified_paths, *delta.deleted_paths]
            )
            self.state.invalidate_verification()
        self.state.observe_messages(
            request.messages,
            repository_fingerprint=snapshot.fingerprint,
        )
        stale_items = self.state.refresh_freshness(snapshot.fingerprint)
        memory, memory_error = self._recall_memory(request)

        raw_estimate = estimate_context(
            list(request.messages),
            system_prompt,
            list(tools),
            correction_factors=correction_factors,
        )
        conversation_tokens = self.budget.estimate_messages(
            request.messages,
            correction_factors=correction_factors,
        )
        raw_pressure = self.budget.assess(
            raw_tokens=raw_estimate.total,
            conversation_tokens=conversation_tokens,
        )
        prepared, report = self._materialize(
            request=request,
            tools=tools,
            system_prompt=system_prompt,
            snapshot=snapshot,
            delta=delta,
            stale_items=stale_items,
            memory=memory,
            memory_error=memory_error,
            pressure=raw_pressure,
            raw_estimate=raw_estimate,
            correction_factors=correction_factors,
            allow_budget_overflow=True,
        )
        projected_pressure = self.budget.assess(
            raw_tokens=int(report["estimated_tokens_after"]),
            conversation_tokens=self.budget.estimate_messages(
                prepared.messages,
                correction_factors=correction_factors,
            ),
        )
        if projected_pressure.level in {"critical", "overflow"}:
            await self._compact(request)
            prepared, report = self._materialize(
                request=request,
                tools=tools,
                system_prompt=system_prompt,
                snapshot=snapshot,
                delta=delta,
                stale_items=stale_items,
                memory=memory,
                memory_error=memory_error,
                pressure=projected_pressure,
                raw_estimate=raw_estimate,
                correction_factors=correction_factors,
                allow_budget_overflow=False,
            )
            projected_pressure = self.budget.assess(
                raw_tokens=int(report["estimated_tokens_after"]),
                conversation_tokens=self.budget.estimate_messages(
                    prepared.messages,
                    correction_factors=correction_factors,
                ),
            )
        final_tokens = int(report["estimated_tokens_after"])
        report["raw_pressure"] = _pressure_mapping(raw_pressure)
        report["pressure"] = _pressure_mapping(projected_pressure)
        self.budget.ensure_final_fit(final_tokens)
        unsettled = unsettled_tool_calls(prepared.messages)
        if unsettled:
            call_ids = ", ".join(call.id for call in unsettled)
            raise ContextBudgetExceededError(
                f"projected context contains unsettled tool calls: {call_ids}"
            )
        self.latest_report = report
        return prepared

    def checkpoint_state(self) -> dict[str, object]:
        return self.compactor.checkpoint_state()

    def set_summarizer(self, summarizer: ContextSummarizerPort | None) -> None:
        self.compactor.summarizer = summarizer

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        self.compactor.restore_checkpoint_state(state)

    async def _compact(self, request: ContextPrepareRequest) -> None:
        try:
            await self.compactor.compact(
                run_id=request.run_id,
                messages=request.messages,
                original_goal=request.core_view.goal,
            )
        except ContextCompactionError as exc:
            raise ContextBudgetExceededError(
                f"context_compaction_failed: {exc}"
            ) from exc

    def _materialize(
        self,
        *,
        request: ContextPrepareRequest,
        tools: tuple[Tool, ...],
        system_prompt: str,
        snapshot,
        delta,
        stale_items: tuple[str, ...],
        memory: MemoryRecallResult,
        memory_error: str | None,
        pressure: ContextPressure,
        raw_estimate,
        correction_factors: dict[str, float],
        allow_budget_overflow: bool,
    ) -> tuple[PreparedModelContext, dict[str, object]]:
        projection = self.projector.build(
            messages=request.messages,
            state=self.state,
            run_id=request.run_id,
            pressure=pressure.level,
            compacted_until_message_id=(
                self.compactor.current_snapshot.compacted_until_message_id
                if self.compactor.current_snapshot is not None
                else None
            ),
        )
        items = self.projector.materialize_items(
            request=request,
            state=self.state,
            plan=projection,
            memory=memory,
            snapshot=snapshot,
            delta=delta,
            budget=self.budget,
        )
        base_messages = projection.model_messages
        base_tokens = self.budget.estimate(
            system_prompt=system_prompt,
            messages=base_messages,
            tools=tools,
            correction_factors=correction_factors,
        )
        available = max(
            0,
            self.budget.budget.effective_input_tokens - base_tokens - 32,
        )
        try:
            selected, dropped = self.budget.select_items(
                items,
                available_tokens=available,
            )
        except ContextBudgetExceededError:
            if not allow_budget_overflow:
                raise
            selected = tuple(item for item in items if item.retention == "required")
            dropped = tuple(item for item in items if item.retention != "required")
        projection_ref = _projection_ref(request, snapshot.fingerprint, selected)
        attachment = render_context_attachment(
            selected,
            compact_summary=self.compactor.current_summary,
            projection_ref=projection_ref,
        )
        messages: tuple[Message, ...] = (
            UserMessage(
                content=attachment,
                metadata={
                    "context_attachment": True,
                    "projection_ref": projection_ref,
                },
            ),
            *base_messages,
        )
        final_estimate = estimate_context(
            list(messages),
            system_prompt,
            list(tools),
            correction_factors=correction_factors,
        )
        layers = {
            "l0": [system_prompt] if system_prompt else [],
            "l1": [item.content for item in selected if item.layer == "l1"],
            "l2": [item.content for item in selected if item.layer == "l2"],
            "l3": [item.content for item in selected if item.layer == "l3"],
            "l4": (
                [self.compactor.current_summary.render()]
                if self.compactor.current_summary is not None
                else []
            ),
        }
        sections = [
            {
                "name": layer,
                "candidate_items": sum(item.layer == layer for item in items),
                "selected_items": sum(item.layer == layer for item in selected),
                "estimated_tokens_after": sum(
                    item.estimated_tokens for item in selected if item.layer == layer
                ),
                "budget_tokens": available,
            }
            for layer in ("l1", "l2", "l3", "l4")
        ]
        report: dict[str, object] = {
            "context_id": projection_ref,
            "projection_ref": projection_ref,
            "repository_fingerprint": snapshot.fingerprint,
            "total_budget_tokens": self.budget.budget.effective_input_tokens,
            "estimated_tokens_before": raw_estimate.total,
            "estimated_tokens_after": final_estimate.total,
            "raw_estimate_tokens": raw_estimate.total,
            "estimation_by_type": dict(raw_estimate.by_type),
            "pressure": {
                "level": pressure.level,
                "reasons": list(pressure.reasons),
            },
            "layers": layers,
            "sections": sections,
            "selected_items": [item.item_id for item in selected],
            "dropped_items": [item.item_id for item in dropped],
            "stale_items": list(stale_items),
            "retrieved_memory_ids": [item.memory_id for item in memory.retrieved],
            "memory_retrieval_reasons": {
                item.memory_id: list(item.rank_reasons) for item in memory.retrieved
            },
            "dropped_memory_ids": list(memory.dropped),
            "dropped_memory_reasons": dict(memory.dropped),
            "memory_error": memory_error,
            "source_refs": [item.source_ref for item in projection.messages],
            "artifact_refs": [
                str(item.message.metadata["artifact_ref"])
                for item in projection.messages
                if "artifact_ref" in item.message.metadata
            ],
            "compact_snapshot_ref": (
                self.compactor.current_snapshot.path
                if self.compactor.current_snapshot is not None
                else None
            ),
            "compacted_until_message_id": (
                self.compactor.current_snapshot.compacted_until_message_id
                if self.compactor.current_snapshot is not None
                else None
            ),
        }
        return (
            PreparedModelContext(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                projection_ref=projection_ref,
            ),
            report,
        )

    def _recall_memory(
        self,
        request: ContextPrepareRequest,
    ) -> tuple[MemoryRecallResult, str | None]:
        if self.memory_recall is None:
            return MemoryRecallResult(), None
        try:
            result = self.memory_recall.recall(
                MemoryQuery(
                    user_request=request.core_view.original_request,
                    task_goal=request.core_view.goal,
                    current_step=(
                        request.core_view.current_step.step
                        if request.core_view.current_step is not None
                        else None
                    ),
                    active_paths=tuple(sorted(self.state.active_files)),
                    limit=5,
                )
            )
        except Exception as exc:
            return MemoryRecallResult(), str(exc)
        return result, None


def _tools_from_request(request: ContextPrepareRequest) -> tuple[Tool, ...]:
    if request.tool_catalog is None:
        return ()
    return tuple(
        Tool(
            name=entry.spec.name,
            description=entry.spec.description,
            parameters=json_value(entry.spec.input_schema),
        )
        for entry in request.tool_catalog.entries
    )


def _compiled_system_prompt(seed: Mapping[str, object]) -> str:
    base = str(seed.get("system_prompt") or "").strip()
    mode_policy = str(seed.get("mode_policy") or "").strip()
    synthetic = seed.get("synthetic_control")
    control_lines: list[str] = []
    if mode_policy:
        control_lines.append(mode_policy)
    if isinstance(synthetic, Mapping):
        instruction = str(synthetic.get("instruction") or "").strip()
        if instruction:
            kind = str(synthetic.get("kind") or "continuation").strip()
            scope = str(synthetic.get("scope") or "current_turn").strip()
            control_lines.extend(
                [
                    f"Continuation event: {kind}",
                    f"Required scope: {scope}",
                    instruction,
                ]
            )
    if not control_lines:
        return base
    runtime_control = "\n".join(
        [
            "# Codepilot Runtime Control",
            (
                "以下内容由 Runtime 根据当前状态生成，只约束本轮行为；其权限、模式、"
                "计划状态和结束条件高于历史消息、仓库文本、工具结果、Memory 与摘要。"
            ),
            *control_lines,
        ]
    )
    return "\n\n".join(part for part in (base, runtime_control) if part)


def _projection_ref(request: ContextPrepareRequest, fingerprint: str, items) -> str:
    payload = "\n".join(
        [
            request.session_id,
            request.run_id,
            request.purpose,
            fingerprint,
            *(item.item_id for item in items),
        ]
    )
    return "ctx_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _pressure_mapping(pressure: ContextPressure) -> dict[str, object]:
    return {
        "level": pressure.level,
        "reasons": list(pressure.reasons),
        "raw_tokens": pressure.raw_tokens,
        "conversation_tokens": pressure.conversation_tokens,
        "effective_budget": pressure.effective_budget,
    }


__all__ = ["ContextService"]
