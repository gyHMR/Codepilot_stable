from __future__ import annotations

# 新手导读：ContextGovernor 是上下文投影治理唯一入口，每轮模型调用前生成 PreparedAgentContext。
# 关注点：它串联 snapshot、memory、pressure policy、projector、checkpoint 和 context ledger。

"""统一上下文治理入口。"""

import hashlib
import inspect
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codepilot.core.contracts import AgentContext, ContextPreparationRequest, PreparedAgentContext
from codepilot.llm.estimation import estimate_context_tokens
from codepilot.protocols import (
    ContextCheckpoint,
    ContextPressure,
    ContextReport,
)
from codepilot.sessions.memory.records import MemoryQuery, MemoryRecall
from codepilot.sessions.storage import SessionLayout

from .checkpoint import ContextCheckpointManager
from .compactor import ContextCompactRequest, ContextCompactResult
from .ledger import ToolArtifactLedger
from .policy import ContextPressurePolicy
from .projector import (
    ContextProjector,
    context_mode,
    current_task_goal,
    latest_user_text,
    next_action,
    optional_signal,
    section_reports,
    selected_item_summaries,
    tokens_by_layer,
    tool_output_tokens,
    verification_state,
)
from .repository_tracker import RepositoryTracker
from .snapshot import SessionSnapshotBuilder
from .state import SessionContextState


class ContextGovernor:
    """从完整 Session 状态投影出本轮模型可消费的 ContextView。"""

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        session_id: str,
        state: SessionContextState | None = None,
        memory_retriever: Any | None = None,
        pressure_policy: ContextPressurePolicy | None = None,
        context_compactor: Any | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.state = state or SessionContextState(workspace_dir=self.workspace_dir)
        self.memory_retriever = memory_retriever
        self.pressure_policy = pressure_policy or ContextPressurePolicy()
        self.context_compactor = context_compactor
        self.repository = RepositoryTracker(self.workspace_dir)
        self.ledger = ToolArtifactLedger(
            workspace_dir=self.workspace_dir,
            session_id=session_id,
        )
        self.checkpoints = ContextCheckpointManager(
            workspace_dir=self.workspace_dir,
            session_id=session_id,
        )
        self.snapshot_builder = SessionSnapshotBuilder(
            workspace_dir=self.workspace_dir,
            state=self.state,
            repository=self.repository,
            ledger=self.ledger,
            checkpoints=self.checkpoints,
        )
        self.projector = ContextProjector(ledger=self.ledger)
        self.context_ledger_file = self.layout.context_ledger_file

    async def prepare(
        self,
        context: AgentContext,
        request: ContextPreparationRequest,
    ) -> PreparedAgentContext:
        """准备一次 LLM 调用上下文。"""

        snapshot = self.snapshot_builder.build(context)
        memory_recall = self._recall_memory(context)
        memory_lines = self._render_recalled_memory(memory_recall)

        output_tokens = tool_output_tokens(context.messages)
        history_tokens = estimate_context_tokens(context.messages, "")
        before_tokens = estimate_context_tokens(
            context.messages,
            context.system_prompt,
            context.tools,
        )
        pressure = self.pressure_policy.evaluate(
            request,
            estimated_tokens=before_tokens,
            tool_output_tokens=output_tokens,
            history_tokens=history_tokens,
        )

        evidence_lines = self.projector.render_evidence(
            evidence=self.state.evidence,
            artifacts=snapshot.artifact_refs,
            stale_items=snapshot.stale_items,
        )
        goal = current_task_goal(context)
        checkpoint_created: ContextCheckpoint | None = None
        latest_checkpoint = snapshot.latest_checkpoint
        compact_result: ContextCompactResult | None = None
        compact_summary = ""
        emergency_trim = False
        projection = None
        after_tokens = before_tokens
        for _ in range(5):
            if pressure.level == "critical" and checkpoint_created is None:
                checkpoint_created = self.checkpoints.create(
                    goal=goal,
                    active_files=snapshot.active_files,
                    changed_files=snapshot.changed_files,
                    key_evidence=evidence_lines[:8],
                    verification_state=verification_state(
                        snapshot.stale_items,
                        evidence_lines,
                    ),
                    next_actions=[next_action(context)],
                    source_refs=[ref.path for ref in snapshot.artifact_refs],
                )
                latest_checkpoint = checkpoint_created

            projection = self.projector.project(
                context=context,
                pressure=pressure,
                checkpoint=latest_checkpoint,
                active_files=snapshot.active_files,
                changed_files=snapshot.changed_files,
                memory_lines=memory_lines,
                evidence_lines=evidence_lines,
                compact_result=compact_result,
                emergency_trim=emergency_trim,
            )
            if (
                pressure.level == "critical"
                and self.context_compactor is not None
                and compact_result is None
            ):
                compact_result = await self._compact_context(
                    pressure=pressure,
                    goal=goal,
                    projection=projection,
                    artifact_refs=snapshot.artifact_refs,
                )
                compact_summary = compact_result.recovery_summary
                pressure = _merge_pressure_reasons(
                    pressure,
                    ["llm_compact"],
                )
                continue

            after_tokens = estimate_context_tokens(
                projection.messages,
                projection.system_prompt,
                context.tools,
            )
            projected_tool_tokens = tool_output_tokens(projection.messages)
            projected_history_tokens = estimate_context_tokens(projection.messages, "")
            projected_pressure = self.pressure_policy.evaluate(
                request,
                estimated_tokens=after_tokens,
                tool_output_tokens=projected_tool_tokens,
                history_tokens=projected_history_tokens,
            )
            final_pressure = _merge_pressure_reasons(
                projected_pressure,
                [
                    *pressure.reasons,
                    *(["llm_compact"] if compact_result is not None else []),
                    *(["emergency_context_trim"] if emergency_trim else []),
                ],
            )
            if (
                compact_result is not None
                and projected_pressure.level == "critical"
                and not emergency_trim
            ):
                emergency_trim = True
                pressure = final_pressure
                continue
            if (
                compact_result is not None
                and emergency_trim
                and projected_pressure.level == "critical"
            ):
                raise RuntimeError(
                    "context remains critical after compact and emergency trim"
                )
            if _pressure_rank(projected_pressure.level) <= _pressure_rank(
                pressure.level,
            ):
                if not (
                    self.context_compactor is None
                    and pressure.level == "critical"
                ):
                    pressure = final_pressure
                break
            pressure = _merge_pressure_reasons(
                projected_pressure,
                [*pressure.reasons, "projected_context_pressure"],
            )
        else:
            raise RuntimeError("context preparation did not converge")

        assert projection is not None
        view = projection.view
        prepared_messages = projection.messages
        system_prompt = projection.system_prompt
        prefix_hash = _hash_text("\n".join(view.stable_rules))
        dynamic_hash = _hash_text(
            "\n".join(
                [
                    *view.task_state,
                    *view.working_set,
                    *view.recalled_memory,
                    *view.conversation,
                ]
            )
        )
        report = ContextReport(
            context_id=f"ctx_{hashlib.sha256(dynamic_hash.encode()).hexdigest()[:16]}",
            repository_fingerprint=snapshot.repository_snapshot.fingerprint,
            total_budget_tokens=pressure.effective_budget,
            estimated_tokens_before=before_tokens,
            estimated_tokens_after=after_tokens,
            sections=section_reports(view, pressure.effective_budget),
            selected_items=selected_item_summaries(
                view,
                active_files=snapshot.active_files,
                changed_files=snapshot.changed_files,
                active_file_records=list(self.state.active_files.values()),
                evidence=self.state.evidence,
                artifacts=snapshot.artifact_refs,
                memory_ids=[item.record.id for item in memory_recall.retrieved],
            ),
            stale_items=snapshot.stale_items,
            repository_delta=snapshot.repository_delta,
            retrieved_memory_ids=[item.record.id for item in memory_recall.retrieved],
            memory_retrieval_reasons={
                item.record.id: list(item.reasons)
                for item in memory_recall.retrieved
            },
            context_mode=context_mode(context),
            pressure=pressure,
            context_view=view,
            checkpoint_used=latest_checkpoint
            if latest_checkpoint is not checkpoint_created
            else None,
            checkpoint_created=checkpoint_created,
            artifact_refs=snapshot.artifact_refs,
            tokens_by_layer=tokens_by_layer(view),
            compact_summary=compact_summary,
            prefix_hash=prefix_hash,
            dynamic_hash=dynamic_hash,
        )
        if checkpoint_created is not None or compact_summary:
            self._write_recovery_summary(checkpoint_created, compact_summary)
        self._append_context_view(report)
        return PreparedAgentContext(
            system_prompt=system_prompt,
            messages=prepared_messages,
            tools=list(context.tools),
            report=report,
        )

    def finalize_run(self, _result: object) -> None:
        """Run 结束后的治理扩展点；记忆沉淀仍由现有 MemoryWriter 负责。"""

    def _recall_memory(self, context: AgentContext) -> MemoryRecall:
        if self.memory_retriever is None:
            return MemoryRecall()
        query = MemoryQuery(
            text=latest_user_text(context.messages),
            active_paths=sorted(self.state.active_files),
            task_phase=optional_signal(context, "phase"),
            action_intent=optional_signal(context, "action_intent"),
            recent_error=optional_signal(context, "recent_error_code"),
            retrieval_mode=context_mode(context),
        )
        if hasattr(self.memory_retriever, "recall"):
            return self.memory_retriever.recall(query)
        return MemoryRecall()

    def _render_recalled_memory(self, recall: MemoryRecall) -> list[str]:
        lines: list[str] = []
        if recall.pinned_text:
            lines.append(f"[Pinned memory] {recall.pinned_text}")
        for item in [*recall.always, *recall.selected]:
            label = {
                "correction": "Correction",
                "constraint": "Constraint",
                "decision": "Decision",
                "experience": "Experience",
            }.get(item.record.type, "Memory")
            lines.append(
                f"[{label}] {item.record.content} "
                f"[reasons={', '.join(item.reasons)}]"
            )
        return lines

    async def _compact_context(
        self,
        *,
        pressure: ContextPressure,
        goal: str,
        projection: Any,
        artifact_refs: list[Any],
    ) -> ContextCompactResult:
        request = ContextCompactRequest(
            session_id=self.session_id,
            goal=goal,
            pressure=pressure,
            task_state_lines=list(projection.view.task_state),
            working_set_lines=list(projection.view.working_set),
            memory_lines=list(projection.view.recalled_memory),
            conversation_lines=list(projection.view.conversation),
            artifact_refs=list(artifact_refs),
            token_budget=max(128, pressure.effective_budget // 3),
        )
        compact_fn = getattr(self.context_compactor, "compact")
        value = compact_fn(request)
        if inspect.isawaitable(value):
            value = await value
        return _coerce_compact_result(value)

    def _append_context_view(self, report: ContextReport) -> None:
        self.context_ledger_file.parent.mkdir(parents=True, exist_ok=True)
        compact_summary = report.compact_summary or (
            _checkpoint_recovery_summary(report.checkpoint_created)
            if report.checkpoint_created is not None
            else ""
        )
        with self.context_ledger_file.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(
                json.dumps(
                    {
                        "type": "context_view",
                        "context_id": report.context_id,
                        "pressure": asdict(report.pressure) if report.pressure else None,
                        "tokens_by_layer": dict(report.tokens_by_layer),
                        "selected_items": list(report.selected_items),
                        "dropped_items": [
                            asdict(item) for item in report.dropped_items
                        ],
                        "memory_ids": list(report.retrieved_memory_ids),
                        "checkpoint_created": (
                            asdict(report.checkpoint_created)
                            if report.checkpoint_created
                            else None
                        ),
                        "artifact_refs": [
                            asdict(item) for item in report.artifact_refs
                        ],
                        "compact_summary": compact_summary,
                        "prefix_hash": report.prefix_hash,
                        "dynamic_hash": report.dynamic_hash,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _write_recovery_summary(
        self,
        checkpoint: ContextCheckpoint | None,
        compact_summary: str = "",
    ) -> None:
        path = self.layout.task_state_file
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            state = raw if isinstance(raw, dict) else {}
        else:
            state = {
                "schema_version": 1,
                "task_id": None,
                "raw_user_request": "",
                "current_mode": "build",
                "approval_state": "none",
                "goal": None,
                "proposed_plan": None,
                "approved_plan": None,
                "current_step_id": None,
                "steps": [],
                "verification_status": "unknown",
                "evidence_refs": [],
                "blocked_reason": None,
            }
        state["recovery_summary"] = compact_summary or _checkpoint_recovery_summary(
            checkpoint,
        )
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _pressure_rank(level: str) -> int:
    return {"normal": 0, "tight": 1, "critical": 2}.get(level, 1)


def _merge_pressure_reasons(
    pressure: ContextPressure,
    reasons: list[str],
) -> ContextPressure:
    return ContextPressure(
        level=pressure.level,
        effective_budget=pressure.effective_budget,
        estimated_tokens=pressure.estimated_tokens,
        reasons=_dedupe([*pressure.reasons, *reasons]),
    )


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _checkpoint_recovery_summary(checkpoint: ContextCheckpoint | None) -> str:
    if checkpoint is None:
        return ""
    parts = [f"Goal: {checkpoint.goal}"]
    if checkpoint.active_files:
        parts.append(f"Active files: {', '.join(checkpoint.active_files[:8])}")
    if checkpoint.changed_files:
        parts.append(f"Changed files: {', '.join(checkpoint.changed_files[:8])}")
    if checkpoint.key_evidence:
        parts.append(f"Evidence: {' | '.join(checkpoint.key_evidence[:4])}")
    if checkpoint.verification_state:
        parts.append(f"Verification: {checkpoint.verification_state}")
    if checkpoint.next_actions:
        parts.append(f"Next: {', '.join(checkpoint.next_actions[:4])}")
    return "\n".join(parts)


def _coerce_compact_result(value: object) -> ContextCompactResult:
    if isinstance(value, ContextCompactResult):
        return value
    if isinstance(value, dict):
        return ContextCompactResult(
            recovery_summary=str(value.get("recovery_summary") or ""),
            task_state_lines=_string_list(value.get("task_state_lines")),
            working_set_lines=_string_list(value.get("working_set_lines")),
            memory_lines=_string_list(value.get("memory_lines")),
            conversation_lines=_string_list(value.get("conversation_lines")),
            evidence_refs=_string_list(value.get("evidence_refs")),
            raw=dict(value),
        )
    return ContextCompactResult(recovery_summary=str(value or ""))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if str(item).strip()]


__all__ = ["ContextGovernor"]
