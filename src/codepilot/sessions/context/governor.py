from __future__ import annotations

"""统一上下文治理入口。"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from codepilot.core import AgentContext, ContextPreparationRequest, PreparedAgentContext
from codepilot.llm.overflow import estimate_context_tokens
from codepilot.protocols import (
    ContextCheckpoint,
    ContextReport,
)
from codepilot.sessions.memory.records import DURABLE_MEMORY_KINDS, MemoryQuery, RetrievedMemory
from codepilot.sessions.memory.rendering import render_memory

from .checkpoint import ContextCheckpointManager
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
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.state = state or SessionContextState(workspace_dir=self.workspace_dir)
        self.memory_retriever = memory_retriever
        self.pressure_policy = pressure_policy or ContextPressurePolicy()
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
        self.context_views_file = (
            self.workspace_dir
            / ".codepilot"
            / "sessions"
            / session_id
            / "context_views.jsonl"
        )

    async def prepare(
        self,
        context: AgentContext,
        request: ContextPreparationRequest,
    ) -> PreparedAgentContext:
        """准备一次 LLM 调用上下文。"""

        snapshot = self.snapshot_builder.build(context)
        recalled_memories = self._recall_memory(context)
        memory_lines = self._render_recalled_memory(recalled_memories)
        pinned = self._pinned_memory()
        if pinned:
            memory_lines.insert(0, f"[Pinned project memory] {pinned}")

        output_tokens = tool_output_tokens(context.messages)
        history_tokens = estimate_context_tokens(context.messages, "")
        estimated_tokens = estimate_context_tokens(
            context.messages,
            context.system_prompt,
        )
        pressure = self.pressure_policy.evaluate(
            request,
            estimated_tokens=estimated_tokens,
            tool_output_tokens=output_tokens,
            history_tokens=history_tokens,
        )

        evidence_lines = self.projector.render_evidence(
            evidence=self.state.evidence,
            artifacts=snapshot.artifact_refs,
            stale_items=snapshot.stale_items,
        )
        goal = latest_user_text(context.messages) or current_task_goal(context)
        checkpoint_created: ContextCheckpoint | None = None
        latest_checkpoint = snapshot.latest_checkpoint
        if pressure.level == "critical":
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
        )
        view = projection.view
        prepared_messages = projection.messages
        system_prompt = projection.system_prompt
        before_tokens = estimate_context_tokens(context.messages, context.system_prompt)
        after_tokens = estimate_context_tokens(prepared_messages, system_prompt)
        prefix_hash = _hash_text("\n".join(view.stable_rules))
        dynamic_hash = _hash_text(
            "\n".join(
                [
                    *view.working_state,
                    *view.recalled_memory,
                    *view.evidence,
                    *view.recent_messages,
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
            selected_items=selected_item_summaries(view),
            stale_items=snapshot.stale_items,
            repository_delta=snapshot.repository_delta,
            retrieved_memory_ids=[item.record.id for item in recalled_memories],
            memory_retrieval_reasons={
                item.record.id: list(item.reasons)
                for item in recalled_memories
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
            prefix_hash=prefix_hash,
            dynamic_hash=dynamic_hash,
        )
        self._append_context_view(report)
        return PreparedAgentContext(
            system_prompt=system_prompt,
            messages=prepared_messages,
            tools=list(context.tools),
            report=report,
        )

    def finalize_run(self, _result: object) -> None:
        """Run 结束后的治理扩展点；记忆沉淀仍由现有 MemoryWriter 负责。"""

    def _recall_memory(self, context: AgentContext) -> list[RetrievedMemory]:
        if self.memory_retriever is None:
            return []
        if hasattr(self.memory_retriever, "validate_freshness"):
            self.memory_retriever.validate_freshness()
        query = MemoryQuery(
            text=latest_user_text(context.messages),
            active_paths=sorted(self.state.active_files),
            task_phase=optional_signal(context, "phase"),
            action_intent=optional_signal(context, "action_intent"),
            recent_error=optional_signal(context, "recent_error_code"),
            retrieval_mode=context_mode(context),
        )
        retrieved = self.memory_retriever.retrieve(query)
        return [
            item
            for item in retrieved
            if item.record.kind in DURABLE_MEMORY_KINDS and item.record.status == "active"
        ]

    def _pinned_memory(self) -> str:
        if self.memory_retriever is None or not hasattr(self.memory_retriever, "pinned_memory"):
            return ""
        value = self.memory_retriever.pinned_memory()
        return value if isinstance(value, str) else ""

    def _render_recalled_memory(self, items: list[RetrievedMemory]) -> list[str]:
        lines: list[str] = []
        for item in items:
            lines.append(
                f"{render_memory(item.record)} "
                f"[score={item.score}; reasons={', '.join(item.reasons)}]"
            )
        return lines

    def _append_context_view(self, report: ContextReport) -> None:
        self.context_views_file.parent.mkdir(parents=True, exist_ok=True)
        with self.context_views_file.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(
                json.dumps(
                    {
                        "context_id": report.context_id,
                        "pressure": asdict(report.pressure) if report.pressure else None,
                        "tokens_by_layer": dict(report.tokens_by_layer),
                        "checkpoint_created": (
                            asdict(report.checkpoint_created)
                            if report.checkpoint_created
                            else None
                        ),
                        "artifact_refs": [
                            asdict(item) for item in report.artifact_refs
                        ],
                        "prefix_hash": report.prefix_hash,
                        "dynamic_hash": report.dynamic_hash,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = ["ContextGovernor"]
