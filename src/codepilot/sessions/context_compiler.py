from __future__ import annotations

"""Compile a fresh, budgeted model context before every LLM call.

This module belongs to sessions because it uses long-lived session state,
memory, evidence, repository freshness, and the current Agent run's messages
to prepare the next model input.
"""

import re
import uuid
from dataclasses import dataclass, replace

from codepilot.core.types import (
    AgentContext,
    ContextPreparationRequest,
    PreparedAgentContext,
)
from codepilot.llm.overflow import estimate_context_tokens
from codepilot.protocols import (
    ContextItem,
    ContextReport,
    ContextSectionReport,
    DroppedContextItem,
    RepositoryDelta,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from .context_state import SessionContextState
from .memory import (
    MemoryQuery,
    MemoryRetriever,
    RetrievedMemory,
    render_memory,
)
from .repository_tracker import RepositoryTracker, render_repository_snapshot


_REPOSITORY_SECTION = re.compile(
    r"(?:^|\n\n)## Repository Context\n.*?(?=\n\n(?:## |当前日期：)|\Z)",
    re.DOTALL,
)
_STATIC_MEMORY_SECTION = re.compile(
    r"(?:^|\n\n)长期记忆（MEMORY）：\n.*?(?=\n\n(?:## |当前日期：)|\Z)",
    re.DOTALL,
)


@dataclass(frozen=True)
class ContextPolicy:
    safety_margin_tokens: int = 1024
    repository_ratio: float = 0.10
    active_files_ratio: float = 0.15
    recent_evidence_ratio: float = 0.17
    memory_ratio: float = 0.15
    history_ratio: float = 0.28
    task_ratio: float = 0.15
    minimum_input_budget: int = 1024

    def input_budget(self, request: ContextPreparationRequest) -> int:
        available = (
            request.model_context_window
            - request.model_max_output_tokens
            - self.safety_margin_tokens
        )
        hard_cap = max(
            128,
            request.model_context_window - request.model_max_output_tokens,
        )
        return min(max(self.minimum_input_budget, available), hard_cap)


class ContextCompiler:
    def __init__(
        self,
        *,
        workspace: str,
        state: SessionContextState,
        policy: ContextPolicy | None = None,
        memory_retriever: MemoryRetriever | None = None,
    ) -> None:
        self.state = state
        self.policy = policy or ContextPolicy()
        self.repository = RepositoryTracker(workspace)
        self.memory_retriever = memory_retriever

    def bind_memory_retriever(self, retriever: MemoryRetriever) -> None:
        self.memory_retriever = retriever

    def clone(self) -> "ContextCompiler":
        return ContextCompiler(
            workspace=str(self.repository.workspace),
            state=SessionContextState(workspace_dir=self.state.workspace_dir),
            policy=self.policy,
        )

    async def compile(
        self,
        context: AgentContext,
        request: ContextPreparationRequest,
    ) -> PreparedAgentContext:
        snapshot, delta = self.repository.refresh(self.state.last_repository_snapshot)
        self._apply_repository_delta(delta)
        self.state.last_repository_snapshot = snapshot

        for message in context.messages:
            if isinstance(message, ToolResultMessage):
                self.state.observe_tool_result(
                    message,
                    repository_fingerprint=snapshot.fingerprint,
                )

        stale_items = self.state.validate_sources(snapshot.fingerprint)
        if self.memory_retriever is not None:
            self.memory_retriever.validate_freshness()
            stale_items.extend(
                f"memory:{record.id}:{record.status}"
                for record in [
                    *self.memory_retriever.store.load_session(),
                    *self.memory_retriever.store.load_project(),
                ]
                if record.status == "stale"
            )
        total_budget = self.policy.input_budget(request)
        repository_budget = int(total_budget * self.policy.repository_ratio)
        active_budget = int(total_budget * self.policy.active_files_ratio)
        evidence_budget = int(total_budget * self.policy.recent_evidence_ratio)
        memory_budget = int(total_budget * self.policy.memory_ratio)
        history_budget = int(total_budget * self.policy.history_ratio)

        repository_text = _truncate_to_tokens(
            render_repository_snapshot(snapshot, delta),
            repository_budget,
        )
        active_items = self._active_file_items()
        selected_active, dropped_active = _select_items(
            "active_files",
            active_items,
            active_budget,
        )
        evidence_items = self._evidence_items()
        selected_evidence, dropped_evidence = _select_items(
            "recent_evidence",
            evidence_items,
            evidence_budget,
        )
        memory_items, retrieved_memories = self._memory_items(context)
        selected_memory, dropped_memory = _select_items(
            "memory",
            memory_items,
            memory_budget,
        )

        governance_text = _render_governance_context(
            selected_active,
            selected_evidence,
            selected_memory,
            stale_items,
        )
        system_prompt = _replace_repository_context(
            context.system_prompt,
            repository_text,
        )
        if governance_text:
            system_prompt = f"{system_prompt.rstrip()}\n\n{governance_text}"
        if context.current_task:
            system_prompt = _append_current_task(system_prompt, context.current_task)

        selected_messages, dropped_history = _select_message_suffix(
            context.messages,
            history_budget,
        )
        before_tokens = estimate_context_tokens(context.messages, context.system_prompt)
        after_tokens = estimate_context_tokens(selected_messages, system_prompt)

        section_reports = [
            ContextSectionReport(
                name="repository_state",
                budget_tokens=repository_budget,
                candidate_items=1,
                selected_items=1,
                estimated_tokens_before=_estimate_text_tokens(
                    render_repository_snapshot(snapshot, delta)
                ),
                estimated_tokens_after=_estimate_text_tokens(repository_text),
                reduction_policy="regenerate_short_snapshot",
            ),
            _item_section_report(
                "active_files",
                active_budget,
                active_items,
                selected_active,
                "drop_low_relevance",
            ),
            _item_section_report(
                "recent_evidence",
                evidence_budget,
                evidence_items,
                selected_evidence,
                "drop_stale_then_oldest",
            ),
            _item_section_report(
                "memory",
                memory_budget,
                memory_items,
                selected_memory,
                "retrieve_then_drop_low_score",
            ),
            ContextSectionReport(
                name="history",
                budget_tokens=history_budget,
                candidate_items=len(context.messages),
                selected_items=len(selected_messages),
                estimated_tokens_before=estimate_context_tokens(context.messages, ""),
                estimated_tokens_after=estimate_context_tokens(selected_messages, ""),
                reduction_policy="retain_recent_suffix",
            ),
            ContextSectionReport(
                name="current_request",
                budget_tokens=int(total_budget * self.policy.task_ratio),
                candidate_items=1 if _latest_user_message(context.messages) else 0,
                selected_items=1 if _latest_user_message(selected_messages) else 0,
                estimated_tokens_before=_latest_user_tokens(context.messages),
                estimated_tokens_after=_latest_user_tokens(selected_messages),
                reduction_policy="never_drop",
            ),
        ]
        report = ContextReport(
            context_id=f"ctx_{uuid.uuid4().hex}",
            repository_fingerprint=snapshot.fingerprint,
            total_budget_tokens=total_budget,
            estimated_tokens_before=before_tokens,
            estimated_tokens_after=after_tokens,
            sections=section_reports,
            stale_items=stale_items,
            dropped_items=[
                *dropped_active,
                *dropped_evidence,
                *dropped_memory,
                *dropped_history,
            ],
            repository_delta=delta,
            retrieved_memory_ids=[item.record.id for item in retrieved_memories],
            memory_retrieval_reasons={
                item.record.id: list(item.reasons)
                for item in retrieved_memories
            },
        )
        return PreparedAgentContext(
            system_prompt=system_prompt,
            messages=selected_messages,
            tools=list(context.tools),
            report=report,
        )

    def _apply_repository_delta(self, delta: RepositoryDelta) -> None:
        changed = [*delta.modified_paths, *delta.deleted_paths]
        if changed:
            self.state.invalidate_paths(changed)
        if delta.changed:
            self.state.invalidate_verification()

    def _active_file_items(self) -> list[ContextItem]:
        role_score = {
            "target": 100,
            "test": 80,
            "dependency": 70,
            "config": 60,
            "reference": 40,
        }
        items: list[ContextItem] = []
        for active in self.state.active_files.values():
            content = (
                f"{active.path} | role={active.role} | reason={active.reason} | "
                f"accesses={active.access_count} | "
                f"source_hash={(active.source_hash or 'unknown')[:12]}"
            )
            items.append(
                ContextItem(
                    id=f"active:{active.path}",
                    kind="active_file",
                    content=content,
                    source="session_context_state",
                    trust="derived",
                    priority=role_score.get(active.role, 30) + min(active.access_count, 10),
                    estimated_tokens=_estimate_text_tokens(content),
                    path=active.path,
                    source_hash=active.source_hash,
                    freshness="fresh" if active.source_hash else "unknown",
                )
            )
        return items

    def _evidence_items(self) -> list[ContextItem]:
        trust_score = {
            "observed": 100,
            "derived": 80,
            "user_given": 60,
            "model_claim": 20,
        }
        items: list[ContextItem] = []
        for index, evidence in enumerate(self.state.evidence):
            if evidence.freshness in {"stale", "missing"}:
                continue
            content = (
                f"[{evidence.trust}] {evidence.source}"
                f"{f' ({evidence.path})' if evidence.path else ''}: "
                f"{evidence.content}"
            )
            items.append(
                ContextItem(
                    id=f"evidence:{index}:{evidence.source}",
                    kind=evidence.kind,
                    content=content,
                    source=evidence.source,
                    trust=evidence.trust,  # type: ignore[arg-type]
                    priority=trust_score.get(evidence.trust, 10) + index,
                    estimated_tokens=_estimate_text_tokens(content),
                    path=evidence.path,
                    source_hash=evidence.source_hash,
                    freshness=evidence.freshness,  # type: ignore[arg-type]
                )
            )
        return items

    def _memory_items(
        self,
        context: AgentContext,
    ) -> tuple[list[ContextItem], list[RetrievedMemory]]:
        if self.memory_retriever is None:
            return [], []
        latest = _latest_user_message(context.messages)
        query_text = _user_message_text(latest) if latest is not None else ""
        retrieved = self.memory_retriever.retrieve(
            MemoryQuery(
                text=query_text,
                active_paths=list(self.state.active_files),
            )
        )
        items: list[ContextItem] = []
        pinned = self.memory_retriever.pinned_memory()
        if pinned:
            items.append(
                ContextItem(
                    id="memory:pinned",
                    kind="memory.project.pinned",
                    content=f"[Pinned project memory] {pinned}",
                    source=".codepilot/MEMORY.md",
                    trust="user_given",
                    priority=1000,
                    estimated_tokens=_estimate_text_tokens(pinned),
                    freshness="fresh",
                )
            )
        for item in retrieved:
            record = item.record
            path = record.related_paths[0] if record.related_paths else None
            source_hash = record.source_hashes.get(path) if path else None
            trust = "observed" if record.trust == "verified" else record.trust
            content = (
                f"{render_memory(record)} "
                f"[score={item.score}; reasons={', '.join(item.reasons)}]"
            )
            items.append(
                ContextItem(
                    id=f"memory:{record.id}",
                    kind=f"memory.{record.kind}",
                    content=content,
                    source=record.source,
                    trust=trust,  # type: ignore[arg-type]
                    priority=item.score,
                    estimated_tokens=_estimate_text_tokens(content),
                    path=path,
                    source_hash=source_hash,
                    freshness="fresh",
                )
            )
        return items, retrieved


def _select_items(
    section: str,
    items: list[ContextItem],
    budget: int,
) -> tuple[list[ContextItem], list[DroppedContextItem]]:
    selected: list[ContextItem] = []
    dropped: list[DroppedContextItem] = []
    used = 0
    for item in sorted(items, key=lambda candidate: candidate.priority, reverse=True):
        if item.freshness in {"stale", "missing"}:
            dropped.append(
                DroppedContextItem(item.id, section, "stale", item.source)
            )
            continue
        if selected and used + item.estimated_tokens > budget:
            dropped.append(
                DroppedContextItem(item.id, section, "over_budget", item.source)
            )
            continue
        if not selected and item.estimated_tokens > budget:
            item = replace(
                item,
                content=_truncate_to_tokens(item.content, budget),
                estimated_tokens=budget,
            )
        selected.append(item)
        used += item.estimated_tokens
    return selected, dropped


def _select_message_suffix(
    messages: list,
    budget: int,
) -> tuple[list, list[DroppedContextItem]]:
    if not messages:
        return [], []
    selected_reversed = []
    used = 0
    for message in reversed(messages):
        message_tokens = estimate_context_tokens([message], "")
        if selected_reversed and used + message_tokens > budget:
            break
        selected_reversed.append(message)
        used += message_tokens
    selected = list(reversed(selected_reversed))

    latest_user = _latest_user_message(messages)
    if latest_user is not None and latest_user not in selected:
        selected.append(latest_user)
        selected.sort(key=messages.index)

    selected_ids = {id(message) for message in selected}
    dropped = [
        DroppedContextItem(
            item_id=f"message:{index}",
            section="history",
            reason="over_budget",
            source=type(message).__name__,
        )
        for index, message in enumerate(messages)
        if id(message) not in selected_ids
    ]
    return selected, dropped


def _render_governance_context(
    active: list[ContextItem],
    evidence: list[ContextItem],
    memory: list[ContextItem],
    stale_items: list[str],
) -> str:
    lines = ["## Compiled Task Context"]
    if active:
        lines.append("### Active files")
        lines.extend(f"- {item.content}" for item in active)
    if evidence:
        lines.append("### Recent trusted evidence")
        lines.extend(f"- {item.content}" for item in evidence)
    if memory:
        lines.append("### Recalled memory")
        lines.extend(f"- {item.content}" for item in memory)
    if stale_items:
        lines.append("### Invalidated context")
        lines.extend(
            f"- [Derived] {item}; re-read or re-run before treating it as current."
            for item in stale_items[:20]
        )
    return "\n".join(lines) if len(lines) > 1 else ""


def _replace_repository_context(system_prompt: str, repository_text: str) -> str:
    cleaned = _STATIC_MEMORY_SECTION.sub("", system_prompt)
    cleaned = _REPOSITORY_SECTION.sub("", cleaned).strip()
    return f"{cleaned}\n\n{repository_text}".strip()


def _append_current_task(system_prompt: str, current_task: str) -> str:
    if "## Current Task" in system_prompt:
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n{current_task}".strip()


def _item_section_report(
    name: str,
    budget: int,
    candidates: list[ContextItem],
    selected: list[ContextItem],
    policy: str,
) -> ContextSectionReport:
    return ContextSectionReport(
        name=name,
        budget_tokens=budget,
        candidate_items=len(candidates),
        selected_items=len(selected),
        estimated_tokens_before=sum(item.estimated_tokens for item in candidates),
        estimated_tokens_after=sum(item.estimated_tokens for item in selected),
        reduction_policy=policy,
    )


def _latest_user_message(messages: list) -> UserMessage | None:
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            return message
    return None


def _latest_user_tokens(messages: list) -> int:
    latest = _latest_user_message(messages)
    return estimate_context_tokens([latest], "") if latest is not None else 0


def _user_message_text(message: UserMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.text
        for block in message.content
        if isinstance(block, TextContent)
    )


def _estimate_text_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _truncate_to_tokens(text: str, budget: int) -> str:
    max_chars = max(1, budget * 4)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...<context section truncated>..."


__all__ = ["ContextCompiler", "ContextPolicy"]
