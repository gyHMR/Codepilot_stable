from __future__ import annotations

"""每次 LLM 调用前编译新鲜的、带预算的模型上下文。

本模块属于 sessions 层，因为它使用长期会话状态、记忆、证据、
仓库新鲜度和当前 Agent Run 的消息来准备下一次模型输入。
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
from ..memory import (
    MemoryQuery,
    MemoryRetriever,
    RetrievedMemory,
    render_memory,
)
from .state import SessionContextState
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
    """上下文预算策略：控制各段落的 token 分配比例。"""
    safety_margin_tokens: int = 1024      # 安全余量 token 数
    repository_ratio: float = 0.10        # 仓库上下文占比
    active_files_ratio: float = 0.15      # 活跃文件占比
    recent_evidence_ratio: float = 0.17   # 近期证据占比
    memory_ratio: float = 0.15            # 记忆占比
    history_ratio: float = 0.28           # 历史消息占比
    task_ratio: float = 0.15              # 当前任务占比
    minimum_input_budget: int = 1024      # 最小输入预算

    def input_budget(self, request: ContextPreparationRequest) -> int:
        """计算可用的输入 token 预算。"""
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
    """上下文编译器：在每次 LLM 调用前将仓库、文件、证据、记忆编译为带预算的上下文。"""

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
        """绑定记忆检索器（Session 初始化后调用）。"""
        self.memory_retriever = retriever

    def clone(self) -> "ContextCompiler":
        """克隆编译器（用于 Session 分支时创建独立副本）。"""
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
        """编译上下文：在每次 LLM 调用前，将所有上下文源编译为带预算的输入。

        完整流程：
        1. 刷新仓库快照，计算与上次的差异（delta）
        2. 观察消息中的工具结果，更新活跃文件和证据
        3. 校验所有来源的新鲜度（文件哈希、记忆状态、验证结果）
        4. 按 ContextPolicy 的比例分配 token 预算到各段落
        5. 为每个段落选择条目（按优先级排序，超预算则丢弃）
        6. 组装 system_prompt（仓库上下文 + 治理上下文 + 当前任务）
        7. 选择历史消息后缀（保留最近的消息，丢弃较早的）
        8. 生成 ContextReport（含裁剪统计和丢弃记录）

        返回 PreparedAgentContext（system_prompt + messages + tools + report）。
        """
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
        """将仓库差异应用到会话状态：使变更路径的摘要和证据失效。"""
        changed = [*delta.modified_paths, *delta.deleted_paths]
        if changed:
            self.state.invalidate_paths(changed)
        if delta.changed:
            self.state.invalidate_verification()

    def _active_file_items(self) -> list[ContextItem]:
        """将活跃文件转换为上下文条目列表（按角色和访问次数评分）。

        角色优先级：target(100) > test(80) > dependency(70) > config(60) > reference(40)
        访问次数作为加分项（最多 +10）。
        """
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
        """将上下文证据转换为上下文条目列表（按信任度评分，跳过过时证据）。

        信任度优先级：observed(100) > derived(80) > user_given(60) > model_claim(20)
        """
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
        """检索相关记忆并转换为上下文条目列表。

        包含两部分：
        1. 固定记忆（pinned_memory）：用户手动维护的 MEMORY.md，优先级 1000
        2. 检索到的记忆：根据当前用户消息和活跃文件路径查询，按评分排序

        返回值：(上下文条目列表, 检索到的记忆列表用于 report)
        """
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
    """按优先级选择条目，直到预算用尽。

    选择策略：
    1. 按 priority 降序排列
    2. 跳过 freshness 为 stale/missing 的条目
    3. 如果累计 token 超出预算，丢弃后续条目
    4. 如果第一个条目就超预算，截断其内容

    返回：(选中的条目, 被丢弃的条目记录)
    """
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
    """从消息列表末尾选择后缀，保留最近的消息直到预算用尽。

    策略：
    1. 从末尾向前遍历，累加 token 直到超出预算
    2. 确保最新的用户消息一定被保留（即使超预算也会追加）
    3. 被丢弃的消息记录在 dropped 列表中

    返回：(选中的消息列表, 被丢弃的消息记录)
    """
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
