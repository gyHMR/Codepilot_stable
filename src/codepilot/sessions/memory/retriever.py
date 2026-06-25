from __future__ import annotations

"""检索相关结构化记忆，用于动态上下文编译。"""

import re
from pathlib import Path

from .files import load_global_memory
from .records import MemoryQuery, MemoryRecord, RetrievedMemory
from .rendering import render_memory
from .store import MemoryStore
from .writer import MemoryWriter

PROJECT_ALWAYS_RECALL_CATEGORIES = frozenset({"project_constraint", "user_preference"})


class MemoryRetriever:
    """记忆检索器：根据查询文本和活跃文件路径检索相关记忆并评分排序。"""

    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]:
        """检索与查询相关的记忆，按评分排序返回。

        评分规则（各项累加）：
        - 关联路径匹配: +40（与当前活跃文件相关）
        - 关键词匹配: +10/词，最高 +30
        - 信任度 verified/observed: +20
        - project 作用域: +10
        - 信任度 model_claim: -20（模型自称的低可信度）

        过滤规则：
        - 跳过 status != "active" 的记录
        - 只检索 durable memory: project / decision / experience
        - 跳过 legacy state: task / file / failure

        最终按 kind 限制数量（experience:2, decision:2, project:3）。
        """
        records = [*self.store.load_session(), *self.store.load_project()]
        ranked = [
            scored
            for record in records
            if (scored := score_memory_record(record, query)) is not None
        ]

        ranked.sort(
            key=lambda item: (item.score, item.record.updated_at),
            reverse=True,
        )
        return _apply_kind_limits(ranked, query.limit)

    def validate_freshness(self) -> list[MemoryRecord]:
        """校验记忆新鲜度（委托给 MemoryWriter.validate_freshness）。"""
        return MemoryWriter(
            store=self.store,
            workspace_dir=self.workspace_dir,
        ).validate_freshness()

    def pinned_memory(self) -> str:
        """加载用户固定的项目记忆（.codepilot/MEMORY.md）。"""
        return load_global_memory(self.workspace_dir)


def score_memory_record(
    record: MemoryRecord,
    query: MemoryQuery,
) -> RetrievedMemory | None:
    """为单条记忆计算与查询的相关性；不相关或不可检索时返回 None。"""

    if record.retrieval_exclusion_reason() is not None:
        return None

    query_terms = _terms(query.text)
    active_paths = {Path(path).as_posix() for path in query.active_paths}
    related_paths = {Path(path).as_posix() for path in record.related_paths}
    score = 0
    reasons: list[str] = []

    related = active_paths.intersection(related_paths)
    if related:
        score += 40
        reasons.append(f"related_path:{sorted(related)[0]}")

    keyword_matches = sorted(_keyword_matches(query_terms, _terms(render_memory(record))))
    if keyword_matches:
        score += min(30, len(keyword_matches) * 10)
        reasons.append(f"keyword:{keyword_matches[0]}")

    if record.kind == "project":
        gate_reason = _project_retrieval_gate(record, query, related, keyword_matches)
        if gate_reason is None:
            return None
        reasons.append(gate_reason)

    if record.trust in {"verified", "observed"}:
        score += 20
        reasons.append(f"trust:{record.trust}")
    if record.scope == "project":
        score += 10
        reasons.append("project_memory")

    if query.retrieval_mode == "qa" and record.kind in {"decision", "project"}:
        score += 20
        reasons.append(f"mode:qa_{record.kind}_memory")
    elif query.retrieval_mode == "repair" and record.kind in {"experience", "failure"}:
        score += 15
        reasons.append(f"mode:repair_{record.kind}_memory")
    elif query.retrieval_mode == "verify" and record.kind == "experience":
        score += 10
        reasons.append("mode:verify_experience_memory")

    if record.kind == "experience":
        applies = {
            str(item)
            for item in record.content.get("applies_when", [])
            if isinstance(item, str)
        }
        if query.task_phase and f"phase:{query.task_phase}" in applies:
            score += 25
            reasons.append(f"phase:{query.task_phase}")
        if query.action_intent and f"intent:{query.action_intent}" in applies:
            score += 20
            reasons.append(f"intent:{query.action_intent}")
        if query.recent_error and f"error:{query.recent_error}" in applies:
            score += 30
            reasons.append(f"error:{query.recent_error}")
        maturity = str(record.content.get("maturity", "active"))
        if maturity == "verified":
            score += 20
            reasons.append("maturity:verified")
        elif maturity == "candidate":
            score -= 30
            reasons.append("maturity:candidate")

    if record.trust == "model_claim":
        score -= 20
        reasons.append("model_claim_penalty")

    if score <= 0:
        return None
    return RetrievedMemory(record=record, score=score, reasons=reasons)


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w./-]{2,}", text, flags=re.UNICODE)
    }


def _keyword_matches(query_terms: set[str], record_terms: set[str]) -> set[str]:
    """匹配查询词和记录词，兼容中文短语的包含关系。"""

    matches = set(query_terms.intersection(record_terms))
    for query_term in query_terms:
        for record_term in record_terms:
            if query_term == record_term:
                continue
            if query_term in record_term or record_term in query_term:
                matches.add(record_term)
    return matches


def _project_retrieval_gate(
    record: MemoryRecord,
    query: MemoryQuery,
    related_paths: set[str],
    keyword_matches: list[str],
) -> str | None:
    """Require an explicit reason before project memory may enter context."""

    category = str(record.content.get("category", ""))
    if record.content.get("always_recall") is True:
        return "project_gate:always_recall"
    if category in PROJECT_ALWAYS_RECALL_CATEGORIES:
        return f"project_gate:{category}"
    if related_paths:
        return "project_gate:related_path"
    if keyword_matches:
        return "project_gate:keyword"
    if query.retrieval_mode == "qa" and category in PROJECT_ALWAYS_RECALL_CATEGORIES:
        return f"project_gate:qa_{category}"
    return None


def _apply_kind_limits(
    ranked: list[RetrievedMemory],
    total_limit: int,
) -> list[RetrievedMemory]:
    limits = {
        "experience": 2,
        "decision": 2,
        "project": 3,
    }
    counts: dict[str, int] = {}
    selected: list[RetrievedMemory] = []
    for item in ranked:
        kind = item.record.kind
        if counts.get(kind, 0) >= limits.get(kind, total_limit):
            continue
        selected.append(item)
        counts[kind] = counts.get(kind, 0) + 1
        if len(selected) >= total_limit:
            break
    return selected


__all__ = ["MemoryRetriever", "score_memory_record"]
