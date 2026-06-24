from __future__ import annotations

"""检索相关结构化记忆，用于动态上下文编译。"""

import re
from pathlib import Path

from .files import load_global_memory
from .records import DURABLE_MEMORY_KINDS, MemoryQuery, MemoryRecord, RetrievedMemory
from .rendering import render_memory
from .store import MemoryStore
from .writer import MemoryWriter


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
        query_terms = _terms(query.text)
        active_paths = {Path(path).as_posix() for path in query.active_paths}
        ranked: list[RetrievedMemory] = []
        for record in records:
            if record.status != "active":
                continue
            if record.kind not in DURABLE_MEMORY_KINDS:
                continue
            if (
                record.kind == "experience"
                and record.content.get("maturity") == "candidate"
            ):
                continue
            score = 0
            reasons: list[str] = []
            related = active_paths.intersection(record.related_paths)
            if related:
                score += 40
                reasons.append(f"related_path:{sorted(related)[0]}")
            record_terms = _terms(render_memory(record))
            keyword_matches = sorted(query_terms.intersection(record_terms))
            if keyword_matches:
                score += min(30, len(keyword_matches) * 10)
                reasons.append(f"keyword:{keyword_matches[0]}")
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
            if score > 0:
                ranked.append(RetrievedMemory(record=record, score=score, reasons=reasons))

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


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w./-]{2,}", text, flags=re.UNICODE)
    }


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


__all__ = ["MemoryRetriever"]
