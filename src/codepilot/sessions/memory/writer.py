from __future__ import annotations

"""Admission and consolidation rules for durable memory."""

import uuid
from dataclasses import dataclass
from pathlib import Path

from codepilot.protocols import AgentRunResult, ToolResultMessage

from .experience import ExperienceExtractor, MemoryConsolidator, memory_key_for_text
from .files import sanitize_memory_text
from .records import MemoryRecord
from .store import MemoryStore


PROJECT_CONSTRAINT_KNOWLEDGE = (
    "Codepilot 是学生学习与求职展示项目；后续设计应优先保持清晰、"
    "可解释、可演示，避免生产级复杂平台化。"
)


@dataclass(frozen=True)
class MemoryAdmissionDecision:
    should_store: bool
    reason: str
    kind: str | None = None
    key: str | None = None
    text: str | None = None
    triggers: list[str] | None = None


def decide_prompt_memory_admission(text: str) -> MemoryAdmissionDecision:
    """Decide whether prompt text contains durable long-term knowledge."""

    safe_text = sanitize_memory_text(text, limit=1200)
    if not safe_text:
        return MemoryAdmissionDecision(False, "empty_after_sanitization")
    if _is_correction(safe_text):
        clean = _strip_memory_marker(safe_text)
        return MemoryAdmissionDecision(
            should_store=True,
            reason="user_correction",
            kind="correction",
            key=memory_key_for_text("constraint", clean),
            text=clean,
            triggers=_triggers_for_text(clean),
        )
    if _is_explicit_memory(safe_text):
        clean = _strip_memory_marker(safe_text)
        return MemoryAdmissionDecision(
            should_store=True,
            reason="explicit_memory_request",
            kind="constraint",
            key=memory_key_for_text("constraint", clean),
            text=clean,
            triggers=["always", *_triggers_for_text(clean)],
        )
    if _is_project_boundary_constraint(safe_text):
        return MemoryAdmissionDecision(
            should_store=True,
            reason="durable_project_constraint",
            kind="constraint",
            key="constraint:project_boundary",
            text=PROJECT_CONSTRAINT_KNOWLEDGE,
            triggers=["always", "topic:architecture"],
        )
    return MemoryAdmissionDecision(False, "ordinary_task_prompt")


class MemoryWriter:
    """Durable memory writer.

    The writer only admits long-term knowledge.  Task progress, file freshness,
    tool logs, and transient failures belong to task/context/run stores.
    """

    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def admit_prompt_memory(
        self,
        text: str,
        *,
        run_id: str | None = None,
    ) -> MemoryRecord | None:
        decision = decide_prompt_memory_admission(text)
        if not decision.should_store or not decision.kind or not decision.key or not decision.text:
            return None
        source = "user"
        evidence_refs = [f"run:{run_id}"] if run_id else []
        record = MemoryRecord(
            id=_new_memory_id(),
            scope="project",
            kind=decision.kind,  # type: ignore[arg-type]
            key=decision.key,
            text=decision.text,
            triggers=decision.triggers or [],
            evidence_refs=evidence_refs,
            source=source,
        )
        return MemoryConsolidator(self.store).upsert_project_record(record)

    def observe_tool_result(
        self,
        _message: ToolResultMessage,
        *,
        run_id: str | None = None,
    ) -> list[MemoryRecord]:
        _ = run_id
        return []

    def finalize_run(self, result: AgentRunResult) -> list[MemoryRecord]:
        extractor = ExperienceExtractor()
        consolidator = MemoryConsolidator(self.store)
        records: list[MemoryRecord] = []
        for candidate in extractor.extract(result):
            records.append(
                consolidator.upsert_experience(
                    candidate,
                    run_id=result.run_id,
                )
            )
        return records

    def add_project(self, text: str) -> MemoryRecord:
        content = sanitize_memory_text(text, limit=1600)
        if not content:
            raise ValueError("Memory content is empty after sensitive-data filtering")
        kind = "decision" if _looks_like_decision(content) else "constraint"
        record = MemoryRecord(
            id=_new_memory_id(),
            scope="project",
            kind=kind,  # type: ignore[arg-type]
            key=memory_key_for_text(kind, content),
            text=_strip_decision_marker(content),
            triggers=["always", *_triggers_for_text(content)] if kind == "constraint" else _triggers_for_text(content),
            source="command",
        )
        return MemoryConsolidator(self.store).upsert_project_record(record)

    def promote(self, memory_id: str) -> MemoryRecord:
        source = self.store.get(memory_id)
        if source is None:
            raise ValueError(f"Memory not found: {memory_id}")
        if source.status != "active":
            raise ValueError("Only active memory can be promoted")
        if source.scope != "session" or source.kind != "experience":
            raise ValueError("Only session experience memory can be promoted")
        promoted = MemoryRecord(
            id=_new_memory_id(),
            scope="project",
            kind="experience",
            key=source.key,
            text=source.text,
            triggers=list(source.triggers),
            related_paths=list(source.related_paths),
            evidence_refs=list(source.evidence_refs),
            source="promoted",
            supersedes=[source.id],
            occurrences=source.occurrences,
        )
        return MemoryConsolidator(self.store).upsert_project_record(promoted)


def _is_explicit_memory(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "请记住",
            "记住：",
            "记住:",
            "remember:",
            "remember that",
            "以后",
        )
    )


def _is_correction(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "纠正",
            "更正",
            "不是",
            "而是",
            "不要再",
            "以后不要",
            "actually",
            "correction",
        )
    )


def _is_project_boundary_constraint(text: str) -> bool:
    markers = ("学生", "求职", "学习", "生产级", "过度设计", "复杂设计")
    if not any(marker in text for marker in markers):
        return False
    if (
        "生产级" not in text
        and "过度设计" not in text
        and "复杂设计" not in text
    ):
        return False
    return any(
        marker in text
        for marker in (
            "不要",
            "不做",
            "不是生产级",
            "非生产级",
            "避免",
            "别",
            "无需",
            "不需要",
            "不按生产级",
        )
    )


def _looks_like_decision(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("decision:") or text.startswith("决策：") or text.startswith("决策:")


def _strip_decision_marker(text: str) -> str:
    for marker in ("decision:", "Decision:", "决策：", "决策:"):
        if text.startswith(marker):
            return text[len(marker):].strip()
    return text


def _strip_memory_marker(text: str) -> str:
    cleaned = text.strip()
    for marker in (
        "请记住：",
        "请记住:",
        "记住：",
        "记住:",
        "纠正一下：",
        "纠正一下:",
        "纠正：",
        "纠正:",
        "更正：",
        "更正:",
        "remember:",
        "correction:",
    ):
        if cleaned.lower().startswith(marker.lower()):
            return cleaned[len(marker):].strip()
    return cleaned


def _triggers_for_text(text: str) -> list[str]:
    triggers: list[str] = []
    lowered = text.lower()
    if "context" in lowered or "上下文" in text:
        triggers.append("topic:context")
    if "memory" in lowered or "记忆" in text:
        triggers.append("topic:memory")
    if "architecture" in lowered or "架构" in text or "设计" in text:
        triggers.append("topic:architecture")
    if "edit" in lowered:
        triggers.append("intent:edit_file")
    if "pytest" in lowered or "验证" in text or "测试" in text:
        triggers.append("intent:verify")
    return triggers


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


__all__ = [
    "MemoryAdmissionDecision",
    "MemoryWriter",
    "decide_prompt_memory_admission",
]
