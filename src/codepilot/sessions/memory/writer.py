from __future__ import annotations

"""从用户提示、工具结果和 Run 结果中写入结构化记忆。"""

import uuid
from pathlib import Path

from codepilot.protocols import AgentRunResult, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path

from .experience import ExperienceExtractor, MemoryConsolidator
from .files import sanitize_memory_text
from .records import MemoryRecord, MemoryStatus
from .store import MemoryStore


class MemoryWriter:
    """Durable memory writer.

    Task progress and transient file/tool evidence are handled by neighboring
    session/context modules. This writer only admits stable project knowledge
    and verified reusable experience.
    """

    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def remember_task(self, text: str, *, run_id: str | None = None) -> MemoryRecord | None:
        """Admit durable user-provided project knowledge from a prompt.

        Ordinary task progress is intentionally not stored as durable memory.
        Session task recovery is handled by ``TaskRecoveryStore``.
        """
        safe_text = sanitize_memory_text(text, limit=1200)
        return self._maybe_remember_project_constraint(safe_text, run_id=run_id)

    def observe_tool_result(
        self,
        message: ToolResultMessage,
        *,
        run_id: str | None = None,
    ) -> list[MemoryRecord]:
        """Observe a tool result without promoting transient evidence to memory."""
        created: list[MemoryRecord] = []
        if message.workspace_changed:
            self.invalidate_paths(message.affected_paths)
        return created

    def finalize_run(self, result: AgentRunResult) -> list[MemoryRecord]:
        """Extract verified reusable experience after a run finishes."""
        return self._extract_experience(result)

    def _extract_experience(self, result: AgentRunResult) -> list[MemoryRecord]:
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
        """添加项目级知识记忆（scope=project，持久化到 project.jsonl）。"""
        content = sanitize_memory_text(text, limit=1600)
        if not content:
            raise ValueError("Memory content is empty after sensitive-data filtering")
        record = MemoryRecord(
            id=_new_memory_id(),
            kind="project",
            scope="project",
            content={"knowledge": content, "pinned": False},
            source="user_command",
            trust="user_given",
        )
        return self.store.update(record)

    def _maybe_remember_project_constraint(
        self,
        text: str,
        *,
        run_id: str | None,
    ) -> MemoryRecord | None:
        markers = ("学生", "求职", "学习", "生产级", "过度设计", "复杂设计")
        if not any(marker in text for marker in markers):
            return None
        if "生产级" not in text and "过度设计" not in text and "复杂设计" not in text:
            return None
        if not _expresses_non_production_constraint(text):
            return None
        knowledge = (
            "Codepilot 是学生学习与求职展示项目；后续设计应优先保持清晰、"
            "可解释、可演示，避免生产级复杂平台化。"
        )
        for record in self.store.load_project():
            if (
                record.kind == "project"
                and record.content.get("category") == "project_constraint"
                and record.content.get("knowledge") == knowledge
            ):
                record.source_run_id = run_id
                return self.store.update(record)
        return self.store.update(
            MemoryRecord(
                id=_new_memory_id(),
                kind="project",
                scope="project",
                content={
                    "category": "project_constraint",
                    "knowledge": knowledge,
                },
                source="user_correction",
                source_run_id=run_id,
                trust="user_given",
            )
        )

    def promote(self, memory_id: str) -> MemoryRecord:
        """将 session 级 durable memory 提升为 project 级。"""
        source = self.store.get(memory_id)
        if source is None:
            raise ValueError(f"Memory not found: {memory_id}")
        if source.status != "active":
            raise ValueError("Only active memory can be promoted")
        promoted = MemoryRecord(
            id=_new_memory_id(),
            kind=source.kind,
            scope="project",
            content=dict(source.content),
            source=f"promoted:{source.id}",
            source_run_id=source.source_run_id,
            related_paths=list(source.related_paths),
            source_hashes=dict(source.source_hashes),
            trust=source.trust,
        )
        self.store.update(promoted)
        return promoted

    def invalidate_paths(self, paths: list[str]) -> list[MemoryRecord]:
        """Mark legacy path-bound memory stale when a tool changes the workspace."""
        normalized = {Path(path).as_posix() for path in paths}
        changed: list[MemoryRecord] = []
        for record in self.store.load_session():
            if (
                record.kind == "file"
                and record.status == "active"
                and normalized.intersection(record.related_paths)
            ):
                record.status = "stale"
                self.store.update(record)
                changed.append(record)
        return changed

    def validate_freshness(self) -> list[MemoryRecord]:
        """Validate legacy file memories so old stores remain safe to load."""
        changed: list[MemoryRecord] = []
        for record in [*self.store.load_session(), *self.store.load_project()]:
            if record.kind != "file" or record.status not in {"active", "stale"}:
                continue
            fresh = True
            for path, source_hash in record.source_hashes.items():
                state = file_state_for_path(self.workspace_dir, path)
                if not state.get("exists") or state.get("sha256") != source_hash:
                    fresh = False
                    break
            target_status: MemoryStatus = "active" if fresh else "stale"
            if record.status != target_status:
                record.status = target_status
                self.store.update(record)
                changed.append(record)
        return changed


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def _expresses_non_production_constraint(text: str) -> bool:
    negative_markers = (
        "不要",
        "不做",
        "不是生产级",
        "非生产级",
        "避免",
        "别",
        "无需",
        "不需要",
        "不按生产级",
        "不要生产级",
        "避免生产级",
        "避免过度设计",
    )
    return any(marker in text for marker in negative_markers)


__all__ = ["MemoryWriter"]
