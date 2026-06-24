from __future__ import annotations

"""轻量经验提炼与合并。

第一版只从确定性闭环中提炼经验：工具失败、后续成功、并且验证通过。
不让 LLM 自由总结，避免把无证据推测写入长期记忆。
"""

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from codepilot.protocols import AgentRunResult, ToolResultMessage

from .records import MemoryRecord, utc_now_iso
from .store import MemoryStore


@dataclass(frozen=True)
class ExperienceCandidate:
    content: dict[str, Any]
    related_paths: list[str]
    trust: str


class ExperienceExtractor:
    """从一次 run 的消息中提炼可复用经验候选。"""

    def extract(self, result: AgentRunResult) -> list[ExperienceCandidate]:
        messages = [
            message for message in result.messages
            if isinstance(message, ToolResultMessage)
        ]
        candidates: list[ExperienceCandidate] = []
        candidates.extend(self._extract_edit_repair(result, messages))
        candidates.extend(self._extract_verification_repair(result, messages))
        return candidates

    def _extract_edit_repair(
        self,
        result: AgentRunResult,
        messages: list[ToolResultMessage],
    ) -> list[ExperienceCandidate]:
        failed_match = next(
            (
                (index, message) for index, message in enumerate(messages)
                if message.tool_name == "edit"
                and message.is_error
                and message.error_code in {"multiple_matches", "unexpected_match_count"}
            ),
            None,
        )
        if failed_match is None:
            return []
        failed_index, failed = failed_match
        succeeded_match = next(
            (
                (index, message)
                for index, message in enumerate(
                    messages[failed_index + 1:],
                    failed_index + 1,
                )
                if message.tool_name == "edit"
                and not message.is_error
                and message.status == "success"
            ),
            None,
        )
        if succeeded_match is None:
            return []
        succeeded_index, succeeded = succeeded_match
        passed_verifications = _passed_verifications_after(messages, succeeded_index)
        if not passed_verifications:
            return []
        paths = _paths_from([failed, succeeded])
        content = {
            "lesson_type": "tool_usage",
            "situation": "edit 工具因为 old_text 不唯一或匹配数不符合预期而失败",
            "failed_attempt": "直接使用不够唯一的 old_text 调用 edit",
            "failure_signal": failed.error_code,
            "better_action": "先 read 目标区域，再使用更长且唯一的 old_text 或 occurrence_index 进行编辑",
            "applies_when": [
                "phase:repair",
                "intent:edit_file",
                f"error:{failed.error_code}",
            ],
            "avoid_when": [],
            "evidence_refs": _evidence_refs([failed, succeeded, *passed_verifications]),
            "maturity": "verified" if result.status == "completed" else "active",
            "occurrence_count": 1,
            "last_seen_at": utc_now_iso(),
        }
        content["fingerprint"] = _fingerprint(content, paths)
        return [ExperienceCandidate(content=content, related_paths=paths, trust="verified")]

    def _extract_verification_repair(
        self,
        result: AgentRunResult,
        messages: list[ToolResultMessage],
    ) -> list[ExperienceCandidate]:
        failed_match = next(
            (
                (index, message) for index, message in enumerate(messages)
                if isinstance(message.verification, dict)
                and message.verification.get("status") == "failed"
            ),
            None,
        )
        if failed_match is None:
            return []
        failed_index, failed = failed_match
        passed_verifications = _passed_verifications_after(messages, failed_index)
        if not passed_verifications:
            return []
        content = {
            "lesson_type": "verification_repair",
            "situation": "修改后验证失败，需要基于失败日志修复",
            "failed_attempt": "第一次修改没有满足验证期望",
            "failure_signal": "verification_failed",
            "better_action": "先读取失败摘要和相关文件，再做最小修复并重新运行同一验证命令",
            "applies_when": [
                "phase:repair",
                "intent:debug_failure",
                "error:verification_failed",
            ],
            "avoid_when": [],
            "evidence_refs": _evidence_refs([failed, *passed_verifications]),
            "maturity": "verified" if result.status == "completed" else "active",
            "occurrence_count": 1,
            "last_seen_at": utc_now_iso(),
        }
        content["fingerprint"] = _fingerprint(content, [])
        return [ExperienceCandidate(content=content, related_paths=[], trust="verified")]


class MemoryConsolidator:
    """将经验候选写入记忆，并按 fingerprint 合并重复经验。"""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def upsert_experience(
        self,
        candidate: ExperienceCandidate,
        *,
        run_id: str | None,
    ) -> MemoryRecord:
        fingerprint = str(candidate.content.get("fingerprint", ""))
        for record in self.store.load_session():
            if (
                record.kind == "experience"
                and record.content.get("fingerprint") == fingerprint
            ):
                record.content["occurrence_count"] = int(
                    record.content.get("occurrence_count", 1)
                ) + 1
                record.content["last_seen_at"] = utc_now_iso()
                evidence = record.content.setdefault("evidence_refs", [])
                if isinstance(evidence, list):
                    for item in candidate.content.get("evidence_refs", []):
                        if item not in evidence:
                            evidence.append(item)
                if candidate.content.get("maturity") == "verified":
                    record.content["maturity"] = "verified"
                    record.trust = "verified"
                record.source_run_id = run_id
                return self.store.update(record)
        return self.store.update(
            MemoryRecord(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                kind="experience",
                scope="session",
                content=dict(candidate.content),
                source="experience_extractor",
                source_run_id=run_id,
                related_paths=list(candidate.related_paths),
                trust="verified" if candidate.trust == "verified" else "observed",
            )
        )


def _passed_verifications_after(
    messages: list[ToolResultMessage],
    index: int,
) -> list[ToolResultMessage]:
    return [
        message for message in messages[index + 1:]
        if isinstance(message.verification, dict)
        and message.verification.get("status") == "passed"
    ]


def _paths_from(messages: list[ToolResultMessage]) -> list[str]:
    paths: list[str] = []
    for message in messages:
        for path in message.affected_paths:
            if path not in paths:
                paths.append(path)
    return paths


def _evidence_refs(messages: list[ToolResultMessage]) -> list[str]:
    refs: list[str] = []
    for message in messages:
        if message.tool_call_id:
            refs.append(f"tool:{message.tool_call_id}")
            if isinstance(message.verification, dict):
                refs.append(f"verification:{message.tool_call_id}")
    return refs


def _fingerprint(content: dict[str, Any], paths: list[str]) -> str:
    raw = "|".join(
        [
            str(content.get("lesson_type", "")),
            str(content.get("failure_signal", "")),
            str(content.get("better_action", "")),
            ",".join(str(item) for item in content.get("applies_when", [])),
            ",".join(paths),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


__all__ = ["ExperienceExtractor", "MemoryConsolidator"]
