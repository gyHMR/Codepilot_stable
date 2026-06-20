from __future__ import annotations

"""会话级别的工作上下文状态。

本模块刻意不作为第二份聊天历史——它只存储紧凑的、与来源绑定的事实，
帮助运行时在每次模型调用前编译上下文。
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codepilot.protocols import RepositorySnapshot, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path


@dataclass
class ActiveFile:
    """活跃文件记录：跟踪会话中被读写过的文件。"""
    path: str                        # 文件路径
    role: str                        # 角色：target/test/dependency/config/reference
    reason: str                      # 记录原因
    source_hash: str | None = None   # 文件内容哈希
    access_count: int = 1            # 访问次数
    last_accessed_at: float = field(default_factory=time.time)


@dataclass
class FileSummary:
    """文件摘要：由 LLM 生成的文件内容摘要。"""
    path: str
    summary: str
    source_hash: str
    relevant_symbols: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    freshness: str = "fresh"         # 新鲜度：fresh/stale/missing


@dataclass
class ContextEvidence:
    """上下文证据：从工具结果、验证等来源收集的结构化事实。"""
    kind: str                        # 证据类型：tool_result/verification 等
    content: str                     # 证据内容
    trust: str                       # 信任级别：observed/derived/user_given/model_claim
    source: str                      # 来源（工具名等）
    source_hash: str | None = None
    workspace_fingerprint: str | None = None
    freshness: str = "unknown"       # 新鲜度
    path: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionContextState:
    """会话上下文状态：维护活跃文件、摘要、证据等运行时事实。"""
    workspace_dir: Path
    active_files: dict[str, ActiveFile] = field(default_factory=dict)
    file_summaries: dict[str, FileSummary] = field(default_factory=dict)
    evidence: list[ContextEvidence] = field(default_factory=list)
    last_repository_snapshot: RepositorySnapshot | None = None
    observed_tool_call_ids: set[str] = field(default_factory=set)

    def observe_tool_result(
        self,
        message: ToolResultMessage,
        *,
        repository_fingerprint: str | None = None,
    ) -> None:
        if message.tool_call_id and message.tool_call_id in self.observed_tool_call_ids:
            return
        if message.tool_call_id:
            self.observed_tool_call_ids.add(message.tool_call_id)
        details = message.details if isinstance(message.details, dict) else {}
        state = details.get("file_state")
        if not isinstance(state, dict):
            state = message.metadata.get("file_state")
        path = state.get("path") if isinstance(state, dict) else None
        source_hash = state.get("sha256") if isinstance(state, dict) else None

        paths = [str(item) for item in message.affected_paths]
        if isinstance(path, str) and path not in paths:
            paths.append(path)

        role = "target" if message.workspace_changed else "reference"
        for item in paths:
            self.touch_file(
                item,
                role=role,
                reason=f"{message.tool_name} tool result",
                source_hash=source_hash if item == path else None,
            )

        if message.workspace_changed:
            self.invalidate_paths(paths)
            self.invalidate_verification()

        text = _tool_result_text(message)
        if text:
            self.evidence.append(
                ContextEvidence(
                    kind="tool_result",
                    content=text,
                    trust="observed",
                    source=message.tool_name,
                    source_hash=source_hash if isinstance(source_hash, str) else None,
                    workspace_fingerprint=repository_fingerprint,
                    freshness="fresh",
                    path=path if isinstance(path, str) else None,
                )
            )
            self.evidence = self.evidence[-80:]

        if message.verification:
            self.evidence.append(
                ContextEvidence(
                    kind="verification",
                    content=str(message.verification),
                    trust="observed",
                    source=message.tool_name,
                    workspace_fingerprint=repository_fingerprint,
                    freshness="fresh",
                )
            )

    def touch_file(
        self,
        path: str,
        *,
        role: str,
        reason: str,
        source_hash: str | None = None,
    ) -> None:
        normalized = Path(path).as_posix()
        current = self.active_files.get(normalized)
        if current is None:
            self.active_files[normalized] = ActiveFile(
                path=normalized,
                role=role,
                reason=reason,
                source_hash=source_hash,
            )
            return
        current.access_count += 1
        current.last_accessed_at = time.time()
        current.reason = reason
        if role == "target" or current.role == "reference":
            current.role = role
        if source_hash:
            current.source_hash = source_hash

    def invalidate_paths(self, paths: list[str]) -> None:
        for path in paths:
            normalized = Path(path).as_posix()
            summary = self.file_summaries.get(normalized)
            if summary is not None:
                summary.freshness = "stale"
            for evidence in self.evidence:
                if evidence.path == normalized:
                    evidence.freshness = "stale"

    def invalidate_verification(self) -> None:
        for evidence in self.evidence:
            if evidence.kind == "verification":
                evidence.freshness = "stale"

    def validate_sources(self, repository_fingerprint: str) -> list[str]:
        stale: list[str] = []
        for path, summary in list(self.file_summaries.items()):
            state = file_state_for_path(self.workspace_dir, path)
            if not state.get("exists"):
                summary.freshness = "missing"
            elif state.get("sha256") != summary.source_hash:
                summary.freshness = "stale"
            else:
                summary.freshness = "fresh"
            if summary.freshness != "fresh":
                stale.append(f"file_summary:{path}:{summary.freshness}")

        for evidence in self.evidence:
            if (
                evidence.kind == "verification"
                and evidence.workspace_fingerprint
                and evidence.workspace_fingerprint != repository_fingerprint
            ):
                evidence.freshness = "stale"
            if evidence.freshness in {"stale", "missing"}:
                stale.append(f"evidence:{evidence.source}:{evidence.freshness}")
        return stale


def _tool_result_text(message: ToolResultMessage, *, limit: int = 1200) -> str:
    parts = [getattr(block, "text", "") for block in message.content]
    text = "".join(part for part in parts if part).strip()
    return text[:limit]


__all__ = [
    "ActiveFile",
    "ContextEvidence",
    "FileSummary",
    "SessionContextState",
]
