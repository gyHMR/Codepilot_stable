"""
会话级别的工作上下文状态模块。

本模块管理 Agent 会话的运行时上下文状态，包括活跃文件、文件摘要、
上下文证据等。它刻意不存储聊天历史，只维护紧凑的、与来源绑定的事实，
帮助运行时在每次模型调用前编译上下文。

设计原则:
    1. **不重复存储聊天历史**: 消息历史由其他模块管理
    2. **紧凑存储**: 只存储必要的事实信息
    3. **来源绑定**: 每个状态都有明确的来源和信任级别
    4. **新鲜度追踪**: 自动检测状态是否过期

主要数据结构:
    - ActiveFile: 活跃文件记录，追踪会话中被读写过的文件
    - FileSummary: 文件摘要，由 LLM 生成的文件内容摘要
    - ContextEvidence: 上下文证据，从工具结果等来源收集的结构化事实
    - SessionContextState: 会话上下文状态，聚合以上所有状态

状态生命周期:
    1. 工具执行结果 → observe_tool_result() → 更新活跃文件和证据
    2. LLM 推理前 → validate_sources() → 检查新鲜度
    3. 上下文编译 → 使用状态信息构建上下文

新鲜度管理:
    - fresh: 状态有效，与磁盘/仓库一致
    - stale: 状态过期，需要更新
    - missing: 文件已被删除

证据信任级别:
    - observed: 直接观察到的（如工具返回结果）
    - derived: 从其他证据推导出的
    - user_given: 用户提供的
    - model_claim: 模型声称的（需要验证）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codepilot.protocols import RepositorySnapshot, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path


@dataclass
class ActiveFile:
    """
    活跃文件记录：跟踪会话中被读写过的文件。

    当 Agent 通过工具访问文件时，会创建或更新 ActiveFile 记录。
    这些记录用于:
    - 追踪哪些文件与当前任务相关
    - 检测文件是否被外部修改
    - 为上下文编译提供文件列表

    属性:
        path: 文件路径（POSIX 格式，如 "src/main.py"）
        role: 文件在任务中的角色
            - "target": 正在修改的目标文件
            - "test": 测试文件
            - "dependency": 依赖文件
            - "config": 配置文件
            - "reference": 只读参考文件
        reason: 记录该文件的原因（如 "read tool result"）
        source_hash: 文件内容的 SHA256 哈希（用于检测变化）
        access_count: 文件被访问的次数
        last_accessed_at: 最后访问时间戳

    示例:
        file = ActiveFile(
            path="src/main.py",
            role="target",
            reason="write tool result",
            source_hash="abc123...",
        )
    """

    # 文件路径（POSIX 格式）
    path: str

    # 文件角色：target/test/dependency/config/reference
    role: str

    # 记录原因
    reason: str

    # 文件内容哈希（SHA256），用于检测文件是否被修改
    source_hash: str | None = None

    # 访问次数，用于判断文件的重要性
    access_count: int = 1

    # 最后访问时间戳
    last_accessed_at: float = field(default_factory=time.time)


@dataclass
class FileSummary:
    """
    文件摘要：由 LLM 生成的文件内容摘要。

    当 Agent 读取文件时，可以生成文件摘要以供后续参考。
    摘要包含文件的关键信息，避免重复读取整个文件。

    属性:
        path: 文件路径（POSIX 格式）
        summary: 文件内容摘要文本
        source_hash: 生成摘要时的文件哈希
        relevant_symbols: 文件中相关的符号列表（如函数名、类名）
        created_at: 摘要创建时间戳
        freshness: 新鲜度状态
            - "fresh": 摘要与当前文件一致
            - "stale": 文件已被修改，摘要过期
            - "missing": 文件已被删除

    示例:
        summary = FileSummary(
            path="src/utils.py",
            summary="工具函数模块，包含日期处理、字符串操作等",
            source_hash="def456...",
            relevant_symbols=["format_date", "parse_string"],
        )
    """

    # 文件路径
    path: str

    # 摘要文本
    summary: str

    # 生成摘要时的文件哈希
    source_hash: str

    # 相关符号列表
    relevant_symbols: list[str] = field(default_factory=list)

    # 创建时间
    created_at: float = field(default_factory=time.time)

    # 新鲜度：fresh/stale/missing
    freshness: str = "fresh"


@dataclass
class ContextEvidence:
    """
    上下文证据：从工具结果、验证等来源收集的结构化事实。

    证据是 Agent 在执行过程中收集的结构化信息，用于:
    - 为 LLM 提供事实依据
    - 追踪信息的来源和可信度
    - 检测信息是否过期

    属性:
        kind: 证据类型
            - "tool_result": 工具执行结果
            - "verification": 验证结果
            - "observation": 观察结果
        content: 证据内容文本
        trust: 信任级别
            - "observed": 直接观察到的（最可信）
            - "derived": 从其他证据推导出的
            - "user_given": 用户提供的
            - "model_claim": 模型声称的（需要验证）
        source: 证据来源（如工具名 "read", "bash" 等）
        source_hash: 相关文件的哈希（可选）
        workspace_fingerprint: 工作区指纹（用于检测工作区变化）
        freshness: 新鲜度状态
            - "fresh": 证据仍然有效
            - "stale": 证据已过期
            - "unknown": 未知
        path: 相关文件路径（可选）
        created_at: 证据创建时间戳

    示例:
        evidence = ContextEvidence(
            kind="tool_result",
            content="文件已成功写入",
            trust="observed",
            source="write",
            path="src/main.py",
            freshness="fresh",
        )
    """

    # 证据类型
    kind: str

    # 证据内容
    content: str

    # 信任级别
    trust: str

    # 来源
    source: str

    # 相关文件哈希
    source_hash: str | None = None

    # 工作区指纹
    workspace_fingerprint: str | None = None

    # 新鲜度
    freshness: str = "unknown"

    # 相关文件路径
    path: str | None = None

    # 创建时间
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionContextState:
    """
    会话上下文状态：维护活跃文件、摘要、证据等运行时事实。

    这是会话级别的状态聚合器，管理 Agent 在会话过程中收集的所有
    上下文信息。它不存储聊天历史，只存储紧凑的事实信息。

    属性:
        workspace_dir: 工作区目录路径
        active_files: 活跃文件字典，键为文件路径
        file_summaries: 文件摘要字典，键为文件路径
        evidence: 上下文证据列表（最多保留 80 条）
        last_repository_snapshot: 最后一次仓库快照
        observed_tool_call_ids: 已观察的工具调用 ID 集合（用于去重）

    主要方法:
        observe_tool_result(): 观察工具结果，更新状态
        touch_file(): 记录文件访问
        invalidate_paths(): 使指定路径的摘要和证据失效
        invalidate_verification(): 使所有验证证据失效
        validate_sources(): 校验所有来源的新鲜度

    使用示例:
        state = SessionContextState(workspace_dir=Path("/path/to/project"))

        # 观察工具结果
        state.observe_tool_result(tool_result_message)

        # 校验新鲜度
        stale_items = state.validate_sources(repository_fingerprint)
    """

    # 工作区目录
    workspace_dir: Path

    # 活跃文件字典：路径 → ActiveFile
    active_files: dict[str, ActiveFile] = field(default_factory=dict)

    # 文件摘要字典：路径 → FileSummary
    file_summaries: dict[str, FileSummary] = field(default_factory=dict)

    # 上下文证据列表（最多 80 条，最新的在后面）
    evidence: list[ContextEvidence] = field(default_factory=list)

    # 最后一次仓库快照
    last_repository_snapshot: RepositorySnapshot | None = None

    # 已观察的工具调用 ID 集合（用于去重）
    observed_tool_call_ids: set[str] = field(default_factory=set)

    def observe_tool_result(
        self,
        message: ToolResultMessage,
        *,
        repository_fingerprint: str | None = None,
    ) -> None:
        """
        观察工具结果，更新活跃文件、证据和新鲜度状态。

        这是状态更新的核心方法，在每次工具执行完成后调用。
        它会提取工具结果中的有用信息，更新各种状态。

        处理流程:
            1. 去重检查：同一 tool_call_id 只处理一次
            2. 提取文件状态信息
            3. 更新活跃文件记录
            4. 如果工具修改了工作区：
               a. 使相关路径的摘要失效
               b. 使所有验证证据失效
            5. 提取工具结果文本作为证据
            6. 提取验证结果作为证据

        Args:
            message: 工具结果消息
                - tool_call_id: 工具调用 ID（用于去重）
                - tool_name: 工具名称
                - content: 结果内容
                - affected_paths: 受影响的文件路径列表
                - workspace_changed: 是否修改了工作区
                - verification: 验证结果（可选）
                - details: 详细信息（包含 file_state）
                - metadata: 元数据
            repository_fingerprint: 可选的仓库指纹，用于关联证据

        证据保留策略:
            - 最多保留 80 条证据
            - 超出时删除最旧的证据
        """

        # 去重检查：同一 tool_call_id 只处理一次
        if message.tool_call_id and message.tool_call_id in self.observed_tool_call_ids:
            return
        if message.tool_call_id:
            self.observed_tool_call_ids.add(message.tool_call_id)

        # 提取文件状态信息
        details = message.details if isinstance(message.details, dict) else {}
        state = details.get("file_state")
        if not isinstance(state, dict):
            state = message.metadata.get("file_state")
        path = state.get("path") if isinstance(state, dict) else None
        source_hash = state.get("sha256") if isinstance(state, dict) else None

        # 收集所有受影响的路径
        paths = [str(item) for item in message.affected_paths]
        if isinstance(path, str) and path not in paths:
            paths.append(path)

        # 确定文件角色：如果工作区被修改则为 target，否则为 reference
        role = "target" if message.workspace_changed else "reference"

        # 更新活跃文件
        for item in paths:
            self.touch_file(
                item,
                role=role,
                reason=f"{message.tool_name} tool result",
                source_hash=source_hash if item == path else None,
            )

        # 如果工具修改了工作区，使相关状态失效
        if message.workspace_changed:
            self.invalidate_paths(paths)
            self.invalidate_verification()

        # 提取工具结果文本作为证据
        text = _tool_result_text(message)
        if text:
            self.evidence.append(
                ContextEvidence(
                    kind="tool_result",
                    content=text,
                    trust="observed",  # 工具结果是直接观察到的
                    source=message.tool_name,
                    source_hash=source_hash if isinstance(source_hash, str) else None,
                    workspace_fingerprint=repository_fingerprint,
                    freshness="fresh",
                    path=path if isinstance(path, str) else None,
                )
            )
            # 保留最近 80 条证据
            self.evidence = self.evidence[-80:]

        # 提取验证结果作为证据
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
        """
        记录文件访问（新建或更新 active_files 条目）。

        Args:
            path: 文件路径
            role: 文件角色（target/test/dependency/config/reference）
            reason: 记录原因
            source_hash: 可选的文件哈希

        处理逻辑:
            - 新文件：创建 ActiveFile 记录
            - 已有文件：
              a. 递增 access_count
              b. 更新 last_accessed_at
              c. 更新 reason
              d. role 升级规则：reference → target（target 优先级更高）
              e. 更新 source_hash（如果提供）
        """

        # 规范化路径为 POSIX 格式
        normalized = Path(path).as_posix()
        current = self.active_files.get(normalized)

        # 新文件：创建记录
        if current is None:
            self.active_files[normalized] = ActiveFile(
                path=normalized,
                role=role,
                reason=reason,
                source_hash=source_hash,
            )
            return

        # 已有文件：更新记录
        current.access_count += 1
        current.last_accessed_at = time.time()
        current.reason = reason

        # role 升级规则：target 优先级高于 reference
        if role == "target" or current.role == "reference":
            current.role = role

        # 更新哈希（如果提供）
        if source_hash:
            current.source_hash = source_hash

    def invalidate_paths(self, paths: list[str]) -> None:
        """
        使指定路径的摘要和证据失效。

        当文件被修改时调用，标记相关状态为 "stale"。

        Args:
            paths: 要失效的文件路径列表
        """

        for path in paths:
            normalized = Path(path).as_posix()

            # 使文件摘要失效
            summary = self.file_summaries.get(normalized)
            if summary is not None:
                summary.freshness = "stale"

            # 使相关证据失效
            for evidence in self.evidence:
                if evidence.path == normalized:
                    evidence.freshness = "stale"

    def invalidate_verification(self) -> None:
        """
        使所有验证证据失效。

        当工作区被修改时调用，因为验证结果可能已过期。
        """

        for evidence in self.evidence:
            if evidence.kind == "verification":
                evidence.freshness = "stale"

    def validate_sources(self, repository_fingerprint: str) -> list[str]:
        """
        校验所有来源的新鲜度，返回过时条目列表。

        检查内容:
            1. file_summaries：对比 SHA256 与磁盘文件
            2. verification 证据：对比 workspace_fingerprint 与当前仓库指纹

        Args:
            repository_fingerprint: 当前仓库的指纹

        Returns:
            过时条目列表，格式为 ["file_summary:path:stale", "evidence:source:missing", ...]

        使用示例:
            stale = state.validate_sources(current_fingerprint)
            if stale:
                print(f"发现 {len(stale)} 个过时条目")
        """

        stale: list[str] = []

        # 检查文件摘要
        for path, summary in list(self.file_summaries.items()):
            # 获取文件当前状态
            state = file_state_for_path(self.workspace_dir, path)

            if not state.get("exists"):
                # 文件已删除
                summary.freshness = "missing"
            elif state.get("sha256") != summary.source_hash:
                # 文件哈希不匹配，已被修改
                summary.freshness = "stale"
            else:
                # 文件未变化
                summary.freshness = "fresh"

            # 记录过时条目
            if summary.freshness != "fresh":
                stale.append(f"file_summary:{path}:{summary.freshness}")

        # 检查验证证据
        for evidence in self.evidence:
            # 检查工作区指纹是否匹配
            if (
                evidence.kind == "verification"
                and evidence.workspace_fingerprint
                and evidence.workspace_fingerprint != repository_fingerprint
            ):
                evidence.freshness = "stale"

            # 记录过时条目
            if evidence.freshness in {"stale", "missing"}:
                stale.append(f"evidence:{evidence.source}:{evidence.freshness}")

        return stale


def _tool_result_text(message: ToolResultMessage, *, limit: int = 1200) -> str:
    """
    提取工具结果的文本内容。

    Args:
        message: 工具结果消息
        limit: 文本长度限制（默认 1200 字符）

    Returns:
        工具结果的文本内容（截断到指定长度）
    """

    # 提取所有文本块
    parts = [getattr(block, "text", "") for block in message.content]
    # 拼接并截断
    text = "".join(part for part in parts if part).strip()
    return text[:limit]


__all__ = [
    "ActiveFile",           # 活跃文件记录
    "ContextEvidence",      # 上下文证据
    "FileSummary",          # 文件摘要
    "SessionContextState",  # 会话上下文状态
]
