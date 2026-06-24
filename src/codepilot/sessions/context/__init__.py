"""
动态会话上下文治理模块。

本模块负责管理 Agent 会话的上下文，包括上下文编译、压缩、
仓库追踪和状态管理等功能。它是 Agent 与 LLM 交互的核心桥梁。

主要功能:
    1. **上下文编译 (ContextCompiler)**
       - 将会话状态编译为 LLM 可理解的上下文格式
       - 管理上下文策略 (ContextPolicy)
       - 处理消息的格式转换和优化

    2. **上下文压缩 (ContextCompaction)**
       - 当上下文超出窗口限制时自动压缩
       - 保留最近消息，将旧消息摘要化
       - 支持 LLM 生成摘要或规则降级

    3. **仓库上下文 (RepositoryBootstrap)**
       - 引导 Agent 理解项目结构
       - 提供 Git 信息和文件摘要
       - 构建仓库级别的上下文

    4. **仓库追踪 (RepositoryTracker)**
       - 追踪仓库文件的变化
       - 检测文件的新鲜度
       - 提供快照比较功能

    5. **状态管理 (SessionContextState)**
       - 管理会话级别的上下文状态
       - 追踪活跃文件和文件摘要
       - 维护上下文证据链

模块结构:
    - compaction.py: 上下文压缩实现
    - compiler.py: 上下文编译器
    - repository_context.py: 仓库上下文引导
    - repository_tracker.py: 仓库变化追踪
    - state.py: 会话上下文状态

典型使用流程:
    # 创建上下文编译器
    compiler = ContextCompiler(policy=ContextPolicy(...))

    # 编译上下文
    context = compiler.compile(messages, tools)

    # 追踪仓库变化
    tracker = RepositoryTracker(workspace_dir)
    tracker.track_changes()

    # 压缩过长的上下文
    if needs_compaction:
        result = build_compacted_context(messages, summary)
"""

from __future__ import annotations

from .compaction import (
    COMPACTION_SYSTEM_PROMPT,
    ContextCompactionResult,
    build_compacted_context,
    fallback_summary,
    format_messages_for_summary,
)
from .compiler import ContextCompiler, ContextPolicy
from .repository_context import (
    GitInfo,
    RepositoryBootstrap,
    build_repository_bootstrap,
    render_repository_context,
)
from .repository_tracker import (
    RepositoryTracker,
    compare_snapshots,
    render_repository_snapshot,
)
from .state import ActiveFile, ContextEvidence, FileSummary, SessionContextState


__all__ = [
    # 状态管理
    "ActiveFile",           # 活跃文件信息
    "ContextEvidence",      # 上下文证据
    "FileSummary",          # 文件摘要
    "SessionContextState",  # 会话上下文状态

    # 上下文编译
    "ContextCompiler",      # 上下文编译器
    "ContextPolicy",        # 上下文策略

    # 上下文压缩
    "COMPACTION_SYSTEM_PROMPT",  # 压缩用系统提示词
    "ContextCompactionResult",   # 压缩结果
    "build_compacted_context",   # 构建压缩上下文
    "fallback_summary",          # 降级摘要
    "format_messages_for_summary",  # 格式化消息用于摘要

    # 仓库上下文
    "GitInfo",                   # Git 信息
    "RepositoryBootstrap",       # 仓库引导
    "build_repository_bootstrap",  # 构建仓库引导
    "render_repository_context",   # 渲染仓库上下文

    # 仓库追踪
    "RepositoryTracker",         # 仓库追踪器
    "compare_snapshots",         # 比较快照
    "render_repository_snapshot",  # 渲染仓库快照
]
