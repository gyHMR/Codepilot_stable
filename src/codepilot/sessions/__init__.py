"""
Codepilot 会话管理模块。

本包是 Codepilot Agent 的核心会话管理模块，负责以下主要功能：

1. **会话生命周期管理 (AgentSession)**
   - 管理长期运行的 Agent 会话
   - 处理会话的创建、恢复和持久化
   - 协调各个子模块的工作

2. **上下文编排 (ContextCompiler)**
   - 编译和组装发送给 LLM 的上下文内容
   - 管理上下文策略和裁剪策略
   - 处理仓库上下文的引导和追踪

3. **历史记录管理 (SessionCheckpoint)**
   - 管理会话的检查点和分支
   - 支持会话状态的回滚和恢复

4. **结构化记忆系统 (Memory*)**
   - MemoryStore: 记忆的持久化存储
   - MemoryWriter: 写入和更新记忆记录
   - MemoryRetriever: 检索相关记忆
   - MemoryQuery: 记忆查询接口
   - MemoryRecord/RetrievedMemory: 记忆数据结构

5. **持久化存储 (SessionStore, RunStore)**
   - SessionStore: 会话级别的持久化存储
   - RunStore: 运行级别的持久化存储
   - FreshnessResult: 存储新鲜度检查结果

典型使用流程:
    # 创建会话选项
    options = AgentSessionOptions(...)

    # 创建会话实例
    session = AgentSession(options)

    # 会话会自动管理上下文编译、记忆检索和持久化

模块导出:
    - AgentSession: 核心会话类
    - AgentSessionOptions: 会话配置选项
    - ConvertToLlmFn: 消息转换函数类型
    - ContextCompiler: 上下文编译器
    - ContextPolicy: 上下文策略
    - SessionCheckpoint: 会话检查点
    - SessionStore/FreshnessResult/RunStore: 持久化相关
    - RepositoryBootstrap/RepositoryTracker: 仓库上下文相关
    - new_session_id: 生成新会话 ID 的工具函数
    - load_global_memory/save_global_memory: 全局记忆的加载和保存
    - Memory*: 记忆系统相关类
"""

from .context import ContextCompiler, ContextPolicy, RepositoryBootstrap, RepositoryTracker
from .history import SessionCheckpoint
from .memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryRetriever,
    MemoryStore,
    MemoryWriter,
    RetrievedMemory,
    load_global_memory,
    save_global_memory,
)
from .persistence import FreshnessResult, RunStore, SessionStore, new_session_id
from .session import AgentSession
from .types import AgentSessionOptions, ConvertToLlmFn

__all__ = [
    # 核心会话类
    "AgentSession",
    # 配置和类型
    "AgentSessionOptions",
    "ConvertToLlmFn",
    # 上下文管理
    "ContextCompiler",
    "ContextPolicy",
    # 历史记录
    "SessionCheckpoint",
    # 持久化存储
    "SessionStore",
    "FreshnessResult",
    "RunStore",
    # 仓库上下文
    "RepositoryBootstrap",
    "RepositoryTracker",
    # 工具函数
    "new_session_id",
    "load_global_memory",
    "save_global_memory",
    # 记忆系统
    "MemoryQuery",
    "MemoryRecord",
    "MemoryRetriever",
    "MemoryStore",
    "MemoryWriter",
    "RetrievedMemory",
]
