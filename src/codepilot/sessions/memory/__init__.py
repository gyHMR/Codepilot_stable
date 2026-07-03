from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：sessions 层是会话事实源，负责消息、run、记忆、上下文投影和任务恢复。

"""结构化会话记忆和项目记忆。"""

from .experience import ExperienceExtractor, MemoryConsolidator
from .files import load_global_memory, sanitize_memory_text, save_global_memory
from .records import (
    MEMORY_SCHEMA_VERSION,
    MemoryKind,
    MemoryQuery,
    MemoryRecall,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryStatus,
    RetrievedMemory,
)
from .rendering import render_memory
from .retriever import MemoryRetriever
from .store import MemoryStore
from .writer import MemoryAdmissionDecision, MemoryWriter, decide_prompt_memory_admission


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "ExperienceExtractor",
    "MemoryKind",
    "MemoryAdmissionDecision",
    "MemoryConsolidator",
    "MemoryQuery",
    "MemoryRecall",
    "MemoryRecord",
    "MemoryRetriever",
    "MemoryScope",
    "MemorySource",
    "MemoryStatus",
    "MemoryStore",
    "MemoryWriter",
    "decide_prompt_memory_admission",
    "RetrievedMemory",
    "load_global_memory",
    "render_memory",
    "sanitize_memory_text",
    "save_global_memory",
]
