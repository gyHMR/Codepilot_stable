from __future__ import annotations

"""跨层共享的上下文治理数据契约。"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

# 上下文新鲜度：新鲜/过时/已丢失/未知
ContextFreshness = Literal["fresh", "stale", "missing", "unknown"]
# 上下文信任度：观察到/推导得出/用户给出/模型声称
ContextTrust = Literal["observed", "derived", "user_given", "model_claim"]
# 上下文条目被丢弃的原因
DroppedContextReason = Literal[
    "duplicate",       # 重复
    "stale",           # 过时
    "low_relevance",   # 相关性低
    "over_budget",     # 超出预算
    "superseded",      # 被替代
    "missing_source",  # 来源丢失
]
_CONTEXT_FRESHNESS_VALUES = frozenset({"fresh", "stale", "missing", "unknown"})
_CONTEXT_TRUST_VALUES = frozenset(
    {"observed", "derived", "user_given", "model_claim"}
)
_DROPPED_CONTEXT_REASONS = frozenset(
    {
        "duplicate",
        "stale",
        "low_relevance",
        "over_budget",
        "superseded",
        "missing_source",
    }
)


@dataclass(frozen=True)
class RepositorySnapshot:
    """仓库快照：记录工作区在某一时刻的结构化状态。"""
    workspace_root: str                                  # 工作区根目录
    project_type: str | None                             # 项目类型（Python/JS 等）
    manifest_files: list[str]                            # 清单文件列表
    top_level_entries: list[str]                         # 顶层目录/文件列表
    test_directories: list[str]                          # 测试目录列表
    instruction_files: list[str]                         # 指令文件列表（CLAUDE.md 等）
    branch: str | None                                   # Git 分支名
    head_sha: str | None                                 # HEAD commit SHA
    git_status: list[str]                                # git status 输出行
    fingerprint: str                                     # 整体指纹（SHA256）
    instruction_hashes: dict[str, str] = field(default_factory=dict)  # 指令文件哈希
    dirty_path_hashes: dict[str, str] = field(default_factory=dict)   # 脏文件哈希


@dataclass(frozen=True)
class RepositoryDelta:
    """仓库快照差异：两次快照之间的变化。"""
    added_paths: list[str] = field(default_factory=list)     # 新增路径
    modified_paths: list[str] = field(default_factory=list)  # 修改路径
    deleted_paths: list[str] = field(default_factory=list)   # 删除路径
    branch_changed: bool = False    # 分支是否变更
    head_changed: bool = False      # HEAD 是否变更
    instructions_changed: bool = False  # 指令文件是否变更

    @property
    def changed(self) -> bool:
        return bool(
            self.added_paths
            or self.modified_paths
            or self.deleted_paths
            or self.branch_changed
            or self.head_changed
            or self.instructions_changed
        )


@dataclass
class ContextItem:
    """上下文条目：一个可被选择或丢弃的上下文片段。"""
    id: str                        # 条目唯一标识
    kind: str                      # 条目类型（active_file/evidence/memory 等）
    content: str                   # 条目内容文本
    source: str                    # 来源
    trust: ContextTrust            # 信任度
    priority: int                  # 优先级（越高越不容易被丢弃）
    estimated_tokens: int          # 估算 token 数
    path: str | None = None        # 关联文件路径
    source_hash: str | None = None # 来源文件哈希
    freshness: ContextFreshness = "unknown"  # 新鲜度

    def __post_init__(self) -> None:
        _ensure_context_trust(self.trust)
        _ensure_context_freshness(self.freshness)


@dataclass(frozen=True)
class ContextSectionReport:
    """上下文段落报告：记录一个段落的裁剪结果。"""
    name: str                      # 段落名称
    budget_tokens: int             # token 预算
    candidate_items: int           # 候选条目数
    selected_items: int            # 最终选中条目数
    estimated_tokens_before: int   # 裁剪前估算 token 数
    estimated_tokens_after: int    # 裁剪后估算 token 数
    reduction_policy: str          # 使用的裁剪策略


@dataclass(frozen=True)
class DroppedContextItem:
    """被丢弃的上下文条目记录。"""
    item_id: str                       # 条目 ID
    section: str                       # 所属段落
    reason: DroppedContextReason       # 丢弃原因
    source: str                        # 来源

    def __post_init__(self) -> None:
        _ensure_dropped_context_reason(self.reason)


@dataclass
class ContextReport:
    """上下文编译报告：记录一次上下文编译的完整裁剪和选择结果。"""
    context_id: str                                              # 编译 ID
    repository_fingerprint: str                                  # 仓库指纹
    total_budget_tokens: int                                     # 总 token 预算
    estimated_tokens_before: int                                 # 编译前估算 token 数
    estimated_tokens_after: int                                  # 编译后估算 token 数
    sections: list[ContextSectionReport] = field(default_factory=list)  # 各段落报告
    selected_items: list[dict[str, Any]] = field(default_factory=list)  # 选中条目摘要
    stale_items: list[str] = field(default_factory=list)         # 过时条目列表
    dropped_items: list[DroppedContextItem] = field(default_factory=list)  # 被丢弃条目
    repository_delta: RepositoryDelta = field(default_factory=RepositoryDelta)  # 仓库差异
    retrieved_memory_ids: list[str] = field(default_factory=list)        # 检索到的记忆 ID
    memory_retrieval_reasons: dict[str, list[str]] = field(default_factory=dict)  # 记忆检索原因
    context_mode: str | None = None
    budget_profile: dict[str, float] = field(default_factory=dict)
    relevance_reasons: dict[str, list[str]] = field(default_factory=dict)
    sanitization: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_context_freshness(value: object) -> ContextFreshness:
    if value not in _CONTEXT_FRESHNESS_VALUES:
        raise ValueError(f"Unknown context freshness: {value}")
    return cast(ContextFreshness, value)


def _ensure_context_trust(value: object) -> ContextTrust:
    if value not in _CONTEXT_TRUST_VALUES:
        raise ValueError(f"Unknown context trust: {value}")
    return cast(ContextTrust, value)


def _ensure_dropped_context_reason(value: object) -> DroppedContextReason:
    if value not in _DROPPED_CONTEXT_REASONS:
        raise ValueError(f"Unknown dropped context reason: {value}")
    return cast(DroppedContextReason, value)


__all__ = [
    "ContextFreshness",
    "ContextItem",
    "ContextReport",
    "ContextSectionReport",
    "ContextTrust",
    "DroppedContextItem",
    "DroppedContextReason",
    "RepositoryDelta",
    "RepositorySnapshot",
]
