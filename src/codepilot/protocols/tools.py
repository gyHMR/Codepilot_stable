from __future__ import annotations

# 新手导读：tools.py 定义模型可见工具 spec、工具调用和工具结果的跨层协议。
# 关注点：注意这里没有 execute 函数；可执行工具属于 tools/contracts.py。

"""
工具相关类型定义。

定义了工具调用全生命周期涉及的类型：
- Tool: 工具定义（模型可见的工具规范）
- ToolCall: 模型发出的工具调用请求
- ToolResult: 工具执行结果
- ToolMetadata: 工具元数据（风险级别、权限要求等）
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Union, cast

from .content import ImageContent, TextContent


# 工具风险级别：用于权限控制和审批流程
ToolRiskLevel = Literal["low", "medium", "high"]
_TOOL_RISK_LEVELS = frozenset({"low", "medium", "high"})

# 工具执行结果状态
ToolResultStatus = Literal["success", "error", "denied", "approval_required", "cancelled"]
_TOOL_RESULT_STATUSES = frozenset(
    {"success", "error", "denied", "approval_required", "cancelled"}
)


@dataclass
class Tool:
    """模型可见的工具定义（Tool Specification）。

    描述一个可供 LLM 调用的工具，包含名称、描述和参数 schema。

    Attributes:
        name: 工具名称（LLM 通过此名称发起调用）。
        description: 工具功能描述（帮助 LLM 理解何时使用此工具）。
        parameters: 参数的 JSON Schema 定义。
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        self.name = _require_tool_spec_text(self.name, field_name="tool name")
        self.description = _require_tool_spec_text(
            self.description,
            field_name="description",
        )
        if not isinstance(self.parameters, dict):
            raise TypeError("Tool parameters must be a dict")
        self.parameters = deepcopy(self.parameters)


@dataclass
class ToolCall:
    """模型发出的归一化工具调用请求。

    当 LLM 决定调用工具时，生成此对象描述要调用的工具和参数。

    Attributes:
        type: 类型标识，固定为 "toolCall"。
        id: 工具调用的唯一 ID（用于匹配 ToolResultMessage）。
        name: 要调用的工具名称。
        arguments: 解析后的参数字典。
        raw_arguments: 原始参数 JSON 字符串（解析失败时保留原文）。
        index: 在同一批工具调用中的序号。
        provider: 来源 provider 标识（可选）。
        metadata: 附加元数据字典。
    """

    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    index: int | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# 工具结果中可包含的内容块类型
ToolResultBlock = Union[TextContent, ImageContent]


@dataclass
class ToolResult:
    """归一化的工具执行结果。

    工具执行完成后返回此对象，包含执行状态、输出内容、影响范围等信息。

    Attributes:
        tool_call_id: 对应的工具调用 ID。
        tool_name: 工具名称。
        content: 结果内容块列表（文本/图片）。
        status: 执行状态。
        is_error: 是否为错误结果（与 status 自动同步）。
        approved: 是否已通过审批。
        approval_id: 审批记录 ID（可选）。
        error_code: 错误代码（可选）。
        exit_code: 进程退出码（可选）。
        affected_paths: 受影响的文件路径列表。
        workspace_changed: 是否修改了工作区文件。
        diff_summary: 变更摘要（可选）。
        verification: 验证结果字典（可选）。
        details: 附加详情（可选）。
        metadata: 附加元数据字典。
    """

    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultBlock] = field(default_factory=list)
    status: ToolResultStatus = "success"
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None
    error_code: str | None = None
    exit_code: int | None = None
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool | None = None
    diff_summary: str | None = None
    verification: dict[str, Any] | None = None
    details: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化后处理：自动同步 is_error 和 status 的一致性。"""
        ensure_tool_result_status(self.status)
        if self.is_error and self.status == "success":
            self.status = "error"
        elif self.status != "success":
            self.is_error = True


@dataclass(frozen=True)
class ToolMetadata:
    """工具元数据（不可变）。

    描述工具的静态属性，用于权限控制、并发调度和审批流程。

    Attributes:
        name: 工具名称。
        category: 工具分类（如 "file"、"shell"、"search"）。
        read_only: 是否为只读工具（不修改文件系统）。
        concurrency_safe: 是否可安全并发执行。
        exclusive: 是否需要独占访问（执行期间阻止其他工具）。
        requires_approval: 是否需要用户审批才能执行。
        risk_level: 风险级别（low/medium/high）。
        resource_scope: 资源作用域（如文件路径模式）。
        network_access: 是否需要网络访问。
        credential_required: 是否需要凭据。
        extra: 扩展字段字典。
    """

    name: str
    category: str
    read_only: bool
    concurrency_safe: bool
    exclusive: bool
    requires_approval: bool
    risk_level: ToolRiskLevel
    resource_scope: tuple[str, ...]
    network_access: bool = False
    credential_required: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_tool_metadata_text(self.name, field_name="name"),
        )
        object.__setattr__(
            self,
            "category",
            _require_tool_metadata_text(self.category, field_name="category"),
        )
        for field_name in (
            "read_only",
            "concurrency_safe",
            "exclusive",
            "requires_approval",
            "network_access",
            "credential_required",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(f"Tool metadata {field_name} must be bool")
        object.__setattr__(self, "risk_level", _ensure_tool_risk_level(self.risk_level))
        object.__setattr__(
            self,
            "resource_scope",
            tuple(_clean_unique_metadata_items(self.resource_scope)),
        )
        if not self.resource_scope:
            raise ValueError("Tool metadata resource_scope cannot be empty")
        if not isinstance(self.extra, dict):
            raise TypeError("Tool metadata extra must be a dict")
        object.__setattr__(self, "extra", dict(self.extra))


def ensure_tool_result_status(value: object) -> ToolResultStatus:
    if value not in _TOOL_RESULT_STATUSES:
        raise ValueError(f"Unknown tool result status: {value}")
    return cast(ToolResultStatus, value)


def coerce_tool_result_status(
    value: object,
    *,
    default: ToolResultStatus,
) -> ToolResultStatus:
    if value in _TOOL_RESULT_STATUSES:
        return cast(ToolResultStatus, value)
    return default


def _clean_tool_spec_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_tool_spec_text(value: object, *, field_name: str) -> str:
    text = _clean_tool_spec_text(value)
    if not text:
        raise ValueError(f"Tool {field_name} cannot be empty")
    return text


def _clean_tool_metadata_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_tool_metadata_text(value: object, *, field_name: str) -> str:
    text = _clean_tool_metadata_text(value)
    if not text:
        raise ValueError(f"tool metadata {field_name} cannot be empty")
    return text


def _ensure_tool_risk_level(value: object) -> ToolRiskLevel:
    text = _clean_tool_metadata_text(value)
    if text not in _TOOL_RISK_LEVELS:
        raise ValueError(f"Unknown tool risk level: {value}")
    return cast(ToolRiskLevel, text)


def _clean_unique_metadata_items(values: tuple[str, ...]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_tool_metadata_text(value)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


__all__ = [
    "Tool",
    "ToolCall",
    "ToolMetadata",
    "ToolResult",
    "ToolResultBlock",
    "ToolResultStatus",
    "ToolRiskLevel",
    "coerce_tool_result_status",
    "ensure_tool_result_status",
]
