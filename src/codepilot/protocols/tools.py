from __future__ import annotations

# 新手导读：tools.py 只定义模型可见工具 spec 和工具结果。
# 关注点：注意这里没有 execute 函数和运行时元数据；可执行工具属于 tools/contracts.py。

"""
工具相关类型定义。

定义工具层跨层共享的模型可见结构和结果结构：
- Tool: 工具定义（模型可见的工具规范）
- ToolResult: 工具执行结果
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Union, cast

from .conversation import ImageContent, TextContent


# 工具风险级别字符串由 tools.contracts.ToolMetadata 使用。
ToolRiskLevel = Literal["low", "medium", "high"]

# 工具执行结果状态
ToolResultStatus = Literal["success", "error", "denied", "approval_required", "cancelled"]
_TOOL_RESULT_STATUSES = frozenset(
    {"success", "error", "denied", "approval_required", "cancelled"}
)

# Soft plan update tool name shared by tools execution and core plan state.
UPDATE_PLAN_TOOL = "update_plan"


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


def ensure_tool_result_status(value: object) -> ToolResultStatus:
    if value not in _TOOL_RESULT_STATUSES:
        raise ValueError(f"Unknown tool result status: {value}")
    return cast(ToolResultStatus, value)


def _clean_tool_spec_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_tool_spec_text(value: object, *, field_name: str) -> str:
    text = _clean_tool_spec_text(value)
    if not text:
        raise ValueError(f"Tool {field_name} cannot be empty")
    return text


__all__ = [
    "Tool",
    "ToolResult",
    "ToolResultBlock",
    "ToolResultStatus",
    "ToolRiskLevel",
    "UPDATE_PLAN_TOOL",
    "ensure_tool_result_status",
]
