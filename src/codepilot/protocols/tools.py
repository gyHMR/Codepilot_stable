from __future__ import annotations

# 新手导读：tools.py 只定义模型可见工具 spec 和对话结果状态。
# 关注点：注意这里没有 execute 函数和运行时元数据；可执行工具属于 tools/contracts.py。

"""
工具相关类型定义。

定义模型调用协议中的可见结构：
- Tool: 工具定义（模型可见的工具规范）
- ToolResultStatus: ToolResultMessage 的对话投影状态

工具执行结果由 codepilot.tools.results.ToolResult 唯一定义。
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast


# 工具执行结果状态
ToolResultStatus = Literal[
    "success",
    "error",
    "denied",
    "approval_required",
    "cancelled",
    "timed_out",
    "interrupted",
]
_TOOL_RESULT_STATUSES = frozenset(
    {
        "success",
        "error",
        "denied",
        "approval_required",
        "cancelled",
        "timed_out",
        "interrupted",
    }
)

# Task Plan tool names shared by tools execution and core plan state.
PROPOSE_PLAN_TOOL = "propose_plan"
CREATE_BUILD_PLAN_TOOL = "create_build_plan"
UPDATE_PLAN_PROGRESS_TOOL = "update_plan_progress"
CLOSE_PLAN_TOOL = "close_plan"
PLAN_TOOL_NAMES = frozenset(
    {
        PROPOSE_PLAN_TOOL,
        CREATE_BUILD_PLAN_TOOL,
        UPDATE_PLAN_PROGRESS_TOOL,
        CLOSE_PLAN_TOOL,
    }
)
PLAN_ITEM_LIMIT = 20


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
    "ToolResultStatus",
    "CLOSE_PLAN_TOOL",
    "CREATE_BUILD_PLAN_TOOL",
    "PLAN_TOOL_NAMES",
    "PROPOSE_PLAN_TOOL",
    "UPDATE_PLAN_PROGRESS_TOOL",
    "ensure_tool_result_status",
]
