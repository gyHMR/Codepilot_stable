from __future__ import annotations

"""
任务控制内部工具模块。

本模块定义了 Agent 在运行期间暴露给 LLM 的任务控制工具。
目前只有一个工具：complete_task_step（完成当前任务步骤）。

工作原理：
    1. 当 TaskController 初始化任务时，会将 complete_task_step 工具添加到工具列表
    2. LLM 在判断当前步骤完成后，调用此工具标记步骤完成
    3. TaskController 通过检测工具结果中的 task_control 元数据来识别完成信号
    4. 任务控制器随后推进到下一个步骤或标记任务完成

注意事项：
    - 此工具不应绕过工作区变更后的必要验证
    - summary 参数是必需的，用于记录完成摘要
    - evidence_refs 是可选的，用于关联证据
"""

from typing import Any

from codepilot.protocols import TextContent
from codepilot.tools import AgentTool, AgentToolResult


# 工具名称常量，用于在工具列表中查找和去重
COMPLETE_TASK_STEP_TOOL = "complete_task_step"


def complete_task_step_tool() -> AgentTool:
    """创建并返回任务步骤完成工具。

    此工具供 LLM 在判断当前任务步骤的验收标准已满足时调用。
    它会触发 TaskController 将当前步骤标记为完成，并推进到下一步。

    使用场景：
        - 调查步骤：收集到足够证据后调用
        - 规划步骤：制定好计划后调用
        - 总结步骤：整理好结论后调用

    注意：
        - 不应用于绕过工作区变更后的必要验证
        - summary 参数必须提供，用于记录完成摘要

    Returns:
        AgentTool: 任务步骤完成工具实例。
    """
    return AgentTool(
        name=COMPLETE_TASK_STEP_TOOL,
        label="Complete task step",
        description=(
            "Mark the current task step as complete when its acceptance criteria "
            "are satisfied. Use this for investigation, planning, or summary "
            "steps after gathering enough evidence. Do not use it to bypass "
            "required verification after workspace changes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short evidence-backed summary of what was completed.",
                },
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional evidence references such as tool:read_1.",
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        execute=_execute_complete_task_step,
        runtime_managed=True,
    )


def has_complete_task_step_tool(tools: list[AgentTool]) -> bool:
    """检查工具列表中是否已包含任务步骤完成工具。

    Args:
        tools: 当前可用的工具列表。

    Returns:
        bool: 如果工具列表中已存在 complete_task_step 工具则返回 True。
    """
    return any(tool.name == COMPLETE_TASK_STEP_TOOL for tool in tools)


async def _execute_complete_task_step(
    tool_call_id: str,
    params: dict[str, Any],
    signal=None,
    on_update=None,
) -> AgentToolResult:
    """执行任务步骤完成工具。

    处理流程：
    1. 从参数中提取 summary 和 evidence_refs
    2. 如果 summary 为空，返回错误结果
    3. 否则返回成功结果，并在 metadata 中携带 task_control 信号

    TaskController 通过检测 metadata.task_control.action == "complete_step"
    来识别这是一个步骤完成信号，而不是普通的工具执行结果。

    Args:
        tool_call_id: 工具调用 ID。
        params: 工具参数（包含 summary 和可选的 evidence_refs）。
        signal: 取消信号（未使用）。
        on_update: 进度更新回调（未使用）。

    Returns:
        AgentToolResult: 工具执行结果。
    """
    # 提取并清理 summary 参数
    summary = " ".join(str(params.get("summary") or "").strip().split())
    # 提取并清理 evidence_refs 参数
    evidence_refs = [
        str(item)
        for item in params.get("evidence_refs", [])
        if isinstance(item, str) and item.strip()
    ]
    # summary 是必需的，为空时返回错误
    if not summary:
        return AgentToolResult(
            content=[TextContent(text="summary is required")],
            status="error",
            is_error=True,
            error_code="missing_summary",
            metadata={
                "task_control": {
                    "action": "complete_step",
                    "valid": False,
                }
            },
        )
    # 返回成功结果，携带 task_control 信号供 TaskController 识别
    return AgentToolResult(
        content=[TextContent(text=f"Current task step completed: {summary}")],
        metadata={
            "task_control": {
                "action": "complete_step",
                "summary": summary[:500],
                "evidence_refs": evidence_refs,
                "tool_call_id": tool_call_id,
            }
        },
    )


__all__ = [
    "COMPLETE_TASK_STEP_TOOL",
    "complete_task_step_tool",
    "has_complete_task_step_tool",
]
