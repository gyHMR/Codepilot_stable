from __future__ import annotations

"""
内置工具：update_plan —— 更新当前会话的软计划面板。

该工具用于向用户展示或更新执行计划（软计划），是一种通信手段而非控制手段。
它不会终止运行，仅为进度展示或计划提案服务。

计划包含整体 summary，且每个项包含：
  - step: 步骤描述文本
  - details: 具体动作、边界和实现意图
  - verification: 该步骤的验证方式
  - status: 步骤状态（pending / in_progress / completed）

关键约束：
  - build 模式同时最多只有一个 in_progress 项
  - plan 模式只提交待审批提案，模型给出的进度状态会被归一化为 pending
  - 计划项总数受 PLAN_ITEM_LIMIT 限制
"""

from collections.abc import Callable
from typing import Any

from codepilot.protocols import PLAN_ITEM_LIMIT, TextContent, UPDATE_PLAN_TOOL
from codepilot.tools.contracts import ToolCallRequest, ToolDefinition, ToolResult
from codepilot.tools.registry import get_builtin_tool_metadata

# 有效的计划项状态集合，用于合法性校验
_ITEM_STATUSES = {"pending", "in_progress", "completed"}


# ---------------------------------------------------------------------------
# 工厂函数：create_plan_tools()
#   - 根据 allow 白名单决定是否注册 update_plan 工具
#   - 定义工具的 JSON Schema 参数规范
#   - 返回 ToolDefinition 列表，供工具注册系统消费
# ---------------------------------------------------------------------------
def create_plan_tools(*, allow: Callable[[str], bool]) -> list[ToolDefinition]:
    """
    创建 update_plan 工具的工厂函数。

    参数:
        allow (Callable[[str], bool]):
            工具白名单回调。传入工具名称，返回该工具是否允许注册。
            若 `allow(UPDATE_PLAN_TOOL)` 返回 False，则返回空列表。

    返回:
        list[ToolDefinition]: 包含 update_plan 工具定义的列表（或空列表）。

    异常:
        ValueError: 当内置元数据中找不到 UPDATE_PLAN_TOOL 对应的元数据时抛出。
    """

    # 检查 update_plan 是否在白名单中，不在则跳过注册
    if not allow(UPDATE_PLAN_TOOL):
        return []

    # 从注册中心获取该工具的预定义元数据（重试策略、超时、权限等）
    metadata = get_builtin_tool_metadata(UPDATE_PLAN_TOOL)
    if metadata is None:
        raise ValueError(f"Missing builtin metadata for {UPDATE_PLAN_TOOL}")

    # ---- 构造并返回 ToolDefinition ----
    return [
        ToolDefinition(
            name=UPDATE_PLAN_TOOL,
            label="Update plan",
            description=(
                "Update the visible soft plan board. This communicates progress "
                "or a proposed plan, but never completes the run."
            ),
            # 参数 JSON Schema 定义
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "整体实现思路、关键决策和范围约束。",
                    },
                    # explanation: 可选的人类可读说明文本，描述本次更新的意图
                    "explanation": {"type": "string"},
                    # plan: 计划项数组，必须包含至少 1 项、至多 PLAN_ITEM_LIMIT 项
                    "plan": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": PLAN_ITEM_LIMIT,
                        "items": {
                            "type": "object",
                            "properties": {
                                # step: 步骤描述（必填）
                                "step": {"type": "string"},
                                "details": {
                                    "type": "string",
                                    "description": "该步骤的具体动作、边界和实现意图。",
                                },
                                "verification": {
                                    "type": "string",
                                    "description": "完成该步骤后应执行的验证。",
                                },
                                # status: 步骤状态（必填），只能是三种预定义状态之一
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["step", "details", "verification", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                # summary 和 plan 为必填参数；explanation 可选
                "required": ["summary", "plan"],
                "additionalProperties": False,
            },
            metadata=metadata,
            # 绑定执行函数
            execute=_execute_update_plan,
        )
    ]


# ---------------------------------------------------------------------------
# 执行函数：_execute_update_plan()
#   - 校验并处理来自模型的计划更新请求
#   - 区分运行模式（plan 模式 vs 执行模式）施加不同的验证规则
#   - 返回成功或错误结果
# ---------------------------------------------------------------------------
async def _execute_update_plan(
    request: ToolCallRequest,
    signal: Any = None,
    on_update: Any = None,
) -> ToolResult:
    """
    执行计划更新操作的核心异步函数。

    处理流程:
        1. 对请求参数进行结构化和业务规则校验（_validated_plan_update）
        2. plan 模式把模型给出的进度状态归一化为 pending，形成待审批提案
        3. 校验失败返回错误结果，成功返回确认结果

    参数:
        request (ToolCallRequest): 工具调用请求，包含 arguments（计划数据）和 current_mode（当前运行模式）。
        signal: 取消信号（未使用）。
        on_update: 进度更新回调（未使用）。

    返回:
        ToolResult:
            - 成功时：content 为 "Plan updated."，metadata 包含验证后的 plan_update 数据
            - 失败时：content 为错误描述文本，status="error"，error_code="invalid_plan_update"
    """
    _ = signal, on_update

    try:
        is_proposal = request.current_mode == "plan"
        plan_update, original_statuses = _validated_plan_update(
            request.arguments,
            proposal=is_proposal,
        )

    except ValueError as exc:
        # 校验失败：返回带有错误信息的 ToolResult，is_error=True 标记为错误结果
        return ToolResult(
            content=[TextContent(text=str(exc))],
            status="error",
            is_error=True,
            error_code="invalid_plan_update",
        )

    metadata: dict[str, Any] = {"plan_update": plan_update}
    if is_proposal and any(status != "pending" for status in original_statuses):
        metadata["plan_normalized"] = True
        metadata["original_plan_statuses"] = original_statuses
        message = "Plan proposed. Item statuses were normalized to pending for approval."
    else:
        message = "Plan updated."

    # 校验通过：返回成功结果，将验证后的计划数据放入 metadata 供下游消费
    return ToolResult(
        content=[TextContent(text=message)],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 校验函数：_validated_plan_update(params)
#   - 对计划更新参数进行全面的结构和业务规则校验
#   - 检验内容包括：
#       1. plan 必须是非空数组
#       2. plan 数组长度不得超过 PLAN_ITEM_LIMIT
#       3. 每个计划项必须是字典类型
#       4. 每个计划项的 step 不能为空
#       5. 每个计划项的 status 必须在 _ITEM_STATUSES 集合中
#       6. 整个计划中 in_progress 状态的项最多只能有 1 个
#   - 所有字符串值经过 _clean_text() 清洗（去首尾空白、合并中间空白）
# ---------------------------------------------------------------------------
def _validated_plan_update(
    params: dict[str, Any],
    *,
    proposal: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """
    验证并清洗计划更新参数。

    参数:
        params (dict[str, Any]): 来自工具调用的原始参数字典，通常包含:
            - "plan": 计划项数组
            - "explanation": 可选的说明文本

    返回:
        tuple[dict[str, Any], list[str]]: 验证通过并清洗后的规范化字典，以及模型原始状态列表。
        规范化字典包含:
            - "explanation": 清洗后的说明文本（可能为空字符串）
            - "plan": 清洗后的计划项列表（每个项包含 "step" 和 "status"）

    异常:
        ValueError: 当任何校验规则不满足时抛出，附带具体错误描述。
    """

    # ---- 规则 1：plan 必须存在且为非空数组 ----
    plan = params.get("plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must contain at least one item")

    # ---- 规则 2：plan 数组长度不得超限 ----
    if len(plan) > PLAN_ITEM_LIMIT:
        raise ValueError(f"plan cannot contain more than {PLAN_ITEM_LIMIT} items")

    summary = _clean_text(params.get("summary"))
    if not summary:
        raise ValueError("summary is required")

    # ---- 清洗 explanation 字段 ----
    # _clean_text 会去除首尾空白并将内部连续空白合并为单个空格
    explanation = _clean_text(params.get("explanation")) or ""

    # ---- 逐项校验 plan 中的每个条目 ----
    items: list[dict[str, str]] = []
    original_statuses: list[str] = []
    in_progress = 0  # 计数器：记录 in_progress 状态的数量

    for index, raw in enumerate(plan):
        # 规则 3：每个计划项必须是字典类型
        if not isinstance(raw, dict):
            raise ValueError(f"plan[{index}] must be an object")

        # 规则 4：step 字段不能为空
        step = _clean_text(raw.get("step"))
        if not step:
            raise ValueError(f"plan[{index}].step is required")
        details = _clean_text(raw.get("details"))
        if not details:
            raise ValueError(f"plan[{index}].details is required")
        verification = _clean_text(raw.get("verification"))
        if not verification:
            raise ValueError(f"plan[{index}].verification is required")

        # 规则 5：status 字段必须为合法状态值
        status = _clean_text(raw.get("status"))
        if status not in _ITEM_STATUSES:
            raise ValueError(f"plan[{index}].status is invalid")
        original_statuses.append(status)

        # 累计 in_progress 数量（用于规则 6 的最终检查）
        if status == "in_progress":
            in_progress += 1

        # 收集清洗后的合法条目
        items.append(
            {
                "step": step,
                "details": details,
                "verification": verification,
                "status": "pending" if proposal else status,
            }
        )

    # ---- 规则 6：同时最多只能有 1 个 in_progress 项 ----
    # 这是业务语义约束：同一时间只能有一个步骤处于进行中状态
    if not proposal and in_progress > 1:
        raise ValueError("plan can contain at most one in_progress item")

    return {
        "summary": summary,
        "explanation": explanation,
        "plan": items,
    }, original_statuses


# ---------------------------------------------------------------------------
# 工具函数：_clean_text(value)
#   - 将任意值转换为清洗后的单行字符串
#   - 处理流程：转字符串 → 去首尾空白 → 将内部连续空白合并为单个空格
#   - 对 None 值返回空字符串
# ---------------------------------------------------------------------------
def _clean_text(value: object) -> str:
    """
    清洗文本值：去除首尾空白并合并内部连续空白。

    该函数用于将用户输入或模型输出的各种格式的文本统一规范化为
    紧凑的单行字符串，避免因多余空格导致的匹配或比较问题。

    参数:
        value (object): 任意类型的输入值。若为 None，返回空字符串。

    返回:
        str: 清洗后的字符串。例如:
            - "  hello   world  " -> "hello world"
            - None -> ""
            - 123 -> "123"
    """
    # 若 value 为 None，直接返回空字符串
    # 否则：str(value) 转为字符串 → strip() 去首尾空白 → split() 按空白分割
    # → " ".join(...) 用单空格重新连接（实现内部空白合并）
    return " ".join(str(value).strip().split()) if value is not None else ""


__all__ = ["create_plan_tools"]
