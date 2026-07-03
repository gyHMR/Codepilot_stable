from __future__ import annotations

# 新手导读：planner 负责把目标整理成可执行步骤，保持轻量而不是引入复杂计划系统。
# 关注点：读它可以理解 Codepilot 如何把用户请求变成任务步骤。

"""
轻量级 LLM 任务规划器模块。

本模块实现了 plan-and-execute 任务控制模式中的规划器。
规划器在 Agent 循环开始前调用 LLM，生成一个结构化的执行计划。

设计原则：
    - 规划器刻意保持小巧：只做一件事——生成计划
    - 使用 LLM 生成 JSON 格式的计划，然后验证和规范化
    - 规划失败时安全降级为单步计划（"完成当前请求"）
    - 最多生成 6 个步骤，保持计划简洁可执行

核心类：
    - PlannedTaskStep: 单个规范化步骤（由规划器生成）
    - TaskPlanDraft: 经过验证的规划器输出（用于初始化 TaskState）
    - TaskPlanner: 规划器主类，负责生成和解析计划

工作流程：
    1. TaskPlanner.generate() 调用 LLM 生成 JSON 计划
    2. parse_plan_message() 解析 LLM 响应中的 JSON
    3. 验证和规范化每个步骤（标题、类型、验收标准等）
    4. 返回 TaskPlanDraft，供 TaskController.initialize() 使用
"""

import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, cast

from codepilot.llm.api_registry import complete_simple
from codepilot.llm.event_stream import AssistantMessageEventStream
from codepilot.protocols import (
    AssistantMessage,
    Context,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    UserMessage,
)

from ..events import maybe_await
from ..types import AgentMessage
from .contracts import PlanningDiscoveryReport
from .state import TASK_STEP_KINDS, TaskStepKind


_MAX_PLANNED_STEPS = 6
_MAX_FIELD_CHARS = 240
_TASK_PLAN_SOURCES = frozenset({"llm", "llm_with_discovery", "fallback"})


@dataclass(frozen=True)
class PlannedTaskStep:
    """任务规划器生成的单个规范化步骤。

    由 TaskPlanner 从 LLM 响应中解析并规范化后创建。
    作为 TaskPlanDraft 的组成部分，最终传递给 TaskController.initialize()。

    Attributes:
        title: 步骤标题（最多 80 字符）。
        kind: 步骤类型（investigate/edit/verify/summarize/other）。
        acceptance: 完成标准（最多 240 字符，可选）。
        verification_hint: 验证方式提示（最多 240 字符，可选）。
    """
    title: str                           # 步骤标题
    kind: TaskStepKind = "other"         # 步骤类型
    acceptance: str | None = None        # 完成标准
    verification_hint: str | None = None # 验证方式提示

    def __post_init__(self) -> None:
        """初始化后校验和规范化：清理文本、截断超长字段。"""
        title = _clean_text(self.title, limit=80)
        if not title:
            raise ValueError("Task plan step title cannot be empty")

        kind = _clean_text(self.kind, limit=40) or "other"
        if kind not in TASK_STEP_KINDS:
            raise ValueError(f"Unknown task step kind: {kind}")

        object.__setattr__(self, "title", title)
        object.__setattr__(self, "kind", cast(TaskStepKind, kind))
        object.__setattr__(
            self,
            "acceptance",
            _clean_text(self.acceptance, limit=_MAX_FIELD_CHARS) or None,
        )
        object.__setattr__(
            self,
            "verification_hint",
            _clean_text(self.verification_hint, limit=_MAX_FIELD_CHARS) or None,
        )


@dataclass(frozen=True)
class TaskPlanDraft:
    """经过验证的规划器输出，用于初始化 TaskState。

    TaskPlanDraft 是模型规划与确定性任务控制之间的边界。
    它只存储经过规范化、非空的步骤，使下游代码可以直接初始化 TaskState，
    而无需重新验证规划器特定的字段。

    Attributes:
        goal: 任务目标（最多 1200 字符）。
        steps: 规范化后的步骤元组（至少 1 个步骤）。
        source: 计划来源（"llm" 表示由 LLM 生成，"fallback" 表示降级计划）。
        fallback_reason: 降级原因，便于评测和审计定位 planner 为什么没有生效。
        fallback_output_preview: 降级时保留的模型输出短预览，便于定位解析失败。
        fallback_parsed_keys: 降级时解析出的顶层 JSON key，便于判断 schema 偏差。
    """
    goal: str                                                              # 任务目标
    steps: tuple[PlannedTaskStep, ...] = field(default_factory=tuple)       # 步骤元组
    source: str = "fallback"                                               # 计划来源
    fallback_reason: str | None = None                                      # 降级原因
    fallback_output_preview: str | None = None                              # 降级输出预览
    fallback_parsed_keys: tuple[str, ...] = field(default_factory=tuple)     # 降级 JSON key

    def __post_init__(self) -> None:
        """初始化后校验：确保目标非空、步骤至少一个、来源合法。"""
        goal = _clean_text(self.goal, limit=1200)
        if not goal:
            raise ValueError("Task plan goal cannot be empty")

        source = _clean_text(self.source, limit=40)
        if source not in _TASK_PLAN_SOURCES:
            raise ValueError(f"Unknown task plan source: {source}")

        steps = tuple(self.steps)
        if not steps:
            raise ValueError("Task plan draft must contain at least one step")
        for step in steps:
            if not isinstance(step, PlannedTaskStep):
                raise TypeError("Task plan steps must be PlannedTaskStep instances")

        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self,
            "fallback_reason",
            _clean_text(self.fallback_reason, limit=_MAX_FIELD_CHARS) or None,
        )
        object.__setattr__(
            self,
            "fallback_output_preview",
            _clean_text(self.fallback_output_preview, limit=500) or None,
        )
        object.__setattr__(
            self,
            "fallback_parsed_keys",
            tuple(
                key
                for item in self.fallback_parsed_keys
                if (key := _clean_text(item, limit=80))
            ),
        )


StreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]


class TaskPlanner:
    """轻量级 LLM 任务规划器：生成和解析任务执行计划。

    使用方式：
        planner = TaskPlanner()
        draft = await planner.generate(
            model=model,
            messages=messages,
            convert_to_llm=convert_to_llm,
            fallback_goal="完成当前请求",
        )
        # draft.steps 包含规范化后的步骤列表
    """

    async def generate(
        self,
        *,
        model: Model,
        messages: list[AgentMessage],
        convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]],
        fallback_goal: str,
        stream_fn: StreamFn | None = None,
        api_key: str | None = None,
        session_id: str | None = None,
        discovery_report: PlanningDiscoveryReport | None = None,
    ) -> TaskPlanDraft:
        """向 LLM 请求生成初始执行计划，失败时返回安全的降级计划。

        流程：
        1. 将消息转换为 LLM 格式
        2. 构建规划专用的系统提示词
        3. 调用 LLM 生成 JSON 格式的计划
        4. 解析和验证计划
        5. 失败时返回单步降级计划

        Args:
            model: LLM 模型信息。
            messages: 当前消息列表。
            convert_to_llm: 消息转换函数。
            fallback_goal: 降级计划的目标描述。
            stream_fn: 可选的流式调用函数。
            api_key: 可选的 API Key。
            session_id: 可选的会话 ID。

        Returns:
            TaskPlanDraft: 经过验证的执行计划。
        """
        try:
            llm_messages = await maybe_await(convert_to_llm(list(messages)))
            context = Context(
                system_prompt=_planner_system_prompt(),
                messages=[
                    *llm_messages,
                    *(
                        [
                            UserMessage(
                                content=_render_discovery_for_planner(discovery_report)
                            )
                        ]
                        if discovery_report is not None
                        else []
                    ),
                    UserMessage(
                        content=(
                            "请为当前用户请求生成一个简洁执行计划。"
                            "只输出 JSON，不要输出 Markdown。"
                        )
                    ),
                ],
                tools=[],
            )
            options = SimpleStreamOptions(api_key=api_key, session_id=session_id)
            if stream_fn is not None:
                stream = await maybe_await(stream_fn(model, context, options))
                message = await stream.result()
            else:
                message = await maybe_await(complete_simple(model, context, options))
            draft = self.parse_plan_message(message, fallback_goal=fallback_goal)
            if (
                discovery_report is not None
                and discovery_report.status == "completed"
                and draft.source == "llm"
            ):
                return TaskPlanDraft(
                    goal=draft.goal,
                    steps=draft.steps,
                    source="llm_with_discovery",
                )
            return draft
        except Exception as exc:
            return self.fallback(
                fallback_goal,
                reason=f"{type(exc).__name__}: {exc}",
            )

    def parse_plan_message(
        self,
        message: AssistantMessage,
        *,
        fallback_goal: str,
    ) -> TaskPlanDraft:
        """从助手消息中解析 JSON 格式的执行计划。

        解析流程：
        1. 提取消息中的文本内容
        2. 尝试解析为 JSON 对象
        3. 提取 goal 和 steps 字段
        4. 验证和规范化每个步骤
        5. 解析失败时返回降级计划

        Args:
            message: LLM 返回的助手消息。
            fallback_goal: 降级计划的目标描述。

        Returns:
            TaskPlanDraft: 经过验证的执行计划。
        """

        text = _assistant_text(message)
        data = _loads_json_object(text)
        if not isinstance(data, dict):
            return self.fallback(
                fallback_goal,
                reason="invalid_json",
                output_preview=text,
            )
        goal = _clean_text(data.get("goal"), limit=1200) or fallback_goal
        raw_steps = _extract_raw_steps(data)
        if not isinstance(raw_steps, list):
            return self.fallback(
                goal,
                reason="missing_steps",
                output_preview=text,
                parsed_keys=_parsed_keys(data),
            )
        steps: list[PlannedTaskStep] = []
        seen: set[str] = set()
        for raw in raw_steps:
            fields = _raw_step_fields(raw)
            if fields is None:
                continue
            title, kind, acceptance, verification_hint = fields
            if not title or title in seen:
                continue
            seen.add(title)
            if kind not in TASK_STEP_KINDS:
                kind = "other"
            planned_kind = cast(TaskStepKind, kind)
            steps.append(
                PlannedTaskStep(
                    title=title,
                    kind=planned_kind,
                    acceptance=acceptance,
                    verification_hint=verification_hint,
                )
            )
            if len(steps) >= _MAX_PLANNED_STEPS:
                break
        if not steps:
            return self.fallback(
                goal,
                reason="empty_steps",
                output_preview=text,
                parsed_keys=_parsed_keys(data),
            )
        return TaskPlanDraft(goal=goal, steps=steps, source="llm")

    def fallback(
        self,
        goal: str,
        *,
        reason: str | None = None,
        output_preview: str | None = None,
        parsed_keys: tuple[str, ...] | None = None,
    ) -> TaskPlanDraft:
        """返回安全的单步降级计划。

        当 LLM 规划失败（解析错误、网络异常等）时，返回一个最简单的
        单步计划，确保任务控制系统仍能正常工作。

        Args:
            goal: 任务目标描述。
            reason: 降级原因。

        Returns:
            TaskPlanDraft: 单步降级计划。
        """
        clean_goal = _clean_text(goal, limit=1200) or "完成当前请求"
        return TaskPlanDraft(
            goal=clean_goal,
            steps=[PlannedTaskStep(title="完成当前请求")],
            source="fallback",
            fallback_reason=reason,
            fallback_output_preview=output_preview,
            fallback_parsed_keys=parsed_keys or (),
        )


def _planner_system_prompt() -> str:
    """返回规划器的系统提示词：指示 LLM 输出 JSON 格式的执行计划。"""
    return (
        "You are Codepilot Task Planner.\n"
        "Create a short plan for a local coding agent. Output JSON only.\n"
        "Schema: {\"goal\": string, \"steps\": [{\"title\": string, "
        "\"kind\": \"investigate|edit|verify|summarize|other\", "
        "\"acceptance\": string|null, \"verification_hint\": string|null}]}.\n"
        "Use 2-5 steps for implementation tasks and at most 6 steps. "
        "Keep steps concrete and executable. Do not include markdown.\n"
        "If a planning discovery report is provided, use those facts as the "
        "primary evidence. Do not invent files or verification commands that "
        "are not supported by the report."
    )


def _render_discovery_for_planner(report: PlanningDiscoveryReport) -> str:
    lines = [
        "Planning discovery report:",
        f"status: {report.status}",
    ]
    if report.facts:
        lines.append("facts:")
        lines.extend(f"- {item}" for item in report.facts)
    if report.relevant_files:
        lines.append("relevant_files:")
        lines.extend(f"- {item}" for item in report.relevant_files)
    if report.risks:
        lines.append("risks:")
        lines.extend(f"- {item}" for item in report.risks)
    if report.verification_hints:
        lines.append("verification_hints:")
        lines.extend(f"- {item}" for item in report.verification_hints)
    if report.open_questions:
        lines.append("open_questions:")
        lines.extend(f"- {item}" for item in report.open_questions)
    return "\n".join(lines)


def _assistant_text(message: AssistantMessage) -> str:
    """从助手消息中提取纯文本内容（过滤掉非文本块）。"""
    return "".join(
        block.text
        for block in message.content
        if isinstance(block, TextContent)
    ).strip()


def _loads_json_object(text: str) -> object:
    """从文本中加载 JSON 对象：先尝试直接解析，失败则尝试提取花括号内的内容。

    LLM 有时会在 JSON 前后添加额外文本（如 "```json\n...\n```"），
    此函数会尝试从文本中提取有效的 JSON。
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试从文本中提取 JSON 对象（匹配最外层的花括号）
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _extract_raw_steps(data: dict[str, object]) -> object:
    """从常见 planner JSON 形态中提取步骤列表。"""
    raw_steps = data.get("steps")
    if isinstance(raw_steps, list):
        return raw_steps

    raw_plan = data.get("plan")
    if isinstance(raw_plan, list):
        return raw_plan
    if isinstance(raw_plan, dict):
        nested_steps = raw_plan.get("steps")
        if isinstance(nested_steps, list):
            return nested_steps

    raw_tasks = data.get("tasks")
    if isinstance(raw_tasks, list):
        return raw_tasks
    return None


def _raw_step_fields(raw: object) -> tuple[str, str, str | None, str | None] | None:
    """把 dict 或字符串步骤规范化为 planner 内部字段。"""
    if isinstance(raw, str):
        title = _clean_text(raw, limit=80)
        if not title:
            return None
        return title, _infer_step_kind(title), None, None
    if not isinstance(raw, dict):
        return None

    title = _clean_text(
        raw.get("title")
        or raw.get("name")
        or raw.get("task")
        or raw.get("description"),
        limit=80,
    )
    if not title:
        return None
    kind = _clean_text(raw.get("kind") or raw.get("type"), limit=40)
    if kind not in TASK_STEP_KINDS:
        kind = _infer_step_kind(title)
    return (
        title,
        kind,
        _clean_text(
            raw.get("acceptance") or raw.get("acceptance_criteria"),
            limit=_MAX_FIELD_CHARS,
        )
        or None,
        _clean_text(
            raw.get("verification_hint") or raw.get("verify") or raw.get("verification"),
            limit=_MAX_FIELD_CHARS,
        )
        or None,
    )


def _infer_step_kind(title: str) -> str:
    text = title.lower()
    if any(token in text for token in ("pytest", "test", "测试", "验证", "检查")):
        return "verify"
    if any(token in text for token in ("修改", "修复", "编辑", "实现", "更新", "fix", "edit")):
        return "edit"
    if any(token in text for token in ("定位", "阅读", "排查", "分析", "调查", "查找", "inspect")):
        return "investigate"
    if any(token in text for token in ("总结", "汇报", "说明", "summarize")):
        return "summarize"
    return "other"


def _parsed_keys(data: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(key) for key in data.keys())


def _clean_text(value: object, *, limit: int) -> str:
    """清理文本：去除多余空白、截断到指定长度。"""
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


__all__ = ["PlannedTaskStep", "TaskPlanDraft", "TaskPlanner"]
