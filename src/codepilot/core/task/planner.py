from __future__ import annotations

"""Small optional planner for turning a prompt into task steps."""

import json
import re
from dataclasses import dataclass
from typing import cast

from codepilot.llm.ports import (
    LLMCompleted,
    LLMCorrelation,
    LLMFailed,
    LLMOptions,
    LLMRequest,
    ModelDescriptor,
    ModelPort,
)
from codepilot.protocols import AssistantMessage, Message, TextContent, UserMessage

from .contracts import PlanningDiscoveryReport
from .state import TASK_STEP_KINDS, TaskStepKind


AgentMessage = Message

_MAX_STEPS = 6
_MAX_FIELD_CHARS = 240
_PLAN_SOURCES = frozenset({"llm", "llm_with_discovery", "fallback"})


@dataclass(frozen=True)
class PlannedTaskStep:
    title: str
    kind: TaskStepKind = "other"
    acceptance: str | None = None
    verification_hint: str | None = None

    def __post_init__(self) -> None:
        title = _compact(self.title, limit=100)
        if not title:
            raise ValueError("Task plan step title cannot be empty")
        kind = _compact(self.kind, limit=40)
        if kind not in TASK_STEP_KINDS:
            kind = _infer_kind(title)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "kind", cast(TaskStepKind, kind))
        object.__setattr__(
            self,
            "acceptance",
            _compact(self.acceptance, limit=_MAX_FIELD_CHARS) or None,
        )
        object.__setattr__(
            self,
            "verification_hint",
            _compact(self.verification_hint, limit=_MAX_FIELD_CHARS) or None,
        )


@dataclass(frozen=True)
class TaskPlanDraft:
    goal: str
    steps: tuple[PlannedTaskStep, ...]
    source: str = "fallback"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        goal = _compact(self.goal, limit=1200)
        if not goal:
            raise ValueError("Task plan goal cannot be empty")
        if self.source not in _PLAN_SOURCES:
            raise ValueError(f"Unknown task plan source: {self.source}")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("Task plan draft must contain at least one step")
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "steps", steps[:_MAX_STEPS])
        object.__setattr__(
            self,
            "fallback_reason",
            _compact(self.fallback_reason, limit=240) or None,
        )


class TaskPlanner:
    async def generate(
        self,
        *,
        model: ModelDescriptor,
        messages: list[AgentMessage],
        model_port: ModelPort | None,
        fallback_goal: str,
        session_id: str | None = None,
        discovery_report: PlanningDiscoveryReport | None = None,
        options: LLMOptions | None = None,
    ) -> TaskPlanDraft:
        if model_port is None:
            return self.fallback(fallback_goal, reason="missing_model_port")
        try:
            planner_messages = [
                *messages,
                *(_discovery_message(discovery_report) if discovery_report else []),
                UserMessage(content="请把当前请求整理成 2-5 个简短执行步骤。只输出 JSON。"),
            ]
            request = LLMRequest(
                model=model,
                messages=tuple(planner_messages),
                system_prompt=_planner_prompt(),
                options=options or LLMOptions(),
                correlation=LLMCorrelation(session_id=session_id or ""),
            )
            assistant: AssistantMessage | None = None
            async for event in model_port.stream(request):
                if isinstance(event, LLMFailed):
                    return self.fallback(fallback_goal, reason=str(event.error))
                if isinstance(event, LLMCompleted):
                    assistant = event.message
            if assistant is None:
                return self.fallback(fallback_goal, reason="empty_planner_response")
            draft = self.parse_plan_message(assistant, fallback_goal=fallback_goal)
            if (
                discovery_report is not None
                and discovery_report.status == "completed"
                and draft.source == "llm"
            ):
                return TaskPlanDraft(draft.goal, draft.steps, source="llm_with_discovery")
            return draft
        except Exception as exc:
            return self.fallback(fallback_goal, reason=f"{type(exc).__name__}: {exc}")

    def parse_plan_message(
        self,
        message: AssistantMessage,
        *,
        fallback_goal: str,
    ) -> TaskPlanDraft:
        data = _load_json(_assistant_text(message))
        if not isinstance(data, dict):
            return self.fallback(fallback_goal, reason="invalid_json")
        goal = _compact(data.get("goal"), limit=1200) or fallback_goal
        steps = _parse_steps(data)
        if not steps:
            return self.fallback(goal, reason="missing_steps")
        return TaskPlanDraft(goal=goal, steps=tuple(steps), source="llm")

    def fallback(self, goal: str, *, reason: str | None = None) -> TaskPlanDraft:
        return TaskPlanDraft(
            goal=_compact(goal, limit=1200) or "完成当前请求",
            steps=(PlannedTaskStep("完成当前请求"),),
            source="fallback",
            fallback_reason=reason,
        )


def _planner_prompt() -> str:
    return (
        "You are Codepilot Task Planner. Output JSON only: "
        '{"goal": string, "steps": [{"title": string, '
        '"kind": "read|plan|edit|verify|summarize|other", '
        '"acceptance": string|null, "verification_hint": string|null}]}. '
        "Keep the plan short and executable."
    )


def _discovery_message(report: PlanningDiscoveryReport) -> list[UserMessage]:
    lines = ["Planning discovery report:", f"status: {report.status}"]
    if report.facts:
        lines.extend(["facts:", *[f"- {item}" for item in report.facts]])
    if report.relevant_files:
        lines.extend(["relevant_files:", *[f"- {item}" for item in report.relevant_files]])
    if report.verification_hints:
        lines.extend(["verification_hints:", *[f"- {item}" for item in report.verification_hints]])
    return [UserMessage(content="\n".join(lines))]


def _assistant_text(message: AssistantMessage) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextContent))


def _load_json(text: str) -> object:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _parse_steps(data: dict[str, object]) -> tuple[PlannedTaskStep, ...]:
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) and isinstance(data.get("plan"), dict):
        raw_steps = data["plan"].get("steps")  # type: ignore[index]
    if not isinstance(raw_steps, list):
        return ()
    steps: list[PlannedTaskStep] = []
    seen: set[str] = set()
    for raw in raw_steps:
        step = _parse_step(raw)
        if step is None or step.title in seen:
            continue
        seen.add(step.title)
        steps.append(step)
        if len(steps) >= _MAX_STEPS:
            break
    return tuple(steps)


def _parse_step(raw: object) -> PlannedTaskStep | None:
    if isinstance(raw, str):
        title = _compact(raw, limit=100)
        return PlannedTaskStep(title, kind=_infer_kind(title)) if title else None
    if not isinstance(raw, dict):
        return None
    title = _compact(
        raw.get("title") or raw.get("name") or raw.get("task") or raw.get("description"),
        limit=100,
    )
    if not title:
        return None
    kind = _compact(raw.get("kind") or raw.get("type"), limit=40)
    if kind not in TASK_STEP_KINDS:
        kind = _infer_kind(title)
    return PlannedTaskStep(
        title=title,
        kind=cast(TaskStepKind, kind),
        acceptance=_compact(raw.get("acceptance") or raw.get("acceptance_criteria"), limit=240) or None,
        verification_hint=_compact(raw.get("verification_hint") or raw.get("verify"), limit=240) or None,
    )


def _infer_kind(value: object) -> TaskStepKind:
    text = _compact(value, limit=160).lower()
    if any(token in text for token in ("pytest", "test", "验证", "测试", "检查")):
        return "verify"
    if any(token in text for token in ("修改", "修复", "实现", "edit", "fix")):
        return "edit"
    if any(token in text for token in ("阅读", "分析", "查找", "inspect", "read")):
        return "read"
    if any(token in text for token in ("计划", "plan")):
        return "plan"
    if any(token in text for token in ("总结", "summarize")):
        return "summarize"
    return "other"


def _compact(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


__all__ = ["PlannedTaskStep", "TaskPlanDraft", "TaskPlanner"]
