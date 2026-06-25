from __future__ import annotations

"""Lightweight LLM planner for plan-and-execute task control.

The planner is intentionally small: it asks the model for a bounded JSON plan,
validates the result, and falls back to a single-step plan when planning fails.
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

from .events import maybe_await
from .task_state import TASK_STEP_KINDS, TaskStepKind
from .types import AgentMessage


_MAX_PLANNED_STEPS = 6
_MAX_FIELD_CHARS = 240
_TASK_PLAN_SOURCES = frozenset({"llm", "fallback"})


@dataclass(frozen=True)
class PlannedTaskStep:
    """A single normalized step proposed by the task planner."""

    title: str
    kind: TaskStepKind = "other"
    acceptance: str | None = None
    verification_hint: str | None = None

    def __post_init__(self) -> None:
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
    """Validated planner output used to initialize TaskState.

    A draft is the boundary between model planning and deterministic task
    control.  It stores only normalized, non-empty steps so downstream code can
    initialize ``TaskState`` without re-validating planner-specific fields.
    """

    goal: str
    steps: tuple[PlannedTaskStep, ...] = field(default_factory=tuple)
    source: str = "fallback"

    def __post_init__(self) -> None:
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


StreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]


class TaskPlanner:
    """Generate and parse a lightweight task plan."""

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
    ) -> TaskPlanDraft:
        """Ask the model for an initial plan, returning a safe fallback on failure."""

        try:
            llm_messages = await maybe_await(convert_to_llm(list(messages)))
            context = Context(
                system_prompt=_planner_system_prompt(),
                messages=[
                    *llm_messages,
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
            return self.parse_plan_message(message, fallback_goal=fallback_goal)
        except Exception:
            return self.fallback(fallback_goal)

    def parse_plan_message(
        self,
        message: AssistantMessage,
        *,
        fallback_goal: str,
    ) -> TaskPlanDraft:
        """Parse a JSON plan from an assistant message."""

        text = _assistant_text(message)
        data = _loads_json_object(text)
        if not isinstance(data, dict):
            return self.fallback(fallback_goal)
        goal = _clean_text(data.get("goal"), limit=1200) or fallback_goal
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            return self.fallback(goal)
        steps: list[PlannedTaskStep] = []
        seen: set[str] = set()
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            title = _clean_text(raw.get("title"), limit=80)
            if not title or title in seen:
                continue
            seen.add(title)
            kind = _clean_text(raw.get("kind"), limit=40) or "other"
            if kind not in TASK_STEP_KINDS:
                kind = "other"
            planned_kind = cast(TaskStepKind, kind)
            steps.append(
                PlannedTaskStep(
                    title=title,
                    kind=planned_kind,
                    acceptance=_clean_text(raw.get("acceptance"), limit=_MAX_FIELD_CHARS) or None,
                    verification_hint=(
                        _clean_text(raw.get("verification_hint"), limit=_MAX_FIELD_CHARS)
                        or None
                    ),
                )
            )
            if len(steps) >= _MAX_PLANNED_STEPS:
                break
        if not steps:
            return self.fallback(goal)
        return TaskPlanDraft(goal=goal, steps=steps, source="llm")

    def fallback(self, goal: str) -> TaskPlanDraft:
        """Return a safe single-step plan."""

        clean_goal = _clean_text(goal, limit=1200) or "完成当前请求"
        return TaskPlanDraft(
            goal=clean_goal,
            steps=[PlannedTaskStep(title="完成当前请求")],
            source="fallback",
        )


def _planner_system_prompt() -> str:
    return (
        "You are Codepilot Task Planner.\n"
        "Create a short plan for a local coding agent. Output JSON only.\n"
        "Schema: {\"goal\": string, \"steps\": [{\"title\": string, "
        "\"kind\": \"investigate|edit|verify|summarize|other\", "
        "\"acceptance\": string|null, \"verification_hint\": string|null}]}.\n"
        "Use 2-5 steps for implementation tasks and at most 6 steps. "
        "Keep steps concrete and executable. Do not include markdown."
    )


def _assistant_text(message: AssistantMessage) -> str:
    return "".join(
        block.text
        for block in message.content
        if isinstance(block, TextContent)
    ).strip()


def _loads_json_object(text: str) -> object:
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


def _clean_text(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


__all__ = ["PlannedTaskStep", "TaskPlanDraft", "TaskPlanner"]
