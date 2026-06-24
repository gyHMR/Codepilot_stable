from __future__ import annotations

"""Lightweight LLM planner for plan-and-execute task control.

The planner is intentionally small: it asks the model for a bounded JSON plan,
validates the result, and falls back to a single-step plan when planning fails.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

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
from .types import AgentMessage


_MAX_PLANNED_STEPS = 6
_MAX_FIELD_CHARS = 240
_VALID_STEP_KINDS = {"investigate", "edit", "verify", "summarize", "other"}


@dataclass(frozen=True)
class PlannedTaskStep:
    """A model-proposed task step."""

    title: str
    kind: str = "other"
    acceptance: str | None = None
    verification_hint: str | None = None


@dataclass(frozen=True)
class TaskPlanDraft:
    """Validated planner output used to initialize TaskState."""

    goal: str
    steps: list[PlannedTaskStep] = field(default_factory=list)
    source: str = "fallback"


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
            if kind not in _VALID_STEP_KINDS:
                kind = "other"
            steps.append(
                PlannedTaskStep(
                    title=title,
                    kind=kind,
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
