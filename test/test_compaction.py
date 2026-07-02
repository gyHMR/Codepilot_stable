from __future__ import annotations

import sys
from pathlib import Path
import asyncio


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _test_model():
    from codepilot.protocols import Model

    return Model(
        id="test-model",
        name="Test Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


def test_agent_session_compacts_context_with_summary_builder(tmp_path: Path) -> None:
    asyncio.run(_run_compaction_case(tmp_path))


def test_compaction_preserves_full_session_history(tmp_path: Path) -> None:
    asyncio.run(_run_history_preservation_case(tmp_path))


def test_compaction_prompt_preserves_working_context_facts() -> None:
    from codepilot.sessions.context.compaction import COMPACTION_SYSTEM_PROMPT

    assert "当前任务目标" in COMPACTION_SYSTEM_PROMPT
    assert "关键文件" in COMPACTION_SYSTEM_PROMPT
    assert "失败原因" in COMPACTION_SYSTEM_PROMPT
    assert "验证" in COMPACTION_SYSTEM_PROMPT
    assert "下一步" in COMPACTION_SYSTEM_PROMPT


def test_context_compaction_decision_names_trigger_reason_and_retention() -> None:
    from codepilot.sessions.context.compaction import decide_context_compaction

    by_message_count = decide_context_compaction(
        message_count=8,
        estimated_tokens=200,
        max_context_messages=5,
        max_context_tokens=0,
        retain_recent_messages=3,
    )
    by_tokens = decide_context_compaction(
        message_count=8,
        estimated_tokens=1200,
        max_context_messages=0,
        max_context_tokens=1000,
        retain_recent_messages=3,
    )
    forced = decide_context_compaction(
        message_count=4,
        estimated_tokens=100,
        max_context_messages=0,
        max_context_tokens=0,
        retain_recent_messages=10,
        force=True,
    )
    too_short = decide_context_compaction(
        message_count=2,
        estimated_tokens=5000,
        max_context_messages=1,
        max_context_tokens=1,
        retain_recent_messages=5,
    )

    assert by_message_count.should_compact is True
    assert by_message_count.reason == "message_threshold"
    assert by_message_count.retain_recent_messages == 3
    assert by_message_count.over_message_limit is True
    assert by_tokens.should_compact is True
    assert by_tokens.reason == "token_threshold"
    assert by_tokens.over_token_limit is True
    assert forced.should_compact is True
    assert forced.reason == "overflow"
    assert forced.retain_recent_messages == 3
    assert too_short.should_compact is False
    assert too_short.reason == "not_enough_messages"


def test_llm_compaction_summary_builds_prompt_and_passes_api_key() -> None:
    asyncio.run(_run_llm_compaction_summary_case())


async def _run_llm_compaction_summary_case() -> None:
    from codepilot.protocols import AssistantMessage, TextContent, UserMessage
    from codepilot.sessions.context.compaction import (
        COMPACTION_SYSTEM_PROMPT,
        build_llm_compaction_summary,
    )

    captured = {}

    async def get_api_key(provider: str) -> str:
        captured["provider"] = provider
        return "secret-key"

    async def complete_fn(model, context, options):
        captured["model"] = model
        captured["context"] = context
        captured["options"] = options
        return AssistantMessage(content=[TextContent(text="compressed facts")])

    summary = await build_llm_compaction_summary(
        [UserMessage(content="keep this fact")],
        model=_test_model(),
        get_api_key=get_api_key,
        complete_fn=complete_fn,
    )

    assert summary == "compressed facts"
    assert captured["provider"] == "unit-test"
    assert captured["context"].system_prompt == COMPACTION_SYSTEM_PROMPT
    assert "keep this fact" in captured["context"].messages[0].content
    assert captured["options"].api_key == "secret-key"
    assert captured["options"].max_tokens == 2000


def test_agent_session_compacts_before_continue_run(tmp_path: Path) -> None:
    asyncio.run(_run_continue_pre_compaction_case(tmp_path))


def test_agent_session_finalizes_run_before_after_hooks(tmp_path: Path) -> None:
    asyncio.run(_run_finalization_lifecycle_case(tmp_path))


async def _run_compaction_case(tmp_path: Path) -> None:
    from codepilot.protocols import AssistantMessage, TextContent, UserMessage
    from codepilot.runtime.types import AgentSessionOptions
    from codepilot.sessions.session import AgentSession

    session = AgentSession(
        AgentSessionOptions(
            model=_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
            messages=[
                UserMessage(content="old user"),
                AssistantMessage(content=[TextContent(text="old assistant")]),
                UserMessage(content="recent user"),
                AssistantMessage(content=[TextContent(text="recent assistant")]),
            ],
            retain_recent_messages=2,
            summary_builder=lambda messages: f"summary-count={len(messages)}",
        )
    )

    try:
        await session._compact_context_if_needed(force=True)
        assert len(session.messages) == 3
        assert isinstance(session.messages[0], UserMessage)
        assert "summary-count=2" in str(session.messages[0].content)
        events = session.store.load_events()
        compacted = [event for event in events if event["type"] == "context_compacted"]
        assert len(compacted) == 1
        report = compacted[0]["report"]
        assert report["reason"] == "overflow"
        assert report["retained_messages"] == 2
        assert report["summary_created"] is True
        assert isinstance(report["tokens_before"], int)
        assert isinstance(report["tokens_after"], int)
    finally:
        session.close()


async def _run_continue_pre_compaction_case(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from codepilot.protocols import AgentRunResult, AssistantMessage, TextContent, UserMessage
    from codepilot.runtime.types import AgentSessionOptions
    from codepilot.sessions.session import AgentSession

    class FakeAgent:
        def __init__(self, messages):
            self.state = SimpleNamespace(
                messages=list(messages),
                system_prompt="sys",
                tools=[],
                model=_test_model(),
            )
            self.message_count_at_continue: int | None = None

        def set_messages(self, messages):
            self.state.messages = list(messages)

        def set_task_recovery_projection(self, task):
            _ = task

        async def continue_run(self, *, run_id=None):
            self.message_count_at_continue = len(self.state.messages)
            return AgentRunResult(
                run_id=run_id or "run_continue",
                session_id="session_continue",
                status="completed",
                stop_reason="final_answer",
                messages=[],
            )

    messages = [
        UserMessage(content="old user"),
        AssistantMessage(content=[TextContent(text="old assistant")]),
        UserMessage(content="recent user"),
        AssistantMessage(content=[TextContent(text="recent assistant")]),
    ]
    session = AgentSession(
        AgentSessionOptions(
            model=_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
            messages=messages,
            max_context_messages=3,
            retain_recent_messages=2,
            summary_builder=lambda older: f"summary-count={len(older)}",
            context_governance_enabled=False,
            memory_enabled=False,
        )
    )
    fake_agent = FakeAgent(messages)
    session.agent = fake_agent  # type: ignore[assignment]

    try:
        await session.continue_run(run_id="run_continue")
        assert fake_agent.message_count_at_continue == 3
        assert isinstance(fake_agent.state.messages[0], UserMessage)
        assert "summary-count=2" in str(fake_agent.state.messages[0].content)
    finally:
        session.close()


async def _run_finalization_lifecycle_case(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from codepilot.protocols import AgentRunResult
    from codepilot.runtime.types import AgentSessionOptions
    from codepilot.sessions.session import AgentSession

    observed_after_hooks: list[tuple[bool, int]] = []

    def after_hook(ctx) -> None:
        observed_after_hooks.append(
            (
                ctx.is_continue,
                len(ctx.session.store.load_run_results(limit=10)),
            )
        )

    class FakeAgent:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                messages=[],
                system_prompt="sys",
                tools=[],
                model=_test_model(),
            )

        def set_messages(self, messages):
            self.state.messages = list(messages)

        def set_task_recovery_projection(self, task):
            _ = task

        async def run(self, text, *, images=None, run_id=None):
            _ = text, images
            return AgentRunResult(
                run_id=run_id or "run_one",
                session_id="session_finalization",
                status="completed",
                stop_reason="final_answer",
                messages=[],
            )

        async def continue_run(self, *, run_id=None):
            return AgentRunResult(
                run_id=run_id or "run_two",
                session_id="session_finalization",
                status="completed",
                stop_reason="final_answer",
                messages=[],
            )

    session = AgentSession(
        AgentSessionOptions(
            model=_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
            memory_enabled=False,
            after_prompt_hooks=[after_hook],
        )
    )
    session.agent = FakeAgent()  # type: ignore[assignment]

    try:
        await session.run("first", run_id="run_one")
        await session.continue_run(run_id="run_two")

        assert observed_after_hooks == [(False, 1), (True, 2)]
        assert [item["run_id"] for item in session.store.load_run_results()] == [
            "run_one",
            "run_two",
        ]
    finally:
        session.close()


async def _run_history_preservation_case(tmp_path: Path) -> None:
    from codepilot.protocols import AssistantMessage, TextContent, UserMessage
    from codepilot.runtime.types import AgentSessionOptions
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.persistence.store import SessionStore

    store = SessionStore(tmp_path, "session_history")
    store.ensure_initialized(model_id="test-model", provider="unit-test", system_prompt="sys")
    store.append_context_message(UserMessage(content="old user"))
    store.append_context_message(AssistantMessage(content=[TextContent(text="old assistant")]))
    store.append_context_message(UserMessage(content="recent user"))
    store.append_context_message(AssistantMessage(content=[TextContent(text="recent assistant")]))
    original_entry_ids = store.list_entry_ids()

    session = AgentSession(
        AgentSessionOptions(
            model=_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
            session_id="session_history",
            retain_recent_messages=2,
            summary_builder=lambda messages: f"summary-count={len(messages)}",
        )
    )

    try:
        await session._compact_context_if_needed(force=True)
        assert len(session.messages) == 3
        assert store.list_entry_ids() == original_entry_ids
        assert len(store.load_session_messages()) == 4
    finally:
        session.close()
