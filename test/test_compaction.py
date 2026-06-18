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
    from codepilot.sessions.compaction import COMPACTION_SYSTEM_PROMPT

    assert "当前任务目标" in COMPACTION_SYSTEM_PROMPT
    assert "关键文件" in COMPACTION_SYSTEM_PROMPT
    assert "失败原因" in COMPACTION_SYSTEM_PROMPT
    assert "验证" in COMPACTION_SYSTEM_PROMPT
    assert "下一步" in COMPACTION_SYSTEM_PROMPT


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


async def _run_history_preservation_case(tmp_path: Path) -> None:
    from codepilot.protocols import AssistantMessage, TextContent, UserMessage
    from codepilot.runtime.types import AgentSessionOptions
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.store import SessionStore

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
