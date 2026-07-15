from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_runtime_frame_conversion() -> None:
    from codepilot.interfaces.web.events import runtime_frame_to_event
    from codepilot.runtime.actions import ProgressFrame

    event = runtime_frame_to_event(
        ProgressFrame(event={"type": "text_delta", "delta": "hi"}),
        session_id="s1",
        sequence=3,
        event_id_factory=lambda: "evt-3",
        clock=lambda: "2026-07-12T00:00:00Z",
    )

    assert event.model_dump() == {
        "event_id": "evt-3",
        "session_id": "s1",
        "run_id": None,
        "type": "message_delta",
        "sequence": 3,
        "timestamp": "2026-07-12T00:00:00Z",
        "data": {"type": "text_delta", "delta": "hi"},
    }


def test_nested_runtime_message_update_is_projected_to_message_delta() -> None:
    from codepilot.interfaces.web.events import runtime_frame_to_event
    from codepilot.runtime.actions import ProgressFrame

    event = runtime_frame_to_event(
        ProgressFrame(
            event={
                "type": "message_update",
                "assistant_message_event": {"type": "text_delta", "delta": "hi"},
            }
        ),
        session_id="s1",
        sequence=1,
        event_id_factory=lambda: "evt-1",
    )

    assert event.type == "message_delta"
    assert event.data == {"type": "text_delta", "delta": "hi"}


def test_approval_projection_normalizes_web_fields() -> None:
    from codepilot.interfaces.web.events import runtime_frame_to_event

    event = runtime_frame_to_event(
        SimpleNamespace(
            kind="approval_required",
            approval={
                "approval_id": "a1",
                "risk": "high",
                "effects": frozenset({"process_spawn", "filesystem_read"}),
            },
        ),
        session_id="s1",
        sequence=1,
    )

    assert event.data["risk_level"] == "high"
    assert sorted(event.data["effects"]) == ["filesystem_read", "process_spawn"]


def test_runtime_frame_kinds_have_stable_web_types() -> None:
    from codepilot.interfaces.web.events import runtime_frame_to_event

    cases = [
        (SimpleNamespace(kind="approval_required", approval={"id": "a1"}), "approval_required"),
        (SimpleNamespace(kind="run_paused", record=SimpleNamespace(run_id="r1"), checkpoint={}), "run_paused"),
        (SimpleNamespace(kind="run_finished", record=SimpleNamespace(run_id="r1")), "run_finished"),
        (SimpleNamespace(kind="command_finished", record=SimpleNamespace(run_id=None)), "command_finished"),
        (SimpleNamespace(kind="cancelled", session_id="s1", cancelled=True, reason="user"), "cancelled"),
        (SimpleNamespace(kind="failed", error=RuntimeError("secretless failure")), "failed"),
    ]

    for index, (frame, expected) in enumerate(cases, start=1):
        event = runtime_frame_to_event(
            frame,
            session_id="s1",
            sequence=index,
            event_id_factory=lambda: f"evt-{index}",
        )
        assert event.type == expected
        assert isinstance(event.data, dict)


def test_event_hub_replays_and_reports_expired_ids() -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.events import EventHub, WebEvent

        hub = EventHub(capacity=2)
        for index in range(1, 4):
            await hub.publish(
                WebEvent(
                    event_id=f"e{index}", session_id="s1", type="progress",
                    sequence=index, timestamp="now", data={"index": index},
                )
            )

        replay = hub.replay_after("e2")
        assert [event.event_id for event in replay.events] == ["e3"]
        assert replay.expired is False
        assert hub.replay_after("e1").expired is True
        assert EventHub().replay_after("event-from-previous-process").expired is True

    asyncio.run(run_case())


def test_event_hub_broadcasts_to_two_subscribers() -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.events import EventHub, WebEvent

        hub = EventHub()
        first = hub.subscribe()
        second = hub.subscribe()
        event = WebEvent(
            event_id="e1", session_id="s1", type="connected",
            sequence=1, timestamp="now", data={},
        )

        await hub.publish(event)

        assert await first.get() == event
        assert await second.get() == event

    asyncio.run(run_case())


def test_event_hub_subscribe_after_has_no_replay_subscription_gap() -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.events import EventHub, WebEvent

        hub = EventHub()
        first = WebEvent(
            event_id="e1", session_id="s1", type="progress",
            sequence=1, timestamp="now", data={},
        )
        second = WebEvent(
            event_id="e2", session_id="s1", type="progress",
            sequence=2, timestamp="now", data={},
        )
        await hub.publish(first)

        replay, queue = hub.subscribe_after("e1")
        await hub.publish(second)

        assert replay.events == ()
        assert await queue.get() == second

    asyncio.run(run_case())
