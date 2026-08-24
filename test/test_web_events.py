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
        "type": "assistant.delta",
        "sequence": 3,
        "timestamp": "2026-07-12T00:00:00Z",
        "data": {"delta": "hi"},
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

    assert event.type == "assistant.delta"
    assert event.data == {"delta": "hi"}


def test_tool_call_deltas_are_projected_as_one_stable_activity() -> None:
    from codepilot.interfaces.web.events import runtime_frame_to_event
    from codepilot.runtime.actions import ProgressFrame

    event = runtime_frame_to_event(
        ProgressFrame(event={
            "type": "message_update",
            "assistant_message_event": {
                "type": "tool_call_delta",
                "toolCall": {"id": "call-1", "name": "read", "arguments": {"path": "a.py"}},
            },
        }),
        session_id="s1", sequence=2, event_id_factory=lambda: "evt-2",
    )
    assert event.type == "activity.updated"
    assert event.data["activity_id"] == "call-1"
    assert event.data["name"] == "read"


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
    assert event.type == "approval.requested"
    assert sorted(event.data["effects"]) == ["filesystem_read", "process_spawn"]


def test_runtime_frame_kinds_have_stable_web_types() -> None:
    from codepilot.interfaces.web.events import runtime_frame_to_event

    cases = [
        (SimpleNamespace(kind="approval_required", approval={"id": "a1"}), "approval.requested"),
        (SimpleNamespace(kind="run_paused", record=SimpleNamespace(run_id="r1"), checkpoint={}), "run.status_changed"),
        (SimpleNamespace(kind="run_finished", record=SimpleNamespace(run_id="r1")), "run.completed"),
        (SimpleNamespace(kind="command_finished", record=SimpleNamespace(run_id=None)), "run.completed"),
        (SimpleNamespace(kind="cancelled", session_id="s1", cancelled=True, reason="user"), "run.completed"),
        (SimpleNamespace(kind="failed", error=RuntimeError("secretless failure")), "error"),
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


def test_failed_core_reason_is_preserved_in_web_error_event() -> None:
    from codepilot.core.contracts import CoreReason
    from codepilot.interfaces.web.events import runtime_frame_to_event
    from codepilot.runtime.actions import FailedFrame
    from codepilot.runtime.errors import runtime_error_payload

    event = runtime_frame_to_event(
        FailedFrame(
            error=runtime_error_payload(
                CoreReason(
                    "plan.reconciliation_exhausted",
                    message="Plan reconciliation was not completed.",
                )
            )
        ),
        session_id="s1",
        sequence=1,
    )

    assert event.type == "error"
    assert event.data["code"] == "plan.reconciliation_exhausted"
    assert event.data["source"] == "core"
    assert event.data["message"] == "Plan reconciliation was not completed."


def test_user_input_pause_projects_interaction_event() -> None:
    from codepilot.interfaces.web.events import runtime_frame_to_event

    frame = SimpleNamespace(
        kind="run_paused",
        record=SimpleNamespace(run_id="r1"),
        checkpoint={
            "waiting": {
                "kind": "user_input",
                "request_id": "question-1",
                "payload": {"prompt": "Choose", "options": ["A", "B"]},
            }
        },
    )
    event = runtime_frame_to_event(frame, session_id="s1", sequence=1)
    assert event.type == "interaction.requested"
    assert event.data["status"] == "waiting_user"
    assert event.data["request_id"] == "question-1"


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


def test_event_hub_overflow_requests_resync_instead_of_silent_unsubscribe() -> None:
    async def run_case() -> None:
        from codepilot.interfaces.web.events import EventHub, WebEvent

        hub = EventHub(subscriber_capacity=1)
        queue = hub.subscribe()
        first = WebEvent(event_id="e1", session_id="s1", type="assistant.delta", sequence=1, timestamp="now", data={"delta": "a"})
        second = WebEvent(event_id="e2", session_id="s1", type="assistant.delta", sequence=2, timestamp="now", data={"delta": "b"})
        await hub.publish(first)
        await hub.publish(second)

        event = await queue.get()
        assert event.type == "sync_required"
        assert event.data["reason"] == "subscriber_queue_overflow"
        assert hub.should_close(queue) is True

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
