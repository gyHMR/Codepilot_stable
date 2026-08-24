"""提供 Session 事件的 SSE 实时订阅与断线重放。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse

from ..events import WebEvent


router = APIRouter(prefix="/api/sessions")


@router.get("/{session_id}/events")
async def session_events(
    request: Request,
    session_id: str,
    last_event_id_header: str | None = Header(None, alias="Last-Event-ID"),
    last_event_id_query: str | None = Query(None, alias="last_event_id"),
) -> StreamingResponse:
    service = request.app.state.web_service
    await service.ensure_open(session_id)
    hub = service.events_for(session_id)
    last_event_id = last_event_id_header or last_event_id_query

    async def stream() -> AsyncIterator[str]:
        replay, queue = hub.subscribe_after(last_event_id)
        if last_event_id is None or replay.expired:
            yield format_sse(
                WebEvent(
                    event_id=uuid4().hex,
                    session_id=session_id,
                    type="session.snapshot",
                    sequence=max(1, hub.latest_sequence + 1),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    data={
                        **service.projection(session_id),
                        "reason": (
                            "initial_subscription"
                            if last_event_id is None
                            else "event_history_expired"
                        ),
                    },
                )
            )
        if not replay.expired:
            for event in replay.events:
                yield format_sse(event)

        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield format_sse(event)
                if event.type == "sync_required" and hub.should_close(queue):
                    return
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def format_sse(event: WebEvent) -> str:
    payload = json.dumps(event.model_dump(), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.type}\ndata: {payload}\n\n"


__all__ = ["format_sse", "router"]
