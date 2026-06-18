from __future__ import annotations

"""Retry helpers for session turns."""

import asyncio
from typing import Awaitable, Callable

from codepilot.core import AgentMessage
from codepilot.llm.types import AssistantMessage


async def run_with_retry(
    *,
    op: Callable[[], Awaitable[list[AgentMessage]]],
    retry_enabled: bool,
    max_retries: int,
    retry_base_delay_ms: int,
    get_final_assistant: Callable[[], AssistantMessage | None],
    append_event: Callable[[dict], None],
) -> list[AgentMessage]:
    attempts = max_retries + 1 if retry_enabled else 1
    last: list[AgentMessage] | None = None

    for attempt in range(attempts):
        messages = await op()
        last = messages

        final_assistant = get_final_assistant()
        if not should_retry(final_assistant) or attempt >= attempts - 1:
            return messages

        delay_ms = int(retry_base_delay_ms * (2**attempt))
        append_event(
            {
                "type": "auto_retry_start",
                "attempt": attempt + 1,
                "max_attempts": attempts,
                "delay_ms": delay_ms,
                "error_message": final_assistant.error_message if final_assistant else "",
            }
        )
        await asyncio.sleep(delay_ms / 1000.0)

    return last or []


def should_retry(message: AssistantMessage | None) -> bool:
    if message is None:
        return False
    if message.stop_reason != "error":
        return False
    if message.error_info is not None:
        return message.error_info.retryable
    error_text = (message.error_message or "").lower()
    if "invalid_api_key" in error_text or "authentication" in error_text or "unauthorized" in error_text:
        return False
    return True
