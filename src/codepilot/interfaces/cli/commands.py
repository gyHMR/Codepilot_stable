from __future__ import annotations

"""CLI slash-command adapter.

This module is intentionally thin: CLI parses user text and delegates the
application command to RuntimeGateway. Command semantics live in sessions;
runtime only dispatches the action and passes read-only capability views.
"""

from typing import TYPE_CHECKING, Any

from codepilot.runtime.actions import CommandFinishedFrame, CommandSubmitted, FailedFrame

if TYPE_CHECKING:
    from codepilot.runtime.gateway import RuntimeGateway


async def handle_cli_command(
    runtime: "RuntimeGateway",
    session_id: str,
    text: str,
) -> Any:
    """Submit a CLI slash command through the runtime application boundary."""

    async for frame in runtime.dispatch(session_id, CommandSubmitted(text=text)):
        if isinstance(frame, CommandFinishedFrame):
            return frame.record
        if isinstance(frame, FailedFrame):
            error = frame.error
            if isinstance(error, dict):
                raise RuntimeError(str(error.get("message") or error.get("code")))
            raise RuntimeError(str(error))
    raise RuntimeError("Runtime command finished without a result")


__all__ = [
    "handle_cli_command",
]
