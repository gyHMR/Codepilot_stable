from __future__ import annotations

from typing import Any


def describe_web_contract() -> dict[str, Any]:
    """Return the planned Web Console contract without starting a server."""

    return {
        "transport": ["http", "websocket"],
        "entrypoint": "codepilot.interfaces.web",
        "delegates_to": ["codepilot.runtime", "codepilot.sessions", "codepilot.tools"],
        "responsibilities": [
            "display_chat_messages",
            "display_tool_events",
            "display_file_tree",
            "display_diffs",
            "display_command_output",
            "submit_tool_approval",
        ],
        "non_responsibilities": [
            "llm_provider_calls",
            "agent_loop",
            "filesystem_mutation",
            "shell_execution",
            "session_persistence",
        ],
    }
