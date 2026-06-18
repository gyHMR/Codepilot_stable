from __future__ import annotations

"""Runtime system prompt assembly."""

from pathlib import Path

from codepilot.tools import AgentTool

from .context_builder import RuntimeContextSources
from .system_prompt import SystemPromptBuildOptions, build_system_prompt


def build_runtime_system_prompt(
    *,
    base_system_prompt: str,
    tools: list[AgentTool],
    context_sources: RuntimeContextSources,
    workspace: Path,
) -> str:
    return build_system_prompt(
        SystemPromptBuildOptions(
            custom_prompt=base_system_prompt or None,
            selected_tools=_canonical_tool_names(tools),
            tool_snippets=context_sources.tool_snippets,
            prompt_guidelines=context_sources.prompt_guidelines,
            append_system_prompt=context_sources.append_system_prompt,
            memory_text=context_sources.memory_text,
            cwd=workspace,
        )
    )


def _canonical_tool_names(tools: list[AgentTool]) -> list[str]:
    aliases = {"list_dir": "ls", "read_file": "read", "write_file": "write"}
    names: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        name = aliases.get(tool.name, tool.name)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names
