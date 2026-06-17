from __future__ import annotations

"""Build runtime context sources consumed by prompt assembly."""

from dataclasses import dataclass
from pathlib import Path

from codepilot.extensions.types import LoadedExtensions
from codepilot.sessions.memory import load_global_memory

from .config_loader import RuntimeConfig


@dataclass(frozen=True)
class RuntimeContextSources:
    prompt_guidelines: list[str]
    append_system_prompt: str | None
    tool_snippets: dict[str, str] | None
    memory_text: str


def build_runtime_context_sources(
    workspace: Path,
    config: RuntimeConfig,
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> RuntimeContextSources:
    prompt_guidelines = [
        *(config.prompt_guidelines or []),
        *loaded_extensions.prompt_guidelines,
        *loaded_skills.prompt_guidelines,
    ]
    prompt_guidelines.extend([f"[skill-diagnostic] {d}" for d in loaded_skills.diagnostics])

    append_system_prompt = config.append_system_prompt
    merged_append_sections = [*loaded_extensions.append_prompts, *loaded_skills.append_prompts]
    if merged_append_sections:
        extension_append = "\n\n".join(merged_append_sections)
        append_system_prompt = (
            f"{append_system_prompt}\n\n{extension_append}".strip()
            if append_system_prompt
            else extension_append
        )

    if config.prompt_debug_sources:
        debug_lines = _build_prompt_debug_lines(loaded_extensions, loaded_skills)
        append_system_prompt = (
            f"{append_system_prompt}\n\n" + "\n".join(debug_lines)
            if append_system_prompt
            else "\n".join(debug_lines)
        )

    return RuntimeContextSources(
        prompt_guidelines=prompt_guidelines,
        append_system_prompt=append_system_prompt,
        tool_snippets=config.tool_snippets,
        memory_text=load_global_memory(workspace),
    )


def _build_prompt_debug_lines(
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> list[str]:
    debug_lines: list[str] = ["## Prompt Sources", "### extensions"]
    debug_lines.extend([f"- {p}" for p in loaded_extensions.loaded_paths] or ["- (none)"])
    debug_lines.append("### skills")
    debug_lines.extend([f"- {p}" for p in loaded_skills.loaded_paths] or ["- (none)"])
    if loaded_extensions.errors or loaded_skills.errors:
        debug_lines.append("### errors")
        debug_lines.extend([f"- {e}" for e in [*loaded_extensions.errors, *loaded_skills.errors]])
    if loaded_skills.diagnostics:
        debug_lines.append("### diagnostics")
        debug_lines.extend([f"- {d}" for d in loaded_skills.diagnostics])
    return debug_lines
