from __future__ import annotations

"""Load and register tools for one opened runtime session."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codepilot.extensions import load_extensions, load_skills
from codepilot.extensions.mcp import MCPManager, create_mcp_manager
from codepilot.protocols.commands import RegisteredCommand
from codepilot.tools.builtins import create_builtin_registrations
from codepilot.tools.contracts import ToolRegistration
from codepilot.tools.registry import ToolRegistry
from codepilot.tools.sandbox import ShellExecutionPolicy

from .config import RuntimeConfig
from .opening import SessionOpenIntent


@dataclass
class RuntimeTools:
    registry: ToolRegistry
    mcp_manager: MCPManager | None = None
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[Any] = field(default_factory=list)
    after_prompt_hooks: list[Any] = field(default_factory=list)
    prompt_guidelines: list[str] = field(default_factory=list)
    append_prompts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_runtime_tools(
    workspace: Path,
    intent: SessionOpenIntent,
    config: RuntimeConfig,
) -> RuntimeTools:
    warnings: list[str] = []
    loaded_extensions = load_extensions(workspace, configured_paths=config.extension_paths)
    loaded_skills = load_skills(workspace, configured_paths=config.skill_paths)
    warnings.extend(f"extension: {error}" for error in loaded_extensions.errors)
    warnings.extend(f"skill: {error}" for error in loaded_skills.errors)
    warnings.extend(f"skill: {item}" for item in loaded_skills.diagnostics)

    mcp_manager = create_mcp_manager(
        config.mcp_servers,
        transport_factory=intent.mcp_transport_factory,
    )

    builtin_tools = create_builtin_registrations(
        workspace,
        enabled_names=config.enabled_builtin_tools,
        edit_require_unique_match=config.edit_require_unique_match,
        shell_policy=ShellExecutionPolicy(
            timeout_seconds=config.shell_timeout_seconds,
            max_timeout_seconds=config.shell_max_timeout_seconds,
            stdout_limit=config.shell_stdout_limit,
            stderr_limit=config.shell_stderr_limit,
            allowed_env=tuple(config.shell_allowed_env or ()),
        ),
    )

    registry = ToolRegistry()
    registry.reserve(
        {
            "propose_plan",
            "create_build_plan",
            "update_plan_progress",
            "close_plan",
            "request_user_input",
            "list_exploration_agents",
            "dispatch_exploration",
        }
    )
    _register_registration_groups(
        registry,
        builtin_tools,
        source="builtin",
        warnings=warnings,
    )
    _register_registration_groups(
        registry,
        list(intent.tools),
        source="caller",
        warnings=warnings,
    )
    _register_registration_groups(
        registry,
        loaded_skills.tools,
        source="skill",
        warnings=warnings,
    )
    _register_registration_groups(
        registry,
        loaded_extensions.tools,
        source="extension",
        warnings=warnings,
    )
    _validate_skill_requirements(
        loaded_skills.skills,
        registry=registry,
        configured_mcp_servers={
            str(item.get("name") or "").strip()
            for item in config.mcp_servers or []
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        },
        warnings=warnings,
    )
    return RuntimeTools(
        registry=registry,
        mcp_manager=mcp_manager if mcp_manager.configured else None,
        commands={**loaded_skills.commands, **loaded_extensions.commands},
        before_prompt_hooks=[*loaded_extensions.before_prompt_hooks, *loaded_skills.before_prompt_hooks],
        after_prompt_hooks=[*loaded_extensions.after_prompt_hooks, *loaded_skills.after_prompt_hooks],
        prompt_guidelines=[
            *(config.prompt_guidelines or []),
            *loaded_extensions.prompt_guidelines,
            *loaded_skills.prompt_guidelines,
        ],
        append_prompts=[
            *([config.append_system_prompt] if config.append_system_prompt else []),
            *loaded_extensions.append_prompts,
            *loaded_skills.append_prompts,
            *(_debug_prompt_sources(loaded_extensions, loaded_skills) if config.prompt_debug_sources else []),
        ],
        warnings=warnings,
    )


def _register_registration_groups(
    registry: ToolRegistry,
    registrations: list[ToolRegistration],
    *,
    source: str,
    warnings: list[str],
) -> None:
    groups: dict[str, list[ToolRegistration]] = {}
    for registration in registrations:
        if not isinstance(registration, ToolRegistration):
            raise TypeError(
                f"{source} tools must be ToolRegistration values, got "
                f"{type(registration).__name__}"
            )
        if registration.source != source:
            raise ValueError(
                f"{source} tool '{registration.spec.name}' declares source "
                f"'{registration.source}'"
            )
        groups.setdefault(registration.owner, []).append(registration)

    for owner, items in groups.items():
        if source == "builtin":
            registry.register_batch(items, owner=owner)
            continue
        for item in items:
            try:
                registry.register(item)
            except Exception as exc:
                warnings.append(
                    f"{source} owner '{owner}' tool '{item.spec.name}' registration failed: {exc}"
                )


def _debug_prompt_sources(extensions: Any, skills: Any) -> list[str]:
    lines: list[str] = ["## Prompt Sources", "### extensions"]
    lines.extend([f"- {path}" for path in extensions.loaded_paths] or ["- (none)"])
    lines.append("### skills")
    lines.extend([f"- {path}" for path in skills.loaded_paths] or ["- (none)"])
    errors = [*extensions.errors, *skills.errors]
    if errors:
        lines.append("### errors")
        lines.extend(f"- {error}" for error in errors)
    if skills.diagnostics:
        lines.append("### diagnostics")
        lines.extend(f"- {item}" for item in skills.diagnostics)
    return ["\n".join(lines)]


def _validate_skill_requirements(
    skills: list[Any],
    *,
    registry: ToolRegistry,
    configured_mcp_servers: set[str],
    warnings: list[str],
) -> None:
    available_tools = {
        entry.spec.name for entry in registry.catalog_snapshot().entries
    }
    for package in skills:
        manifest = package.manifest
        missing_tools = sorted(set(manifest.required_tools) - available_tools)
        missing_mcp = sorted(set(manifest.required_mcp) - configured_mcp_servers)
        if missing_tools:
            warnings.append(
                f"skill '{manifest.name}' requires unavailable tools: "
                + ", ".join(missing_tools)
            )
        if missing_mcp:
            warnings.append(
                f"skill '{manifest.name}' requires unconfigured MCP servers: "
                + ", ".join(missing_mcp)
            )


__all__ = ["RuntimeTools", "build_runtime_tools"]
