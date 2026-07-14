"""将 Skill 清单、提示和资源装配为运行时扩展能力。"""

from __future__ import annotations

"""Expose validated Skill Packages through commands and canonical tools."""

from dataclasses import dataclass
from pathlib import Path

from codepilot.protocols.commands import CommandOutcome, RegisteredCommand
from codepilot.tools import (
    ConcurrencyPolicy,
    DataclassCodec,
    OutputLimits,
    OutputTrustPolicy,
    TextContent,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolHandlerError,
    ToolPolicy,
    ToolRegistration,
    ToolResource,
    ToolSpec,
)

from .skills import SkillCatalog, SkillPackage, SkillPackageError, load_skill_catalog
from .types import LoadedExtensions


SKILL_LOADER_TOOL_NAME = "load_skill"
SKILL_RESOURCE_TOOL_NAME = "read_skill_resource"
_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def load_skills(
    workspace_dir: str | Path,
    configured_paths: list[str] | None = None,
) -> LoadedExtensions:
    catalog = load_skill_catalog(workspace_dir, configured_paths=configured_paths)
    result = LoadedExtensions(
        skills=list(catalog.packages),
        diagnostics=list(catalog.diagnostics),
        loaded_paths=list(catalog.loaded_paths),
        errors=list(catalog.errors),
    )
    if not catalog.packages:
        return result
    result.prompt_guidelines.append(
        "Skills are instruction packages. When a task matches the Available Skills index, "
        "call load_skill before following that workflow. Use read_skill_resource only for "
        "package resources named by the loaded instructions."
    )
    result.append_prompts.append(_render_skill_index(catalog))
    result.tools.extend(
        (
            _create_skill_loader_tool(catalog),
            _create_skill_resource_tool(catalog),
        )
    )
    for package in catalog.packages:
        command = package.manifest.command
        if command:
            result.commands[command] = RegisteredCommand(
                name=command,
                description=package.manifest.description,
                source="skill",
                handler=lambda ctx, _package=package: _skill_command(_package, ctx.raw_text),
            )
    return result


def _render_skill_index(catalog: SkillCatalog) -> str:
    lines = [
        "## Available Skills",
        "Call load_skill with a listed name before applying its workflow.",
    ]
    for package in catalog.packages:
        manifest = package.manifest
        command = f" /{manifest.command}" if manifest.command else ""
        lines.append(
            f"- {manifest.name}{command} (v{manifest.version}, {package.source}): "
            f"{manifest.description}"
        )
    return "\n".join(lines)


def _skill_command(package: SkillPackage, raw_text: str) -> CommandOutcome:
    _, _, request = str(raw_text or "").partition(" ")
    request = request.strip() or "Apply this skill to the current task."
    return CommandOutcome(
        prompt=(
            f"Use the available skill '{package.manifest.name}' for this request. "
            f"Call {SKILL_LOADER_TOOL_NAME} with that exact name before following its workflow.\n\n"
            f"Request: {request}"
        )
    )


@dataclass(frozen=True)
class _SkillLoadInput:
    name: str


@dataclass(frozen=True)
class _SkillLoadOutput:
    skill: str
    version: str
    command: str | None
    source: str
    trust: str
    digest: str
    content: str
    resources: list[str]


@dataclass(frozen=True)
class _SkillResourceInput:
    skill: str
    path: str


@dataclass(frozen=True)
class _SkillResourceOutput:
    skill: str
    path: str
    content: str


def _create_skill_loader_tool(catalog: SkillCatalog) -> ToolRegistration:
    input_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "Exact skill name or slash command from the Available Skills index.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "version": {"type": "string"},
            "command": {"type": ["string", "null"]},
            "source": {"type": "string"},
            "trust": {"type": "string"},
            "digest": {"type": "string"},
            "content": {"type": "string"},
            "resources": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "skill",
            "version",
            "command",
            "source",
            "trust",
            "digest",
            "content",
            "resources",
        ],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input: _SkillLoadInput, request):
            package = _require_package(catalog, input.name)
            resource = ToolResource(f"skill:///{package.manifest.name}/SKILL.md")
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("skill.load",),
                    resources=(resource,),
                    effects=frozenset({"filesystem_read"}),
                    risk="low",
                    reason="Load one explicitly configured skill workflow",
                ),
            )

    async def handler(input: _SkillLoadInput, context) -> _SkillLoadOutput:
        context.cancellation.raise_if_cancelled()
        package = _require_package(catalog, input.name)
        if _skill_mode(context.request.mode) not in package.manifest.allowed_modes:
            raise ToolHandlerError(
                "skill.mode_denied",
                f"Skill {package.manifest.name} is not available in {context.request.mode} mode",
                details={"allowed_modes": sorted(package.manifest.allowed_modes)},
            )
        try:
            content = package.read_instructions()
            resources = list(package.list_resources())
        except SkillPackageError as exc:
            raise ToolHandlerError("skill.read_failed", str(exc)) from exc
        context.effects.report(
            ToolEffect(
                kind="filesystem_read",
                resource=ToolResource(f"skill:///{package.manifest.name}/SKILL.md"),
                operation="read skill instructions",
                status="completed",
                certainty="observed",
            )
        )
        manifest = package.manifest
        return _SkillLoadOutput(
            skill=manifest.name,
            version=manifest.version,
            command=manifest.command,
            source=package.source,
            trust=package.trust,
            digest=package.digest,
            content=_render_loaded_skill(package, content),
            resources=resources,
        )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["content"]),)

    return ToolRegistration(
        version="1.0.0",
        implementation_version="2",
        spec=ToolSpec(
            SKILL_LOADER_TOOL_NAME,
            "Load one validated Skill Package workflow by exact name or slash command. Use after matching the current request to the Available Skills index. Returns the instructions and a bounded list of optional package resources; loading a skill never executes its scripts.",
            input_schema,
            output_schema,
        ),
        category="external",
        source="skill",
        owner="skill:runtime",
        policy=_read_policy(max_content_bytes=300_000),
        input_codec=DataclassCodec(_SkillLoadInput, input_schema),
        output_codec=DataclassCodec(_SkillLoadOutput, output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


def _create_skill_resource_tool(catalog: SkillCatalog) -> ToolRegistration:
    input_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {
            "skill": {"type": "string", "minLength": 1},
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Resource path returned by load_skill under references/, scripts/, or assets/.",
            },
        },
        "required": ["skill", "path"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": _DRAFT,
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["skill", "path", "content"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input: _SkillResourceInput, request):
            _ = request
            package = _require_package(catalog, input.skill)
            try:
                path = package.resolve_resource(input.path)
                relative = path.relative_to(package.root.resolve(strict=True)).as_posix()
            except SkillPackageError as exc:
                raise ValueError(str(exc)) from exc
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("skill.resource.read",),
                    resources=(ToolResource(f"skill:///{package.manifest.name}/{relative}"),),
                    effects=frozenset({"filesystem_read"}),
                    risk="low",
                    reason="Read one declared Skill Package resource",
                ),
            )

    async def handler(input: _SkillResourceInput, context) -> _SkillResourceOutput:
        context.cancellation.raise_if_cancelled()
        package = _require_package(catalog, input.skill)
        if _skill_mode(context.request.mode) not in package.manifest.allowed_modes:
            raise ToolHandlerError(
                "skill.mode_denied",
                f"Skill {package.manifest.name} is not available in {context.request.mode} mode",
            )
        try:
            path = package.resolve_resource(input.path)
            relative = path.relative_to(package.root.resolve(strict=True)).as_posix()
            content = package.read_resource(relative)
        except SkillPackageError as exc:
            raise ToolHandlerError("skill.resource_read_failed", str(exc)) from exc
        resource = ToolResource(f"skill:///{package.manifest.name}/{relative}")
        context.effects.report(
            ToolEffect(
                kind="filesystem_read",
                resource=resource,
                operation="read skill resource",
                status="completed",
                certainty="observed",
            )
        )
        return _SkillResourceOutput(
            skill=package.manifest.name,
            path=relative,
            content=content,
        )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["content"]),)

    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(
            SKILL_RESOURCE_TOOL_NAME,
            "Read one UTF-8 text resource from references/, scripts/, or assets/ in a previously discovered Skill Package. Use only paths returned by load_skill. This tool never executes scripts and rejects absolute paths, traversal, symbolic links, binary files, and oversized content.",
            input_schema,
            output_schema,
        ),
        category="external",
        source="skill",
        owner="skill:runtime",
        policy=_read_policy(max_content_bytes=256_000),
        input_codec=DataclassCodec(_SkillResourceInput, input_schema),
        output_codec=DataclassCodec(_SkillResourceOutput, output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


def _read_policy(*, max_content_bytes: int) -> ToolPolicy:
    return ToolPolicy(
        allowed_modes=frozenset({"plan", "execute", "unrestricted"}),
        declared_effects=frozenset({"filesystem_read"}),
        required_permissions=frozenset(),
        base_risk="low",
        approval="never",
        timeout=TimeoutPolicy(5_000, 5_000),
        concurrency=ConcurrencyPolicy(mode="parallel"),
        output_limits=OutputLimits(
            max_data_bytes=max_content_bytes + 64_000,
            max_content_bytes=max_content_bytes,
        ),
        # Skill paths are explicitly enabled instruction sources, unlike remote tool data.
        output_trust=OutputTrustPolicy(default_content_trust="trusted"),
    )


def _require_package(catalog: SkillCatalog, name: str) -> SkillPackage:
    package = catalog.find(name)
    if package is None:
        raise ToolHandlerError(
            "skill.not_found",
            f"Skill not found: {name}",
            details={"available": [item.manifest.name for item in catalog.packages]},
        )
    return package


def _skill_mode(runtime_mode: str) -> str:
    return "execute" if runtime_mode == "unrestricted" else runtime_mode


def _render_loaded_skill(package: SkillPackage, content: str) -> str:
    manifest = package.manifest
    return (
        f"Loaded skill {manifest.name} v{manifest.version} "
        f"(source: {package.source}, trust: {package.trust}).\n"
        "Follow the skill instructions below. Package scripts are resources only; any execution "
        "must be an explicit call to an available ToolRuntime-managed command or shell tool.\n\n"
        f"{content}"
    )


__all__ = [
    "SKILL_LOADER_TOOL_NAME",
    "SKILL_RESOURCE_TOOL_NAME",
    "load_skills",
]
