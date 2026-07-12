from __future__ import annotations

# 新手导读：skills.py 负责发现 Markdown skill，并把它们变成命令、索引和按需加载工具。
# 关注点：skill 是轻量扩展方式，不需要写 Python 代码。

"""技能加载器：发现工作区中的 .md 技能文件，启动时只暴露索引，正文按需加载。"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codepilot.protocols.commands import RegisteredCommand
from codepilot.tools import (
    ConcurrencyPolicy,
    DataclassCodec,
    OutputLimits,
    OutputTrustPolicy,
    TextContent,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolHandlerError,
    ToolPolicy,
    ToolRegistration,
    ToolSpec,
)

from .types import LoadedExtensions, SkillSpec

SKILL_LOADER_TOOL_NAME = "load_skill"


def discover_skill_paths(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> list[Path]:
    """发现技能文件路径：扫描默认目录和配置路径中的 .md 文件。"""
    workspace = Path(workspace_dir)
    paths: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        paths.append(resolved)

    default_dir = workspace / ".codepilot" / "skills"
    if default_dir.exists() and default_dir.is_dir():
        for path in sorted(default_dir.glob("*.md")):
            _add(path)

    for raw in configured_paths or []:
        target = Path(raw)
        if not target.is_absolute():
            target = workspace / raw
        if target.exists() and target.is_dir():
            for path in sorted(target.glob("*.md")):
                _add(path)
        elif target.exists() and target.is_file() and target.suffix.lower() == ".md":
            _add(target)

    return paths


def load_skills(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> LoadedExtensions:
    """加载所有技能：解析 .md 文件，注册命令，并生成紧凑技能索引。"""
    result = LoadedExtensions()
    seen_cmds: dict[str, str] = {}
    for path in discover_skill_paths(workspace_dir, configured_paths=configured_paths):
        try:
            raw_text = path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue
            meta, text = _parse_skill_frontmatter(raw_text)
            title = str(meta.get("name") or _extract_title(text) or path.stem).strip()
            if not title:
                title = path.stem
            cmd = str(meta.get("command") or f"skill:{_slugify(title)}").strip().lstrip("/")
            if not cmd:
                cmd = f"skill:{_slugify(path.stem)}"
            desc = str(meta.get("description") or f"Run skill: {title}").strip()
            skill = SkillSpec(
                name=title,
                command_name=cmd,
                description=desc,
                content=text,
                source_path=str(path),
            )
            if cmd in seen_cmds:
                result.diagnostics.append(f"skill command conflict: /{cmd} from {path} overrides {seen_cmds[cmd]}")
            seen_cmds[cmd] = str(path)

            result.skills.append(skill)
            result.commands[cmd] = RegisteredCommand(
                name=cmd,
                description=desc,
                source="skill",
                handler=lambda ctx, _skill=skill: _render_skill_prompt(_skill, ctx.raw_text),
            )
            result.loaded_paths.append(str(path))
        except Exception as exc:
            result.errors.append(f"{path}: {exc}")
    if result.skills:
        result.prompt_guidelines.append(
            "Skills are available in the Available Skills index; use load_skill "
            "with a listed name or command before following a skill workflow."
        )
        result.append_prompts.append(_render_skill_index(result.skills))
        result.tools.append(_create_skill_loader_tool(result.skills))
    return result


def _render_skill_index(skills: list[SkillSpec]) -> str:
    """渲染启动提示词中的紧凑技能目录，不包含 skill 正文。"""

    lines = [
        "## Available Skills",
        (
            "These skills are discoverable capabilities. When a task clearly matches "
            "one, call load_skill with its name or command to load the full workflow."
        ),
    ]
    for skill in skills:
        lines.append(f"- /{skill.command_name}: {skill.name} - {skill.description}")
    return "\n".join(lines)


@dataclass(frozen=True)
class _SkillLoadInput:
    name: str


@dataclass(frozen=True)
class _SkillLoadOutput:
    skill: str
    command: str
    content: str


def _create_skill_loader_tool(skills: list[SkillSpec]) -> ToolRegistration:
    """创建模型可调用的 canonical skill 正文加载工具。"""

    lookup: dict[str, SkillSpec] = {}
    for skill in skills:
        lookup[_skill_lookup_key(skill.name)] = skill
        lookup[_skill_lookup_key(skill.command_name)] = skill

    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "Skill name or slash command from the Available Skills index.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "command": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["skill", "command", "content"],
        "additionalProperties": False,
    }

    class Resolver:
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("skill.load",),
                    resources=(),
                    effects=frozenset(),
                    risk="low",
                    reason="Load one already discovered skill workflow",
                ),
            )

    async def handler(input: _SkillLoadInput, context) -> _SkillLoadOutput:
        context.cancellation.raise_if_cancelled()
        skill = lookup.get(_skill_lookup_key(input.name))
        if skill is None:
            raise ToolHandlerError(
                "skill.not_found",
                f"Skill not found: {input.name}",
                details={
                    "requested": input.name,
                    "available": [item.command_name for item in skills],
                },
            )
        return _SkillLoadOutput(
            skill=skill.name,
            command=skill.command_name,
            content=_render_loaded_skill(skill),
        )

    class Renderer:
        def render(self, data):
            return (TextContent(text=data["content"]),)

    input_codec = DataclassCodec(_SkillLoadInput, input_schema)
    output_codec = DataclassCodec(_SkillLoadOutput, output_schema)
    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(
            SKILL_LOADER_TOOL_NAME,
            (
                "Load the full Markdown workflow for one discovered skill. Use when the current task "
                "matches an entry in the Available Skills index. The name must identify a listed skill "
                "or slash command. Returns the selected skill name, command, and workflow content."
            ),
            input_schema,
            output_schema,
        ),
        category="external",
        source="skill",
        owner="skill:loader",
        policy=ToolPolicy(
            allowed_modes=frozenset({"plan", "execute", "unrestricted"}),
            declared_effects=frozenset(),
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(5_000, 5_000),
            concurrency=ConcurrencyPolicy(mode="parallel"),
            output_limits=OutputLimits(max_content_bytes=256_000),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=input_codec,
        output_codec=output_codec,
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


def _extract_title(text: str) -> str | None:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()
    return None


def _parse_skill_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip().lower()
        val = v.strip().strip("'").strip('"')
        if key and val:
            meta[key] = val
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "skill"


def _skill_lookup_key(value: object) -> str:
    return str(value or "").strip().lstrip("/").lower()


def _render_loaded_skill(skill: SkillSpec) -> str:
    return (
        f"Loaded skill {skill.name} (command: /{skill.command_name}).\n"
        "Follow the skill content below and produce actionable results.\n\n"
        f"{skill.content}"
    )


def _render_skill_prompt(skill: SkillSpec, raw_text: str) -> str:
    cmd_text = raw_text.strip() if raw_text else f"/{skill.command_name}"
    return (
        f"Applied skill {skill.name} (command: {cmd_text}).\n"
        "Follow the skill content below and produce actionable results.\n\n"
        f"{skill.content}"
    )
