from __future__ import annotations

import asyncio
import json
from pathlib import Path


def test_runtime_tools_catalog_is_filtered_by_current_mode(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.tools import build_runtime_tools

    config = load_runtime_config(
        SessionOpenIntent(workspace_dir=tmp_path, current_mode="read")
    )
    loaded = build_runtime_tools(tmp_path, SessionOpenIntent(workspace_dir=tmp_path), config)

    names = {tool.name for tool in loaded.specs}
    assert {"ls", "read", "grep", "find", "workspace_status"} <= names
    assert {"propose_plan", "create_build_plan", "update_plan_progress", "close_plan"}.isdisjoint(names)
    assert "write" not in names
    assert "bash" not in names


def test_runtime_tools_registers_caller_tools_without_reserved_name_compat(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.tools import build_runtime_tools
    custom = _caller_registration("custom_echo", owner="caller:custom")
    reserved = _caller_registration("read", owner="caller:reserved")

    intent = SessionOpenIntent(workspace_dir=tmp_path, tools=[custom, reserved])
    loaded = build_runtime_tools(tmp_path, intent, load_runtime_config(intent))

    assert loaded.registry.entry("custom_echo") is not None
    assert loaded.registry.entry("read").source == "builtin"
    assert any("registration failed" in warning for warning in loaded.warnings)


def test_skill_loader_tool_loads_discovered_skill_content(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.tools import build_runtime_tools
    from codepilot.tools.contracts import ToolExecutionRequest
    from codepilot.tools.runtime import ToolRuntime

    skill_dir = tmp_path / ".codepilot" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "demo.md").write_text(
        "\n".join(
            [
                "---",
                "name: Demo Skill",
                "command: demo",
                "description: Demo workflow",
                "---",
                "# Demo",
                "Use this workflow.",
            ]
        ),
        encoding="utf-8",
    )

    intent = SessionOpenIntent(workspace_dir=tmp_path)
    loaded = build_runtime_tools(tmp_path, intent, load_runtime_config(intent))
    runtime = ToolRuntime(loaded.registry)
    entry = loaded.registry.entry("load_skill")
    assert entry is not None

    observation = asyncio.run(
        runtime.execute(
            ToolExecutionRequest(
                run_id="run1",
                session_id="session1",
                tool_call_id="skill1",
                tool_name="load_skill",
                arguments={"name": "demo"},
                mode="execute",
                registration_id=entry.registration_id,
            )
        )
    )

    assert observation.status == "success"
    assert "Use this workflow." in observation.content[0].text
    assert observation.data["command"] == "demo"


def test_grep_searches_when_path_is_specific_file(tmp_path: Path) -> None:
    from codepilot.tools.builtins import create_builtin_registrations
    from codepilot.tools.contracts import ToolExecutionRequest
    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.runtime import ToolRuntime

    source = tmp_path / "src" / "register.py"
    source.parent.mkdir()
    source.write_text(
        "class UserRegister:\n    pass\n",
        encoding="utf-8",
        newline="\n",
    )
    registry = ToolRegistry()
    registration = create_builtin_registrations(tmp_path, enabled_names=["grep"])[0]
    registration_id = registry.register(registration)

    result = asyncio.run(
        ToolRuntime(registry).execute(
            ToolExecutionRequest(
                run_id="run1",
                session_id="session1",
                tool_call_id="grep1",
                tool_name="grep",
                arguments={
                    "path": "src/register.py",
                    "pattern": "class UserRegister",
                },
                mode="execute",
                registration_id=registration_id,
            )
        )
    )

    assert result.status == "success"
    assert "src/register.py:1:class UserRegister:" in result.content[0].text
    assert result.data["details"]["scanned_files"] == 1
    assert result.data["details"]["match_count"] == 1


def test_tools_json_limits_enabled_builtin_tools(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.tools import build_runtime_tools

    config_dir = tmp_path / ".codepilot"
    config_dir.mkdir()
    (config_dir / "tools.json").write_text(
        json.dumps({"enabled": ["read", "workspace_status", "create_build_plan", "close_plan"]}),
        encoding="utf-8",
    )

    intent = SessionOpenIntent(workspace_dir=tmp_path)
    loaded = build_runtime_tools(tmp_path, intent, load_runtime_config(intent))

    assert {tool.name for tool in loaded.specs} == {
        "read",
        "workspace_status",
    }


def _caller_registration(name: str, *, owner: str):
    from codepilot.tools import (
        ConcurrencyPolicy,
        JsonObjectCodec,
        OutputLimits,
        OutputTrustPolicy,
        TextContent,
        TimeoutPolicy,
        ToolAccessRequest,
        ToolAccessResolution,
        ToolPolicy,
        ToolRegistration,
        ToolSpec,
    )

    schema = {"type": "object", "properties": {}, "additionalProperties": False}

    class Resolver:
        def resolve(self, input, request):
            _ = request
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(name,),
                    resources=(),
                    effects=frozenset(),
                    risk="low",
                    reason="caller test",
                ),
            )

    async def handler(input, context):
        _ = input, context
        return {}

    class Renderer:
        def render(self, data):
            _ = data
            return (TextContent("ok"),)

    codec = JsonObjectCodec(schema)
    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(name, "Canonical caller registration used by runtime tools tests.", schema, schema),
        category="external",
        source="caller",
        owner=owner,
        policy=ToolPolicy(
            allowed_modes=frozenset({"plan", "execute"}),
            declared_effects=frozenset(),
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(1_000, 1_000),
            concurrency=ConcurrencyPolicy("parallel"),
            output_limits=OutputLimits(),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=codec,
        output_codec=codec,
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )
