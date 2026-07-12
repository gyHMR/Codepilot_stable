from __future__ import annotations

import asyncio
from pathlib import Path


def test_flat_skill_file_is_rejected_without_legacy_fallback(tmp_path: Path) -> None:
    from codepilot.extensions import load_skill_catalog

    skill = tmp_path / "legacy.md"
    skill.write_text("# legacy", encoding="utf-8")

    catalog = load_skill_catalog(tmp_path, configured_paths=[str(skill)])

    assert catalog.packages == []
    assert len(catalog.errors) == 1
    assert "flat skill files are not supported" in catalog.errors[0]


def test_manifest_is_strict_and_directory_name_must_match(tmp_path: Path) -> None:
    from codepilot.extensions import load_skill_catalog

    package = tmp_path / "wrong-directory"
    package.mkdir()
    (package / "SKILL.md").write_text(
        """---
name: expected-name
version: 1.0.0
description: Strict package.
unknown_field: true
---
Instructions.
""",
        encoding="utf-8",
    )

    catalog = load_skill_catalog(tmp_path, configured_paths=[str(package)])

    assert catalog.packages == []
    assert "unknown skill manifest fields" in catalog.errors[0]


def test_skill_resource_rejects_traversal(tmp_path: Path) -> None:
    from codepilot.extensions import load_skills
    from codepilot.tools import ToolExecutionRequest, ToolRegistry, ToolRuntime

    package = _write_package(tmp_path / "safe-skill")
    loaded = load_skills(tmp_path, configured_paths=[str(package)])
    resource_tool = loaded.tools[1]
    registry = ToolRegistry()
    registration_id = registry.register(resource_tool)
    runtime = ToolRuntime(registry)

    result = asyncio.run(
        runtime.execute(
            ToolExecutionRequest(
                run_id="run-skill-resource",
                session_id="session-skill-resource",
                tool_call_id="call-skill-resource",
                tool_name="read_skill_resource",
                arguments={"skill": "safe-skill", "path": "../outside.txt"},
                mode="plan",
                registration_id=registration_id,
            )
        )
    )

    assert result.status == "denied"
    assert result.error is not None
    assert result.error.kind == "validation"


def test_skill_mode_is_enforced_by_runtime_handler(tmp_path: Path) -> None:
    from codepilot.extensions import load_skills
    from codepilot.tools import ToolExecutionRequest, ToolRegistry, ToolRuntime

    package = _write_package(tmp_path / "safe-skill", allowed_modes="  - execute")
    registration = load_skills(tmp_path, configured_paths=[str(package)]).tools[0]
    registry = ToolRegistry()
    registration_id = registry.register(registration)
    runtime = ToolRuntime(registry)

    result = asyncio.run(
        runtime.execute(
            ToolExecutionRequest(
                run_id="run-skill-mode",
                session_id="session-skill-mode",
                tool_call_id="call-skill-mode",
                tool_name="load_skill",
                arguments={"name": "safe-skill"},
                mode="plan",
                registration_id=registration_id,
            )
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "skill.mode_denied"


def test_command_prompt_is_followed_by_a_normal_model_run() -> None:
    from codepilot.runtime.actions import PromptSubmitted
    from codepilot.runtime.gateway import RuntimeGateway
    from codepilot.sessions.contracts import SessionCommandRecord

    class RecordingGateway(RuntimeGateway):
        def __init__(self) -> None:
            self.prompt: PromptSubmitted | None = None

        async def _run_prompt(self, session, action):
            _ = session
            self.prompt = action
            yield action

    gateway = RecordingGateway()
    record = SessionCommandRecord(
        session_id="session-skill-command",
        command="/safe-skill inspect this",
        handled=True,
        data={"prompt": "Call load_skill and inspect this."},
    )

    async def follow_up():
        return [
            frame
            async for frame in gateway._follow_up_from_command(object(), record)
        ]

    frames = asyncio.run(follow_up())

    assert frames == [gateway.prompt]
    assert gateway.prompt is not None
    assert gateway.prompt.text == "Call load_skill and inspect this."


def _write_package(path: Path, *, allowed_modes: str = "  - plan\n  - execute") -> Path:
    path.mkdir()
    (path / "references").mkdir()
    (path / "references" / "guide.md").write_text("Guide", encoding="utf-8")
    (path / "SKILL.md").write_text(
        f"""---
name: safe-skill
version: 1.0.0
description: A safe test skill.
allowed_modes:
{allowed_modes}
---
Follow the workflow.
""",
        encoding="utf-8",
    )
    return path
