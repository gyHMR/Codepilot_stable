from __future__ import annotations

import asyncio
from pathlib import Path


def test_repository_bootstrap_recognizes_python_project_without_reading_files(tmp_path: Path) -> None:
    from codepilot.sessions.workspace import build_repository_bootstrap, render_repository_context

    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "test").mkdir()
    secret = tmp_path / "secret.py"
    secret.write_text("API_KEY='should-not-appear'", encoding="utf-8")

    bootstrap = build_repository_bootstrap(tmp_path)
    rendered = render_repository_context(bootstrap)

    assert bootstrap.project_type == "Python"
    assert "pyproject.toml" in bootstrap.manifest_files
    assert "test/" in bootstrap.test_directories
    assert "src/" in bootstrap.top_level_entries
    assert "Repository Context" in rendered
    assert "should-not-appear" not in rendered
    assert "API_KEY" not in rendered


def test_repository_bootstrap_limits_top_level_entries(tmp_path: Path) -> None:
    from codepilot.sessions.workspace import build_repository_bootstrap

    for index in range(40):
        (tmp_path / f"entry_{index:02d}.txt").write_text("x", encoding="utf-8")

    bootstrap = build_repository_bootstrap(tmp_path)

    assert len(bootstrap.top_level_entries) == 30


def test_runtime_prompt_keeps_repository_context_with_custom_prompt(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.builder import build_runtime_tools, build_system_prompt

    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    intent = SessionOpenIntent(
        workspace_dir=tmp_path,
        system_prompt="Custom system prompt",
        load_workspace_resources=False,
    )
    config = load_runtime_config(intent)
    tools = build_runtime_tools(tmp_path, intent, config)
    prompt = build_system_prompt(workspace=tmp_path, config=config, tools=tools)

    assert prompt.startswith("Custom system prompt")
    assert "Repository Context" in prompt
    assert "Project type: Python" in prompt


def test_runtime_prompt_includes_skill_index_without_skill_body(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.builder import build_runtime_tools, build_system_prompt

    skill_package = tmp_path / "focused-review"
    skill_package.mkdir()
    (skill_package / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: focused-review",
                "version: 1.0.0",
                "command: focused-review",
                "description: Use this when reviewing a focused code change.",
                "---",
                "# Focused Review",
                "SECRET_SKILL_BODY_MARKER",
            ]
        ),
        encoding="utf-8",
    )
    intent = SessionOpenIntent(
        workspace_dir=tmp_path,
        skill_paths=[str(skill_package)],
        load_workspace_resources=False,
    )
    config = load_runtime_config(intent)
    tools = build_runtime_tools(tmp_path, intent, config)
    prompt = build_system_prompt(workspace=tmp_path, config=config, tools=tools)

    assert "Available Skills" in prompt
    assert "/focused-review" in prompt
    assert "Use this when reviewing a focused code change." in prompt
    assert "SECRET_SKILL_BODY_MARKER" not in prompt


def test_runtime_config_default_tool_call_batch_limit_supports_agent_batches(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config

    config = load_runtime_config(
        SessionOpenIntent(workspace_dir=tmp_path, load_workspace_resources=False)
    )

    assert config.max_tool_calls_per_turn == 16


def test_default_runtime_prompt_describes_coding_agent_workflow(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.builder import build_runtime_tools, build_system_prompt

    intent = SessionOpenIntent(workspace_dir=tmp_path, load_workspace_resources=False)
    config = load_runtime_config(intent)
    tools = build_runtime_tools(tmp_path, intent, config)

    prompt = build_system_prompt(workspace=tmp_path, config=config, tools=tools)

    assert "本地代码仓库中工作的 coding agent" in prompt
    assert "canonical Task Plan" in prompt
    assert "Codepilot Runtime Control" in prompt
    assert "不编造文件、符号、调用链" in prompt
    assert "验证" in prompt
    assert "当前模式：" not in prompt


def test_base_prompt_does_not_embed_plan_mode_policy(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.builder import build_runtime_tools, build_system_prompt

    intent = SessionOpenIntent(
        workspace_dir=tmp_path,
        current_mode="plan",
        load_workspace_resources=False,
    )
    config = load_runtime_config(intent)
    tools = build_runtime_tools(tmp_path, intent, config)

    prompt = build_system_prompt(workspace=tmp_path, config=config, tools=tools)

    assert "当前模式：" not in prompt
    assert "当前 mode=plan" not in prompt
    assert "- write:" not in prompt
    assert "- edit:" not in prompt
    assert "- apply_patch:" not in prompt


def test_runtime_context_adapter_returns_typed_prepared_context(tmp_path: Path) -> None:
    from codepilot.core.contracts import (
        ContextPrepareRequest,
        CoreContextView,
        PreparedModelContext,
    )
    from codepilot.llm.ports import ModelDescriptor
    from codepilot.protocols import Model
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
    from codepilot.sessions.contracts import SessionOptions, SessionRunIntent

    model = Model(
        id="unit",
        name="Unit",
        api="unit",
        provider="unit",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )

    async def run_case() -> PreparedModelContext:
        coordinator = RuntimeSessionCoordinator(
            SessionOptions(
                model=model,
                workspace_dir=tmp_path,
                session_id="session_context_adapter",
                memory_enabled=False,
            )
        )
        try:
            prepared_run = await coordinator._prepare_run(  # noqa: SLF001
                SessionRunIntent(text="inspect the project", request_id="request_1"),
                run_id="run_context_adapter",
                model=ModelDescriptor(provider="unit", model_id="unit"),
            )
            context_port = prepared_run.context_port
            assert context_port is not None
            loop_input = prepared_run.loop_input
            result = context_port.prepare(
                ContextPrepareRequest(
                    session_id=loop_input.session_id,
                    run_id=loop_input.run_id,
                    purpose="reasoning",
                    directive="core.reasoning",
                    messages=loop_input.messages,
                    core_view=CoreContextView.from_state(
                        loop_input.state,
                        loop_input.mode,
                    ),
                    model=loop_input.model,
                    tool_catalog=None,
                    seed=loop_input.context_seed,
                )
            )
            return await result if asyncio.iscoroutine(result) else result
        finally:
            coordinator.close()

    prepared = asyncio.run(run_case())

    assert isinstance(prepared, PreparedModelContext)
    assert prepared.projection_ref
    assert prepared.messages
