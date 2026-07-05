from __future__ import annotations

from pathlib import Path


def test_repository_bootstrap_recognizes_python_project_without_reading_files(tmp_path: Path) -> None:
    from codepilot.runtime.assembly import build_repository_bootstrap, render_repository_context

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
    from codepilot.runtime.assembly import build_repository_bootstrap

    for index in range(40):
        (tmp_path / f"entry_{index:02d}.txt").write_text("x", encoding="utf-8")

    bootstrap = build_repository_bootstrap(tmp_path)

    assert len(bootstrap.top_level_entries) == 30


def test_runtime_prompt_keeps_repository_context_with_custom_prompt(tmp_path: Path) -> None:
    from codepilot.runtime.assembly import RuntimeContext
    from codepilot.runtime.assembly import build_runtime_system_prompt

    prompt = build_runtime_system_prompt(
        base_system_prompt="Custom system prompt",
        tools=[],
        runtime_context=RuntimeContext(
            repository_context="## Repository Context\n- Project type: Python",
            prompt_guidelines=[],
            append_sections=[],
            tool_snippets={},
            memory_text="",
        ),
        workspace=tmp_path,
    )

    assert prompt.startswith("Custom system prompt")
    assert "Repository Context" in prompt
    assert "Project type: Python" in prompt


def test_runtime_prompt_includes_skill_index_without_skill_body(tmp_path: Path) -> None:
    from codepilot.extensions import load_skills
    from codepilot.extensions.types import LoadedExtensions
    from codepilot.runtime.assembly import RuntimeConfig
    from codepilot.runtime.assembly import build_runtime_context
    from codepilot.runtime.assembly import build_runtime_system_prompt

    skill_file = tmp_path / "review.md"
    skill_file.write_text(
        "\n".join(
            [
                "---",
                "name: Focused Review",
                "command: focused-review",
                "description: Use this when reviewing a focused code change.",
                "---",
                "# Focused Review",
                "SECRET_SKILL_BODY_MARKER",
            ]
        ),
        encoding="utf-8",
    )
    loaded_skills = load_skills(tmp_path, configured_paths=[str(skill_file)])
    config = RuntimeConfig(
        system_prompt="",
        thinking_level="off",
        tool_execution="parallel",
        task_mode="edit",
        planning_budget_profile="balanced",
        retry_enabled=True,
        max_retries=2,
        retry_base_delay_ms=1200,
        read_only_mode=False,
        block_dangerous_bash=True,
        bash_allow_patterns=None,
        bash_block_patterns=None,
        edit_require_unique_match=True,
        extension_paths=[],
        skill_paths=[str(skill_file)],
        mcp_servers=[],
        prompt_guidelines=None,
        append_system_prompt=None,
        prompt_debug_sources=False,
        tool_snippets=None,
        enabled_builtin_tools=None,
    )

    context = build_runtime_context(
        tmp_path,
        config,
        LoadedExtensions(),
        loaded_skills,
    )
    prompt = build_runtime_system_prompt(
        base_system_prompt="",
        tools=[],
        runtime_context=context,
        workspace=tmp_path,
    )

    assert "Available Skills" in prompt
    assert "/focused-review" in prompt
    assert "Use this when reviewing a focused code change." in prompt
    assert "SECRET_SKILL_BODY_MARKER" not in prompt
