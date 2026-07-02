from __future__ import annotations

from pathlib import Path


def test_repository_bootstrap_recognizes_python_project_without_reading_files(tmp_path: Path) -> None:
    from codepilot.runtime.bootstrap.context import build_repository_bootstrap, render_repository_context

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
    from codepilot.runtime.bootstrap.context import build_repository_bootstrap

    for index in range(40):
        (tmp_path / f"entry_{index:02d}.txt").write_text("x", encoding="utf-8")

    bootstrap = build_repository_bootstrap(tmp_path)

    assert len(bootstrap.top_level_entries) == 30


def test_runtime_prompt_keeps_repository_context_with_custom_prompt(tmp_path: Path) -> None:
    from codepilot.runtime.bootstrap.context import RuntimeContext
    from codepilot.runtime.bootstrap.prompt import build_runtime_system_prompt

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
