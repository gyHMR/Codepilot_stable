from __future__ import annotations

from pathlib import Path


def test_repository_bootstrap_recognizes_python_project_without_reading_files(tmp_path: Path) -> None:
    from codepilot.sessions.store import build_repository_bootstrap, render_repository_context

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
    from codepilot.sessions.store import build_repository_bootstrap

    for index in range(40):
        (tmp_path / f"entry_{index:02d}.txt").write_text("x", encoding="utf-8")

    bootstrap = build_repository_bootstrap(tmp_path)

    assert len(bootstrap.top_level_entries) == 30


def test_runtime_prompt_keeps_repository_context_with_custom_prompt(tmp_path: Path) -> None:
    from codepilot.runtime import SessionOpenIntent
    from codepilot.runtime.config import load_runtime_config
    from codepilot.runtime.prompt import build_system_prompt
    from codepilot.runtime.tools import build_runtime_tools

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
    from codepilot.runtime.prompt import build_system_prompt
    from codepilot.runtime.tools import build_runtime_tools

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
    intent = SessionOpenIntent(
        workspace_dir=tmp_path,
        skill_paths=[str(skill_file)],
        load_workspace_resources=False,
    )
    config = load_runtime_config(intent)
    tools = build_runtime_tools(tmp_path, intent, config)
    prompt = build_system_prompt(workspace=tmp_path, config=config, tools=tools)

    assert "Available Skills" in prompt
    assert "/focused-review" in prompt
    assert "Use this when reviewing a focused code change." in prompt
    assert "SECRET_SKILL_BODY_MARKER" not in prompt
