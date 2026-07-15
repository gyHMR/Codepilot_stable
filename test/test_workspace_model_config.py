from __future__ import annotations

import asyncio
import json

import pytest

from codepilot.interfaces.cli.main import (
    _check_model_config,
    _show_config,
    _init_model_config,
    build_parser,
)
from codepilot.runtime.builder import build_runtime_session
from codepilot.runtime.config import WorkspaceResourceLoader, load_runtime_config
from codepilot.runtime.model import resolve_runtime_model
from codepilot.runtime import SessionOpenIntent


def _write_model_config(
    workspace,
    *,
    api_key: str = "local-key",
    model_id: str = "deepseek-chat",
) -> None:
    root = workspace / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.local.json").write_text(
        json.dumps(
            {
                "api": "openai-compatible",
                "provider": "deepseek",
                "model_id": model_id,
                "base_url": "https://api.deepseek.com/v1",
                "api_key": api_key,
                "api_key_env": "DEEPSEEK_API_KEY",
                "context_window": 64000,
                "max_tokens": 8192,
                "reasoning": False,
                "vision": False,
            }
        ),
        encoding="utf-8",
    )


def test_workspace_model_config_loads_openai_compatible_deepseek(tmp_path) -> None:
    _write_model_config(tmp_path)
    model = WorkspaceResourceLoader(tmp_path).load().model

    assert model is not None
    assert model.api == "openai-compatible"
    assert model.provider == "deepseek"
    assert model.to_model().base_url == "https://api.deepseek.com/v1"
    assert model.resolve_api_key() == "local-key"


def test_environment_key_overrides_local_key(tmp_path, monkeypatch) -> None:
    _write_model_config(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    model = WorkspaceResourceLoader(tmp_path).load().model

    assert model is not None
    assert model.resolve_api_key() == "environment-key"


def test_runtime_resolves_workspace_model_and_key(tmp_path) -> None:
    _write_model_config(tmp_path)
    intent = SessionOpenIntent(workspace_dir=tmp_path)
    resolved = resolve_runtime_model(intent, load_runtime_config(intent))

    assert resolved.model.provider == "deepseek"
    assert resolved.get_api_key is not None
    assert resolved.get_api_key("deepseek") == "local-key"


def test_runtime_restores_custom_workspace_model_from_local_config(tmp_path) -> None:
    from codepilot.sessions.contracts import ModelRef
    from codepilot.sessions.service import CreateSessionRequest, SessionStateService

    _write_model_config(tmp_path, model_id="deepseek-v4-flash")
    SessionStateService(tmp_path).create_session(
        CreateSessionRequest(
            workspace_root=str(tmp_path),
            model=ModelRef(provider="deepseek", model="deepseek-v4-flash"),
            current_mode="plan",
            system_prompt_hash="test-system-prompt",
            session_id="session_custom_model",
        )
    )

    intent = SessionOpenIntent(
        workspace_dir=tmp_path,
        session_id="session_custom_model",
    )
    resolved = resolve_runtime_model(intent, load_runtime_config(intent))

    assert resolved.model.id == "deepseek-v4-flash"
    assert resolved.model.context_window == 64000
    assert resolved.get_api_key is not None
    assert resolved.get_api_key("deepseek") == "local-key"


def test_factory_does_not_persist_api_key(tmp_path) -> None:
    _write_model_config(tmp_path, api_key="secret-value")
    intent = SessionOpenIntent(workspace_dir=tmp_path)
    session = build_runtime_session(intent)
    resolved = resolve_runtime_model(intent, load_runtime_config(intent))
    try:
        assert resolved.get_api_key is not None
        assert resolved.get_api_key("deepseek") == "secret-value"
        session_file = (
            tmp_path
            / ".codepilot"
            / "sessions"
            / session.session_id
            / "session.json"
        )
        assert "secret-value" not in session_file.read_text(encoding="utf-8")
    finally:
        session.controller.close()


def test_init_config_creates_editable_template(tmp_path) -> None:
    _init_model_config(tmp_path)
    path = tmp_path / ".codepilot" / "model.local.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["api"] == "openai-compatible"
    assert raw["provider"] == "deepseek"
    assert raw["api_key"] == ""


def test_cli_exposes_local_config_commands() -> None:
    args = build_parser().parse_args(["config", "init"])

    assert args.command == "config"
    assert args.config_action == "init"


def test_cli_defaults_leave_runtime_config_unspecified() -> None:
    args = build_parser().parse_args(["--prompt", "hello"])

    assert args.prompt == "hello"
    assert args.model is None
    assert args.permission_mode is None
    assert args.current_mode is None


def test_cli_interactive_opens_runtime_session_and_runs_repl(tmp_path, monkeypatch) -> None:
    from codepilot.interfaces.cli import main as cli_main

    captured = {}

    class FakeRuntime:
        def open_session(self, intent):
            captured["intent"] = intent

            class Handle:
                session_id = "session_1"

            return Handle()

        async def close_all(self):
            captured["closed"] = True

    async def fake_run_repl(runtime, session_id, **kwargs):
        captured["run_mode"] = "repl"
        captured["session_id"] = session_id

    monkeypatch.setattr(cli_main, "RuntimeGateway", FakeRuntime)
    monkeypatch.setattr(cli_main, "run_repl", fake_run_repl)

    args = build_parser().parse_args(["--cwd", str(tmp_path)])

    assert asyncio.run(cli_main._run_from_args(args)) == 0
    assert captured["run_mode"] == "repl"
    assert captured["session_id"] == "session_1"
    assert captured["intent"].workspace_dir == tmp_path
    assert captured["closed"] is True


def test_cli_rejects_removed_legacy_options() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "print", "--prompt", "hello"])


def test_cli_help_uses_cyber_command_deck(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "CP // COMMAND DECK" in output
    assert "C P" in output
    assert "rpc" in output
    assert "--prompt" in output


def test_cli_config_help_uses_cyber_config_deck(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["config", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "CP // CONFIG DECK" in output
    assert "explain <key>" in output
    assert "check" in output


def test_config_check_and_show_use_sanitized_human_output(tmp_path, capsys) -> None:
    _write_model_config(tmp_path, api_key="secret-value")

    _check_model_config(tmp_path)
    _show_config(tmp_path)

    output = capsys.readouterr().out
    assert "CP // CONFIG CHECK" in output
    assert "CP // MODEL CONFIG" in output
    assert "deepseek-chat" in output
    assert "local-file (do not commit)" in output
    assert "secret-value" not in output


def test_restored_session_identity_overrides_workspace_settings(tmp_path) -> None:
    from codepilot.runtime.config import load_session_open_metadata
    from codepilot.sessions.contracts import ModelRef
    from codepilot.sessions.service import CreateSessionRequest, SessionStateService

    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps(
            {
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "system_prompt": "workspace prompt",
            }
        ),
        encoding="utf-8",
    )
    SessionStateService(tmp_path).create_session(
        CreateSessionRequest(
            workspace_root=str(tmp_path),
            model=ModelRef(provider="deepseek", model="deepseek-v4-pro"),
            current_mode="build",
            system_prompt_hash="test-system-prompt",
            session_id="session_restore",
        )
    )

    intent = SessionOpenIntent(workspace_dir=tmp_path, session_id="session_restore")
    config = load_runtime_config(intent)
    resolved = resolve_runtime_model(intent, config)

    assert load_session_open_metadata(tmp_path, "session_restore") is not None
    assert resolved.model.provider == "deepseek"
    assert resolved.model.id == "deepseek-v4-pro"
    assert config.system_prompt == "workspace prompt"
    assert config.sources["system_prompt"].kind == "project"


def test_explicit_false_and_empty_values_override_workspace_config(tmp_path) -> None:
    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps(
            {
                "retry_enabled": True,
                "tool_permission_mode": "ask",
                "prompt_debug_sources": True,
                "extension_paths": ["workspace-extension"],
            }
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(
        SessionOpenIntent(
            workspace_dir=tmp_path,
            retry_enabled=False,
            tool_permission_mode="workspace-write",
            prompt_debug_sources=False,
            extension_paths=[],
        ),
    )

    assert config.retry_enabled is False
    assert config.tool_permission_mode == "workspace-write"
    assert config.prompt_debug_sources is False
    assert config.extension_paths == []
    assert config.sources["retry_enabled"].kind == "cli"


def test_removed_shell_security_settings_are_rejected(tmp_path) -> None:
    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps({"block_dangerous_bash": False}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Removed runtime settings"):
        load_runtime_config(SessionOpenIntent(workspace_dir=tmp_path))


def test_workspace_values_fall_back_to_defaults_with_sources(tmp_path) -> None:
    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps({"max_tool_calls_per_turn": 3}),
        encoding="utf-8",
    )

    config = load_runtime_config(SessionOpenIntent(workspace_dir=tmp_path))

    assert config.max_tool_calls_per_turn == 3
    assert config.max_retries == 2
    assert config.sources["max_tool_calls_per_turn"].kind == "project"
    assert config.sources["max_retries"].kind == "default"


def test_workspace_settings_can_select_current_mode(tmp_path) -> None:
    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps({"current_mode": "plan"}),
        encoding="utf-8",
    )

    config = load_runtime_config(SessionOpenIntent(workspace_dir=tmp_path))

    assert config.current_mode == "plan"
    assert config.sources["current_mode"].kind == "project"


def test_workspace_settings_can_select_planning_budget_profile(tmp_path) -> None:
    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps({"planning_budget_profile": "wide"}),
        encoding="utf-8",
    )

    config = load_runtime_config(SessionOpenIntent(workspace_dir=tmp_path))

    assert config.current_mode == "build"
    assert config.planning_budget_profile == "wide"
    assert config.sources["planning_budget_profile"].kind == "project"


def test_read_current_mode_forces_read_only_permission(tmp_path) -> None:
    config = load_runtime_config(
        SessionOpenIntent(workspace_dir=tmp_path, current_mode="read")
    )

    assert config.current_mode == "read"
    assert config.tool_permission_mode == "read-only"


def test_read_current_mode_forces_workspace_write_override_to_read_only(tmp_path) -> None:
    config = load_runtime_config(
        SessionOpenIntent(
            workspace_dir=tmp_path,
            current_mode="read",
            tool_permission_mode="workspace-write",
        ),
    )

    assert config.current_mode == "read"
    assert config.tool_permission_mode == "read-only"


def test_plan_current_mode_forces_read_only_permission(tmp_path) -> None:
    config = load_runtime_config(
        SessionOpenIntent(workspace_dir=tmp_path, current_mode="plan")
    )

    assert config.current_mode == "plan"
    assert config.tool_permission_mode == "read-only"


def test_plan_current_mode_forces_workspace_write_override_to_read_only(tmp_path) -> None:
    config = load_runtime_config(
        SessionOpenIntent(
            workspace_dir=tmp_path,
            current_mode="plan",
            tool_permission_mode="workspace-write",
        ),
    )

    assert config.current_mode == "plan"
    assert config.tool_permission_mode == "read-only"
