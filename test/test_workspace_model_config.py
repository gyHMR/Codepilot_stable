from __future__ import annotations

import json

import pytest

from codepilot.interfaces.cli.main import _init_model_config, build_parser
from codepilot.runtime.assembly import create_agent_session
from codepilot.runtime.model_resolver import resolve_model
from codepilot.runtime.resources import WorkspaceResourceLoader
from codepilot.runtime.types import CreateAgentSessionOptions


def _write_model_config(workspace, *, api_key: str = "local-key") -> None:
    root = workspace / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.local.json").write_text(
        json.dumps(
            {
                "api": "openai-compatible",
                "provider": "deepseek",
                "model_id": "deepseek-chat",
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
    resolved = resolve_model(
        CreateAgentSessionOptions(workspace_dir=tmp_path),
        inputs=_runtime_inputs(tmp_path),
    )

    assert resolved.model.provider == "deepseek"
    assert resolved.get_api_key is not None
    assert resolved.get_api_key("deepseek") == "local-key"


def test_factory_does_not_persist_api_key(tmp_path) -> None:
    _write_model_config(tmp_path, api_key="secret-value")
    session = create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path))
    try:
        assert session.get_api_key is not None
        assert session.get_api_key("deepseek") == "secret-value"
        assert "secret-value" not in session.store.session_file.read_text(encoding="utf-8")
    finally:
        session.close()


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
    assert args.task_mode is None


def test_cli_rejects_removed_legacy_options() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "print", "--prompt", "hello"])


def test_restored_session_identity_overrides_workspace_settings(tmp_path) -> None:
    from codepilot.runtime.config import read_restored_session_meta, resolve_runtime_config
    from codepilot.sessions.persistence.store import SessionStore

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
    store = SessionStore(tmp_path, "session_restore")
    store.ensure_initialized(model_id="deepseek-v4-pro", provider="deepseek", system_prompt="restored prompt")

    options = CreateAgentSessionOptions(workspace_dir=tmp_path, session_id="session_restore")
    inputs = _runtime_inputs(tmp_path, session_id="session_restore")
    resolved = resolve_model(options, inputs=inputs)
    config = resolve_runtime_config(options, inputs=inputs)

    assert read_restored_session_meta(tmp_path, "session_restore") is not None
    assert resolved.model.provider == "deepseek"
    assert resolved.model.id == "deepseek-v4-pro"
    assert config.system_prompt == "restored prompt"
    assert config.sources["system_prompt"] == "restored_session"


def test_explicit_false_and_empty_values_override_workspace_config(tmp_path) -> None:
    from codepilot.runtime.config import resolve_runtime_config

    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps(
            {
                "retry_enabled": True,
                "read_only_mode": True,
                "block_dangerous_bash": True,
                "prompt_debug_sources": True,
                "bash_allow_patterns": ["pytest"],
                "extension_paths": ["workspace-extension"],
            }
        ),
        encoding="utf-8",
    )

    config = resolve_runtime_config(
        CreateAgentSessionOptions(
            workspace_dir=tmp_path,
            retry_enabled=False,
            read_only_mode=False,
            block_dangerous_bash=False,
            prompt_debug_sources=False,
            bash_allow_patterns=[],
            extension_paths=[],
        ),
        inputs=_runtime_inputs(tmp_path),
    )

    assert config.retry_enabled is False
    assert config.read_only_mode is False
    assert config.block_dangerous_bash is False
    assert config.prompt_debug_sources is False
    assert config.bash_allow_patterns == []
    assert config.extension_paths == []
    assert config.sources["retry_enabled"] == "options"
    assert config.sources["bash_allow_patterns"] == "options"


def test_workspace_values_fall_back_to_defaults_with_sources(tmp_path) -> None:
    from codepilot.runtime.config import resolve_runtime_config

    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps({"max_tool_calls_per_turn": 3, "tool_execution": "sequential"}),
        encoding="utf-8",
    )

    config = resolve_runtime_config(
        CreateAgentSessionOptions(workspace_dir=tmp_path),
        inputs=_runtime_inputs(tmp_path),
    )

    assert config.max_tool_calls_per_turn == 3
    assert config.tool_execution == "sequential"
    assert config.max_retries == 2
    assert config.sources["max_tool_calls_per_turn"] == "workspace"
    assert config.sources["tool_execution"] == "workspace"
    assert config.sources["max_retries"] == "default"


def test_workspace_settings_can_select_task_mode(tmp_path) -> None:
    from codepilot.runtime.config import resolve_runtime_config

    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps({"task_mode": "plan"}),
        encoding="utf-8",
    )

    config = resolve_runtime_config(
        CreateAgentSessionOptions(workspace_dir=tmp_path),
        inputs=_runtime_inputs(tmp_path),
    )

    assert config.task_mode == "plan"
    assert config.sources["task_mode"] == "workspace"


def test_workspace_settings_can_select_planning_budget_profile(tmp_path) -> None:
    from codepilot.runtime.config import resolve_runtime_config

    root = tmp_path / ".codepilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps({"planning_budget_profile": "wide"}),
        encoding="utf-8",
    )

    config = resolve_runtime_config(
        CreateAgentSessionOptions(workspace_dir=tmp_path),
        inputs=_runtime_inputs(tmp_path),
    )

    assert config.task_mode == "edit"
    assert config.planning_budget_profile == "wide"
    assert config.sources["planning_budget_profile"] == "workspace"


def test_read_task_mode_forces_read_only_permission(tmp_path) -> None:
    from codepilot.runtime.config import resolve_runtime_config

    config = resolve_runtime_config(
        CreateAgentSessionOptions(workspace_dir=tmp_path, task_mode="read"),
        inputs=_runtime_inputs(tmp_path),
    )

    assert config.task_mode == "read"
    assert config.read_only_mode is True
    assert config.tool_permission_mode == "read-only"


def test_read_task_mode_rejects_workspace_write_override(tmp_path) -> None:
    from codepilot.runtime.config import resolve_runtime_config

    with pytest.raises(ValueError, match="task_mode=read"):
        resolve_runtime_config(
            CreateAgentSessionOptions(
                workspace_dir=tmp_path,
                task_mode="read",
                tool_permission_mode="workspace-write",
            ),
            inputs=_runtime_inputs(tmp_path),
        )


def _runtime_inputs(tmp_path, *, session_id: str | None = None):
    from codepilot.runtime.config import RuntimeInputs

    return RuntimeInputs(
        workspace=tmp_path,
        resources=WorkspaceResourceLoader(tmp_path).load(),
        restored_meta=(
            __import__("codepilot.sessions.persistence.store", fromlist=["SessionStore"])
            .SessionStore(tmp_path, session_id)
            .read_meta()
            if session_id
            else None
        ),
    )
