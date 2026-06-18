from __future__ import annotations

import json

from codepilot.interfaces.cli.cli import _init_model_config, build_parser
from codepilot.runtime.factory import create_agent_session
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
        resources=WorkspaceResourceLoader(tmp_path).load(),
        restored_meta=None,
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
        assert "secret-value" not in session.store.meta_file.read_text(encoding="utf-8")
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
    args = build_parser().parse_args(["--init-config"])

    assert args.init_config is True
