from __future__ import annotations

"""Human-facing ``codepilot config`` commands."""

import json
from pathlib import Path

from codepilot.runtime import SessionOpenIntent
from codepilot.runtime.config import (
    UnknownRuntimeConfigKeyError,
    check_workspace_model_config,
    describe_workspace_config,
    explain_session_open_config,
)

from .render import create_console, render_key_value_panel


_MODEL_CONFIG_TEMPLATE = {
    "api": "openai-compatible",
    "provider": "deepseek",
    "model_id": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "",
    "api_key_env": "DEEPSEEK_API_KEY",
    "context_window": 64000,
    "max_tokens": 8192,
    "reasoning": False,
    "vision": False,
}


def run_config_command(
    action: str,
    *,
    workspace: str | Path,
    key: str | None = None,
    intent: SessionOpenIntent | None = None,
) -> None:
    """Run one ``codepilot config`` action."""

    if action == "init":
        init_model_config(workspace)
        return
    if action == "show":
        show_config(workspace)
        return
    if action == "check":
        check_model_config(workspace)
        return
    if action == "explain":
        if intent is None:
            raise ValueError("config explain requires a session intent")
        explain_config(intent, key)
        return
    raise ValueError(f"Unknown config action: {action}")


def init_model_config(workspace: str | Path) -> None:
    """Create the editable local model config template."""

    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    config_file = Path(workspace) / ".codepilot" / "model.local.json"
    if config_file.exists():
        raise ValueError(f"Model config already exists: {config_file}")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(_MODEL_CONFIG_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    content = Text()
    content.append("Created ", style="success")
    content.append(str(config_file), style="path")
    content.append("\nEdit api_key or configure api_key_env, then run `codepilot`.", style="value")
    content.append("\nDo not commit this file to version control.", style="warning")
    create_console().print(
        Panel(
            content,
            title="[panel.title]CP // CONFIG INIT[/panel.title]",
            border_style="success",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def check_model_config(workspace: str | Path) -> None:
    """Validate the local model config and render a concise report."""

    view = check_workspace_model_config(workspace)
    render_key_value_panel(
        "Config Check",
        list(view.rows),
        border_style=view.border_style,
    )


def show_config(workspace: str | Path) -> None:
    """Show sanitized model and runtime settings."""

    from rich.panel import Panel
    from rich.table import Table

    console = create_console()
    view = describe_workspace_config(workspace)

    model_table = Table(show_header=True, box=None, padding=(0, 2))
    model_table.add_column("Key", style="label", width=12)
    model_table.add_column("Value", style="value")
    for key, value in view.model_rows:
        model_table.add_row(str(key), str(value))
    console.print(
        Panel(
            model_table,
            title="[panel.title]CP // MODEL CONFIG[/panel.title]",
            border_style="panel.border",
        )
    )

    settings_table = Table(show_header=True, box=None, padding=(0, 2))
    settings_table.add_column("Key", style="label", width=25)
    settings_table.add_column("Value", style="value")
    for key, value in view.settings_rows:
        settings_table.add_row(str(key), str(value))
    console.print(
        Panel(
            settings_table,
            title="[panel.title]CP // SETTINGS[/panel.title]",
            border_style="panel.border",
        )
    )


def explain_config(intent: SessionOpenIntent, key: str | None) -> None:
    """Render where one runtime configuration value came from."""

    from rich.panel import Panel
    from rich.table import Table

    console = create_console()
    if not key:
        console.print("[error]Usage: codepilot config explain <key>[/error]")
        console.print("[muted2]Available keys: model, provider, model_id, thinking_level, tool_execution, etc.[/muted2]")
        return

    try:
        resolved = explain_session_open_config(intent, key)
    except UnknownRuntimeConfigKeyError:
        console.print(f"[error]Unknown config key: {key}[/error]")
        return
    except KeyError as exc:
        raise ValueError(str(exc).strip("'")) from exc

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="label", width=12)
    table.add_column("Value", style="value")
    table.add_row("Key", key)
    table.add_row("Value", str(resolved.value))
    table.add_row("Source", format_config_source(resolved.source))
    console.print(
        Panel(
            table,
            title=f"[panel.title]CP // CONFIG: {key}[/panel.title]",
            border_style="panel.border",
        )
    )


def format_config_source(source: object) -> str:
    """Format the runtime config source for terminal display."""

    kind = str(getattr(source, "kind", "unknown"))
    location = getattr(source, "location", None)
    labels = {
        "cli": "CLI argument",
        "session": "restored session",
        "project": "project",
        "default": "built-in default",
    }
    label = labels.get(kind, kind)
    return f"{label}:{location}" if location else label


__all__ = [
    "check_model_config",
    "explain_config",
    "format_config_source",
    "init_model_config",
    "run_config_command",
    "show_config",
]
