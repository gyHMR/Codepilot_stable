from __future__ import annotations

"""Shared terminal UI helpers for the Codepilot CLI."""

import os
from typing import Iterable


CP_CIRCUIT_MARK = (
    "╭─ C P ─╮\n"
    "│ ╭╮ ╭─ │\n"
    "│ ╰╯ ╰─ │\n"
    "╰─╼╾─╼╾╯"
)

PLAIN_CP_MARK = (
    "+- C P -+\n"
    "| () <- |\n"
    "| [] -> |\n"
    "+-------+"
)


def no_color_requested() -> bool:
    """Return True when the environment asks for plain terminal output."""

    return bool(os.getenv("NO_COLOR"))


def create_console(*, no_color: bool | None = None):
    """Create a Rich console with Codepilot's cyber-terminal theme."""

    from rich.console import Console

    return Console(
        theme=create_theme(),
        no_color=no_color_requested() if no_color is None else no_color,
    )


def create_theme():
    """Build the shared Rich theme without importing Rich at module import time."""

    from rich.theme import Theme

    return Theme({
        "brand": "bold #22d3ee",
        "brand.hot": "bold #f0abfc",
        "brand.dim": "#0891b2",
        "panel.border": "#155e75",
        "panel.title": "bold #67e8f9",
        "label": "#94a3b8",
        "value": "#e5e7eb",
        "muted": "#64748b",
        "muted2": "#94a3b8",
        "info": "#7dd3fc",
        "warning": "#fbbf24",
        "error": "bold #fb7185",
        "success": "#86efac",
        "cancelled": "#fbbf24",
        "tool": "#67e8f9",
        "model": "#c084fc",
        "path": "#93c5fd",
    })


def cyber_title(text: str) -> str:
    return f"CP // {text.upper()}"


def render_key_value_panel(
    title: str,
    rows: Iterable[tuple[str, object]],
    *,
    console=None,
    border_style: str = "panel.border",
) -> None:
    """Render a compact two-column panel for human CLI output."""

    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    target = console or create_console()
    table = Table.grid(padding=(0, 2))
    table.add_column(style="label", no_wrap=True)
    table.add_column(style="value")
    for key, value in rows:
        table.add_row(str(key), str(value))
    target.print(
        Panel(
            table,
            title=f"[panel.title]{cyber_title(title)}[/panel.title]",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def format_plain_panel(title: str, rows: Iterable[tuple[str, object]]) -> list[str]:
    lines = [f"+-- {cyber_title(title)} " + "-" * 28]
    for key, value in rows:
        lines.append(f"| {key:<12} {value}")
    lines.append("+" + "-" * 48)
    return lines


def format_help_text(*, prog: str = "codepilot", no_color: bool | None = None) -> str:
    """Return a branded argparse-compatible help screen."""

    plain = no_color_requested() if no_color is None else no_color
    mark = PLAIN_CP_MARK if plain else CP_CIRCUIT_MARK
    return f"""{mark}

{cyber_title("Command Deck")}

Usage:
  {prog} [options]
  {prog} -p "explain this function"
  {prog} config <init|show|check|explain> [key]
  {prog} rpc

Options:
  -p, --prompt TEXT          Single prompt mode; prints only the assistant reply
  --cwd, --workspace PATH    Workspace directory (default: current directory)
  --resume SESSION_ID        Resume an existing session
  --model PROVIDER/MODEL     Override model for this run
  --permission-mode MODE     read-only | workspace-write | ask
  --task-mode MODE           read | edit | plan
  --planning-budget PROFILE  conservative | balanced | wide
  --verbose                  Show debug events and config sources
  --no-color                 Disable colored terminal UI
  --version                  Show version and exit

Commands:
  config                     Manage local model/runtime configuration
  rpc                        Start JSONL RPC mode; no human UI is emitted
"""


def format_config_help_text(*, prog: str = "codepilot config") -> str:
    mark = PLAIN_CP_MARK if no_color_requested() else CP_CIRCUIT_MARK
    return f"""{mark}

{cyber_title("Config Deck")}

Usage:
  {prog} init
  {prog} show
  {prog} check
  {prog} explain <key>

Actions:
  init       Create .codepilot/model.local.json
  show       Display sanitized model and settings
  check      Validate model configuration and credentials
  explain    Show where a runtime config value came from
"""


def format_error_text(message: str, *, usage: str | None = None) -> str:
    lines = [cyber_title("Argument Error"), f"Error: {message}"]
    if usage:
        lines.append("")
        lines.append(usage.strip())
    return "\n".join(lines) + "\n"


__all__ = [
    "CP_CIRCUIT_MARK",
    "PLAIN_CP_MARK",
    "create_console",
    "create_theme",
    "cyber_title",
    "format_config_help_text",
    "format_error_text",
    "format_help_text",
    "format_plain_panel",
    "no_color_requested",
    "render_key_value_panel",
]
