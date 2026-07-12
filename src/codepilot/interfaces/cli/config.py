from __future__ import annotations

"""面向人类用户的 ``codepilot config`` 子命令。

本文件负责配置相关 CLI 交互：
- ``init``：生成可编辑的本地模型配置模板。
- ``show``：展示脱敏后的模型配置和运行时设置。
- ``check``：检查 workspace 中的模型配置是否可用。
- ``explain``：解释某个配置值来自 CLI、session、项目文件还是默认值。

配置的真实解析逻辑在 ``codepilot.runtime.config``。CLI 层只调用 runtime 暴露的视图，
再把结果渲染给用户，避免出现“CLI 一套配置规则、runtime 一套配置规则”。
"""

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
"""``codepilot config init`` 写入 ``.codepilot/model.local.json`` 的默认模板。

模板故意包含 ``api_key`` 和 ``api_key_env`` 两种形式，方便新手直接填写 key，
也支持把密钥放在环境变量中。写文件时使用 ``ensure_ascii=False`` 和 UTF-8，
保证中文注释或未来扩展字段不会被破坏。
"""


def run_config_command(
    action: str,
    *,
    workspace: str | Path,
    key: str | None = None,
    intent: SessionOpenIntent | None = None,
) -> None:
    """执行一个 ``codepilot config`` 子动作。

    Args:
        action: 用户输入的配置动作，当前支持 ``init``、``show``、``check``、``explain``。
        workspace: 当前项目目录；配置文件会放在该目录下的 ``.codepilot``。
        key: ``explain`` 动作要解释的配置键，例如 ``model`` 或 ``thinking_level``。
        intent: CLI 已经构造好的 ``SessionOpenIntent``。``explain`` 需要它来复用
            runtime 的完整配置解析规则。

    Raises:
        ValueError: 动作未知，或 ``explain`` 缺少 session intent。
    """

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
    """创建可编辑的本地模型配置模板。

    Args:
        workspace: 配置所在的 workspace 目录。

    Raises:
        ValueError: 当 ``.codepilot/model.local.json`` 已存在时抛出，避免覆盖用户密钥。

    该方法只负责写入模板和提示用户下一步；真实配置读取仍由 runtime 层完成。
    """

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
    """检查本地模型配置并渲染简短报告。

    Args:
        workspace: 要检查的 workspace 目录。

    ``check_workspace_model_config`` 会返回适合展示的 rows 和边框样式，CLI 只负责渲染。
    """

    view = check_workspace_model_config(workspace)
    render_key_value_panel(
        "Config Check",
        list(view.rows),
        border_style=view.border_style,
    )


def show_config(workspace: str | Path) -> None:
    """展示脱敏后的模型配置和运行时设置。

    Args:
        workspace: 要读取配置的 workspace 目录。

    注意这里展示的是 runtime 返回的 view，敏感字段已经由 runtime/config 做过处理；
    CLI 不应该自行读取原始密钥文件。
    """

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
    """展示某个运行时配置值的来源。

    Args:
        intent: 当前 CLI 参数构造出的 session 打开意图。它包含 workspace、模型覆盖、
            权限模式等信息，是 runtime 解释配置来源的输入。
        key: 要解释的配置键。为空时输出使用方法，不抛异常。

    ``explain`` 适合排查“为什么实际用的是这个模型/权限模式/配置值”这类问题。
    """

    from rich.panel import Panel
    from rich.table import Table

    console = create_console()
    if not key:
        console.print("[error]Usage: codepilot config explain <key>[/error]")
        console.print("[muted2]Available keys: model, provider, model_id, thinking_level, etc.[/muted2]")
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
    """把 runtime 配置来源对象格式化成人类可读文本。

    Args:
        source: runtime 返回的来源对象，通常具有 ``kind`` 和可选 ``location`` 字段。

    Returns:
        形如 ``CLI argument``、``project:path``、``built-in default`` 的字符串。
    """

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
