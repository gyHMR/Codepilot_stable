from __future__ import annotations

"""
Codepilot CLI 命令行入口。

设计原则：
- 参数只用于启动和临时覆盖，长期策略进入配置文件
- 高频操作简单，低频高级配置仍可访问
- CLI 是适配层，不复制 Agent、Session 和 Tool 的核心逻辑

示例：
    codepilot                              # 交互式模式（默认）
    codepilot -p "解释 main 函数"           # 单次输出模式
    codepilot --cwd ./project              # 指定工作区
    codepilot --resume SESSION_ID          # 恢复会话
    codepilot --model deepseek/deepseek-chat  # 临时覆盖模型
    codepilot rpc                          # 启动 RPC 模式
    codepilot config init                  # 初始化配置
    codepilot config show                  # 查看配置
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Sequence

from codepilot.runtime.resources import WorkspaceResourceLoader
from codepilot.runtime.assembly import (
    UnknownRuntimeConfigKeyError,
    explain_runtime_config,
)
from codepilot.runtime.service import RuntimeService
from codepilot.runtime.types import ConfigValueSource, CreateAgentSessionOptions

from .runner import RunOptions, run


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


def _init_model_config(workspace: str | Path) -> None:
    """初始化模型配置文件。"""
    config_file = Path(workspace) / ".codepilot" / "model.local.json"
    if config_file.exists():
        raise ValueError(f"Model config already exists: {config_file}")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(_MODEL_CONFIG_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Created {config_file}")
    print("Edit this file, replace api_key (or configure api_key_env), then run `codepilot`.")
    print("Warning: Do not commit this file to version control.")


def _check_model_config(workspace: str | Path) -> None:
    """检查模型配置。"""
    loader = WorkspaceResourceLoader(workspace)
    model = loader.load().model
    if model is None:
        raise ValueError(f"Model config not found: {loader.model_file}")
    if model.api_key_env and os.getenv(model.api_key_env):
        credential_source = f"environment:{model.api_key_env}"
    elif model.api_key:
        credential_source = "local-file (do not commit)"
    else:
        credential_source = "missing"
    print(f"config    = {loader.model_file}")
    print(f"api       = {model.api}")
    print(f"provider  = {model.provider}")
    print(f"model_id  = {model.model_id}")
    print(f"base_url  = {model.base_url}")
    print(f"credential = {credential_source}")
    if credential_source == "missing":
        print("status    = MISSING_CREDENTIAL")
        print(f"Set environment variable {model.api_key_env} or add api_key to {loader.model_file}")
    else:
        print("status    = valid")


def _show_config(workspace: str | Path) -> None:
    """显示当前配置（脱敏）。"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    loader = WorkspaceResourceLoader(workspace)
    loaded = loader.load()
    model = loaded.model
    settings = loaded.settings

    # 模型配置表格
    model_table = Table(show_header=True, box=None, padding=(0, 2))
    model_table.add_column("Key", style="bold cyan", width=12)
    model_table.add_column("Value")

    if model:
        model_table.add_row("provider", model.provider)
        model_table.add_row("model_id", model.model_id)
        model_table.add_row("base_url", model.base_url)
        model_table.add_row("api", model.api)
        if model.api_key_env and os.getenv(model.api_key_env):
            model_table.add_row("credential", f"env:{model.api_key_env}")
        elif model.api_key:
            model_table.add_row("credential", "local-file (do not commit)")
        else:
            model_table.add_row("credential", "[red]MISSING[/red]")
    else:
        model_table.add_row("status", "[dim](not configured)[/dim]")

    console.print(Panel(model_table, title="[bold]Model Config[/bold]", border_style="blue"))

    # 设置表格（只显示非 None 的配置）
    settings_table = Table(show_header=True, box=None, padding=(0, 2))
    settings_table.add_column("Key", style="bold cyan", width=25)
    settings_table.add_column("Value")

    if settings:
        import dataclasses
        if dataclasses.is_dataclass(settings):
            for field in dataclasses.fields(settings):
                value = getattr(settings, field.name)
                # 跳过 None 值和敏感信息
                if value is None:
                    continue
                if "key" in field.name.lower() or "secret" in field.name.lower():
                    continue
                settings_table.add_row(field.name, str(value))

    if settings_table.row_count == 0:
        settings_table.add_row("status", "[dim](using defaults)[/dim]")

    console.print(Panel(settings_table, title="[bold]Settings[/bold]", border_style="blue"))


def _explain_config(options: CreateAgentSessionOptions, key: str | None) -> None:
    """解释配置项的来源。

    使用 RuntimeConfigResolver 追踪每个配置项的最终值和来源。
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    if not key:
        console.print("[red]Usage: codepilot config explain <key>[/red]")
        console.print("Available keys: model, provider, model_id, thinking_level, tool_execution, etc.")
        return

    try:
        resolved = explain_runtime_config(options, key)
    except UnknownRuntimeConfigKeyError:
        console.print(f"[red]Unknown config key: {key}[/red]")
    except KeyError as exc:
        raise ValueError(str(exc).strip("'")) from exc
    else:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Field", style="bold", width=12)
        table.add_column("Value")
        table.add_row("Key", key)
        table.add_row("Value", str(resolved.value))
        table.add_row("Source", _format_source(resolved.source))
        console.print(Panel(table, title=f"[bold]Config: {key}[/bold]", border_style="blue"))


def _format_source(source: ConfigValueSource) -> str:
    """格式化配置来源显示。"""
    source_map = {
        "cli": "CLI argument",
        "session": "restored session",
        "project": "project",
        "default": "built-in default",
    }
    label = source_map.get(source.kind, source.kind)
    return f"{label}:{source.location}" if source.location else label


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    一级参数只保留高频、启动时需要的配置。
    低频配置通过配置文件或未来的 --set 机制覆盖。
    """
    parser = argparse.ArgumentParser(
        description="Codepilot - Local AI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""\
Examples:
  codepilot                              Interactive mode (default)
  codepilot -p "explain this function"   Single prompt mode
  codepilot --cwd ./project              Specify workspace
  codepilot --resume SESSION_ID          Resume session
  codepilot --model deepseek/deepseek-chat  Override model
  codepilot rpc                          Start RPC mode
  codepilot config init                  Initialize config
  codepilot config show                  Show current config
""",
    )

    # ── 核心参数（推荐使用） ─────────────────────────────────────

    parser.add_argument(
        "-p", "--prompt",
        default=None,
        help="Single prompt mode: run once and exit",
    )
    parser.add_argument(
        "--cwd", "--workspace",
        default=".",
        dest="workspace",
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--resume", "--session-id",
        default=None,
        dest="session_id",
        help="Resume existing session by ID",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model (format: provider/model-id)",
    )
    parser.add_argument(
        "--permission-mode",
        default=None,
        choices=["read-only", "workspace-write", "ask"],
        help="Permission mode (default: workspace-write)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show debug events and config sources",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable colored output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.3",
    )

    # ── 子命令 ──────────────────────────────────────────────────

    subparsers = parser.add_subparsers(dest="command")

    # config 子命令
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument(
        "config_action",
        choices=["init", "show", "check", "explain"],
        help="Config action to perform",
    )
    config_parser.add_argument(
        "config_key",
        nargs="?",
        default=None,
        help="Config key to explain (for 'explain' action)",
    )

    # rpc 子命令
    subparsers.add_parser("rpc", help="Start RPC mode (JSONL protocol)")

    return parser


def _resolve_run_mode(args: argparse.Namespace) -> str:
    """解析运行模式。

    优先级：-p 参数、rpc 子命令、默认交互模式。
    """
    if args.prompt:
        return "print"
    if args.command == "rpc":
        return "rpc"
    return "interactive"


def _resolve_model_id(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """解析模型标识。

    --model 必须使用 provider/model-id 格式。

    返回:
        (provider, model_id) 元组
    """
    if args.model:
        parts = args.model.split("/", 1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError("--model must use provider/model-id format")
        return parts[0], parts[1]
    return None, None


def _resolve_permission_mode(args: argparse.Namespace) -> str | None:
    """解析权限模式。"""
    return args.permission_mode


async def _run_from_args(args: argparse.Namespace) -> int:
    """根据解析后的命令行参数执行 Agent 会话。

    主要流程：
    1. 处理 config 子命令
    2. 将 CLI 参数转换为 CreateAgentSessionOptions
    3. 通过 RuntimeService 创建 Agent 会话
    4. 执行会话管理操作或正常运行
    5. 确保会话在退出时正确关闭
    """
    workspace = Path(args.workspace)

    # ── 处理 config 子命令 ──────────────────────────────────────

    if args.command == "config":
        if args.config_action == "init":
            _init_model_config(workspace)
            return 0
        if args.config_action == "show":
            _show_config(workspace)
            return 0
        if args.config_action == "check":
            _check_model_config(workspace)
            return 0
        if args.config_action == "explain":
            provider, model_id = _resolve_model_id(args)
            _explain_config(
                CreateAgentSessionOptions(
                    workspace_dir=workspace,
                    provider=provider,
                    model_id=model_id,
                    session_id=args.session_id,
                    read_only_mode=(
                        args.permission_mode == "read-only"
                        if args.permission_mode is not None
                        else None
                    ),
                    tool_permission_mode=args.permission_mode,
                ),
                args.config_key,
            )
            return 0

    # ── 解析运行模式和模型 ──────────────────────────────────────

    run_mode = _resolve_run_mode(args)
    provider, model_id = _resolve_model_id(args)
    permission_mode = _resolve_permission_mode(args)

    # ── 构建会话配置 ────────────────────────────────────────────

    options = CreateAgentSessionOptions(
        workspace_dir=workspace,
        provider=provider,
        model_id=model_id,
        session_id=args.session_id,
        read_only_mode=(
            permission_mode == "read-only"
            if permission_mode is not None
            else None
        ),
        tool_permission_mode=permission_mode,
    )
    if run_mode == "interactive":
        from .approval import CliApprovalProvider

        options.approval_provider = CliApprovalProvider()

    # ── 创建会话并运行 ──────────────────────────────────────────

    runtime = RuntimeService()
    handle = runtime.create_session(options)

    try:
        await run(
            RunOptions(
                mode=run_mode,
                session_id=handle.session_id,
                runtime=runtime,
                prompt=args.prompt,
                verbose=args.verbose,
                no_color=args.no_color,
            )
        )
    finally:
        # 关闭所有会话（包括 fork 出来的新会话）
        await runtime.aclose_all()

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口函数。

    解析命令行参数，启动异步事件循环执行 Agent 会话。
    捕获 ValueError 并通过 parser.error() 输出友好的错误信息。
    """
    from codepilot.runtime.service import SessionBusyError, SessionNotFoundError, EmptyInputError

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        return asyncio.run(_run_from_args(args))
    except KeyboardInterrupt:
        print("\nBye.")
        return 130
    except SessionNotFoundError as exc:
        parser.error(str(exc))
        return 2
    except SessionBusyError as exc:
        print(f"Error: {exc}")
        return 1
    except EmptyInputError as exc:
        print(f"Error: {exc}")
        return 1
    except ValueError as exc:
        parser.error(str(exc))
        return 2


# 脚本直接运行时的入口点
if __name__ == "__main__":
    raise SystemExit(main())
