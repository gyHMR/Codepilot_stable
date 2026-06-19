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
from codepilot.runtime.service import RuntimeService
from codepilot.runtime.types import CreateAgentSessionOptions

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


def _explain_config(workspace: str | Path, key: str | None) -> None:
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

    # 加载配置并解析来源
    from codepilot.runtime.config import load_runtime_inputs, resolve_runtime_config
    from codepilot.runtime.types import CreateAgentSessionOptions

    options = CreateAgentSessionOptions(workspace_dir=workspace)
    inputs = load_runtime_inputs(options)
    resolved = resolve_runtime_config(options, inputs)

    # 特殊处理 model 相关配置
    if key in ("model", "provider", "model_id"):
        _explain_model_config(workspace, key, resolved, inputs)
        return

    # 通用配置项解释
    if key in resolved.sources:
        source = resolved.sources[key]
        value = getattr(resolved, key, None)

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Field", style="bold", width=12)
        table.add_column("Value")
        table.add_row("Key", key)
        table.add_row("Value", str(value))
        table.add_row("Source", _format_source(source))
        console.print(Panel(table, title=f"[bold]Config: {key}[/bold]", border_style="blue"))
    else:
        console.print(f"[red]Unknown config key: {key}[/red]")


def _explain_model_config(
    workspace: str | Path,
    key: str,
    resolved: Any,
    inputs: Any,
) -> None:
    """解释模型相关配置的来源。"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    loader = WorkspaceResourceLoader(workspace)
    loaded = loader.load()
    model = loaded.model

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold", width=12)
    table.add_column("Value")

    if key == "model":
        # 显示完整的模型标识
        if model:
            model_id = f"{model.provider}/{model.model_id}"
            table.add_row("Value", model_id)
            table.add_row("Source", "project:.codepilot/model.local.json")
        else:
            table.add_row("Value", "(not configured)")
            table.add_row("Source", "default")
    elif key == "provider":
        if model:
            table.add_row("Value", model.provider)
            table.add_row("Source", "project:.codepilot/model.local.json")
        else:
            table.add_row("Value", "(not configured)")
    elif key == "model_id":
        if model:
            table.add_row("Value", model.model_id)
            table.add_row("Source", "project:.codepilot/model.local.json")
        else:
            table.add_row("Value", "(not configured)")

    console.print(Panel(table, title=f"[bold]Config: {key}[/bold]", border_style="blue"))


def _format_source(source: str) -> str:
    """格式化配置来源显示。"""
    source_map = {
        "options": "CLI argument",
        "restored_session": "restored session",
        "workspace": "project:.codepilot/settings.json",
        "default": "built-in default",
    }
    return source_map.get(source, source)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    一级参数只保留高频、启动时需要的配置。
    低频配置通过配置文件或未来的 --set 机制覆盖。
    """
    parser = argparse.ArgumentParser(
        description="Codepilot - Local AI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        choices=["read-only", "workspace-write"],
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

    # ── 高级参数（兼容旧接口） ──────────────────────────────────

    advanced_group = parser.add_argument_group("advanced options (legacy)")

    advanced_group.add_argument(
        "--mode",
        choices=["print", "interactive", "rpc"],
        default=None,
        help=argparse.SUPPRESS,  # 隐藏，使用子命令替代
    )
    advanced_group.add_argument(
        "--provider",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--model-id",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--system-prompt",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--thinking-level",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--tool-execution",
        choices=["parallel", "sequential"],
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--read-only",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--no-tool-events",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    # --session-id 已在核心参数中定义为 --resume
    advanced_group.add_argument(
        "--init-config",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--check-config",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--max-context-messages",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--retain-recent-messages",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--no-retry",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--retry-base-delay-ms",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--allow-dangerous-bash",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--bash-allow-pattern",
        action="append",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--bash-block-pattern",
        action="append",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--relaxed-edit",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--disable-workspace-resources",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    # 会话管理命令（保留兼容）
    advanced_group.add_argument(
        "--list-entries",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--show-tree",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--fork-entry",
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_group.add_argument(
        "--switch-entry",
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser


def _resolve_run_mode(args: argparse.Namespace) -> str:
    """解析运行模式。

    优先级：
    1. -p 参数 → print 模式
    2. rpc 子命令 → rpc 模式
    3. --mode 参数（兼容）
    4. 默认 → interactive 模式
    """
    if args.prompt:
        return "print"
    if args.command == "rpc":
        return "rpc"
    if args.mode:
        return args.mode
    return "interactive"


def _resolve_model_id(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """解析模型标识。

    支持两种格式：
    1. --model provider/model-id → 拆分为 provider 和 model_id
    2. --provider + --model-id（兼容旧接口）

    返回:
        (provider, model_id) 元组
    """
    if args.model:
        parts = args.model.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            # 没有 / 分隔符，整个作为 model_id
            return None, args.model
    return args.provider, args.model_id


def _resolve_permission_mode(args: argparse.Namespace) -> str:
    """解析权限模式。"""
    if args.permission_mode:
        return args.permission_mode
    if args.read_only:
        return "read-only"
    return "workspace-write"


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
            _explain_config(workspace, args.config_key)
            return 0

    # ── 处理旧的 init-config / check-config 参数 ────────────────

    if args.init_config:
        _init_model_config(workspace)
        return 0

    if args.check_config:
        _check_model_config(workspace)
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
        system_prompt=args.system_prompt,
        session_id=args.session_id,
        thinking_level=args.thinking_level,
        tool_execution=args.tool_execution,
        max_context_messages=args.max_context_messages,
        max_context_tokens=args.max_context_tokens,
        retain_recent_messages=args.retain_recent_messages,
        retry_enabled=None if args.no_retry is None else False,
        max_retries=args.max_retries,
        retry_base_delay_ms=args.retry_base_delay_ms,
        read_only_mode=(permission_mode == "read-only"),
        block_dangerous_bash=None if args.allow_dangerous_bash is None else False,
        bash_allow_patterns=args.bash_allow_pattern,
        bash_block_patterns=args.bash_block_pattern,
        edit_require_unique_match=None if args.relaxed_edit is None else False,
        load_workspace_resources=not args.disable_workspace_resources,
    )

    # ── 创建会话并运行 ──────────────────────────────────────────

    runtime = RuntimeService()
    session = runtime.create_session(options).session

    try:
        # 会话管理操作（兼容旧接口）
        if args.switch_entry:
            session.switch_to_entry(str(args.switch_entry))
            print(json.dumps({"type": "switch_entry", "session_id": session.session_id, "entry_id": args.switch_entry}))
            return 0

        if args.fork_entry:
            forked = session.fork_from_entry(str(args.fork_entry))
            try:
                print(json.dumps({
                    "type": "forked",
                    "from_session_id": session.session_id,
                    "from_entry_id": args.fork_entry,
                    "new_session_id": forked.session_id,
                }, ensure_ascii=False))
            finally:
                forked.close()
            return 0

        if args.list_entries:
            print(json.dumps({
                "session_id": session.session_id,
                "entry_ids": session.list_entry_ids(),
            }, ensure_ascii=False))
            return 0

        if args.show_tree:
            print(json.dumps({
                "session_id": session.session_id,
                "tree": session.get_session_tree(),
            }, ensure_ascii=False))
            return 0

        # 正常运行
        await run(
            RunOptions(
                mode=run_mode,
                session=session,
                prompt=args.prompt,
                verbose=args.verbose,
                no_color=args.no_color,
            )
        )
    finally:
        runtime.close_session(session.session_id)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口函数。

    解析命令行参数，启动异步事件循环执行 Agent 会话。
    捕获 ValueError 并通过 parser.error() 输出友好的错误信息。
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        return asyncio.run(_run_from_args(args))
    except KeyboardInterrupt:
        print("\nBye.")
        return 130
    except ValueError as exc:
        parser.error(str(exc))
        return 2


# 脚本直接运行时的入口点
if __name__ == "__main__":
    raise SystemExit(main())
