from __future__ import annotations

"""Codepilot 命令行入口。

本文件负责把用户在终端里输入的命令行参数翻译成 runtime 层能理解的动作：

1. ``build_parser`` 定义 ``codepilot`` 支持哪些参数和子命令。
2. ``build_session_intent`` 把解析后的参数转换成 ``SessionOpenIntent``。
3. ``_run_from_args`` 根据模式选择 config / rpc / 单次 prompt / 交互 REPL。
4. ``main`` 负责进程级错误处理和退出码。

注意：CLI 层只做“输入解释”和“入口分流”，不直接管理 agent loop、工具执行或 session 存储。
"""

import argparse
import asyncio
import inspect
import sys
from pathlib import Path
from typing import Sequence

from codepilot.runtime import RuntimeGateway, SessionOpenIntent

from .config import (
    check_model_config as _check_model_config,
    init_model_config as _init_model_config,
    run_config_command,
    show_config as _show_config,
)
from .interactive import run_once, run_repl
from .render import format_config_help_text, format_error_text, format_help_text
from .rpc import run_rpc


class CodepilotArgumentParser(argparse.ArgumentParser):
    """带 Codepilot 风格帮助文本的参数解析器。

    ``argparse.ArgumentParser`` 默认帮助文本偏通用，这里重写 ``format_help`` 和
    ``error``，让主命令与 ``config`` 子命令使用统一的终端文案。
    """

    def __init__(self, *args: object, help_kind: str = "main", **kwargs: object) -> None:
        """创建解析器。

        Args:
            *args: 透传给 ``argparse.ArgumentParser`` 的位置参数。
            help_kind: 帮助文本类型。``"main"`` 表示主命令帮助，
                ``"config"`` 表示 ``codepilot config`` 子命令帮助。
            **kwargs: 透传给 ``argparse.ArgumentParser`` 的关键字参数。
        """
        self.help_kind = help_kind
        super().__init__(*args, **kwargs)

    def format_help(self) -> str:
        """返回 CLI 定制帮助文本。

        argparse 在用户执行 ``--help`` 时会调用该方法。这里根据 ``help_kind`` 区分
        主命令和 config 子命令，避免在业务代码里散落多套帮助文案。
        """
        if self.help_kind == "config":
            return format_config_help_text(prog=self.prog)
        return format_help_text(prog=self.prog)

    def error(self, message: str) -> None:
        """格式化参数错误并以退出码 2 结束进程。

        Args:
            message: argparse 发现的错误文本，例如缺少参数或参数值非法。
        """
        self.print_usage()
        self._print_message(format_error_text(message), sys.stderr)
        self.exit(2)


def build_parser() -> argparse.ArgumentParser:
    """构造 ``codepilot`` 命令行参数解析器。

    返回值是 argparse 解析器，调用方通常是 ``main``。参数只描述用户意图，
    不在这里打开 runtime 或执行命令，方便测试和复用。
    """

    parser = CodepilotArgumentParser(
        description="Codepilot - Local AI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )

    # 普通运行参数：决定本次会话使用哪个 workspace、model、权限模式和运行模式。
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Single prompt mode: run once and exit",
    )
    parser.add_argument(
        "--cwd",
        "--workspace",
        default=".",
        dest="workspace",
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--resume",
        "--session-id",
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
        "--mode",
        default=None,
        dest="current_mode",
        choices=["read", "plan", "build"],
        help="Mode for this session (default: build)",
    )
    parser.add_argument(
        "--planning-budget",
        default=None,
        choices=["conservative", "balanced", "wide"],
        dest="planning_budget_profile",
        help="Planning discovery budget profile (default: balanced)",
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

    # 子命令只做入口分流。config 仍然会经过 SessionOpenIntent，
    # 因为 explain 需要复用 runtime 的配置解析规则。
    subparsers = parser.add_subparsers(dest="command", parser_class=CodepilotArgumentParser)
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        help_kind="config",
    )
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
    subparsers.add_parser("rpc", help="Start RPC mode (JSONL protocol)")
    web_parser = subparsers.add_parser("web", help="Start local Web workspace")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000)
    web_parser.add_argument("--workspace", default=".")
    web_parser.add_argument("--reload", action="store_true", default=False)
    return parser


def build_session_intent(args: argparse.Namespace) -> SessionOpenIntent:
    """把 CLI 参数翻译成 runtime 的会话打开契约。

    Args:
        args: ``build_parser`` 解析出的 argparse 命名空间。

    Returns:
        ``SessionOpenIntent``。runtime 层只需要理解这个 DTO，不需要知道 argparse。

    这里是 CLI 到 runtime 的关键边界：路径、模型覆盖、权限模式、运行模式都会在这里
    汇总成明确字段，而不是让 runtime 回头解析命令行参数。
    """

    provider, model_id = _resolve_model_id(args.model)
    return SessionOpenIntent(
        workspace_dir=Path(args.workspace),
        provider=provider,
        model_id=model_id,
        session_id=args.session_id,
        tool_permission_mode=args.permission_mode,
        current_mode=args.current_mode,
        planning_budget_profile=args.planning_budget_profile,
    )


async def _run_from_args(args: argparse.Namespace) -> int:
    """根据解析后的参数运行对应 CLI 模式。

    Args:
        args: argparse 解析结果。

    Returns:
        进程退出码。正常完成返回 0，参数错误由 ``main`` 捕获后返回 2。

    分流规则：
    - ``config``：执行配置管理命令，不进入 agent 对话。
    - ``rpc``：启动 JSONL RPC 协议，供外部程序调用。
    - ``--prompt``：单次提问，运行完即退出。
    - 默认：进入交互式 REPL。
    """

    if args.command == "web":
        from codepilot.interfaces.web.main import WebServerOptions, run_web_server

        server_result = run_web_server(
            WebServerOptions(
                host=args.host,
                port=args.port,
                workspace=Path(args.workspace),
                reload=args.reload,
            )
        )
        if inspect.isawaitable(server_result):
            await server_result
        return 0

    intent = build_session_intent(args)
    if args.command == "config":
        run_config_command(
            args.config_action,
            workspace=Path(args.workspace),
            key=args.config_key,
            intent=intent,
        )
        return 0

    # RuntimeGateway 是 CLI 进入 runtime 层的唯一对象。CLI 不直接创建 core/session/tool。
    runtime = RuntimeGateway()
    handle = runtime.open_session(intent)
    try:
        if args.command == "rpc":
            await run_rpc(runtime, handle.session_id)
        elif args.prompt:
            await run_once(runtime, handle.session_id, args.prompt)
        else:
            await run_repl(
                runtime,
                handle.session_id,
                verbose=args.verbose,
                no_color=args.no_color,
            )
    finally:
        await runtime.close_all()
    return 0


def _resolve_model_id(model: str | None) -> tuple[str | None, str | None]:
    """解析 ``--model provider/model-id`` 参数。

    Args:
        model: 用户传入的模型覆盖值；未传时为 ``None``。

    Returns:
        ``(provider, model_id)``。没有覆盖时两个值都为 ``None``，让 runtime 使用配置默认值。

    Raises:
        ValueError: 当格式不是 ``provider/model-id`` 或任一部分为空时抛出。
    """
    if model is None:
        return None, None
    parts = model.split("/", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError("--model must use provider/model-id format")
    return parts[0].strip(), parts[1].strip()


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并启动 CLI。

    Args:
        argv: 可选参数列表。测试时可传入显式列表；正常运行时为 ``None``，
            argparse 会读取 ``sys.argv``。

    Returns:
        进程退出码：0 表示成功，130 表示用户 Ctrl+C 中断，2 表示参数错误。
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


if __name__ == "__main__":
    raise SystemExit(main())
