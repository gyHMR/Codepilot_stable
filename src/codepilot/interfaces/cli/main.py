from __future__ import annotations

"""Codepilot command line entrypoint."""

import argparse
import asyncio
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
    """ArgumentParser with branded human-facing help and errors."""

    def __init__(self, *args: object, help_kind: str = "main", **kwargs: object) -> None:
        self.help_kind = help_kind
        super().__init__(*args, **kwargs)

    def format_help(self) -> str:
        if self.help_kind == "config":
            return format_config_help_text(prog=self.prog)
        return format_help_text(prog=self.prog)

    def error(self, message: str) -> None:
        self.print_usage()
        self._print_message(format_error_text(message), sys.stderr)
        self.exit(2)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = CodepilotArgumentParser(
        description="Codepilot - Local AI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
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
        "--task-mode",
        default=None,
        choices=["read", "plan", "build"],
        help="Task mode for this session (default: build)",
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
    return parser


def build_session_intent(args: argparse.Namespace) -> SessionOpenIntent:
    """Translate parsed CLI args into the runtime session-open contract."""

    provider, model_id = _resolve_model_id(args.model)
    permission_mode = args.permission_mode
    return SessionOpenIntent(
        workspace_dir=Path(args.workspace),
        provider=provider,
        model_id=model_id,
        session_id=args.session_id,
        read_only_mode=(
            permission_mode == "read-only"
            if permission_mode is not None
            else None
        ),
        tool_permission_mode=permission_mode,
        task_mode=args.task_mode,
        planning_budget_profile=args.planning_budget_profile,
    )


async def _run_from_args(args: argparse.Namespace) -> int:
    """Run the selected CLI mode from parsed args."""

    intent = build_session_intent(args)
    if args.command == "config":
        run_config_command(
            args.config_action,
            workspace=Path(args.workspace),
            key=args.config_key,
            intent=intent,
        )
        return 0

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
    if model is None:
        return None, None
    parts = model.split("/", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError("--model must use provider/model-id format")
    return parts[0].strip(), parts[1].strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI args and run the selected mode."""

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
