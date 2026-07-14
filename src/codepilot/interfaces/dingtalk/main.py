"""DingTalk 适配器的命令行入口与服务启动逻辑。"""

from __future__ import annotations

# 新手导读：main.py 是独立的 codepilot-dingtalk 入口，不复用 CLI parser。
# 关注点：钉钉远程入口不能改变 codepilot、codepilot -p、RPC 或斜杠命令行为。

"""Command entrypoint for ``codepilot-dingtalk``."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence

from .bridge import DingTalkBridge
from .schemas import DingTalkBridgeConfig
from .transport import create_stream_transport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codepilot-dingtalk",
        description="Run Codepilot through a DingTalk Stream bot.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser(
        "serve",
        help="start DingTalk Stream bridge",
        description="start DingTalk Stream bridge",
    )
    serve.add_argument(
        "--cwd",
        "--workspace",
        default=".",
        dest="workspace",
        help="Workspace directory bound to this bridge.",
    )
    serve.add_argument(
        "--session-id",
        default=None,
        help="Resume or bind to a specific Codepilot session id.",
    )
    serve.add_argument(
        "--model",
        default=None,
        help="Override model with provider/model-id.",
    )
    serve.add_argument(
        "--allowed-user",
        action="append",
        default=[],
        help="DingTalk sender staff id allowed to control this bridge.",
    )
    serve.add_argument(
        "--allow-dirty",
        action="store_true",
        default=False,
        help="Allow remote runs when the git workspace has user changes.",
    )
    serve.add_argument(
        "--verbose-events",
        action="store_true",
        default=False,
        help="Send non-final tool progress events to DingTalk.",
    )
    return parser


async def serve(args: argparse.Namespace) -> int:
    client_id = _required_env("DINGTALK_CLIENT_ID")
    client_secret = _required_env("DINGTALK_CLIENT_SECRET")
    provider, model_id = _parse_model(args.model)
    allowed_users = _allowed_users(args.allowed_user)
    config = DingTalkBridgeConfig(
        workspace_dir=str(Path(args.workspace).resolve()),
        allowed_users=allowed_users,
        session_id=args.session_id,
        provider=provider,
        model_id=model_id,
        allow_dirty=args.allow_dirty,
        verbose_events=args.verbose_events,
    )
    bridge = DingTalkBridge(config=config)
    transport = create_stream_transport(
        client_id=client_id,
        client_secret=client_secret,
    )
    await transport.start(bridge)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "serve":
            return asyncio.run(serve(args))
        parser.error(f"unknown command: {args.command}")
        return 2
    except KeyboardInterrupt:
        print("\nBye.")
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"codepilot-dingtalk: {exc}", file=sys.stderr)
        return 2


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value


def _parse_model(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    parts = value.split("/", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError("--model must use provider/model-id format")
    return parts[0].strip(), parts[1].strip()


def _allowed_users(cli_values: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    values.extend(cli_values)
    env_value = os.getenv("CODEPILOT_DINGTALK_ALLOWED_USERS", "")
    for chunk in env_value.replace(";", ",").split(","):
        if chunk.strip():
            values.append(chunk)
    if not values:
        raise ValueError(
            "Configure --allowed-user or CODEPILOT_DINGTALK_ALLOWED_USERS before starting DingTalk bridge"
        )
    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "serve"]
