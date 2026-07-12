from __future__ import annotations

import asyncio
from pathlib import Path


def test_web_subcommand_defaults_to_localhost_and_current_workspace() -> None:
    from codepilot.interfaces.cli.main import build_parser

    args = build_parser().parse_args(["web"])

    assert args.command == "web"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.workspace == "."
    assert args.reload is False


def test_web_subcommand_accepts_server_options() -> None:
    from codepilot.interfaces.cli.main import build_parser

    args = build_parser().parse_args(
        [
            "web",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--workspace",
            "repo",
            "--reload",
        ]
    )

    assert (args.host, args.port, args.workspace, args.reload) == (
        "0.0.0.0",
        9000,
        "repo",
        True,
    )


def test_web_subcommand_dispatches_without_opening_cli_session(monkeypatch, tmp_path) -> None:
    from codepilot.interfaces.cli.main import _run_from_args, build_parser

    captured = []
    monkeypatch.setattr(
        "codepilot.interfaces.web.main.run_web_server",
        lambda options: captured.append(options),
    )
    args = build_parser().parse_args(["web", "--workspace", str(tmp_path)])

    result = asyncio.run(_run_from_args(args))

    assert result == 0
    assert captured[0].workspace == Path(tmp_path).resolve()


def test_non_loopback_host_warns_about_missing_authentication(capsys, tmp_path) -> None:
    from codepilot.interfaces.web.main import WebServerOptions

    WebServerOptions(host="0.0.0.0", port=8000, workspace=tmp_path)

    assert "no authentication" in capsys.readouterr().err.lower()
