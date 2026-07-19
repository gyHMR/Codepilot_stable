from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


def test_web_dev_cli_options_enable_reload() -> None:
    from codepilot.interfaces.cli.main import build_parser
    from codepilot.interfaces.web.main import WebServerOptions

    args = build_parser().parse_args(
        ["web", "--dev", "--port", "9010", "--frontend-port", "5199"]
    )
    options = WebServerOptions(
        host=args.host,
        port=args.port,
        workspace=Path(args.workspace),
        reload=args.reload,
        dev=args.dev,
        frontend_port=args.frontend_port,
    )

    assert options.dev is True
    assert options.reload is True
    assert options.frontend_port == 5199


def test_dev_commands_use_current_python_and_selected_ports(tmp_path: Path) -> None:
    from codepilot.interfaces.web.main import (
        WebServerOptions,
        build_dev_commands,
    )

    web_root = tmp_path / "web"
    web_root.mkdir()
    (web_root / "package.json").write_text("{}", encoding="utf-8")
    source_root = tmp_path / "src"
    source_root.mkdir()
    options = WebServerOptions(
        port=9010,
        workspace=tmp_path,
        dev=True,
        frontend_port=5199,
    )

    backend, frontend = build_dev_commands(
        options,
        source_root=source_root,
        web_root=web_root,
        npm_executable="npm.cmd",
    )

    assert backend[:3] == (sys.executable, "-m", "uvicorn")
    assert "9010" in backend
    assert str(source_root) in backend
    assert frontend == (
        "npm.cmd",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        "5199",
        "--strictPort",
    )


def test_dev_layout_requires_source_checkout() -> None:
    from codepilot.interfaces.web.main import resolve_dev_layout

    with TemporaryDirectory() as directory:
        module_file = (
            Path(directory)
            / "site-packages"
            / "codepilot"
            / "interfaces"
            / "web"
            / "main.py"
        )

        with pytest.raises(RuntimeError, match="source checkout"):
            resolve_dev_layout(module_file)


def test_dev_environment_targets_workspace_and_backend(tmp_path: Path) -> None:
    from codepilot.interfaces.web.main import WebServerOptions, build_web_environment

    options = WebServerOptions(port=9010, workspace=tmp_path, dev=True)

    environment = build_web_environment(options, base={"PATH": "test-path"})

    assert environment["PATH"] == "test-path"
    assert environment["CODEPILOT_WEB_WORKSPACE"] == str(tmp_path.resolve())
    assert environment["CODEPILOT_WEB_BACKEND_URL"] == "http://127.0.0.1:9010"


def test_supervisor_stops_sibling_when_process_exits(monkeypatch) -> None:
    from codepilot.interfaces.web import main as web_main

    class FakeProcess:
        def __init__(self, pid: int, result: int, wait_forever: bool = False) -> None:
            self.pid = pid
            self.returncode = None
            self._result = result
            self._wait_forever = wait_forever

        async def wait(self) -> int:
            if self._wait_forever:
                await asyncio.Event().wait()
            self.returncode = self._result
            return self._result

    backend = FakeProcess(101, 3)
    frontend = FakeProcess(102, 0, wait_forever=True)
    stopped: list[int] = []

    async def fake_stop(process) -> None:
        stopped.append(process.pid)

    monkeypatch.setattr(web_main, "_stop_process", fake_stop)

    async def run_case() -> None:
        with pytest.raises(RuntimeError, match="backend.*3"):
            await web_main.supervise_processes(
                (("backend", backend), ("frontend", frontend))
            )

    asyncio.run(run_case())
    assert stopped == [102]
