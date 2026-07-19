"""Web 服务的配置、启动和关闭入口。"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias


Command: TypeAlias = tuple[str, ...]
NamedProcess: TypeAlias = tuple[str, asyncio.subprocess.Process]
_PROCESS_STOP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class WebServerOptions:
    """Web 服务监听地址、端口和 Runtime 配置。"""

    host: str = "127.0.0.1"
    port: int = 8000
    workspace: Path = Path(".")
    reload: bool = False
    dev: bool = False
    frontend_port: int = 5173

    def __post_init__(self) -> None:
        host = self.host.strip()
        if not host:
            raise ValueError("Web host is required")
        if not 1 <= self.port <= 65535:
            raise ValueError("Web port must be between 1 and 65535")
        if not 1 <= self.frontend_port <= 65535:
            raise ValueError("Web frontend port must be between 1 and 65535")
        if self.dev and self.frontend_port == self.port:
            raise ValueError("Web backend and frontend ports must be different")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())
        if self.dev:
            object.__setattr__(self, "reload", True)
        if host not in {"127.0.0.1", "localhost", "::1"}:
            print(
                "Warning: Codepilot Web has no authentication; non-loopback access is unsafe.",
                file=sys.stderr,
            )


def build_dev_commands(
    options: WebServerOptions,
    *,
    source_root: Path,
    web_root: Path,
    npm_executable: str,
) -> tuple[Command, Command]:
    """Build the reload backend and Vite commands used by development mode."""

    if not (Path(web_root) / "package.json").is_file():
        raise ValueError(f"Web frontend is missing from {web_root}")
    backend = _build_backend_command(options, source_root=source_root)
    frontend = (
        npm_executable,
        "run",
        "dev",
        "--",
        "--host",
        options.host,
        "--port",
        str(options.frontend_port),
        "--strictPort",
    )
    return backend, frontend


def _build_backend_command(
    options: WebServerOptions,
    *,
    source_root: Path,
) -> Command:
    return (
        sys.executable,
        "-m",
        "uvicorn",
        "codepilot.interfaces.web.app:create_app_from_env",
        "--factory",
        "--reload",
        "--host",
        options.host,
        "--port",
        str(options.port),
        "--reload-dir",
        str(Path(source_root).resolve()),
    )


def resolve_dev_layout(module_file: Path = Path(__file__)) -> tuple[Path, Path]:
    """Locate ``src`` and ``web`` in a Codepilot source checkout."""

    for root in Path(module_file).resolve().parents:
        source_root = root / "src"
        web_root = root / "web"
        if (
            (source_root / "codepilot" / "interfaces" / "web").is_dir()
            and (web_root / "package.json").is_file()
        ):
            return source_root, web_root
    raise RuntimeError(
        "codepilot web --dev requires a Codepilot source checkout containing "
        "src/codepilot and web/package.json"
    )


def build_web_environment(
    options: WebServerOptions,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the shared backend/frontend development environment."""

    environment = dict(os.environ if base is None else base)
    environment["CODEPILOT_WEB_WORKSPACE"] = str(options.workspace)
    environment["CODEPILOT_WEB_BACKEND_URL"] = _backend_url(options)
    return environment


def _backend_url(options: WebServerOptions) -> str:
    host = options.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{options.port}"


def _source_root(module_file: Path = Path(__file__)) -> Path:
    for candidate in Path(module_file).resolve().parents:
        if (candidate / "codepilot" / "interfaces" / "web").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the installed Codepilot source package")


def _npm_executable() -> str:
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm") or shutil.which(
        "npm"
    )
    if npm is None:
        raise RuntimeError("codepilot web --dev requires Node.js and npm on PATH")
    return npm


def _ensure_frontend_dependencies(web_root: Path) -> None:
    if not (web_root / "node_modules").is_dir():
        raise RuntimeError(
            f"Web dependencies are missing; run 'npm install' in {web_root}"
        )


async def _spawn_process(
    command: Command,
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> asyncio.subprocess.Process:
    platform_options: dict[str, Any]
    if os.name == "nt":
        platform_options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        platform_options = {"start_new_session": True}
    return await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=dict(environment),
        **platform_options,
    )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        with suppress(ProcessLookupError, OSError):
            process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), _PROCESS_STOP_TIMEOUT_SECONDS)
        return
    except asyncio.TimeoutError:
        pass
    await _kill_process_tree(process)


async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), 2.0)


async def supervise_processes(processes: tuple[NamedProcess, ...]) -> None:
    """Wait for the first child exit and stop every remaining sibling."""

    waits = {
        asyncio.create_task(process.wait(), name=f"web-dev:{name}"): (name, process)
        for name, process in processes
    }
    try:
        done, _pending = await asyncio.wait(
            waits, return_when=asyncio.FIRST_COMPLETED
        )
        completed = next(iter(done))
        name, _process = waits[completed]
        return_code = completed.result()
        if return_code:
            raise RuntimeError(
                f"Web dev {name} process exited with code {return_code}"
            )
        raise RuntimeError(f"Web dev {name} process exited unexpectedly")
    finally:
        for task in waits:
            if not task.done():
                task.cancel()
        await asyncio.gather(*waits, return_exceptions=True)
        await asyncio.gather(
            *(
                _stop_process(process)
                for _name, process in processes
                if process.returncode is None
            ),
            return_exceptions=True,
        )


async def _run_reload_server(options: WebServerOptions) -> None:
    source_root = _source_root()
    environment = build_web_environment(options)
    process = await _spawn_process(
        _build_backend_command(options, source_root=source_root),
        cwd=options.workspace,
        environment=environment,
    )
    try:
        return_code = await process.wait()
        if return_code:
            raise RuntimeError(f"Web reload server exited with code {return_code}")
    finally:
        await _stop_process(process)


async def _run_dev_server(options: WebServerOptions) -> None:
    source_root, web_root = resolve_dev_layout()
    _ensure_frontend_dependencies(web_root)
    npm = _npm_executable()
    backend_command, frontend_command = build_dev_commands(
        options,
        source_root=source_root,
        web_root=web_root,
        npm_executable=npm,
    )
    environment = build_web_environment(options)
    print(f"Codepilot Web API: {_backend_url(options)}")
    print(f"Codepilot Web dev UI: {_frontend_url(options)}")

    backend: asyncio.subprocess.Process | None = None
    frontend: asyncio.subprocess.Process | None = None
    try:
        backend = await _spawn_process(
            backend_command,
            cwd=options.workspace,
            environment=environment,
        )
        frontend = await _spawn_process(
            frontend_command,
            cwd=web_root,
            environment=environment,
        )
        await supervise_processes((("backend", backend), ("frontend", frontend)))
    finally:
        await asyncio.gather(
            *(
                _stop_process(process)
                for process in (backend, frontend)
                if process is not None
            ),
            return_exceptions=True,
        )


def _frontend_url(options: WebServerOptions) -> str:
    host = options.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{options.frontend_port}"


async def run_web_server(options: WebServerOptions) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            'Web dependencies are missing; install Codepilot with pip install -e ".[web]"'
        ) from exc

    if options.dev:
        await _run_dev_server(options)
        return
    if options.reload:
        await _run_reload_server(options)
        return

    os.environ["CODEPILOT_WEB_WORKSPACE"] = str(options.workspace)
    config = uvicorn.Config(
        "codepilot.interfaces.web.app:create_app_from_env",
        host=options.host,
        port=options.port,
        reload=options.reload,
        factory=True,
    )
    await uvicorn.Server(config).serve()


__all__ = ["WebServerOptions", "build_dev_commands", "run_web_server"]
