from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WebServerOptions:
    host: str = "127.0.0.1"
    port: int = 8000
    workspace: Path = Path(".")
    reload: bool = False

    def __post_init__(self) -> None:
        host = self.host.strip()
        if not host:
            raise ValueError("Web host is required")
        if not 1 <= self.port <= 65535:
            raise ValueError("Web port must be between 1 and 65535")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())
        if host not in {"127.0.0.1", "localhost", "::1"}:
            print(
                "Warning: Codepilot Web has no authentication; non-loopback access is unsafe.",
                file=sys.stderr,
            )


async def run_web_server(options: WebServerOptions) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            'Web dependencies are missing; install Codepilot with pip install -e ".[web]"'
        ) from exc

    os.environ["CODEPILOT_WEB_WORKSPACE"] = str(options.workspace)
    config = uvicorn.Config(
        "codepilot.interfaces.web.app:create_app_from_env",
        host=options.host,
        port=options.port,
        reload=options.reload,
        factory=True,
    )
    await uvicorn.Server(config).serve()


__all__ = ["WebServerOptions", "run_web_server"]
