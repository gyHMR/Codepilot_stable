from __future__ import annotations

"""CLI interface adapter package."""

from .interactive import run_once, run_repl
from .main import build_parser
from .render import CliStartupState, SimpleRenderer, TerminalRenderer, build_startup_state
from .rpc import run_rpc

__all__ = [
    "CliStartupState",
    "SimpleRenderer",
    "TerminalRenderer",
    "build_parser",
    "build_startup_state",
    "run_once",
    "run_repl",
    "run_rpc",
]
