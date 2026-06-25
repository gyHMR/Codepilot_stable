"""CLI 接口适配器包。

提供命令行交互能力，包括参数解析、交互式 REPL、单次输出和 RPC 模式。
"""

from .main import build_parser
from .renderer import SimpleRenderer, TerminalRenderer
from .runner import RunOptions, run, run_interactive, run_print, run_rpc
from .startup import CliStartupState, build_startup_state

__all__ = [
    "CliStartupState",
    "RunOptions",
    "SimpleRenderer",
    "TerminalRenderer",
    "build_parser",
    "build_startup_state",
    "run",
    "run_interactive",
    "run_print",
    "run_rpc",
]
