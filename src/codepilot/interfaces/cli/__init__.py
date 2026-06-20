"""CLI 接口适配器包。

提供命令行交互能力，包括参数解析、交互式 REPL、单次输出和 RPC 模式。
"""

from .cli import build_parser, main
from .runner import RunOptions, run, run_interactive, run_print, run_rpc

__all__ = ["RunOptions", "build_parser", "main", "run", "run_interactive", "run_print", "run_rpc"]
