# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：interfaces 层只做 CLI/Web 输入输出适配，不复制 core/tools/runtime 的业务逻辑。

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
