"""CLI 界面包：把用户输入和 Runtime 帧转换为终端交互。"""

from __future__ import annotations

"""CLI interface adapter package.

本文件是 ``codepilot.interfaces.cli`` 包的公开出口。

它不负责执行业务逻辑，只把 CLI 层外部需要使用的入口函数和渲染器集中导出：
- ``run_once`` / ``run_repl``：人类终端模式的运行入口。
- ``run_rpc``：给编辑器、脚本等非人类客户端使用的 JSONL RPC 入口。
- ``build_parser``：命令行参数解析器构造函数。
- ``TerminalRenderer`` / ``SimpleRenderer``：CLI 层的显示适配器。

这里的导出应当保持克制，避免把 CLI 内部辅助函数变成跨层依赖。
"""

from .interactive import run_once, run_repl
from .main import build_parser
from .render import CliStartupState, SimpleRenderer, TerminalRenderer, build_startup_state
from .rpc import run_rpc

# CLI 包级 public API。其他层如果确实需要使用 CLI，优先从这里导入稳定入口；
# 文件内部的解析、渲染、错误转换等辅助函数不在这里暴露。
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
