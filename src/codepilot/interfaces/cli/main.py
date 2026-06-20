"""Codepilot CLI 入口模块。

导出 build_parser 和 main，供 __main__.py 和外部脚本调用。
"""

from .cli import build_parser, main

__all__ = ["build_parser", "main"]
