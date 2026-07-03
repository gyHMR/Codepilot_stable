from __future__ import annotations

# 新手导读：CLI 类型别名描述输入、输出和运行模式这些界面层概念。
# 关注点：这些类型留在 interfaces，避免 runtime 知道终端细节。

"""CLI adapter type aliases.

These aliases describe how the command-line interface reads user input,
writes terminal output, and selects an interaction mode.  They intentionally
live in the interface layer rather than ``codepilot.runtime.contracts`` because
runtime services should not know about terminal concerns.
"""

from typing import Callable, Literal


RunMode = Literal["print", "interactive", "rpc"]
OutputFn = Callable[[str], None]
InputFn = Callable[[str], str]


__all__ = ["InputFn", "OutputFn", "RunMode"]
