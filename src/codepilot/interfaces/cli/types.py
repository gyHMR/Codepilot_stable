from __future__ import annotations

"""CLI adapter type aliases.

These aliases describe how the command-line interface reads user input,
writes terminal output, and selects an interaction mode.  They intentionally
live in the interface layer rather than ``codepilot.runtime.types`` because
runtime services should not know about terminal concerns.
"""

from typing import Callable, Literal


RunMode = Literal["print", "interactive", "rpc"]
OutputFn = Callable[[str], None]
InputFn = Callable[[str], str]


__all__ = ["InputFn", "OutputFn", "RunMode"]
