"""定义扩展向运行系统注册命令、工具和生命周期钩子的 API。"""

from __future__ import annotations

# 新手导读：ExtensionAPI 是 Python 扩展 register(api) 能拿到的注册入口。
# 关注点：扩展通过它登记工具、hook、命令和提示文本。

"""扩展 API：供扩展的 register(api) 函数调用，注册工具、钩子、命令和提示词。"""

from dataclasses import replace

from codepilot.protocols.commands import CommandHandler, LifecycleHook, RegisteredCommand
from codepilot.tools import ToolRegistration

from .types import LoadedExtensions


class ExtensionAPI:
    """扩展 API：扩展通过 register(api) 函数获取此对象来注册能力。"""

    def __init__(self, *, owner: str = "extension:anonymous") -> None:
        self._owner = str(owner).strip()
        if not self._owner:
            raise ValueError("extension owner cannot be empty")
        self._tools: list[ToolRegistration] = []
        self._prompt_guidelines: list[str] = []
        self._append_prompts: list[str] = []
        self._commands: dict[str, RegisteredCommand] = {}
        self._before_prompt_hooks: list[LifecycleHook] = []
        self._after_prompt_hooks: list[LifecycleHook] = []

    def register_tool(self, tool: ToolRegistration) -> None:
        if not isinstance(tool, ToolRegistration):
            raise TypeError("ExtensionAPI.register_tool expects ToolRegistration")
        self._tools.append(replace(tool, source="extension", owner=self._owner))

    def add_prompt_guideline(self, guideline: str) -> None:
        text = guideline.strip()
        if text:
            self._prompt_guidelines.append(text)

    def append_system_prompt(self, text: str) -> None:
        content = text.strip()
        if content:
            self._append_prompts.append(content)

    def register_command(self, name: str, handler: CommandHandler, description: str | None = None) -> None:
        cmd = name.strip().lstrip("/")
        if not cmd:
            return
        self._commands[cmd] = RegisteredCommand(
            name=cmd,
            handler=handler,
            description=description,
            source="extension",
        )

    def on_before_prompt(self, hook: LifecycleHook) -> None:
        self._before_prompt_hooks.append(hook)

    def on_after_prompt(self, hook: LifecycleHook) -> None:
        self._after_prompt_hooks.append(hook)

    def snapshot(self) -> LoadedExtensions:
        """将当前注册的所有能力快照为 LoadedExtensions。"""
        return LoadedExtensions(
            tools=list(self._tools),
            prompt_guidelines=list(self._prompt_guidelines),
            append_prompts=list(self._append_prompts),
            commands=dict(self._commands),
            before_prompt_hooks=list(self._before_prompt_hooks),
            after_prompt_hooks=list(self._after_prompt_hooks),
        )
