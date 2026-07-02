from __future__ import annotations

"""
交互式 Shell 模块。

基于 prompt_toolkit 实现高级终端交互能力：
- 输入历史持久化
- / 命令补全
- 多行输入（Shift+Enter 换行）
- 快捷键支持
- 动态提示符

设计原则：
- Shell 只负责输入收集，不处理业务逻辑
- 通过回调函数与上层通信
- 历史文件存储在 .codepilot/history
"""

from pathlib import Path
from typing import Callable, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from codepilot.interfaces.cli.commands import RuntimeCommand, builtin_commands


# 自定义样式
CODEPILOT_STYLE = Style.from_dict({
    "prompt": "bold #67e8f9",
    "continuation": "#64748b",
    "bottom-toolbar": "bg:#0b0f10 #94a3b8",
    "completion-menu.completion": "bg:#111827 #cbd5e1",
    "completion-menu.completion.current": "bg:#164e63 #ecfeff bold",
    "completion-menu.meta.completion": "bg:#111827 #64748b",
    "completion-menu.meta.completion.current": "bg:#164e63 #a5f3fc",
})

class CommandCompleter(Completer):
    """命令补全器。

    Shell 只消费运行时命令元数据，不维护第二份命令名称或描述。
    真正的命令语义由 runtime.command_registry 统一定义。
    """

    def __init__(self, commands: Iterable[RuntimeCommand]) -> None:
        self.commands = tuple(commands)

    def get_completions(
        self,
        document: "Document",
        complete_event: "CompleteEvent",
    ) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()

        # 只在输入以 / 开头时补全
        if not text.startswith("/"):
            return

        # 只补全命令名；命令参数由各命令自己的帮助文本解释。
        word = text[1:]
        if " " in word:
            return

        for cmd in self.commands:
            if cmd.name.startswith(word):
                yield Completion(
                    cmd.name,
                    start_position=-len(word),
                    display_meta=cmd.description,
                )


class InteractiveShell:
    """交互式 Shell。

    基于 prompt_toolkit 实现高级终端交互。

    Attributes:
        session: prompt_toolkit PromptSession 实例。
        multiline: 是否启用多行输入模式。
    """

    def __init__(
        self,
        *,
        history_dir: str | Path | None = None,
        multiline: bool = False,
    ) -> None:
        """初始化 Shell。

        Args:
            history_dir: 历史文件存储目录（默认 .codepilot/）。
            multiline: 是否启用多行输入模式。
        """
        self.multiline = multiline

        # 设置历史文件
        if history_dir is None:
            history_dir = Path.cwd() / ".codepilot"
        history_path = Path(history_dir) / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建补全器
        completer = CommandCompleter(builtin_commands())

        # 创建快捷键绑定
        bindings = KeyBindings()

        @bindings.add("enter")
        def _(event: "KeyPressEvent") -> None:
            """Enter 提交输入，Shift+Enter 换行。"""
            if self.multiline:
                # 多行模式：Enter 提交，Shift+Enter 换行
                event.current_buffer.validate_and_handle()
            else:
                # 单行模式：Enter 提交
                event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        def _(event: "KeyPressEvent") -> None:
            """Alt+Enter 在单行模式下换行。"""
            event.current_buffer.insert_text("\n")

        @bindings.add("c-c")
        def _(event: "KeyPressEvent") -> None:
            """Ctrl+C 结束当前输入，让上层决定退出或取消任务。"""
            event.app.exit(exception=KeyboardInterrupt)

        # 创建 PromptSession
        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            completer=completer,
            key_bindings=bindings,
            style=CODEPILOT_STYLE,
            multiline=multiline,
        )

    async def prompt(
        self,
        *,
        prompt_text: str = "› ",
        bottom_toolbar: str | None = None,
    ) -> str:
        """显示提示符并获取用户输入（异步版本）。

        Args:
            prompt_text: 提示符文本（支持 HTML 格式）。
            bottom_toolbar: 底部工具栏文本。

        Returns:
            用户输入的文本（已去除首尾空白）。

        Raises:
            EOFError: 用户按下 Ctrl+D 时抛出。
            KeyboardInterrupt: 用户按下 Ctrl+C 时抛出。
        """
        try:
            # 使用 HTML 格式的提示符
            if not prompt_text.startswith("<"):
                prompt_text = f"<prompt>{prompt_text}</prompt>"

            text = await self.session.prompt_async(
                HTML(prompt_text),
                bottom_toolbar=HTML(bottom_toolbar) if bottom_toolbar else None,
            )
            return text.strip()
        except (EOFError, KeyboardInterrupt):
            raise

    async def prompt_yes_no(
        self,
        question: str,
        *,
        default: bool = False,
    ) -> bool:
        """显示是/否确认提示（异步版本）。

        Args:
            question: 问题文本。
            default: 默认值（直接按 Enter 时使用）。

        Returns:
            用户选择的结果。
        """
        suffix = " [Y/n] " if default else " [y/N] "
        try:
            answer = await self.session.prompt_async(
                HTML(f"<prompt>{question}{suffix}</prompt>"),
            )
            answer = answer.strip().lower()
            if not answer:
                return default
            return answer in ("y", "yes", "是")
        except (EOFError, KeyboardInterrupt):
            return False

    async def prompt_choice(
        self,
        question: str,
        choices: list[str],
        *,
        default: int = 0,
    ) -> int:
        """显示选择提示（异步版本）。

        Args:
            question: 问题文本。
            choices: 选项列表。
            default: 默认选项索引。

        Returns:
            用户选择的选项索引。
        """
        # 显示选项
        print(f"\n{question}")
        for i, choice in enumerate(choices):
            marker = ">" if i == default else " "
            print(f"  {marker} [{i + 1}] {choice}")

        try:
            answer = await self.session.prompt_async(
                HTML(f"<prompt>Select [1-{len(choices)}] (default: {default + 1}): </prompt>"),
            )
            answer = answer.strip()
            if not answer:
                return default
            idx = int(answer) - 1
            if 0 <= idx < len(choices):
                return idx
            return default
        except (EOFError, KeyboardInterrupt, ValueError):
            return default


def create_shell(
    *,
    history_dir: str | Path | None = None,
    multiline: bool = False,
    no_color: bool = False,
) -> InteractiveShell | None:
    """创建 InteractiveShell 实例。

    如果 prompt_toolkit 不可用或指定 no_color，返回 None。

    Args:
        history_dir: 历史文件存储目录。
        multiline: 是否启用多行输入模式。
        no_color: 是否禁用颜色（禁用时不使用 prompt_toolkit）。

    Returns:
        InteractiveShell 实例或 None。
    """
    if no_color:
        return None

    try:
        return InteractiveShell(history_dir=history_dir, multiline=multiline)
    except Exception:
        # prompt_toolkit 初始化失败时返回 None
        return None
