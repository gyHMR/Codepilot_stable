"""解析 DingTalk 文本中的 Session 和运行控制命令。"""

from __future__ import annotations

# 新手导读：commands.py 只把钉钉文本解析成远程控制命令。
# 关注点：真正执行任务和审批恢复都在 bridge/runtime 中完成。

"""DingTalk text command parsing."""

from .schemas import DingTalkCommand


MAX_DINGTALK_PROMPT_CHARS = 12000


def parse_dingtalk_command(text: str) -> DingTalkCommand:
    """Parse one DingTalk text message into a bridge command."""

    raw = str(text or "").strip()
    if not raw:
        return DingTalkCommand(action="unknown", raw_text="")
    parts = raw.split(maxsplit=1)
    key = parts[0].strip().lower().lstrip("/")
    arg = parts[1].strip() if len(parts) > 1 else ""

    if key == "cp" and arg:
        return DingTalkCommand(
            action="prompt",
            raw_text=raw,
            prompt=arg[:MAX_DINGTALK_PROMPT_CHARS],
        )
    if key == "approve" and arg:
        return DingTalkCommand(action="approve", raw_text=raw, approval_id=arg.split()[0])
    if key == "deny" and arg:
        return DingTalkCommand(action="deny", raw_text=raw, approval_id=arg.split()[0])
    if key == "status":
        return DingTalkCommand(action="status", raw_text=raw)
    if key == "cancel":
        return DingTalkCommand(action="cancel", raw_text=raw)
    if key == "help":
        return DingTalkCommand(action="help", raw_text=raw)
    return DingTalkCommand(action="unknown", raw_text=raw)


def help_text() -> str:
    """Return concise DingTalk command help."""

    from .render import render_help

    return render_help()


__all__ = ["MAX_DINGTALK_PROMPT_CHARS", "help_text", "parse_dingtalk_command"]
