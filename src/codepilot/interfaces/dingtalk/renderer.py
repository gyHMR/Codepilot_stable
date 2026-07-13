from __future__ import annotations

# 新手导读：renderer.py 把 Agent/Runtime 事件压缩成适合钉钉聊天窗口阅读的短消息。
# 关注点：远程通道默认只发摘要，并在发送前做脱敏和截断。

"""Render runtime events into DingTalk-safe text replies."""

from dataclasses import asdict, is_dataclass
import json
from typing import Any

from codepilot.observability import redact_artifact


MAX_DINGTALK_REPLY_CHARS = 1800
MAX_ARGUMENT_SUMMARY_CHARS = 280


def safe_reply(text: object) -> str:
    """Redact and truncate text before it leaves the local process."""

    redacted = redact_artifact(str(text))
    if not isinstance(redacted, str):
        redacted = str(redacted)
    if len(redacted) <= MAX_DINGTALK_REPLY_CHARS:
        return redacted
    return redacted[: MAX_DINGTALK_REPLY_CHARS - 32].rstrip() + "\n...[truncated]"


def render_event(event: dict[str, Any], *, verbose: bool = False) -> list[str]:
    """Render one Agent event into zero or more DingTalk messages."""

    event_type = event.get("type")
    if event_type in {"tool_completed", "tool_failed", "tool_interrupted"}:
        approval = _approval_from_tool_event(event)
        if approval:
            return [safe_reply(approval)]
        if verbose:
            tool = event.get("tool_name") or "tool"
            status = event.get("status") or _field(event.get("result"), "status") or "done"
            return [safe_reply(f"Tool {tool}: {status}")]
    if event_type == "tool_started" and verbose:
        tool = event.get("tool_name") or "tool"
        return [safe_reply(f"Tool {tool}: started")]
    if event_type == "agent_end":
        return [render_run_summary(event.get("result") or event)]
    if event_type == "error":
        message = event.get("message") or event.get("error") or "runtime error"
        return [safe_reply(f"Error: {message}")]
    return []


def render_run_accepted(*, session_id: str, prompt: str) -> str:
    """Render the immediate acknowledgement for a remote prompt."""

    lines = [
        "### Codepilot run accepted",
        "",
        f"- session_id: `{session_id}`",
        "- permission_mode: `ask`",
        f"- prompt: {_inline_code(_shorten(prompt, 120))}",
        "",
        "正在运行，完成后会发送结果摘要。",
    ]
    return safe_reply("\n".join(lines))


def render_run_summary(result: object) -> str:
    """Render a final run result or result-like event."""

    run_id = _field(result, "run_id") or "(unknown)"
    status = _field(result, "status") or "(unknown)"
    affected = _field(result, "affected_paths") or _field(result, "affectedPaths") or []
    changed = _field(result, "workspace_changed")
    if changed is None:
        changed = _field(result, "workspaceChanged")
    lines = [
        "### Codepilot run finished",
        "",
        f"- run_id: `{run_id}`",
        f"- status: `{status}`",
        f"- workspace_changed: `{bool(changed)}`",
    ]
    if affected:
        paths = [str(item) for item in list(affected)[:8]]
        lines.append("- affected_paths: " + ", ".join(f"`{path}`" for path in paths))
        if len(affected) > 8:
            lines.append(f"- affected_paths_more: `{len(affected) - 8}`")
    return safe_reply("\n".join(lines))


def render_approval_received(*, approval_id: str, decision: str) -> str:
    """Render the immediate acknowledgement for an approval command."""

    display_decision = "approved" if decision == "approve" else "denied"
    return safe_reply(
        "\n".join(
            [
                "### Tool approval received",
                "",
                f"- approval_id: `{approval_id}`",
                f"- decision: `{display_decision}`",
                "",
                "正在继续执行，完成后会发送结果摘要。",
            ]
        )
    )


def render_approval_decision(
    *,
    approval_id: str,
    decision: str,
    result: object | None = None,
) -> str:
    display_decision = "approved" if decision == "approve" else "denied"
    lines = [
        f"### Tool approval {display_decision} submitted",
        "",
        f"- approval_id: `{approval_id}`",
    ]
    if result is not None:
        lines.append(render_run_summary(result))
    return safe_reply("\n".join(lines))


def render_status(
    *,
    workspace_dir: str,
    session_id: str | None,
    active_run: bool,
    pending_approvals: list[object],
    workspace_state: str,
) -> str:
    """Render current DingTalk bridge status."""

    lines = [
        "### Codepilot DingTalk status",
        "",
        f"- workspace: `{workspace_dir}`",
        f"- session_id: `{session_id or '(not created)'}`",
        f"- active_run: `{active_run}`",
        "- permission_mode: `ask`",
        f"- workspace_state: `{workspace_state}`",
        f"- pending_approvals: `{len(pending_approvals)}`",
    ]
    approval_ids = [
        str(_field(item, "approval_id") or "")
        for item in pending_approvals[:5]
    ]
    approval_ids = [item for item in approval_ids if item]
    if approval_ids:
        lines.append("- approval_ids: " + ", ".join(f"`{item}`" for item in approval_ids))
    if pending_approvals:
        lines.extend(["", "**pending approval details**"])
        for item in pending_approvals[:5]:
            approval_id = _field(item, "approval_id") or ""
            tool_name = _field(item, "tool_name") or "tool"
            run_id = _field(item, "run_id") or "(unknown)"
            reason = _field(item, "reason") or ""
            detail = f"- `{approval_id}` tool=`{tool_name}` run_id=`{run_id}`"
            if reason:
                detail += f" reason={_inline_code(_shorten(reason, 120))}"
            lines.append(detail)
    return safe_reply("\n".join(lines))


def render_pending_approval_followup(pending_approvals: list[object]) -> str:
    """Render a follow-up message when approval resumes into another approval."""

    first = pending_approvals[0] if pending_approvals else {}
    approval_id = str(_field(first, "approval_id") or "")
    tool_name = str(_field(first, "tool_name") or "tool")
    run_id = str(_field(first, "run_id") or "(unknown)")
    reason = str(_field(first, "reason") or "")
    lines = [
        "### More tool approval required",
        "",
        f"- approval_id: `{approval_id}`",
        f"- tool: `{tool_name}`",
        f"- run_id: `{run_id}`",
    ]
    if reason:
        lines.append(f"- reason: {_inline_code(_shorten(reason, 120))}")
    lines.extend(
        [
            "",
            "**继续审批**",
            f"- `approve {approval_id}`",
            f"- `deny {approval_id}`",
        ]
    )
    return safe_reply("\n".join(lines))


def render_help() -> str:
    """Render grouped DingTalk command help for phone chat."""

    return safe_reply(
        "\n".join(
            [
                "### Codepilot DingTalk commands",
                "",
                "**任务**",
                "- `cp <prompt>` 发起一次本地 Codepilot 任务",
                "",
                "**审批**",
                "- `approve <approval_id>` 允许当前工具调用",
                "- `deny <approval_id>` 拒绝当前工具调用",
                "",
                "**状态**",
                "- `status` 查看 session、工作区和待审批项",
                "",
                "**取消**",
                "- `cancel` 尝试取消当前运行",
                "",
                "**帮助**",
                "- `help` 显示这份命令说明",
            ]
        )
    )


def _approval_from_tool_event(event: dict[str, Any]) -> str | None:
    result = event.get("result")
    status = event.get("status") or _field(result, "status")
    approval_id = event.get("approval_id")
    if status != "approval_required" or not approval_id:
        return None
    tool = event.get("tool_name") or "tool"
    reason = event.get("error_reason") or "approval_required"
    risk = (
        event.get("risk_level")
        or "unknown"
    )
    args = event.get("args") or _field(result, "arguments") or _field(result, "args") or {}
    return "\n".join(
        [
            "### Tool approval required",
            "",
            f"- tool: `{tool}`",
            f"- risk: `{risk}`",
            f"- approval_id: `{approval_id}`",
            f"- reason: {reason}",
            f"- args: `{_summarize_args(args)}`",
            "",
            "**回复审批命令**",
            f"- `approve {approval_id}`",
            f"- `deny {approval_id}`",
        ]
    )


def _summarize_args(args: object) -> str:
    if not isinstance(args, dict) or not args:
        return "{}"
    summary: dict[str, object] = {}
    for key, value in list(args.items())[:6]:
        if isinstance(value, str):
            summary[str(key)] = _shorten(value, 80)
        elif isinstance(value, (int, float, bool)) or value is None:
            summary[str(key)] = value
        else:
            summary[str(key)] = _shorten(str(value), 80)
    text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    return _shorten(text, MAX_ARGUMENT_SUMMARY_CHARS)


def _shorten(text: object, limit: int) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 14)].rstrip() + "...[truncated]"


def _inline_code(text: str) -> str:
    return "`" + text.replace("`", "'") + "`"


def _field(value: object, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    if is_dataclass(value):
        return asdict(value).get(name)
    return getattr(value, name, None)


__all__ = [
    "MAX_DINGTALK_REPLY_CHARS",
    "render_approval_decision",
    "render_approval_received",
    "render_event",
    "render_help",
    "render_pending_approval_followup",
    "render_run_accepted",
    "render_run_summary",
    "render_status",
    "safe_reply",
]
