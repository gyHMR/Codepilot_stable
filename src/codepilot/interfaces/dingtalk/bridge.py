from __future__ import annotations

# 新手导读：bridge.py 是钉钉消息和 Runtime UserAction/RuntimeFrame 之间的薄适配层。
# 关注点：这里做远程入口守门，真正的 Agent 执行仍然交给 runtime/session/tools 主链。

"""DingTalk remote-control bridge."""

import asyncio
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from pathlib import Path
import subprocess
from typing import Any, AsyncIterator

from codepilot.runtime.actions import (
    ApprovalDecided,
    ApprovalRequiredFrame,
    CancelledFrame,
    FailedFrame,
    ProgressFrame,
    PromptSubmitted,
    RunCancelled,
    RunFinishedFrame,
)
from codepilot.runtime import RuntimeGateway, SessionOpenIntent

from .audit import DingTalkAuditLogger
from .commands import parse_dingtalk_command
from .renderer import (
    render_approval_received,
    render_event,
    render_help,
    render_pending_approval_followup,
    render_run_accepted,
    render_run_summary,
    render_status,
    safe_reply,
)
from .schemas import (
    DingTalkBridgeConfig,
    DingTalkCommand,
    DingTalkInboundMessage,
    DingTalkOutboundFormat,
    DingTalkOutboundMessage,
)


_MAX_TRACKED_MESSAGE_IDS = 1000


class DingTalkBridge:
    """Bind one DingTalk bot process to one workspace and one runtime session."""

    def __init__(
        self,
        *,
        config: DingTalkBridgeConfig,
        runtime: RuntimeGateway | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or RuntimeGateway()
        self._session_id = config.session_id
        self._session_ready = False
        self._message_states: OrderedDict[str, str] = OrderedDict()
        self._run_lock = asyncio.Lock()
        self.audit = DingTalkAuditLogger(config.workspace_dir)

    async def handle_message(
        self,
        inbound: DingTalkInboundMessage,
    ) -> list[DingTalkOutboundMessage]:
        """Handle one normalized DingTalk message and return safe text replies."""

        return [reply async for reply in self.iter_replies(inbound)]

    async def iter_replies(
        self,
        inbound: DingTalkInboundMessage,
    ) -> AsyncIterator[DingTalkOutboundMessage]:
        """Stream safe replies for one normalized DingTalk message."""

        if inbound.message_id in self._message_states:
            self.audit.record(
                "message_rejected",
                inbound=inbound,
                command="duplicate",
                status="duplicate",
                reason="duplicate_message",
                session_id=self._session_id,
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            return
        command = parse_dingtalk_command(inbound.text)
        self.audit.record(
            "message_received",
            inbound=inbound,
            command=command.action,
            status="received",
            session_id=self._session_id,
            workspace_state=_remote_workspace_state(self.config.workspace_dir),
        )
        self._remember_message(inbound.message_id, "processing")

        try:
            async for reply in self._dispatch_message(inbound, command):
                yield reply
        except Exception as exc:
            self._forget_message(inbound.message_id)
            self.audit.record(
                "bridge_error",
                inbound=inbound,
                command=command.action,
                session_id=self._session_id,
                status="error",
                reason=str(exc),
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield self._reply(
                inbound,
                f"Codepilot run failed: {exc}",
                format="markdown",
                title="Run Failed",
            )
            return
        self._remember_message(inbound.message_id, "completed")

    async def _dispatch_message(
        self,
        inbound: DingTalkInboundMessage,
        command: DingTalkCommand,
    ) -> AsyncIterator[DingTalkOutboundMessage]:
        if inbound.sender_id not in self.config.allowed_users:
            self.audit.record(
                "message_rejected",
                inbound=inbound,
                command=getattr(command, "action", None),
                session_id=self._session_id,
                status="rejected",
                reason="unauthorized_sender",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield (
                self._reply(
                    inbound,
                    "DingTalk sender is not authorized for this Codepilot bridge.",
                )
            )
            return

        if command.action == "prompt":
            async for reply in self._handle_prompt(inbound, command.prompt):
                yield reply
            return
        if command.action in {"approve", "deny"}:
            async for reply in self._handle_approval(
                inbound,
                approval_id=command.approval_id,
                decision=command.action,
            ):
                yield reply
            return
        if command.action == "status":
            self.audit.record(
                "status_requested",
                inbound=inbound,
                command=command.action,
                session_id=self._session_id,
                status="accepted",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield self._reply(
                inbound,
                self._status_text(),
                format="markdown",
                title="Codepilot Status",
            )
            return
        if command.action == "cancel":
            self.audit.record(
                "cancel_requested",
                inbound=inbound,
                command=command.action,
                session_id=self._session_id,
                status="accepted",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield self._reply(
                inbound,
                await self._cancel_text(),
                format="markdown",
                title="Codepilot Cancel",
            )
            return
        if command.action == "help":
            yield self._reply(
                inbound,
                render_help(),
                format="markdown",
                title="Codepilot Help",
            )
            return
        yield (
            self._reply(
                inbound,
                "Unknown DingTalk command. Send `help` for supported commands.",
            )
        )
        self.audit.record(
            "message_rejected",
            inbound=inbound,
            command=command.action,
            session_id=self._session_id,
            status="rejected",
            reason="unknown_command",
            workspace_state=_remote_workspace_state(self.config.workspace_dir),
        )

    async def _handle_prompt(
        self,
        inbound: DingTalkInboundMessage,
        prompt: str,
    ) -> AsyncIterator[DingTalkOutboundMessage]:
        if self._run_lock.locked():
            self.audit.record(
                "message_rejected",
                inbound=inbound,
                command="prompt",
                session_id=self._session_id,
                status="rejected",
                reason="busy",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield (
                self._reply(
                    inbound,
                    "Codepilot is busy with another DingTalk run. Try again after it finishes.",
                )
            )
            return

        if not self.config.allow_dirty:
            workspace_state = _remote_workspace_state(self.config.workspace_dir)
            block_reason = _remote_workspace_block_reason_for_state(workspace_state)
            if block_reason:
                self.audit.record(
                    "message_rejected",
                    inbound=inbound,
                    command="prompt",
                    session_id=self._session_id,
                    status="rejected",
                    reason=_workspace_block_reason_code(workspace_state),
                    workspace_state=workspace_state,
                )
                yield self._reply(inbound, block_reason)
                return

        async with self._run_lock:
            session_id = self._ensure_session()
            self.audit.record(
                "run_accepted",
                inbound=inbound,
                command="prompt",
                session_id=session_id,
                status="accepted",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield self._reply(
                inbound,
                render_run_accepted(session_id=session_id, prompt=prompt),
                format="markdown",
                title="Run Accepted",
            )
            emitted = False
            final_emitted = False
            async for frame in self.runtime.dispatch(
                session_id,
                PromptSubmitted(text=prompt),
            ):
                if isinstance(frame, ProgressFrame):
                    event_dict = _event_to_dict(frame.event)
                    self._audit_runtime_event(inbound, event_dict, command="prompt")
                    if event_dict.get("type") == "agent_end":
                        final_emitted = True
                    for text in render_event(
                        event_dict,
                        verbose=self.config.verbose_events,
                    ):
                        emitted = True
                        yield self._reply(
                            inbound,
                            text,
                            format="markdown",
                            title="Codepilot Run",
                        )
                    continue
                if isinstance(frame, ApprovalRequiredFrame):
                    event_dict = _approval_frame_to_event(frame)
                    self._audit_runtime_event(inbound, event_dict, command="prompt")
                    for text in render_event(event_dict, verbose=True):
                        emitted = True
                        yield self._reply(
                            inbound,
                            text,
                            format="markdown",
                            title="Approval Required",
                        )
                    continue
                if isinstance(frame, RunFinishedFrame):
                    if not final_emitted:
                        self._audit_run_record(inbound, frame.record, command="prompt")
                        emitted = True
                        final_emitted = True
                        yield self._reply(
                            inbound,
                            render_run_summary(frame.record),
                            format="markdown",
                            title="Codepilot Run",
                        )
                    continue
                if isinstance(frame, FailedFrame):
                    self.audit.record(
                        "bridge_error",
                        inbound=inbound,
                        command="prompt",
                        session_id=session_id,
                        status="error",
                        reason=_frame_error_message(frame),
                        workspace_state=_remote_workspace_state(self.config.workspace_dir),
                    )
                    yield self._reply(
                        inbound,
                        f"Codepilot run failed: {_frame_error_message(frame)}",
                        format="markdown",
                        title="Run Failed",
                    )
                    return

            if not emitted:
                yield self._reply(
                    inbound,
                    "Codepilot run finished, but no final summary was emitted.",
                    format="markdown",
                    title="Codepilot Run",
                )

    async def _handle_approval(
        self,
        inbound: DingTalkInboundMessage,
        *,
        approval_id: str,
        decision: str,
    ) -> AsyncIterator[DingTalkOutboundMessage]:
        session_id = self._session_id
        if not session_id:
            self.audit.record(
                "message_rejected",
                inbound=inbound,
                command=decision,
                approval_id=approval_id,
                status="rejected",
                reason="no_active_session",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield (
                self._reply(
                    inbound,
                    "No active Codepilot session is available for tool approval.",
                )
            )
            return
        if self._run_lock.locked():
            self.audit.record(
                "message_rejected",
                inbound=inbound,
                command=decision,
                session_id=session_id,
                approval_id=approval_id,
                status="rejected",
                reason="busy",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield (
                self._reply(
                    inbound,
                    "Codepilot is busy; submit the approval again after the active run finishes.",
                )
            )
            return
        async with self._run_lock:
            self.audit.record(
                "approval_received",
                inbound=inbound,
                command=decision,
                session_id=session_id,
                approval_id=approval_id,
                status="received",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield self._reply(
                inbound,
                render_approval_received(approval_id=approval_id, decision=decision),
                format="markdown",
                title="Approval Received",
            )
            result = None
            async for frame in self.runtime.dispatch(
                session_id,
                ApprovalDecided(
                    approval_id=approval_id,
                    decision=decision,
                ),
            ):
                if isinstance(frame, ProgressFrame):
                    event_dict = _event_to_dict(frame.event)
                    self._audit_runtime_event(inbound, event_dict, command=decision)
                    for text in render_event(
                        event_dict,
                        verbose=self.config.verbose_events,
                    ):
                        yield self._reply(
                            inbound,
                            text,
                            format="markdown",
                            title="Approval Result",
                        )
                    continue
                if isinstance(frame, ApprovalRequiredFrame):
                    event_dict = _approval_frame_to_event(frame)
                    self._audit_runtime_event(inbound, event_dict, command=decision)
                    for text in render_event(event_dict, verbose=True):
                        yield self._reply(
                            inbound,
                            text,
                            format="markdown",
                            title="Approval Required",
                        )
                    continue
                if isinstance(frame, RunFinishedFrame):
                    result = frame.record
                    continue
                if isinstance(frame, FailedFrame):
                    message = _frame_error_message(frame)
                    self.audit.record(
                        "approval_finished",
                        inbound=inbound,
                        command=decision,
                        session_id=session_id,
                        approval_id=approval_id,
                        status="error",
                        reason=message,
                        workspace_state=_remote_workspace_state(self.config.workspace_dir),
                    )
                    yield self._reply(
                        inbound,
                        _approval_error_message(message, approval_id=approval_id),
                        format="markdown",
                        title="Approval Failed",
                    )
                    return

            run_id = _field(result, "run_id") or _field(result, "runId")
            status = _field(result, "status")
            self.audit.record(
                "approval_finished",
                inbound=inbound,
                command=decision,
                session_id=session_id,
                run_id=str(run_id) if run_id else None,
                approval_id=approval_id,
                status=str(status) if status else "completed",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )
            yield self._reply(
                inbound,
                render_run_summary(result),
                format="markdown",
                title="Approval Result",
            )
            pending = self._pending_approvals()
            if str(status) == "waiting_approval" and pending:
                yield self._reply(
                    inbound,
                    render_pending_approval_followup(pending),
                    format="markdown",
                    title="Approval Required",
                )

    def _ensure_session(self) -> str:
        if self._session_ready and self._session_id:
            return self._session_id
        handle = self.runtime.open_session(
            SessionOpenIntent(
                workspace_dir=self.config.workspace_dir,
                provider=self.config.provider,
                model_id=self.config.model_id,
                session_id=self.config.session_id,
                tool_permission_mode="ask",
                load_workspace_resources=self.config.load_workspace_resources,
            )
        )
        self._session_id = handle.session_id
        self._session_ready = True
        return handle.session_id

    def _status_text(self) -> str:
        if self.config.session_id and not self._session_ready:
            self._ensure_session()
        approvals = self._pending_approvals()
        return render_status(
            workspace_dir=self.config.workspace_dir,
            session_id=self._session_id,
            active_run=self._run_lock.locked(),
            pending_approvals=approvals,
            workspace_state=_remote_workspace_state(self.config.workspace_dir),
        )

    def _pending_approvals(self) -> list[object]:
        if self._session_id and hasattr(self.runtime, "describe"):
            view = self.runtime.describe(self._session_id)
            return list(getattr(view, "pending_approvals", ()))
        return []

    def _audit_runtime_event(
        self,
        inbound: DingTalkInboundMessage,
        event: dict[str, Any],
        *,
        command: str,
    ) -> None:
        event_type = event.get("type")
        if event_type == "tool_interrupted":
            result = event.get("result")
            status = event.get("status") or _field(result, "status")
            approval_id = (
                event.get("approvalId")
                or event.get("approval_id")
                or _field(result, "approval_id")
            )
            if status == "approval_required" and approval_id:
                self.audit.record(
                    "approval_requested",
                    inbound=inbound,
                    command=command,
                    session_id=self._session_id,
                    run_id=_event_run_id(event),
                    approval_id=str(approval_id),
                    status="approval_required",
                    reason=event.get("errorReason") or _field(result, "error_code"),
                    workspace_state=_remote_workspace_state(self.config.workspace_dir),
                )
        if event_type == "agent_end":
            result = event.get("result") or event
            run_id = _field(result, "run_id") or _field(result, "runId") or event.get("runId")
            status = _field(result, "status") or event.get("status")
            self.audit.record(
                "run_finished",
                inbound=inbound,
                command=command,
                session_id=self._session_id,
                run_id=str(run_id) if run_id else None,
                status=str(status) if status else "completed",
                workspace_state=_remote_workspace_state(self.config.workspace_dir),
            )

    def _audit_run_record(
        self,
        inbound: DingTalkInboundMessage,
        result: object,
        *,
        command: str,
    ) -> None:
        run_id = _field(result, "run_id") or _field(result, "runId")
        status = _field(result, "status")
        self.audit.record(
            "run_finished",
            inbound=inbound,
            command=command,
            session_id=self._session_id,
            run_id=str(run_id) if run_id else None,
            status=str(status) if status else "completed",
            workspace_state=_remote_workspace_state(self.config.workspace_dir),
        )

    async def _cancel_text(self) -> str:
        if not self._session_id:
            return "No active Codepilot session to cancel."
        async for frame in self.runtime.dispatch(
            self._session_id,
            RunCancelled(reason="dingtalk"),
        ):
            if isinstance(frame, CancelledFrame):
                return (
                    "Active Codepilot run cancelled."
                    if frame.cancelled
                    else "No active Codepilot run."
                )
            if isinstance(frame, FailedFrame):
                return f"Codepilot cancel failed: {_frame_error_message(frame)}"
        return "No active Codepilot run."

    @staticmethod
    def _reply(
        inbound: DingTalkInboundMessage,
        text: object,
        *,
        format: DingTalkOutboundFormat = "text",
        title: str | None = None,
    ) -> DingTalkOutboundMessage:
        return DingTalkOutboundMessage(
            receiver_id=inbound.sender_id,
            conversation_id=inbound.conversation_id,
            text=safe_reply(text),
            format=format,
            title=title,
        )

    def _remember_message(self, message_id: str, state: str) -> None:
        self._message_states[message_id] = state
        self._message_states.move_to_end(message_id)
        while len(self._message_states) > _MAX_TRACKED_MESSAGE_IDS:
            self._message_states.popitem(last=False)

    def _forget_message(self, message_id: str) -> None:
        self._message_states.pop(message_id, None)


def _remote_workspace_block_reason(workspace_dir: str) -> str | None:
    state = _remote_workspace_state(workspace_dir)
    return _remote_workspace_block_reason_for_state(state)


def _remote_workspace_block_reason_for_state(state: str) -> str | None:
    if state == "non_git":
        return (
            "Remote run refused: workspace is not a Git repository. "
            "Start from a Git workspace or use --allow-dirty explicitly."
        )
    if state == "unknown":
        return (
            "Remote run refused: cannot determine Git workspace status. "
            "Check the workspace manually or restart with --allow-dirty."
        )
    if state == "dirty":
        return (
            "Remote run refused: workspace is dirty. "
            "Commit/stash local changes or restart with --allow-dirty."
        )
    return None


def _workspace_block_reason_code(state: str) -> str:
    if state == "non_git":
        return "workspace_non_git"
    if state == "dirty":
        return "workspace_dirty"
    if state == "unknown":
        return "workspace_unknown"
    return "workspace_blocked"


def _remote_workspace_state(workspace_dir: str) -> str:
    workspace = Path(workspace_dir)
    git_root = _git_output(workspace, "rev-parse", "--show-toplevel")
    if git_root is None:
        return "non_git"
    completed = _git_status(workspace)
    if completed is None:
        return "unknown"
    for line in completed.splitlines():
        if not line.strip():
            continue
        path_text = line[3:].strip()
        if not _only_codepilot_paths(path_text):
            return "dirty"
    return "clean"


def _git_output(workspace: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, UnicodeError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_status(workspace: Path) -> str | None:
    return _git_output(
        workspace,
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
    )


def _only_codepilot_paths(path_text: str) -> bool:
    if not path_text:
        return False
    paths = [part.strip().strip('"') for part in path_text.split(" -> ")]
    return all(
        path == ".codepilot"
        or path.startswith(".codepilot/")
        or path.startswith(".codepilot\\")
        for path in paths
    )


def _event_to_dict(event: object) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    if is_dataclass(event):
        return asdict(event)
    if hasattr(event, "to_dict"):
        value = event.to_dict()
        if isinstance(value, dict):
            return value
    return {
        key: value
        for key, value in vars(event).items()
        if not key.startswith("_")
    }


def _field(value: object, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    if is_dataclass(value):
        return asdict(value).get(name)
    return getattr(value, name, None)


def _event_run_id(event: dict[str, Any]) -> str | None:
    result = event.get("result")
    value = event.get("runId") or event.get("run_id") or _field(result, "run_id") or _field(result, "runId")
    return str(value) if value else None


def _approval_frame_to_event(frame: ApprovalRequiredFrame) -> dict[str, Any]:
    approval = frame.approval
    interruption = _field(approval, "interruption") or approval
    risk = _field(interruption, "risk") or _field(approval, "risk")
    risk_level = _field(risk, "level") or _field(interruption, "risk_level") or "unknown"
    approval_id = _field(interruption, "approval_id") or _field(approval, "approval_id")
    tool_name = _field(interruption, "tool_name") or _field(approval, "tool_name") or "tool"
    arguments = (
        _field(interruption, "arguments")
        or _field(interruption, "args")
        or _field(approval, "arguments")
        or {}
    )
    reason = (
        _field(interruption, "reason")
        or _field(approval, "reason")
        or "approval_required"
    )
    return {
        "type": "tool_interrupted",
        "toolName": tool_name,
        "status": "approval_required",
        "riskLevel": str(risk_level),
        "args": arguments,
        "approvalId": approval_id,
        "errorReason": str(reason),
        "result": {
            "status": "approval_required",
            "approval_id": approval_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "risk_level": str(risk_level),
            "error_code": str(reason),
        },
    }


def _frame_error_message(frame: FailedFrame) -> str:
    error = frame.error
    message = _field(error, "message") or _field(error, "error") or error
    return str(message)


def _approval_error_message(message: str, *, approval_id: str) -> str:
    lowered = message.lower()
    if "not found" in lowered:
        return (
            "Tool approval failed: approval_id not found, not bound to this session, "
            f"or no longer resumable.\n- approval_id: `{approval_id}`\n- reason: {message}"
        )
    if "busy" in lowered:
        return (
            "Tool approval failed: the target session is busy.\n"
            f"- approval_id: `{approval_id}`\n- reason: {message}"
        )
    if "session" in lowered and ("not found" in lowered or "not available" in lowered):
        return (
            "Tool approval failed: the target session is not available.\n"
            f"- approval_id: `{approval_id}`\n- reason: {message}"
        )
    return f"Tool approval failed: {message}"


__all__ = ["DingTalkBridge"]
