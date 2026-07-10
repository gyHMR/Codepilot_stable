from __future__ import annotations

import inspect
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from codepilot.core.contracts import (
    AgentContext,
    AgentLoopInput,
    AgentLoopLimits,
    AgentLoopOutcome,
    AgentResumeInput,
    ContextPreparationRequest,
    PreparedContext,
    RetryPolicy,
    RunCorrelation,
)
from codepilot.core.plan import PlanState, ensure_planning_budget_profile, ensure_run_mode
from codepilot.core.runner import maybe_await
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import (
    AgentRunResult,
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.protocols.commands import SessionLifecycleContext, SessionLifecycleView

from .context import (
    ContextGovernor,
    SessionContextState,
    build_context_freshness_notice,
    calibrate_context_usage,
)
from .contracts import (
    PreparedAgentRun,
    RollbackBaselineRef,
    SessionContinuationIntent,
    SessionOptions,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionView,
)
from .conversation import SessionConversationState
from .rollback import GitRollbackBaseline, build_rollback_metadata, capture_git_baseline
from .memory import MemoryRetriever, MemoryStore, MemoryWriteContext, MemoryWriter
from .store import SessionStore, new_session_id
from .plan_state import PlanStateStore


logger = logging.getLogger("codepilot.sessions.runtime")

_TOOL_ITERATION_BUDGET_BY_PROFILE = {
    "conservative": {
        "read": 48,
        "plan": 72,
        "build": 120,
    },
    "balanced": {
        "read": 96,
        "plan": 160,
        "build": 240,
    },
    "wide": {
        "read": 160,
        "plan": 260,
        "build": 400,
    },
}
_CONTINUE_REQUESTS = {
    "继续",
    "继续吧",
    "继续执行",
    "继续任务",
    "继续上次任务",
    "接着",
    "接着做",
    "接着来",
    "接着执行",
    "continue",
    "go on",
    "resume",
}
_PLAN_STATE_EVENT_TYPES = {
    "plan_proposed",
    "plan_approval_required",
    "plan_approved",
    "plan_rejected",
    "plan_updated",
    "plan_completed",
    "plan_abandoned",
}


class SessionRuntime:
    """Live session object.

    A session runtime owns the mutable state needed while the agent is running:
    transcript, persistent stores, context preparation, memory, plan state and
    rollback baselines.  The lifecycle is deliberately readable:

    ``prepare_run`` opens a run, ``commit_run`` writes the result, ``close``
    clears listeners.
    """

    def __init__(self, options: SessionOptions) -> None:
        self.workspace_dir = Path(options.workspace_dir)
        self.get_api_key = options.get_api_key
        self.session_id = options.session_id or new_session_id()
        self.current_mode = ensure_run_mode(options.current_mode)
        self._system_prompt_builder = options.system_prompt_builder
        system_prompt = self._system_prompt_for(
            self.current_mode,
            fallback=options.system_prompt,
        )
        self.store = SessionStore(self.workspace_dir, self.session_id)
        self.store.ensure_initialized(
            model_id=options.model.id,
            provider=options.model.provider,
            system_prompt=system_prompt,
        )
        self._repair_checkpoint_messages()

        persisted = self.store.load_session_messages()
        messages = [*persisted, *options.messages]
        self.store.update_meta(
            {
                "current_mode": self.current_mode,
                "system_prompt_hash": _hash_text(system_prompt),
            }
        )
        self.planning_budget_profile = ensure_planning_budget_profile(
            options.planning_budget_profile
        )
        self.conversation = SessionConversationState(
            model=options.model,
            system_prompt=system_prompt,
            messages=messages,
            thinking_level=options.thinking_level,
            current_mode=self.current_mode,
        )

        self.memory_enabled = bool(options.memory_enabled)
        self.plan_state = PlanStateStore(self.store)
        self.memory_store = MemoryStore(self.store)
        self.memory_writer = MemoryWriter(
            store=self.memory_store,
            workspace_dir=self.workspace_dir,
        )
        self.memory_retriever = MemoryRetriever(
            store=self.memory_store,
            workspace_dir=self.workspace_dir,
        )
        self.context_governor = self._new_context_governor()
        self._custom_prepare_context = options.prepare_context
        self.prepare_context = self._custom_prepare_context or self.context_governor.prepare
        self.latest_context_report: dict[str, Any] | None = None

        self.tool_execution = options.tool_execution
        self.max_tool_calls_per_turn = options.max_tool_calls_per_turn
        self.retry_enabled = options.retry_enabled
        self.max_retries = options.max_retries
        self.retry_base_delay_ms = options.retry_base_delay_ms
        self.extension_commands = dict(options.extension_commands)
        self.before_prompt_hooks = list(options.before_prompt_hooks)
        self.after_prompt_hooks = list(options.after_prompt_hooks)
        self.before_tool_call = options.before_tool_call
        self.after_tool_call = options.after_tool_call
        self.stream_fn = options.stream_fn
        self.convert_to_llm = options.convert_to_llm

        self._last_session_run_record: SessionRunRecord | None = None
        self._rollback_baselines: dict[str, GitRollbackBaseline] = {}
        self._persisted_event_ids: set[str] = set()
        self._persisted_message_object_ids: dict[int, str] = {}

    async def prepare_run(
        self,
        intent: SessionRunIntent,
        *,
        run_id: str,
        model: ModelDescriptor,
    ) -> PreparedAgentRun:
        pending_plan = self.pending_plan_approval()
        effective_mode = ensure_run_mode(intent.mode_hint or self.current_mode)
        if pending_plan is not None:
            effective_mode = "plan"
        self.archive_plan_for_mode_switch(effective_mode, run_id=run_id)
        is_continue = effective_mode != "plan" and self._is_continue_run(intent.text)
        rollback = await self._begin_run(
            text=intent.text,
            run_id=run_id,
            is_continue=is_continue,
        )
        user_message = UserMessage(content=intent.text)
        user_message_id = self.store.append_message(user_message, run_id=run_id)
        user_message.metadata["session_message_id"] = user_message_id
        self.conversation.append_messages([user_message])
        self.store.set_checkpoint(
            {
                "phase": "user_received",
                "state": "user_received",
                "run_id": run_id,
                "message_id": user_message_id,
            }
        )
        if not is_continue:
            if self.memory_enabled:
                self._admit_prompt_memory(
                    intent.text,
                    run_id=run_id,
                    source_message_id=user_message_id,
                )
        run_plan_state = self._plan_state_for_run(
            text=intent.text,
            run_id=run_id,
            mode=effective_mode,
            pending_plan=pending_plan,
        )
        messages = self._messages_for_loop()
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=AgentLoopInput(
                run_id=run_id,
                correlation=RunCorrelation(session_id=self.session_id),
                messages=messages,
                user_prompt=intent.text,
                context=self._loop_context(effective_mode),
                model=model,
                tools=[],
                mode=effective_mode,
                plan_state=run_plan_state,
                limits=self.loop_limits(effective_mode),
                retry_policy=self.retry_policy(),
            ),
            context_port=RuntimeSessionContextPort(self),
            input_messages=[user_message],
            rollback_baseline=self._remember_rollback_baseline(run_id, rollback),
            context_refs={"context": "session_context"},
            memory_refs={"enabled": self.memory_enabled},
            plan_refs={"plan_state": run_plan_state},
        )

    async def prepare_resume(
        self,
        intent: SessionResumeIntent,
        *,
        run_id: str,
        model: ModelDescriptor,
    ) -> PreparedAgentRun:
        pending_approval = self.pending_approval(intent.approval_id)
        messages = self._messages_for_loop()
        event_start_seq, turn_start_seq = self._run_sequence_offsets(run_id)
        run_state = self._checkpoint_run_state(run_id)
        resume_input = AgentResumeInput(
            run_id=run_id,
            correlation=RunCorrelation(session_id=self.session_id),
            messages=messages,
            context=self._loop_context(),
            model=model,
            tools=[],
            approval_id=intent.approval_id,
            decision=intent.decision,
            reason=intent.reason,
            tool_call_id=(
                _optional_text(pending_approval.get("id"))
                if pending_approval is not None
                else None
            ),
            tool_name=(
                _optional_text(pending_approval.get("name"))
                if pending_approval is not None
                else None
            ),
            arguments=(
                dict(pending_approval.get("arguments"))
                if pending_approval is not None
                and isinstance(pending_approval.get("arguments"), dict)
                else {}
            ),
            mode=self.current_mode,
            plan_state=self.active_plan_state(),
            limits=self.loop_limits(),
            retry_policy=self.retry_policy(),
            event_start_seq=event_start_seq,
            turn_start_seq=turn_start_seq,
            run_state=run_state,
        )
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=AgentLoopInput(
                run_id=run_id,
                correlation=RunCorrelation(session_id=self.session_id),
                messages=messages,
                context=self._loop_context(),
                model=model,
                tools=[],
                mode=self.current_mode,
                plan_state=self.active_plan_state(),
                limits=self.loop_limits(),
                retry_policy=self.retry_policy(),
                event_start_seq=event_start_seq,
                turn_start_seq=turn_start_seq,
                run_state=run_state,
            ),
            resume_input=resume_input,
            context_port=RuntimeSessionContextPort(self),
            plan_refs={"plan_state": self.active_plan_state()},
            rollback_baseline=self._rollback_baseline_ref(run_id),
        )

    async def prepare_continuation(
        self,
        intent: SessionContinuationIntent,
        *,
        run_id: str,
        model: ModelDescriptor,
    ) -> PreparedAgentRun:
        if intent.kind in {"tool_approved", "tool_denied"}:
            return await self.prepare_resume(
                SessionResumeIntent(
                    approval_id=intent.approval_id,
                    decision="approve" if intent.kind == "tool_approved" else "deny",
                    reason=intent.reason,
                    run_id=run_id,
                ),
                run_id=run_id,
                model=model,
            )

        checkpoint = self.runtime_checkpoint()
        if checkpoint is None or _optional_text(checkpoint.get("run_id")) != run_id:
            raise ValueError(f"No resumable checkpoint for run: {run_id}")
        run_state = self._checkpoint_run_state(run_id)

        input_messages: list[Message] = []
        if intent.text:
            user_message = UserMessage(content=intent.text)
            message_id = self.store.append_message(user_message, run_id=run_id)
            user_message.metadata["session_message_id"] = message_id
            self.conversation.append_messages([user_message])
            input_messages.append(user_message)

        event_start_seq, turn_start_seq = self._run_sequence_offsets(run_id)
        mode = ensure_run_mode(intent.target_mode or self.current_mode)
        plan = self.context_plan_state_for_mode(mode)
        directive = _continuation_directive(intent.kind)
        messages = self._messages_for_loop()
        loop_input = AgentLoopInput(
            run_id=run_id,
            correlation=RunCorrelation(session_id=self.session_id),
            messages=messages,
            user_prompt=intent.text or _plan_objective(plan) or "",
            context=self._loop_context(
                mode,
                runtime_directive=directive,
                checkpoint_phase=str(checkpoint.get("phase") or intent.kind),
            ),
            model=model,
            tools=[],
            mode=mode,
            plan_state=plan,
            limits=self.loop_limits(mode),
            retry_policy=self.retry_policy(),
            event_start_seq=event_start_seq,
            turn_start_seq=turn_start_seq,
            run_state=run_state,
        )
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=loop_input,
            context_port=RuntimeSessionContextPort(self),
            input_messages=input_messages,
            context_refs={"context": "session_context", "continuation": intent.kind},
            memory_refs={"enabled": self.memory_enabled},
            plan_refs={"plan_state": plan},
            rollback_baseline=self._rollback_baseline_ref(run_id),
        )

    async def commit_run(
        self,
        prepared: PreparedAgentRun,
        outcome: AgentLoopOutcome,
        result: AgentRunResult,
        *,
        store_outcome: bool,
    ) -> SessionRunRecord:
        if store_outcome:
            for event in outcome.events:
                payload = dict(event)
                self._apply_plan_event(payload)
                self._persist_event(payload)
                await self.conversation.dispatch_event(payload)
            committed_messages = list(outcome.new_messages)
            self.conversation.append_messages(committed_messages)
            for message in committed_messages:
                if id(message) not in self._persisted_message_object_ids:
                    message_id = self.store.append_message(message, run_id=result.run_id)
                    _set_session_message_id(message, message_id)
            self.conversation.remember_result(result)

        self.store.append_run_result(result)
        if prepared.rollback_baseline is not None:
            self._write_rollback_metadata(
                result,
                self._rollback_baseline(prepared.rollback_baseline),
            )
            if _is_terminal_outcome(outcome):
                self._discard_rollback_baseline(prepared.rollback_baseline)
        self._finalize_plan_state(outcome)
        self._close_plan_for_terminal_outcome(outcome)
        self._calibrate_context_usage(result)
        if self.memory_enabled:
            self._finalize_memory(result)
        self.context_governor.finalize_run(result)
        if outcome.status == "waiting_approval":
            pass
        elif outcome.stop_reason == "plan_approval_required":
            self._save_plan_approval_checkpoint(outcome)
        elif outcome.stop_reason == "plan_clarification_required":
            self._save_plan_clarification_checkpoint(outcome)
        elif outcome.stop_reason == "plan_incomplete":
            self._save_plan_incomplete_checkpoint(outcome)
        else:
            self.store.set_checkpoint(None)
        self._save_run_state_checkpoint(outcome)
        await self._run_lifecycle_hooks(
            text=_prompt_text(prepared),
            is_continue=_is_continue_text(_prompt_text(prepared)),
            hooks=self.after_prompt_hooks,
        )

        record = SessionRunRecord(
            run_id=result.run_id,
            session_id=prepared.session_id,
            status=result.status,
            stop_reason=result.stop_reason,
            new_messages=list(result.messages),
            final_text=outcome.final_text,
            events=list(outcome.events),
            outcome=outcome,
            snapshots={
                "context": prepared.context_refs,
                "memory": prepared.memory_refs,
                "plan": prepared.plan_refs,
                "rollback": prepared.rollback_baseline,
            },
        )
        self._last_session_run_record = record
        return record

    def describe(self, *, last_run_id: str | None) -> SessionView:
        return SessionView(
            session_id=self.session_id,
            message_count=len(self.conversation.messages),
            last_run_id=last_run_id,
            current_mode=self.current_mode,
            context=self.runtime_state(),
        )

    def runtime_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "message_count": len(self.conversation.messages),
            "entry_ids": self.store.list_entry_ids(),
            "entries": self.store.list_entries(),
            "tree": self.store.get_session_tree(),
            "leaf_id": self.store.get_leaf_id(),
            "current_mode": self.current_mode,
            "planning_budget_profile": self.planning_budget_profile,
            "plan_summary": self.plan_summary(),
            "pending_plan_approval": self.pending_plan_approval(),
        }

    def plan_summary(self) -> dict[str, object] | None:
        state = self.workflow_plan_state()
        if not isinstance(state, dict):
            return None
        items = state.get("items")
        items = items if isinstance(items, list) else []
        total = len([item for item in items if isinstance(item, dict)])
        done = len([
            item
            for item in items
            if isinstance(item, dict) and item.get("status") in {"completed", "done"}
        ])
        active = next(
            (
                str(item.get("step") or "").strip()
                for item in items
                if isinstance(item, dict)
                and item.get("status") in {"in_progress", "active", "pending"}
                and str(item.get("step") or "").strip()
            ),
            "",
        )
        return {
            "plan_id": state.get("plan_id"),
            "status": state.get("status"),
            "objective_preview": _short_text(state.get("objective"), limit=72),
            "total_items": total,
            "done_items": done,
            "active_item_preview": _short_text(active, limit=72),
        }

    def pending_approvals(self) -> list[dict[str, Any]]:
        checkpoint = self._runtime_checkpoint()
        if not checkpoint:
            return []
        run_id = _optional_text(checkpoint.get("run_id"))
        approvals: list[dict[str, Any]] = []
        for call in _checkpoint_pending_calls(checkpoint):
            approval_id = _optional_text(call.get("approval_id"))
            if approval_id is None:
                continue
            approvals.append(
                {
                    **call,
                    "approval_id": approval_id,
                    "run_id": run_id,
                    "session_id": self.session_id,
                }
            )
        return approvals

    def pending_approval(self, approval_id: str) -> dict[str, Any] | None:
        target = _optional_text(approval_id)
        if target is None:
            return None
        for approval in self.pending_approvals():
            if approval.get("approval_id") == target:
                return approval
        return None

    def runtime_checkpoint(self) -> dict[str, Any] | None:
        checkpoint = self._runtime_checkpoint()
        return dict(checkpoint) if checkpoint is not None else None

    def _checkpoint_run_state(self, run_id: str) -> dict[str, object] | None:
        checkpoint = self.runtime_checkpoint()
        if checkpoint is None or _optional_text(checkpoint.get("run_id")) != run_id:
            return None
        run_state = checkpoint.get("run_state")
        if not isinstance(run_state, dict):
            return None
        return dict(run_state)

    def _save_run_state_checkpoint(self, outcome: AgentLoopOutcome) -> None:
        if _is_terminal_outcome(outcome) or not outcome.run_state:
            return
        checkpoint = self.runtime_checkpoint()
        if checkpoint is None:
            checkpoint = {
                "phase": outcome.stop_reason,
                "run_id": outcome.run_id,
            }
        checkpoint_run_id = _optional_text(checkpoint.get("run_id"))
        if checkpoint_run_id is not None and checkpoint_run_id != outcome.run_id:
            return
        checkpoint["run_id"] = outcome.run_id
        checkpoint["run_state"] = dict(outcome.run_state)
        self.store.set_checkpoint(checkpoint)

    def continuation_run_id(self) -> str:
        checkpoint = self.runtime_checkpoint()
        if checkpoint is None:
            raise ValueError("No paused run to continue")
        run_id = _optional_text(checkpoint.get("run_id"))
        if run_id is None:
            raise ValueError("Runtime checkpoint has no run_id")
        return run_id

    def resume_run_id(self, approval_id: str) -> str:
        approval = self.pending_approval(approval_id)
        if approval is None:
            raise ValueError(f"Approval not found: {approval_id}")
        run_id = _optional_text(approval.get("run_id"))
        if run_id is None:
            raise ValueError(f"Approval has no run_id: {approval_id}")
        return run_id

    def pending_plan_approval(self) -> dict[str, Any] | None:
        state = self.workflow_plan_state()
        if isinstance(state, dict) and state.get("status") == "proposed":
            return dict(state)
        return None

    def current_plan_state(self) -> dict[str, Any] | None:
        state = self.plan_state.current()
        if not isinstance(state, dict):
            return None
        return dict(state)

    def workflow_plan_state(self) -> dict[str, Any] | None:
        state = self.current_plan_state()
        if not isinstance(state, dict):
            return None
        if state.get("status") not in {"proposed", "active"}:
            return None
        return dict(state)

    def active_plan_state(self) -> dict[str, object] | None:
        state = self.workflow_plan_state()
        if not isinstance(state, dict):
            return None
        if state.get("status") != "active":
            return None
        active_plan_id = (self.store.read_meta() or {}).get("active_plan_id")
        if active_plan_id != state.get("plan_id"):
            return None
        return dict(state)

    def context_plan_state(self) -> dict[str, Any] | None:
        return self.context_plan_state_for_mode(self.current_mode)

    def context_plan_state_for_mode(self, mode: str) -> dict[str, Any] | None:
        normalized = ensure_run_mode(mode)
        pending = self.pending_plan_approval()
        if pending is not None:
            return pending
        if normalized == "plan":
            return None
        return self.active_plan_state()

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            enabled=bool(self.retry_enabled),
            max_retries=_int_or_default(self.max_retries, default=0),
            base_delay_ms=_int_or_default(self.retry_base_delay_ms, default=0),
        )

    def loop_limits(self, mode: str | None = None) -> AgentLoopLimits:
        normalized = ensure_run_mode(mode or self.current_mode)
        return AgentLoopLimits(
            max_model_turns=self._model_turn_budget(normalized),
            max_tool_iterations=self._tool_iteration_budget(normalized),
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
        )

    def set_current_mode(self, mode: str) -> str:
        normalized = ensure_run_mode(mode)
        if normalized != "plan" and self.pending_plan_approval() is not None:
            raise ValueError(
                f"{normalized} mode is blocked while a proposed plan is waiting for approval. "
                "Use /plan approve or /plan reject."
            )
        checkpoint = self.runtime_checkpoint()
        self.archive_plan_for_mode_switch(
            normalized,
            run_id=(
                _optional_text(checkpoint.get("run_id"))
                if checkpoint is not None
                else None
            ),
        )
        self.conversation.set_current_mode(normalized)
        if normalized != self.current_mode:
            self.current_mode = normalized
            self.store.append_event(
                {
                    "type": "mode_changed",
                    "sessionId": self.session_id,
                    "currentMode": normalized,
                }
            )
        self.store.update_meta(
            {
                "current_mode": normalized,
                "system_prompt_hash": _hash_text(self.conversation.system_prompt),
            }
        )
        return normalized

    def archive_plan_for_mode_switch(
        self,
        target_mode: str,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        normalized = ensure_run_mode(target_mode)
        current = self.current_plan_state()
        if not isinstance(current, dict):
            return None
        if current.get("status") not in {"proposed", "active"}:
            self.store.update_meta({"active_plan_id": None})
            return current
        if normalized != "plan" or current.get("status") != "active":
            return current
        owner_run_id = _optional_text(current.get("owner_run_id"))
        state = self.plan_state.abandon_current(
            run_id=run_id if run_id == owner_run_id else None,
            source="mode_switch",
        )
        if state is not None:
            self._record_plan_event("plan_abandoned", state, run_id=run_id)
        return state

    def approve_current_plan(self, *, switch_to_build: bool = True) -> dict[str, Any] | None:
        before = self.workflow_plan_state()
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return before
        run_id = _optional_text(before.get("owner_run_id"))
        state = self.plan_state.approve_current(run_id=run_id)
        if state is None:
            return None
        self.store.update_meta({"active_plan_id": state.get("plan_id")})
        self._record_plan_event("plan_approved", state, run_id=run_id)
        if switch_to_build:
            self.set_current_mode("build")
        return state

    def reject_current_plan(self) -> dict[str, Any] | None:
        before = self.workflow_plan_state()
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return before
        run_id = _optional_text(before.get("owner_run_id"))
        state = self.plan_state.reject_current(run_id=run_id)
        if state is None:
            return None
        self.store.update_meta({"active_plan_id": None})
        self._record_plan_event("plan_rejected", state, run_id=run_id)
        self.set_current_mode("plan")
        return state

    def abandon_current_plan(self) -> dict[str, Any] | None:
        before = self.current_plan_state()
        if not isinstance(before, dict):
            return before
        state = self.plan_state.abandon_current(source="user_abandoned")
        if state is None:
            return None
        self.store.update_meta({"active_plan_id": None})
        self._record_plan_event("plan_abandoned", state, run_id=None)
        self.store.set_checkpoint(None)
        return state

    def close(self) -> None:
        self.conversation.clear_listeners()

    def record_event(self, event: dict[str, Any]) -> None:
        """Persist a streamed runner event and update the recoverable checkpoint."""

        payload = dict(event)
        self._apply_plan_event(payload)
        self._persist_event(payload)
        self._checkpoint_from_event(payload)

    def _is_continue_run(self, text: str) -> bool:
        return self.active_plan_state() is not None and _is_continue_text(text)

    async def _begin_run(
        self,
        *,
        text: str,
        run_id: str,
        is_continue: bool,
    ) -> GitRollbackBaseline:
        rollback = capture_git_baseline(self.workspace_dir)
        await self._run_lifecycle_hooks(
            text=text,
            is_continue=is_continue,
            hooks=self.before_prompt_hooks,
        )
        self._check_context_freshness()
        return rollback

    def _admit_prompt_memory(
        self,
        text: str,
        *,
        run_id: str | None,
        source_message_id: str | None,
    ) -> None:
        try:
            event_id = f"event_{uuid4().hex[:12]}"
            result = self.memory_writer.admit_prompt_memory(
                text,
                context=MemoryWriteContext(
                    session_id=self.session_id,
                    run_id=run_id,
                    source_message_id=source_message_id,
                    source_event_id=event_id,
                    evidence_refs=[
                        f"event:{event_id}",
                        *([f"message:{source_message_id}"] if source_message_id else []),
                        *([f"run:{run_id}"] if run_id else []),
                    ],
                ),
            )
            if result is None:
                return
            record, decision = result
            self.store.append_event(
                {
                    "type": decision.reason,
                    "eventId": event_id,
                    "sessionId": self.session_id,
                    "runId": run_id,
                    "memoryId": record.id,
                    "status": record.status,
                }
            )
        except Exception as exc:
            logger.warning("failed to admit prompt memory: %s", exc)
            self.store.append_event(
                {
                    "type": "memory_warning",
                    "sessionId": self.session_id,
                    "operation": "prompt_memory_admission",
                    "message": str(exc),
                }
            )

    def _plan_state_for_run(
        self,
        *,
        text: str,
        run_id: str | None,
        mode: str,
        pending_plan: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if pending_plan is not None:
            return pending_plan
        if mode == "plan":
            return None
        return self.active_plan_state()

    def _last_substantive_user_request(self) -> str | None:
        for message in reversed(self.conversation.messages):
            if not isinstance(message, UserMessage):
                continue
            text = _message_text(message)
            if text and not _is_continue_text(text):
                return text
        return None

    def _check_context_freshness(self) -> None:
        freshness = self.store.run_store.evaluate_freshness()
        if not freshness.should_record_event():
            return
        self.store.append_event(
            {
                "type": "context_freshness_checked",
                "sessionId": self.session_id,
                "freshness": freshness.to_event_payload(),
            }
        )
        if freshness.requires_steering():
            notice = build_context_freshness_notice(freshness)
            if notice is not None:
                self.conversation.add_steering_message(notice)

    def _finalize_memory(self, result: AgentRunResult) -> None:
        try:
            event_id = f"event_{uuid4().hex[:12]}"
            records = self.memory_writer.finalize_run(
                result,
                context=MemoryWriteContext(
                    session_id=self.session_id,
                    run_id=result.run_id,
                    source_event_id=event_id,
                    evidence_refs=[f"event:{event_id}", f"run:{result.run_id}"],
                ),
            )
            for record in records:
                self.store.append_event(
                    {
                        "type": "memory_candidate_created",
                        "sessionId": self.session_id,
                        "runId": result.run_id,
                        "memoryId": record.id,
                        "status": record.status,
                    }
                )
        except Exception as exc:
            logger.warning("failed to finalize memory: %s", exc)
            self.store.append_event(
                {
                    "type": "memory_warning",
                    "sessionId": self.session_id,
                    "operation": "finalize_run",
                    "message": str(exc),
                }
            )

    def _finalize_plan_state(self, outcome: AgentLoopOutcome) -> None:
        try:
            payload = _plan_payload(outcome.plan)
            if payload is None:
                return
            previous = self.current_plan_state()
            state = self.plan_state.save(payload)
            self.store.update_meta(
                {"active_plan_id": state.get("plan_id") if state.get("status") == "active" else None}
            )
            if previous == state:
                return
            self._record_plan_event(
                _plan_event_type(previous, state),
                state,
                run_id=outcome.run_id,
            )
        except Exception as exc:
            logger.warning("failed to finalize plan state: %s", exc)
            self.store.append_event(
                {
                    "type": "plan_state_warning",
                    "sessionId": self.session_id,
                    "operation": "plan_state_finalize",
                    "message": str(exc),
                }
            )

    def _close_plan_for_terminal_outcome(self, outcome: AgentLoopOutcome) -> None:
        if outcome.status not in {"failed", "aborted"}:
            return
        state = self.current_plan_state()
        if not isinstance(state, dict) or state.get("status") != "active":
            return
        abandoned = self.plan_state.abandon_current(
            run_id=outcome.run_id,
            source=f"run_{outcome.status}",
        )
        if abandoned is not None:
            self.store.update_meta({"active_plan_id": None})
            self._record_plan_event("plan_abandoned", abandoned, run_id=outcome.run_id)

    def _calibrate_context_usage(self, result: AgentRunResult) -> None:
        try:
            usage = getattr(result, "usage", None)
            actual_input = getattr(usage, "input", 0) if usage is not None else 0
            if not isinstance(actual_input, int) or actual_input <= 0:
                return
            model = self.conversation.model
            calibrate_context_usage(
                workspace_dir=self.workspace_dir,
                provider=getattr(model, "provider", None),
                model=getattr(model, "id", None),
                report=self.latest_context_report,
                actual_input_tokens=actual_input,
            )
        except Exception as exc:
            logger.warning("failed to calibrate context usage: %s", exc)

    def _write_rollback_metadata(
        self,
        result: AgentRunResult,
        baseline: GitRollbackBaseline,
    ) -> None:
        self.store.write_rollback_metadata(
            result.run_id,
            build_rollback_metadata(
                baseline,
                affected_paths=list(result.affected_paths),
                workspace_changed=bool(result.workspace_changed),
            ),
        )

    async def _run_lifecycle_hooks(
        self,
        *,
        text: str,
        is_continue: bool,
        hooks: list,
    ) -> None:
        if not hooks:
            return
        context = SessionLifecycleContext(
            text=text,
            is_continue=is_continue,
            message_count=len(self.conversation.messages),
            session_view=SessionLifecycleView(
                session_id=self.session_id,
                workspace_dir=str(self.workspace_dir),
                message_count=len(self.conversation.messages),
                current_mode=str(self.current_mode),
            ),
        )
        for hook in hooks:
            value = hook(context)
            if inspect.isawaitable(value):
                await value

    def _messages_for_loop(self) -> list[Message]:
        return [
            *self.conversation.messages,
            *self.conversation.drain_steering_messages(),
        ]

    def _persist_event(self, event: dict[str, Any]) -> bool:
        event_id = _event_id(event)
        if event_id is not None and event_id in self._persisted_event_ids:
            return False
        self.store.append_event(dict(event))
        if event_id is not None:
            self._persisted_event_ids.add(event_id)
        return True

    def _apply_plan_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if event.get("type") not in _PLAN_STATE_EVENT_TYPES:
            return None
        payload = _plan_payload(event.get("plan"))
        if payload is None:
            return None
        try:
            state = self.plan_state.save(payload)
        except Exception as exc:
            logger.warning("failed to apply plan event: %s", exc)
            self.store.append_event(
                {
                    "type": "plan_state_warning",
                    "sessionId": self.session_id,
                    "operation": "plan_event_apply",
                    "message": str(exc),
                }
            )
            return None
        self.store.update_meta(
            {"active_plan_id": state.get("plan_id") if state.get("status") == "active" else None}
        )
        event["plan"] = state
        return state

    def _runtime_checkpoint(self) -> dict[str, Any] | None:
        meta = self.store.read_meta() or {}
        checkpoint = meta.get("runtime_checkpoint")
        return checkpoint if isinstance(checkpoint, dict) else None

    def _checkpoint_from_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "plan_approval_required":
            self._save_plan_approval_checkpoint_from_event(event)
            return
        if event.get("type") == "plan_clarification_required":
            run_id = _event_run_id(event)
            if run_id is not None:
                self.store.set_checkpoint(
                    {
                        "phase": "plan_clarification",
                        "run_id": run_id,
                        "reason": event.get("reason"),
                    }
                )
            return
        if (
            event.get("type") == "tool_interrupted"
            and event.get("status") == "approval_required"
        ):
            self._checkpoint_approval_from_event(event)
            return
        if event.get("type") != "message_end":
            return
        message = event.get("message")
        if not isinstance(message, (AssistantMessage, ToolResultMessage)):
            return
        message_id = self._persist_message_from_event(message, event)
        run_id = _event_run_id(event)
        turn_id = _int_or_none(event.get("turnId"))
        if isinstance(message, AssistantMessage):
            tool_calls = [
                block for block in message.content if isinstance(block, ToolCall) and block.id
            ]
            if tool_calls:
                self.store.set_checkpoint(
                    {
                        "phase": "tool_approval",
                        "run_id": run_id,
                        "turn_id": turn_id,
                        "assistant_message_id": message_id,
                        "pending_tool_call_ids": [call.id for call in tool_calls],
                        "pending_tool_calls": [
                            {
                                "id": call.id,
                                "name": call.name,
                                "arguments": dict(call.arguments),
                            }
                            for call in tool_calls
                        ],
                        "completed_tool_result_ids": [],
                    }
                )
                return
            self.store.set_checkpoint(
                {
                    "phase": "final_response",
                    "run_id": run_id,
                    "turn_id": turn_id,
                    "assistant_message_id": message_id,
                }
            )
            return

        checkpoint = self.store.read_meta() or {}
        runtime_checkpoint = checkpoint.get("runtime_checkpoint")
        runtime_checkpoint = runtime_checkpoint if isinstance(runtime_checkpoint, dict) else {}
        pending_ids = [
            str(item)
            for item in runtime_checkpoint.get("pending_tool_call_ids", [])
            if isinstance(item, str)
        ]
        pending_ids = [item for item in pending_ids if item != message.tool_call_id]
        pending_calls = [
            item
            for item in runtime_checkpoint.get("pending_tool_calls", [])
            if isinstance(item, dict) and item.get("id") != message.tool_call_id
        ]
        completed = [
            str(item)
            for item in runtime_checkpoint.get("completed_tool_result_ids", [])
            if isinstance(item, str)
        ]
        completed.append(message_id)
        phase = "tool_approval" if pending_ids else "tools_completed"
        self.store.set_checkpoint(
            {
                "phase": phase,
                "run_id": run_id,
                "turn_id": turn_id,
                "assistant_message_id": runtime_checkpoint.get("assistant_message_id"),
                "pending_tool_call_ids": pending_ids,
                "pending_tool_calls": pending_calls,
                "completed_tool_result_ids": completed,
            }
        )

    def _checkpoint_approval_from_event(self, event: dict[str, Any]) -> None:
        checkpoint = self._runtime_checkpoint()
        if checkpoint is None:
            return
        tool_call_id = _optional_text(event.get("toolCallId"))
        approval_id = _optional_text(event.get("approvalId"))
        if tool_call_id is None or approval_id is None:
            return

        pending_calls = _checkpoint_pending_calls(checkpoint)
        updated = False
        for call in pending_calls:
            if call.get("id") != tool_call_id:
                continue
            call["approval_id"] = approval_id
            call["reason"] = _approval_reason_from_event(event)
            call["risk_level"] = _approval_risk_from_event(event)
            updated = True
            break
        if not updated:
            pending_calls.append(
                {
                    "id": tool_call_id,
                    "name": _optional_text(event.get("toolName")) or "",
                    "arguments": {},
                    "approval_id": approval_id,
                    "reason": _approval_reason_from_event(event),
                    "risk_level": _approval_risk_from_event(event),
                }
            )

        self.store.set_checkpoint(
            {
                **checkpoint,
                "phase": "tool_approval",
                "run_id": _event_run_id(event) or checkpoint.get("run_id"),
                "turn_id": _int_or_none(event.get("turnId")) or checkpoint.get("turn_id"),
                "pending_tool_call_ids": [
                    str(call["id"])
                    for call in pending_calls
                    if isinstance(call.get("id"), str)
                ],
                "pending_tool_calls": pending_calls,
            }
        )

    def _save_plan_approval_checkpoint_from_event(self, event: dict[str, Any]) -> None:
        plan = self.pending_plan_approval()
        if not isinstance(plan, dict):
            return
        self.store.set_checkpoint(
            {
                "phase": "plan_approval",
                "run_id": _event_run_id(event),
                "turn_id": _int_or_none(event.get("turnId")),
                "plan_id": plan.get("plan_id"),
                "reason": _optional_text(event.get("reason")) or "plan_approval_required",
            }
        )

    def _save_plan_approval_checkpoint(self, outcome: AgentLoopOutcome) -> None:
        plan = self.pending_plan_approval()
        if not isinstance(plan, dict):
            return
        self.store.set_checkpoint(
            {
                "phase": "plan_approval",
                "run_id": outcome.run_id,
                "plan_id": plan.get("plan_id"),
                "reason": outcome.stop_reason,
            }
        )

    def _save_plan_clarification_checkpoint(self, outcome: AgentLoopOutcome) -> None:
        self.store.set_checkpoint(
            {
                "phase": "plan_clarification",
                "run_id": outcome.run_id,
                "reason": outcome.stop_reason,
            }
        )

    def _save_plan_incomplete_checkpoint(self, outcome: AgentLoopOutcome) -> None:
        plan = self.workflow_plan_state()
        self.store.set_checkpoint(
            {
                "phase": "plan_incomplete",
                "run_id": outcome.run_id,
                "plan_id": plan.get("plan_id") if isinstance(plan, dict) else None,
                "reason": outcome.stop_reason,
            }
        )

    def _persist_message_from_event(
        self,
        message: AssistantMessage | ToolResultMessage,
        event: dict[str, Any],
    ) -> str:
        existing = self._persisted_message_object_ids.get(id(message))
        if existing is not None:
            return existing
        message_id = self.store.append_message(message, run_id=_event_run_id(event))
        _set_session_message_id(message, message_id)
        self._persisted_message_object_ids[id(message)] = message_id
        return message_id

    def _repair_checkpoint_messages(self) -> None:
        meta = self.store.read_meta() or {}
        checkpoint = meta.get("runtime_checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("phase") != "tool_approval":
            return
        all_pending_calls = _checkpoint_pending_calls(checkpoint)
        approval_pending_calls = [
            call
            for call in all_pending_calls
            if _optional_text(call.get("approval_id")) is not None
        ]
        pending_calls = [
            call
            for call in all_pending_calls
            if _optional_text(call.get("approval_id")) is None
        ]
        if not pending_calls:
            return
        existing_results = {
            message.tool_call_id
            for message in self.store.load_session_messages()
            if isinstance(message, ToolResultMessage)
        }
        created_ids: list[str] = []
        for call in pending_calls:
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id or call_id in existing_results:
                continue
            tool_name = call.get("name") if isinstance(call.get("name"), str) else ""
            created_ids.append(
                self.store.append_message(
                    ToolResultMessage(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        content=[
                            TextContent(
                                text="Error: task was interrupted before this tool returned."
                            )
                        ],
                        status="error",
                        is_error=True,
                        error_code="tool_result_missing",
                    ),
                    run_id=_optional_text(checkpoint.get("run_id")),
                )
            )
        if created_ids:
            self.store.append_event(
                {
                    "type": "checkpoint_restored",
                    "sessionId": self.session_id,
                    "runId": checkpoint.get("run_id"),
                    "checkpoint": checkpoint,
                    "synthetic_tool_result_ids": created_ids,
                }
            )
            self.store.set_checkpoint(
                {
                    "phase": "tool_approval" if approval_pending_calls else "tools_completed",
                    "run_id": checkpoint.get("run_id"),
                    "assistant_message_id": checkpoint.get("assistant_message_id"),
                    "pending_tool_call_ids": [
                        str(call["id"])
                        for call in approval_pending_calls
                        if isinstance(call.get("id"), str)
                    ],
                    "pending_tool_calls": approval_pending_calls,
                    "completed_tool_result_ids": [
                        *[
                            str(item)
                            for item in checkpoint.get("completed_tool_result_ids", [])
                            if isinstance(item, str)
                        ],
                        *created_ids,
                    ],
                }
            )

    def _loop_context(
        self,
        mode: str | None = None,
        *,
        runtime_directive: str = "",
        checkpoint_phase: str = "",
    ) -> PreparedContext:
        normalized = ensure_run_mode(mode or self.current_mode)
        values: dict[str, Any] = {
            "system_prompt": self.conversation.system_prompt,
            "session_id": self.session_id,
            "mode": normalized,
        }
        if runtime_directive:
            values["runtime_directive"] = runtime_directive
        if checkpoint_phase:
            values["checkpoint_phase"] = checkpoint_phase
        return PreparedContext(values)

    def _tool_iteration_budget(self, mode: str | None = None) -> int:
        normalized = ensure_run_mode(mode or self.current_mode)
        return _TOOL_ITERATION_BUDGET_BY_PROFILE[self.planning_budget_profile][
            normalized
        ]

    def _model_turn_budget(self, mode: str | None = None) -> int:
        tool_iterations = self._tool_iteration_budget(mode)
        return tool_iterations + max(16, tool_iterations // 10)

    def _system_prompt_for(self, mode: str, *, fallback: str) -> str:
        if self._system_prompt_builder is None:
            return str(fallback or "")
        return str(self._system_prompt_builder(ensure_run_mode(mode)) or "")

    def _approve_proposed_plan(self) -> None:
        before = self.workflow_plan_state()
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return
        state = self.plan_state.approve_current()
        if state is None:
            return
        self.store.update_meta({"active_plan_id": state.get("plan_id")})
        self._record_plan_event("plan_approved", state, run_id=None)

    def _record_plan_event(
        self,
        event_type: str,
        state: dict[str, Any],
        *,
        run_id: str | None,
    ) -> None:
        self.store.append_event(
            {
                "type": event_type,
                "sessionId": self.session_id,
                "runId": run_id,
                "plan": state,
            }
        )

    def _new_context_governor(self) -> ContextGovernor:
        return ContextGovernor(
            workspace_dir=self.workspace_dir,
            session_id=self.session_id,
            state=SessionContextState(workspace_dir=self.workspace_dir),
            memory_retriever=self.memory_retriever if self.memory_enabled else None,
            plan_state_store=self.plan_state,
            store=self.store,
        )

    def _remember_rollback_baseline(
        self,
        run_id: str,
        baseline: GitRollbackBaseline,
    ) -> RollbackBaselineRef:
        self._rollback_baselines[run_id] = baseline
        return RollbackBaselineRef(session_id=self.session_id, run_id=run_id)

    def _rollback_baseline_ref(self, run_id: str) -> RollbackBaselineRef | None:
        if run_id not in self._rollback_baselines:
            return None
        return RollbackBaselineRef(session_id=self.session_id, run_id=run_id)

    def _rollback_baseline(self, ref: RollbackBaselineRef | None) -> GitRollbackBaseline:
        if ref is None:
            return GitRollbackBaseline(eligible=False, reason="missing_rollback_baseline_ref")
        if ref.session_id != self.session_id:
            return GitRollbackBaseline(eligible=False, reason="rollback_baseline_session_mismatch")
        return self._rollback_baselines.get(
            ref.run_id,
            GitRollbackBaseline(eligible=False, reason="missing_rollback_baseline"),
        )

    def _discard_rollback_baseline(self, ref: RollbackBaselineRef) -> None:
        if ref.session_id == self.session_id:
            self._rollback_baselines.pop(ref.run_id, None)

    def _run_sequence_offsets(self, run_id: str) -> tuple[int, int]:
        events = self.store.run_store.load_events(run_id)
        turn_ids = [
            int(event.get("turnId", 0))
            for event in events
            if isinstance(event.get("turnId"), int)
        ]
        return len(events), max(turn_ids, default=0)


class RuntimeSessionContextPort:
    def __init__(self, session: SessionRuntime) -> None:
        self._session = session

    async def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session
        request_context = request.get("context")
        request_context = request_context if isinstance(request_context, dict) else {}
        run_signals = request_context.get("run_signals")
        request_mode = _optional_text(request.get("mode"))
        session_mode = ensure_run_mode(getattr(session, "current_mode", "build"))
        context_plan_for_mode = getattr(session, "context_plan_state_for_mode", None)
        if callable(context_plan_for_mode):
            plan_state = context_plan_for_mode(request_mode or session_mode)
        else:
            plan_state = session.context_plan_state()
        mode = ensure_run_mode(request_mode or session_mode)
        checkpoint_getter = getattr(session, "runtime_checkpoint", None)
        checkpoint = checkpoint_getter() if callable(checkpoint_getter) else None
        runtime_state: dict[str, object] = {
            "run_id": str(request.get("run_id") or ""),
            "mode": mode,
            "checkpoint_phase": str(
                request_context.get("checkpoint_phase")
                or (checkpoint or {}).get("phase")
                or "running"
            ),
            "mode_policy": _mode_policy(mode),
        }
        runtime_directive = _optional_text(request_context.get("runtime_directive"))
        if runtime_directive:
            runtime_state["directive"] = runtime_directive
        if isinstance(plan_state, dict):
            runtime_state["plan_state_status"] = str(plan_state.get("status") or "")
        if isinstance(run_signals, dict):
            runtime_state["verification_status"] = str(
                run_signals.get("verification_status") or "unknown"
            )
        prepared = await maybe_await(
            session.prepare_context(
                AgentContext(
                    system_prompt=str(request.get("system_prompt", "")),
                    messages=list(request.get("messages", ())),
                    tools=list(request.get("tools", ())),
                    mode=mode,
                    plan_state=plan_state,
                    run_signals=run_signals if isinstance(run_signals, dict) else None,
                    runtime_state=runtime_state,
                ),
                ContextPreparationRequest(
                    session_id=session.session_id,
                    model_context_window=session.conversation.model.context_window,
                    model_max_output_tokens=session.conversation.model.max_tokens,
                    signal={
                        "run_id": request.get("run_id"),
                        "provider": getattr(session.conversation.model, "provider", None),
                        "model": getattr(session.conversation.model, "id", None),
                    },
                ),
            )
        )
        report = prepared.report.to_dict()
        session.latest_context_report = report
        session.store.append_event(
            {
                "type": "context_projected",
                "sessionId": session.session_id,
                "runId": request.get("run_id"),
                "report": report,
            }
        )
        memory_ids = report.get("retrieved_memory_ids")
        if session.memory_enabled and isinstance(memory_ids, list) and memory_ids:
            session.store.append_event(
                {
                    "type": "memory_retrieved",
                    "sessionId": session.session_id,
                    "runId": request.get("run_id"),
                    "memoryIds": memory_ids,
                    "reasons": report.get("memory_retrieval_reasons", {}),
                }
            )
        return {
            "system_prompt": prepared.system_prompt,
            "messages": list(prepared.messages),
            "tools": list(prepared.tools),
            "context_report": report,
        }

    def record_preflight(self, report: dict[str, int], *, run_id: str | None = None) -> None:
        session = self._session
        payload = {
            "type": "context_preflight",
            "context_id": (session.latest_context_report or {}).get("context_id"),
            "run_id": run_id,
            "created_at": _utc_now_iso(),
            "runner_preflight": dict(report),
        }
        session.store.append_context_ledger(payload)
        if session.latest_context_report is not None:
            session.latest_context_report["runner_preflight"] = dict(report)


def new_run_id() -> str:
    return f"run_{uuid4().hex[:12]}"


def _continuation_directive(kind: str) -> str:
    return {
        "plan_approved": (
            "The user approved the canonical plan. Continue execution in build mode "
            "from the first unfinished plan item without redesigning or restating the plan."
        ),
        "plan_rejected": (
            "The user rejected the proposed plan. Ask one concise question about what "
            "should change; do not create a replacement plan until feedback is provided."
        ),
        "plan_feedback": (
            "The latest user message is feedback on the proposed plan. Revise the same "
            "plan with update_plan, then summarize the canonical revision."
        ),
        "plan_clarification": (
            "Continue planning from the user's clarification. Publish a decision-complete "
            "plan with update_plan when enough information is available."
        ),
        "mode_changed": "Continue the same task using the current mode policy.",
        "automatic_continuation": "Continue the same task from the saved checkpoint.",
    }.get(kind, "")


def _mode_policy(mode: str) -> str:
    if mode == "plan":
        return (
            "当前 mode=plan。你仍是同一个 Coding Agent，处理同一个用户任务，但本轮只允许只读调查和方案设计，禁止修改工作区。"
            "Plan 的职责是根据用户原始请求探索仓库事实，生成可交给 Build 直接执行的代码修改方案；"
            "不是规划如何写方案，也不是执行方案。先分离对象级任务和控制级指令：代码、行为、测试和配置目标属于对象级；"
            "“给方案”“先分析”“不要修改”等只决定当前交付方式与权限，不能成为 execution_objective 或计划步骤。"
            "先探索相关实现、调用方、配置和测试边界；发布计划前必须已有成功的仓库探索证据，信息不足时先只读调查或追问。"
            "遇到多文件改动、长文件（数百行以上）或跨模块调用链与测试边界时，应优先用 "
            "list_exploration_agents 复用缓存；缓存不足时用 dispatch_exploration 并行派发只读 Subagent。"
            "用 update_plan 发布或修订 proposed plan：execution_objective 必须描述 Build 要完成的软件工作；"
            "summary 应概括任务理解、当前实现依据、目标设计、影响范围、风险和验证思路；"
            "items 只能描述批准后实际要执行的代码修改与验证，不能写分析需求、查看代码、撰写方案、回复用户或等待审批。"
            "所有条目保持 pending，发布后等待显式审批；不得执行实现、自行切换到 Build 或推进步骤状态。"
            "自然语言反馈仅用于修订同一个 proposed plan。"
        )
    if mode == "read":
        return (
            "当前 mode=read。你仍是同一个 Coding Agent，处理同一个用户任务，但本轮只做只读探索、定位、解释、审查和状态说明。"
            "目标是回答用户当前问题，并区分代码事实、合理推断和建议；不要默认生成实施计划。"
            "不得修改工作区、运行会产生副作用的命令，不得创建、推进或完成 Task Plan，也不得继续执行未完成步骤。"
            "可以引用当前 Task Plan 作为背景，但它不改变 read 模式的只读边界。代码定位优先用 read/grep/find。"
        )
    return (
        "当前 mode=build。你仍是同一个 Coding Agent，处理同一个用户任务，本轮可以在权限允许范围内读取、修改、运行命令并验证。"
        "已有 active plan 时，其中的执行目标、完成标准和步骤是本次任务的执行契约，必须直接推进而不是重新制定方案；"
        "用户的“给方案”等控制级表达不能替换执行目标。只有用户明确要求修改，或当前步骤已发生"
        "五次有效实现/验证失败，或实际代码与计划基础明显不一致时，才可用 update_plan 修订同一计划。"
        "没有计划时，仅对复杂任务按需创建 active soft plan；简单任务可直接实现和验证。精确修改优先 apply_patch，"
        "单点替换用 edit，新建或整体重写才用 write。代码定位优先用 read/grep/find，shell 主要用于测试和项目命令。"
        "执行 active plan 时尽量每完成一个主要步骤就更新状态；如果发现范围超出、风险升高或计划不再适用，先修订计划或请求用户确认。"
        "最终答复前必须依据 completion criteria、实际改动和最新验证结果调用 update_plan 收尾，完成时将 snapshot.status 设为 completed。"
    )


def _plan_objective(plan: object) -> str:
    if not isinstance(plan, dict):
        return ""
    return _optional_text(plan.get("objective")) or ""


def runtime_retry_policy(session: Any) -> RetryPolicy:
    return session.retry_policy()


def _is_continue_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.strip().lower().strip("。.!！?？").split())
    return normalized in _CONTINUE_REQUESTS


def _is_terminal_outcome(outcome: AgentLoopOutcome) -> bool:
    return outcome.status not in {"waiting_user", "waiting_approval"}


def _message_text(message: UserMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return " ".join(content.strip().split())
    return " ".join(
        block.text.strip()
        for block in content
        if isinstance(block, TextContent) and block.text.strip()
    )


def _prompt_text(prepared: PreparedAgentRun) -> str:
    for message in prepared.input_messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
    return ""


def _int_or_default(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _event_id(event: dict[str, Any]) -> str | None:
    value = event.get("eventId") or event.get("event_id")
    return value if isinstance(value, str) and value else None


def _event_run_id(event: dict[str, Any]) -> str | None:
    value = event.get("runId") or event.get("run_id")
    return value if isinstance(value, str) and value else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _plan_payload(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    raw = getattr(value, "__dict__", None)
    if isinstance(raw, dict):
        return dict(raw)
    return None


def _short_text(value: object, *, limit: int) -> str:
    text = str(value).strip() if value is not None else ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _needs_plan_approval_notice(mode: str, plan_state: object) -> bool:
    return (
        mode == "build"
        and isinstance(plan_state, dict)
        and plan_state.get("status") == "proposed"
    )


def _checkpoint_pending_calls(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    calls = checkpoint.get("pending_tool_calls")
    if isinstance(calls, list):
        result: list[dict[str, Any]] = []
        for item in calls:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            call = {
                "id": str(item.get("id")),
                "name": str(item.get("name") or ""),
                "arguments": dict(item.get("arguments")) if isinstance(item.get("arguments"), dict) else {},
            }
            for field_name in ("approval_id", "reason", "risk_level"):
                value = _optional_text(item.get(field_name))
                if value is not None:
                    call[field_name] = value
            result.append(call)
        return result
    ids = checkpoint.get("pending_tool_call_ids")
    if not isinstance(ids, list):
        return []
    return [{"id": item, "name": "", "arguments": {}} for item in ids if isinstance(item, str)]


def _approval_reason_from_event(event: dict[str, Any]) -> str:
    direct = _optional_text(event.get("reason")) or _optional_text(event.get("errorReason"))
    if direct is not None:
        return direct
    result = event.get("result")
    metadata = result.get("metadata") if isinstance(result, dict) else None
    if isinstance(metadata, dict):
        decision = metadata.get("permission_decision")
        if isinstance(decision, dict):
            reason = _optional_text(decision.get("reason"))
            if reason is not None:
                return reason
    return ""


def _approval_risk_from_event(event: dict[str, Any]) -> str:
    direct = _optional_text(event.get("riskLevel"))
    if direct is not None:
        return direct
    result = event.get("result")
    metadata = result.get("metadata") if isinstance(result, dict) else None
    if isinstance(metadata, dict):
        decision = metadata.get("permission_decision")
        if isinstance(decision, dict):
            risk = _optional_text(decision.get("risk_level"))
            if risk is not None:
                return risk
    return "unknown"


def _set_session_message_id(message: Message, message_id: str) -> None:
    metadata = getattr(message, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault("session_message_id", message_id)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _plan_event_type(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> str:
    old_status = previous.get("status") if isinstance(previous, dict) else None
    new_status = current.get("status")
    if new_status == "proposed" and old_status != "proposed":
        return "plan_proposed"
    if new_status == "completed" and old_status != "completed":
        return "plan_completed"
    if new_status == "rejected" and old_status != "rejected":
        return "plan_rejected"
    if new_status == "abandoned" and old_status != "abandoned":
        return "plan_abandoned"
    return "plan_updated"


__all__ = [
    "RuntimeSessionContextPort",
    "SessionRuntime",
    "new_run_id",
    "runtime_retry_policy",
]
