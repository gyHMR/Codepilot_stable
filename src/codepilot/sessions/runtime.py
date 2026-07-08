from __future__ import annotations

import inspect
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

_TOOL_ITERATION_BUDGET_BY_MODE = {
    "read": 24,
    "plan": 32,
    "build": 48,
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
        self.store = SessionStore(self.workspace_dir, self.session_id)
        self.store.ensure_initialized(
            model_id=options.model.id,
            provider=options.model.provider,
            system_prompt=options.system_prompt,
        )
        self._repair_checkpoint_messages()

        persisted = self.store.load_session_messages()
        messages = [*persisted, *options.messages]
        self.current_mode = ensure_run_mode(options.current_mode)
        self.store.update_meta({"current_mode": self.current_mode})
        self.planning_budget_profile = ensure_planning_budget_profile(
            options.planning_budget_profile
        )
        self.conversation = SessionConversationState(
            model=options.model,
            system_prompt=options.system_prompt,
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
        is_continue = self._is_continue_run(intent.text)
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
        run_plan_state = self.active_plan_state() or self._run_local_plan_seed(
            intent.text,
            run_id=run_id,
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
                context=self._loop_context(),
                model=model,
                tools=[],
                mode=ensure_run_mode(intent.mode_hint or self.current_mode),
                plan_state=run_plan_state,
                limits=self.loop_limits(),
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
        rollback = await self._begin_run(text="", run_id=run_id, is_continue=True)
        messages = self._messages_for_loop()
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
            ),
            resume_input=resume_input,
            context_port=RuntimeSessionContextPort(self),
            rollback_baseline=self._remember_rollback_baseline(run_id, rollback),
            plan_refs={"plan_state": self.active_plan_state()},
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
                self._persist_event(dict(event))
                await self.conversation.dispatch_event(event)
            committed_messages = list(outcome.new_messages)
            self.conversation.append_messages(committed_messages)
            for message in committed_messages:
                if id(message) not in self._persisted_message_object_ids:
                    message_id = self.store.append_message(message, run_id=result.run_id)
                    _set_session_message_id(message, message_id)
            self.conversation.remember_result(result)

        self.store.append_run_result(result)
        self._write_rollback_metadata(
            result,
            self._take_rollback_baseline(prepared.rollback_baseline),
        )
        self._finalize_plan_state(outcome)
        self._calibrate_context_usage(result)
        if self.memory_enabled:
            self._finalize_memory(result)
        self.context_governor.finalize_run(result)
        if outcome.status != "waiting_approval":
            self.store.set_checkpoint(None)
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

    def active_plan_state(self) -> dict[str, object] | None:
        state = self.plan_state.current()
        if state is None:
            return None
        if state.get("status") in {"none", "rejected", "abandoned"}:
            return None
        active_plan_id = (self.store.read_meta() or {}).get("active_plan_id")
        if active_plan_id != state.get("plan_id"):
            return None
        return dict(state)

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            enabled=bool(self.retry_enabled),
            max_retries=_int_or_default(self.max_retries, default=0),
            base_delay_ms=_int_or_default(self.retry_base_delay_ms, default=0),
        )

    def loop_limits(self) -> AgentLoopLimits:
        return AgentLoopLimits(
            max_tool_iterations=self._tool_iteration_budget(),
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
        )

    def set_current_mode(self, mode: str) -> str:
        normalized = ensure_run_mode(mode)
        if normalized != self.current_mode:
            self.current_mode = normalized
            self.conversation.set_current_mode(normalized)
            self.store.append_event(
                {
                    "type": "mode_changed",
                    "sessionId": self.session_id,
                    "currentMode": normalized,
                }
            )
            self.store.update_meta({"current_mode": normalized})
        return normalized

    def approve_current_plan(self, *, switch_to_build: bool = True) -> dict[str, Any] | None:
        before = self.plan_state.current()
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return before
        state = self.plan_state.approve_current()
        if state is None:
            return None
        self._record_plan_event("plan_approved", state, run_id=None)
        if switch_to_build:
            self.set_current_mode("build")
        return state

    def reject_current_plan(self) -> dict[str, Any] | None:
        before = self.plan_state.current()
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return before
        state = self.plan_state.reject_current()
        if state is None:
            return None
        self._record_plan_event("plan_rejected", state, run_id=None)
        return state

    def abandon_current_plan(self) -> dict[str, Any] | None:
        before = self.plan_state.current()
        if not isinstance(before, dict):
            return before
        state = self.plan_state.abandon_current()
        if state is None:
            return None
        self._record_plan_event("plan_abandoned", state, run_id=None)
        return state

    def close(self) -> None:
        self.conversation.clear_listeners()

    def record_event(self, event: dict[str, Any]) -> None:
        """Persist a streamed runner event and update the recoverable checkpoint."""

        self._persist_event(event)
        self._checkpoint_from_event(event)

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

    def _begin_plan_state(self, text: str, *, run_id: str | None) -> None:
        try:
            state = self.plan_state.begin(
                text,
                origin_mode=self.current_mode,
                run_id=run_id,
            )
            self._record_plan_event("plan_updated", state, run_id=run_id)
        except Exception as exc:
            logger.warning("failed to begin plan state: %s", exc)
            self.store.append_event(
                {
                    "type": "plan_state_warning",
                    "sessionId": self.session_id,
                    "operation": "plan_state_begin",
                    "message": str(exc),
                }
            )

    def _run_local_plan_seed(self, text: str, *, run_id: str | None) -> dict[str, Any]:
        return PlanState.new(
            objective=text,
            origin_mode=self.current_mode,
            run_id=run_id,
        ).to_dict()

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
            if outcome.plan is None:
                return
            if (
                getattr(outcome.plan, "status", None) == "none"
                and not getattr(outcome.plan, "items", [])
            ):
                return
            previous = self.plan_state.current()
            state = self.plan_state.save(outcome.plan.__dict__)
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

    def _runtime_checkpoint(self) -> dict[str, Any] | None:
        meta = self.store.read_meta() or {}
        checkpoint = meta.get("runtime_checkpoint")
        return checkpoint if isinstance(checkpoint, dict) else None

    def _checkpoint_from_event(self, event: dict[str, Any]) -> None:
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
                        "phase": "awaiting_tools",
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
        phase = "awaiting_tools" if pending_ids else "tools_completed"
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
                "phase": "awaiting_tools",
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
        if not isinstance(checkpoint, dict) or checkpoint.get("phase") != "awaiting_tools":
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
                    "phase": "awaiting_tools" if approval_pending_calls else "tools_completed",
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

    def _loop_context(self) -> PreparedContext:
        return PreparedContext(
            {
                "system_prompt": self.conversation.system_prompt,
                "session_id": self.session_id,
            }
        )

    def _tool_iteration_budget(self) -> int:
        return _TOOL_ITERATION_BUDGET_BY_MODE[self.current_mode]

    def _approve_proposed_plan(self) -> None:
        before = self.plan_state.current()
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return
        state = self.plan_state.approve_current()
        if state is None:
            return
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

    def _take_rollback_baseline(self, ref: RollbackBaselineRef | None) -> GitRollbackBaseline:
        if ref is None:
            return GitRollbackBaseline(eligible=False, reason="missing_rollback_baseline_ref")
        if ref.session_id != self.session_id:
            return GitRollbackBaseline(eligible=False, reason="rollback_baseline_session_mismatch")
        return self._rollback_baselines.pop(
            ref.run_id,
            GitRollbackBaseline(eligible=False, reason="missing_rollback_baseline"),
        )


class RuntimeSessionContextPort:
    def __init__(self, session: SessionRuntime) -> None:
        self._session = session

    async def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session
        request_context = request.get("context")
        request_context = request_context if isinstance(request_context, dict) else {}
        run_signals = request_context.get("run_signals")
        plan_state = session.active_plan_state()
        prepared = await maybe_await(
            session.prepare_context(
                AgentContext(
                    system_prompt=str(request.get("system_prompt", "")),
                    messages=list(request.get("messages", ())),
                    tools=list(request.get("tools", ())),
                    plan_state=plan_state,
                    run_signals=run_signals if isinstance(run_signals, dict) else None,
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


def runtime_retry_policy(session: Any) -> RetryPolicy:
    return session.retry_policy()


def _is_continue_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.strip().lower().strip("。.!！?？").split())
    return normalized in _CONTINUE_REQUESTS


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
