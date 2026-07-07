from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Callable
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
    TaskStrategy,
)
from codepilot.core.loop import maybe_await
from codepilot.core.task import ensure_planning_budget_profile, ensure_task_mode
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import AgentEvent, AgentRunResult, Message, TextContent, UserMessage
from codepilot.protocols.commands import SessionLifecycleContext, SessionLifecycleView

from .context.freshness import build_context_freshness_notice
from .context.governor import ContextGovernor, calibrate_context_usage
from .context.state import SessionContextState
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
from .history.git_rollback import GitRollbackBaseline, build_rollback_metadata, capture_git_baseline
from .memory import MemoryRetriever, MemoryStore, MemoryWriteContext, MemoryWriter
from .storage import SessionStore, new_session_id
from .task_state import TaskStateStore


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
    transcript, persistent stores, context preparation, memory, task state and
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

        persisted = self.store.load_session_messages()
        messages = [*persisted, *options.messages]
        self.task_mode = ensure_task_mode(options.task_mode)
        self.planning_budget_profile = ensure_planning_budget_profile(
            options.planning_budget_profile
        )
        self.conversation = SessionConversationState(
            model=options.model,
            system_prompt=options.system_prompt,
            messages=messages,
            thinking_level=options.thinking_level,
            task_mode=self.task_mode,
        )

        self.memory_enabled = bool(options.memory_enabled)
        self.task_control_enabled = bool(options.task_control_enabled)
        self.task_state = TaskStateStore(self.store)
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
        self.max_task_replans_per_run = options.max_task_replans_per_run
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
                task_strategy=self._task_strategy(mode_hint=intent.mode_hint),
                limits=self.loop_limits(),
                retry_policy=self.retry_policy(),
            ),
            context_port=RuntimeSessionContextPort(self),
            input_messages=[UserMessage(content=intent.text)],
            rollback_baseline=self._remember_rollback_baseline(run_id, rollback),
            context_refs={"context": "session_context"},
            memory_refs={"enabled": self.memory_enabled},
            task_refs={"task_state": self.active_task_state()},
        )

    async def prepare_resume(
        self,
        intent: SessionResumeIntent,
        *,
        run_id: str,
        model: ModelDescriptor,
    ) -> PreparedAgentRun:
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
            task_strategy=self._task_strategy(),
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
                task_strategy=self._task_strategy(),
                limits=self.loop_limits(),
                retry_policy=self.retry_policy(),
            ),
            resume_input=resume_input,
            context_port=RuntimeSessionContextPort(self),
            rollback_baseline=self._remember_rollback_baseline(run_id, rollback),
            task_refs={"task_state": self.active_task_state()},
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
                self.store.append_event(dict(event))
                await self.conversation.dispatch_event(event)
            committed_messages = [*prepared.input_messages, *outcome.new_messages]
            self.conversation.append_messages(committed_messages)
            for message in committed_messages:
                self.store.append_message(message)
            self.conversation.remember_result(result)

        self.store.append_run_result(result)
        self._write_rollback_metadata(
            result,
            self._take_rollback_baseline(prepared.rollback_baseline),
        )
        self._finalize_task_state(result)
        self._calibrate_context_usage(result)
        if self.memory_enabled:
            self._finalize_memory(result)
        self.context_governor.finalize_run(result)
        await self._run_lifecycle_hooks(
            text=_prompt_text(prepared),
            is_continue=not prepared.input_messages,
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
                "task": prepared.task_refs,
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
            task_mode=self.task_mode,
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
            "task_mode": self.task_mode,
            "planning_budget_profile": self.planning_budget_profile,
        }

    def active_task_state(self) -> dict[str, object] | None:
        state = self.task_state.current()
        return dict(state) if state is not None else None

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

    def set_task_mode(self, mode: str) -> str:
        normalized = ensure_task_mode(mode)
        if normalized != self.task_mode:
            self.task_mode = normalized
            self.conversation.set_task_mode(normalized)
            self.store.append_event(
                {
                    "type": "mode_changed",
                    "sessionId": self.session_id,
                    "currentMode": normalized,
                }
            )
        self.task_state.apply_event({"type": "mode_changed", "current_mode": normalized})
        return normalized

    def rebind_store(self, store: SessionStore) -> None:
        self.store = store
        self.session_id = store.session_id
        self.task_state = TaskStateStore(store)
        self.memory_store = MemoryStore(store)
        self.memory_writer = MemoryWriter(store=self.memory_store, workspace_dir=self.workspace_dir)
        self.memory_retriever = MemoryRetriever(store=self.memory_store, workspace_dir=self.workspace_dir)
        self.context_governor = self._new_context_governor()
        self.prepare_context = self._custom_prepare_context or self.context_governor.prepare

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        return self.conversation.subscribe(listener)

    def close(self) -> None:
        self.conversation.clear_listeners()

    # Existing tests and helper code use underscored names for the live object.
    def _subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        return self.subscribe(listener)

    def _close(self) -> None:
        self.close()

    def _rebind_store(self, store: SessionStore) -> None:
        self.rebind_store(store)

    def _active_task_state(self) -> dict[str, object] | None:
        return self.active_task_state()

    def _is_continue_run(self, text: str) -> bool:
        return self.active_task_state() is not None and _is_continue_text(text)

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
        if not is_continue:
            if self.memory_enabled:
                self._admit_prompt_memory(text, run_id=run_id)
            self._begin_task_state(text, run_id=run_id)
        else:
            self._recover_continue_task_state(run_id=run_id)
        self._check_context_freshness()
        return rollback

    def _admit_prompt_memory(self, text: str, *, run_id: str | None) -> None:
        try:
            event_id = f"event_{uuid4().hex[:12]}"
            result = self.memory_writer.admit_prompt_memory(
                text,
                context=MemoryWriteContext(
                    session_id=self.session_id,
                    run_id=run_id,
                    source_event_id=event_id,
                    evidence_refs=[f"event:{event_id}", *([f"run:{run_id}"] if run_id else [])],
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

    def _begin_task_state(self, text: str, *, run_id: str | None) -> None:
        try:
            state = self.task_state.begin(
                text,
                current_mode=self.task_mode,
                run_id=run_id,
            )
            self.store.append_event(
                {
                    "type": "task_state_updated",
                    "sessionId": self.session_id,
                    "runId": run_id,
                    "goal": state.get("goal"),
                }
            )
        except Exception as exc:
            logger.warning("failed to begin task state: %s", exc)
            self.store.append_event(
                {
                    "type": "task_state_warning",
                    "sessionId": self.session_id,
                    "operation": "task_state_begin",
                    "message": str(exc),
                }
            )

    def _recover_continue_task_state(self, *, run_id: str | None) -> None:
        state = self.task_state.current()
        if state is None:
            return
        goal_is_continue = _task_state_goal_is_continue(state)
        task_is_closed = _task_state_all_steps_completed(state)
        if not goal_is_continue and not task_is_closed:
            return
        recovered_request = (
            self._last_substantive_user_request()
            if goal_is_continue
            else _task_state_request(state) or self._last_substantive_user_request()
        )
        if recovered_request is None:
            return
        recovered = self.task_state.begin(
            recovered_request,
            current_mode=self.task_mode,
            run_id=run_id,
        )
        self.store.append_event(
            {
                "type": "task_state_updated",
                "sessionId": self.session_id,
                "runId": run_id,
                "goal": recovered.get("goal"),
                "recoveredFrom": "continue_request",
            }
        )

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

    def _finalize_task_state(self, result: AgentRunResult) -> None:
        try:
            state = self.task_state.current()
            if state is None:
                return
            if result.status == "completed" and result.task is not None and result.task.completion_satisfied:
                state = _state_from_task_summary(state, result)
                self.task_state.save(state)
            self.store.append_event(
                {
                    "type": "task_state_updated",
                    "sessionId": self.session_id,
                    "runId": result.run_id,
                    "goal": state.get("goal"),
                    "completionSatisfied": _completion_satisfied(state),
                }
            )
        except Exception as exc:
            logger.warning("failed to finalize task state: %s", exc)
            self.store.append_event(
                {
                    "type": "task_state_warning",
                    "sessionId": self.session_id,
                    "operation": "task_state_finalize",
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
                task_mode=str(self.task_mode),
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

    def _loop_context(self) -> PreparedContext:
        return PreparedContext(
            {
                "system_prompt": self.conversation.system_prompt,
                "session_id": self.session_id,
            }
        )

    def _task_strategy(self, *, mode_hint: str | None = None) -> TaskStrategy:
        return TaskStrategy(
            enabled=self.task_control_enabled,
            mode=mode_hint or self.task_mode,
            task_state=self.active_task_state(),
            planning_budget_profile=self.planning_budget_profile,
            max_replans_per_run=_int_or_default(
                self.max_task_replans_per_run,
                default=2,
            ),
        )

    def _tool_iteration_budget(self) -> int:
        configured = getattr(self, "max_tool_iterations", None)
        if isinstance(configured, int) and not isinstance(configured, bool) and configured >= 0:
            return configured
        return _TOOL_ITERATION_BUDGET_BY_MODE[self.task_mode]

    def _new_context_governor(self) -> ContextGovernor:
        return ContextGovernor(
            workspace_dir=self.workspace_dir,
            session_id=self.session_id,
            state=SessionContextState(workspace_dir=self.workspace_dir),
            memory_retriever=self.memory_retriever if self.memory_enabled else None,
            task_state_store=self.task_state,
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
        task_signal = request_context.get("task_signal")
        task_state = (
            session.active_task_state()
            if hasattr(session, "active_task_state")
            else session._active_task_state()
        )
        prepared = await maybe_await(
            session.prepare_context(
                AgentContext(
                    system_prompt=str(request.get("system_prompt", "")),
                    messages=list(request.get("messages", ())),
                    tools=list(request.get("tools", ())),
                    task_state=task_state,
                    task_signal=task_signal if isinstance(task_signal, dict) else None,
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
                "type": "context_prepared",
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


def new_run_id() -> str:
    return f"run_{uuid4().hex[:12]}"


def runtime_loop_limits(session: Any) -> AgentLoopLimits:
    if hasattr(session, "loop_limits"):
        return session.loop_limits()
    configured = getattr(session, "max_tool_iterations", None)
    if isinstance(configured, int) and not isinstance(configured, bool) and configured >= 0:
        iterations = configured
    else:
        mode = ensure_task_mode(getattr(session, "task_mode", "build"))
        iterations = _TOOL_ITERATION_BUDGET_BY_MODE[mode]
    return AgentLoopLimits(
        max_tool_iterations=iterations,
        max_tool_calls_per_turn=getattr(session, "max_tool_calls_per_turn", None),
    )


def runtime_retry_policy(session: Any) -> RetryPolicy:
    return session.retry_policy()


def _state_from_task_summary(
    state: dict[str, object],
    result: AgentRunResult,
) -> dict[str, object]:
    summary = result.task
    if summary is None:
        return state
    next_state = dict(state)
    next_state["task_id"] = summary.task_id or next_state.get("task_id")
    summary_goal = summary.goal if not _is_continue_text(summary.goal) else ""
    next_state["raw_user_request"] = summary_goal or next_state.get("raw_user_request")
    next_state["goal"] = {
        "value": summary_goal or str(next_state.get("raw_user_request") or ""),
        "source": "builder",
        "confidence": "inferred",
    }
    signal = summary.control_signal if isinstance(summary.control_signal, dict) else {}
    mode = signal.get("mode")
    if mode in {"read", "plan", "build"}:
        next_state["current_mode"] = mode
    current_step_id = signal.get("current_step_id")
    next_state["current_step_id"] = current_step_id if isinstance(current_step_id, str) else None
    next_state["steps"] = _summary_steps(summary)
    next_state["verification_status"] = "passed" if summary.completion_satisfied else "unknown"
    next_state["blocked_reason"] = (
        summary.completion_reason
        if summary.blocked_steps and summary.completion_reason
        else None
    )
    next_state["source_run_id"] = result.run_id
    return next_state


def _summary_steps(summary: Any) -> list[dict[str, object]]:
    details = summary.step_details if isinstance(summary.step_details, dict) else {}
    rows: list[tuple[str, str]] = [
        *((title, "completed") for title in summary.completed_steps),
        *((title, "blocked") for title in summary.blocked_steps),
        *((title, "pending") for title in summary.pending_steps),
    ]
    steps: list[dict[str, object]] = []
    seen: set[str] = set()
    for title, status in rows:
        if title in seen:
            continue
        seen.add(title)
        detail = details.get(title)
        detail = detail if isinstance(detail, dict) else {}
        steps.append(
            {
                "id": f"step_{len(steps) + 1}",
                "title": title,
                "kind": detail.get("kind") if detail.get("kind") in {"read", "plan", "edit", "verify", "summarize", "other"} else "other",
                "status": status,
                "acceptance": detail.get("acceptance") if isinstance(detail.get("acceptance"), str) else None,
                "verification_hint": detail.get("verification_hint") if isinstance(detail.get("verification_hint"), str) else None,
                "summary": detail.get("summary") if isinstance(detail.get("summary"), str) else None,
                "evidence_refs": [],
                "failure_count": 0,
            }
        )
    return steps


def _completion_satisfied(state: dict[str, object]) -> bool | None:
    steps = state.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    return all(isinstance(step, dict) and step.get("status") == "completed" for step in steps)


def _task_state_goal_is_continue(state: dict[str, object]) -> bool:
    raw_goal = state.get("goal")
    goal = raw_goal.get("value") if isinstance(raw_goal, dict) else raw_goal
    return _is_continue_text(state.get("raw_user_request")) or _is_continue_text(goal)


def _task_state_all_steps_completed(state: dict[str, object]) -> bool:
    steps = state.get("steps")
    return (
        isinstance(steps, list)
        and bool(steps)
        and all(isinstance(step, dict) and step.get("status") == "completed" for step in steps)
    )


def _task_state_request(state: dict[str, object]) -> str | None:
    raw_request = state.get("raw_user_request")
    if isinstance(raw_request, str) and raw_request.strip() and not _is_continue_text(raw_request):
        return " ".join(raw_request.strip().split())
    raw_goal = state.get("goal")
    goal = raw_goal.get("value") if isinstance(raw_goal, dict) else raw_goal
    if isinstance(goal, str) and goal.strip() and not _is_continue_text(goal):
        return " ".join(goal.strip().split())
    return None


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


__all__ = [
    "RuntimeSessionContextPort",
    "SessionRuntime",
    "new_run_id",
    "runtime_loop_limits",
    "runtime_retry_policy",
]
