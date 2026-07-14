from __future__ import annotations

import asyncio
import inspect
import hashlib
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from codepilot.core.contracts import (
    CoreLimits,
    CoreOutcome,
    CoreRunInput,
    ModelEntry,
    ToolResultEntry,
)
from codepilot.core.commands import (
    AbandonPlan,
    ApprovePlan,
    ApprovePlanRevision,
    CoreCommand,
    RejectPlan,
    RejectPlanRevision,
)
from codepilot.core.plan import (
    ensure_planning_budget_profile,
    ensure_run_mode,
    load_plan_state,
)
from codepilot.core.reducer import ReductionContext, apply_core_command
from codepilot.core.state import CoreState, load_core_state
from codepilot.core.tool_step import interrupted_tool_results
from codepilot.core.transcript import last_assistant_message, unsettled_tool_calls
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import (
    AgentRunResult,
    ImageContent,
    Message,
    TextContent,
    UserMessage,
)
from codepilot.protocols.commands import SessionLifecycleContext, SessionLifecycleView

from codepilot.sessions.context import (
    ContextBudgetConfig,
    ContextService,
    calibrate_context_usage,
)
from codepilot.sessions.contracts import (
    ComponentCheckpoint,
    ModelRef,
    PreparedAgentRun,
    RecoveryRequest,
    RollbackBaselineRef,
    SessionContinuationIntent,
    SessionOptions,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionView,
    WorkspaceEffectsSnapshot,
)
from .registry import SessionConversationState
from codepilot.sessions.rollback import GitRollbackBaseline, build_rollback_metadata, capture_git_baseline
from codepilot.sessions.memory import (
    MemoryProposal,
    MemoryProposalBatch,
    MemoryService,
)
from codepilot.sessions.workspace import capture_workspace_checkpoint
from codepilot.sessions.service import (
    BeginRunRequest,
    CommitRunBoundaryRequest,
    CreateSessionRequest,
    ResumeRunRequest,
    SessionStateService,
    new_session_id,
)
from codepilot.tools.security import ApprovalResponse
from codepilot.tools.codecs import json_value
from .session_state_adapter import RuntimeSessionStateAdapter
from .contracts import project_core_domain_event


logger = logging.getLogger("codepilot.runtime.session_coordinator")

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
class RuntimeSessionCoordinator:
    """Live session object.

    A session runtime owns the mutable state needed while the agent is running:
    transcript, persistent stores, context preparation, memory, plan state and
    rollback baselines.  ``RunCoordinator`` owns the public Run lifecycle;
    this object only exposes private preparation/commit primitives to it and
    keeps Session facts and component state together.
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
        self.state_service = SessionStateService(self.workspace_dir)
        session_state = (
            self.state_service.get_session(self.session_id)
            if options.session_id is not None
            else None
        )
        if session_state is None:
            session_state = self.state_service.create_session(
                CreateSessionRequest(
                    workspace_root=str(self.workspace_dir),
                    model=ModelRef(
                        provider=options.model.provider,
                        model=options.model.id,
                    ),
                    current_mode=self.current_mode,
                    system_prompt_hash=_hash_text(system_prompt),
                    session_id=self.session_id,
                )
            )
        self.session_state = session_state
        self.current_mode = ensure_run_mode(session_state.current_mode)
        persisted = [
            record.message
            for record in self.state_service.load_messages(self.session_id)
        ]
        messages = [*persisted, *options.messages]
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
        self.memory_service = MemoryService(
            workspace_dir=self.workspace_dir,
        )
        self._pending_memory_proposals: dict[str, tuple[MemoryProposal, ...]] = {}
        self._memory_proposal_errors: dict[str, str] = {}
        self._rollback_baselines: dict[str, GitRollbackBaseline] = {}
        self.context_service = self._new_context_service(options)
        self._restore_active_checkpoint()

        self.max_tool_calls_per_turn = options.max_tool_calls_per_turn
        self.retry_enabled = options.retry_enabled
        self.max_retries = options.max_retries
        self.retry_base_delay_ms = options.retry_base_delay_ms
        self.run_timeout_seconds = options.run_timeout_seconds
        self.extension_commands = dict(options.extension_commands)
        self.before_prompt_hooks = list(options.before_prompt_hooks)
        self.after_prompt_hooks = list(options.after_prompt_hooks)
        self.stream_fn = options.stream_fn
        self.convert_to_llm = options.convert_to_llm

    async def _prepare_run(
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
        run_plan_state = self._plan_state_for_run(
            text=intent.text,
            run_id=run_id,
            mode=effective_mode,
            pending_plan=pending_plan,
        )
        initial_core_state = _core_state_for_run(intent.text, run_plan_state)
        is_continue = effective_mode != "plan" and self._is_continue_run(intent.text)
        rollback = capture_git_baseline(self.workspace_dir)
        user_message = _user_message(intent)
        begun = self.state_service.begin_run(
            BeginRunRequest(
                session_id=self.session_id,
                request_id=intent.request_id,
                run_id=run_id,
                user_message=user_message,
                initial_core_state=initial_core_state.to_dict(),
                workspace=capture_workspace_checkpoint(self.workspace_dir),
                components=(
                    ComponentCheckpoint(
                        owner="rollback",
                        schema_version=1,
                        state=_rollback_baseline_state(rollback),
                    ),
                ),
            ),
            expected_session_revision=self.session_state.revision,
        )
        self.session_state = begun.session
        if not begun.reused:
            await self._run_lifecycle_hooks(
                text=intent.text,
                is_continue=is_continue,
                hooks=self.before_prompt_hooks,
            )
        user_message_id = begun.message.message_id
        if isinstance(begun.message.message, UserMessage):
            user_message = begun.message.message
        if begun.reused:
            self._restore_rollback_checkpoint(begun.run)
            rollback_ref = self._rollback_baseline_ref(run_id)
        else:
            rollback_ref = self._remember_rollback_baseline(run_id, rollback)
        user_message.metadata["session_message_id"] = user_message_id
        if begun.reused:
            self.conversation.set_messages(
                [record.message for record in self.state_service.load_messages(self.session_id)]
            )
        else:
            self.conversation.append_messages([user_message])
        state_port = RuntimeSessionStateAdapter(
            self.state_service,
            begun.session,
            begun.run,
            context_state=self.context_service.checkpoint_state,
            workspace_state=self._capture_workspace_checkpoint,
        )
        if not is_continue and not begun.reused:
            if self.memory_enabled:
                self._admit_prompt_memory(
                    intent.text,
                    run_id=run_id,
                    source_message_id=user_message_id,
                    state_port=state_port,
                )
        messages = self._messages_for_loop()
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=CoreRunInput(
                session_id=self.session_id,
                run_id=run_id,
                entry=ModelEntry(),
                messages=tuple(messages),
                state=load_core_state(
                    begun.run.core_state,
                    original_request=intent.text,
                ),
                model=model,
                mode=effective_mode,
                limits=self.core_limits(effective_mode),
                context_seed=self._loop_context(effective_mode),
            ),
            context_port=self.context_service,
            state_port=state_port,
            input_messages=[] if begun.reused else [user_message],
            rollback_baseline=rollback_ref,
            context_refs={"context": "session_context"},
            memory_refs={"enabled": self.memory_enabled},
            plan_refs={"plan_state": run_plan_state},
        )

    async def _prepare_resume(
        self,
        intent: SessionResumeIntent,
        *,
        run_id: str,
        model: ModelDescriptor,
        tools: Any | None = None,
    ) -> PreparedAgentRun:
        recovery = self.state_service.inspect_recovery(
            RecoveryRequest(
                session_id=self.session_id,
                run_id=run_id,
                expected_waiting_kind="tool_approval",
            )
        )
        if recovery.bundle is None:
            raise ValueError(f"Run is not recoverable: {run_id}")
        if tools is None:
            raise ValueError("Tool approval recovery requires a Tool port")
        self._restore_tool_checkpoint(tools, recovery.bundle.run)
        challenge = tools.approval_challenge(intent.approval_id)
        if challenge is None:
            raise ValueError(f"Approval not found: {intent.approval_id}")
        if challenge.run_id != run_id or challenge.session_id != self.session_id:
            raise ValueError("Approval does not belong to the recovered Run")
        self._ensure_workspace_recovery(recovery.bundle.workspace_status)
        self._restore_context_checkpoint(recovery.bundle.run)
        self._restore_rollback_checkpoint(recovery.bundle.run)
        prepared_resume = tools.prepare_resume(
            ApprovalResponse(
                approval_id=intent.approval_id,
                request_fingerprint=challenge.request_fingerprint,
                decision=intent.decision,
                reason=intent.reason,
            )
        )
        resumed_run = self.state_service.resume_run(
            ResumeRunRequest(
                session_id=self.session_id,
                run_id=run_id,
                checkpoint_id=recovery.bundle.run.checkpoint.checkpoint_id,  # type: ignore[union-attr]
                request_id=intent.approval_id,
                components=(
                    ComponentCheckpoint(
                        owner="tools",
                        schema_version=1,
                        state=json_value(prepared_resume.checkpoint_state),
                    ),
                ),
            ),
            expected_run_revision=recovery.bundle.run.revision,
        )
        self.session_state = recovery.bundle.session
        state_port = RuntimeSessionStateAdapter(
            self.state_service,
            self.session_state,
            resumed_run,
            context_state=self.context_service.checkpoint_state,
            workspace_state=self._capture_workspace_checkpoint,
        )
        messages = [record.message for record in recovery.bundle.messages]
        self.conversation.set_messages(messages)
        result = tools.execute_prepared_resume(prepared_resume.resume_id)
        if inspect.isawaitable(result):
            result = await result
        loop_input = CoreRunInput(
            session_id=self.session_id,
            run_id=run_id,
            entry=ToolResultEntry((result,)),
            messages=tuple(messages),
            state=resumed_run.core_state,
            model=model,
            mode=self.current_mode,
            limits=self.core_limits(),
            context_seed=self._loop_context(),
        )
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=loop_input,
            context_port=self.context_service,
            state_port=state_port,
            plan_refs={"plan_state": self.active_plan_state()},
            rollback_baseline=self._rollback_baseline_ref(run_id),
        )

    async def _prepare_continuation(
        self,
        intent: SessionContinuationIntent,
        *,
        run_id: str,
        model: ModelDescriptor,
        tools: Any | None = None,
    ) -> PreparedAgentRun:
        if intent.kind in {"tool_approved", "tool_denied"}:
            return await self._prepare_resume(
                SessionResumeIntent(
                    approval_id=intent.approval_id,
                    decision="approve" if intent.kind == "tool_approved" else "deny",
                    reason=intent.reason,
                    run_id=run_id,
                ),
                run_id=run_id,
                model=model,
                tools=tools,
            )

        recovery = self.state_service.inspect_recovery(
            RecoveryRequest(session_id=self.session_id, run_id=run_id)
        )
        if recovery.bundle is None or recovery.bundle.run.checkpoint is None:
            raise ValueError(f"No resumable checkpoint for run: {run_id}")
        self._ensure_workspace_recovery(recovery.bundle.workspace_status)
        self._restore_context_checkpoint(recovery.bundle.run)
        self._restore_rollback_checkpoint(recovery.bundle.run)
        pending_tool_resume = None
        if tools is not None:
            self._restore_tool_checkpoint(tools, recovery.bundle.run)
            pending_tool_resume = tools.pending_prepared_resume()
        checkpoint = recovery.bundle.run.checkpoint
        waiting = checkpoint.waiting
        if pending_tool_resume is not None:
            if intent.kind != "automatic_continuation":
                raise ValueError(
                    "Prepared Tool resume requires automatic continuation"
                )
            resumed_run = recovery.bundle.run
        else:
            resumed_run = self.state_service.resume_run(
                ResumeRunRequest(
                    session_id=self.session_id,
                    run_id=run_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    request_id=waiting.request_id if waiting is not None else None,
                ),
                expected_run_revision=recovery.bundle.run.revision,
            )
        self.session_state = recovery.bundle.session
        input_messages: list[Message] = []
        if intent.text:
            user_message = UserMessage(content=intent.text)
            committed = self.state_service.commit_run_boundary(
                CommitRunBoundaryRequest(
                    commit_id=f"{run_id}:continuation_input:{resumed_run.revision}",
                    kind="progress",
                    session_id=self.session_id,
                    run_id=run_id,
                    expected_run_revision=resumed_run.revision,
                    expected_session_revision=self.session_state.revision,
                    phase="model",
                    resume_point="before_model",
                    core_state=resumed_run.core_state,
                    new_messages=(user_message,),
                    workspace=(resumed_run.checkpoint.workspace if resumed_run.checkpoint else None),
                )
            )
            resumed_run = committed.run
            self.session_state = committed.session
            user_message.metadata["session_message_id"] = committed.committed_messages[0].message_id
            self.conversation.append_messages([user_message])
            input_messages.append(user_message)
        state_port = RuntimeSessionStateAdapter(
            self.state_service,
            self.session_state,
            resumed_run,
            context_state=self.context_service.checkpoint_state,
            workspace_state=self._capture_workspace_checkpoint,
        )
        mode = ensure_run_mode(intent.target_mode or self.current_mode)
        plan = self.context_plan_state_for_mode(mode)
        synthetic_control = _continuation_control(intent.kind)
        messages = [record.message for record in self.state_service.load_messages(self.session_id)]
        self.conversation.set_messages(messages)
        if pending_tool_resume is not None:
            result = tools.execute_prepared_resume(pending_tool_resume.resume_id)
            if inspect.isawaitable(result):
                result = await result
            entry = ToolResultEntry((result,))
        else:
            entry = _continuation_entry(checkpoint, waiting, messages)
        loop_input = CoreRunInput(
            session_id=self.session_id,
            run_id=run_id,
            entry=entry,
            messages=tuple(messages),
            state=resumed_run.core_state,
            model=model,
            mode=mode,
            limits=self.core_limits(mode),
            context_seed=self._loop_context(
                mode,
                synthetic_control=synthetic_control,
                checkpoint_phase=waiting.kind if waiting is not None else intent.kind,
            ),
        )
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=loop_input,
            context_port=self.context_service,
            state_port=state_port,
            input_messages=input_messages,
            context_refs={"context": "session_context", "continuation": intent.kind},
            memory_refs={"enabled": self.memory_enabled},
            plan_refs={"plan_state": plan},
            rollback_baseline=self._rollback_baseline_ref(run_id),
        )

    async def _commit_run(
        self,
        prepared: PreparedAgentRun,
        outcome: CoreOutcome,
        result: AgentRunResult,
        *,
        events: tuple[dict[str, object], ...] = (),
        store_outcome: bool,
    ) -> SessionRunRecord:
        state_port = (
            prepared.state_port
            if isinstance(prepared.state_port, RuntimeSessionStateAdapter)
            else None
        )
        if state_port is not None:
            if outcome.status != "waiting":
                terminal_status = {
                    "completed": "completed",
                    "failed": "failed",
                    "aborted": "cancelled",
                    "cancelled": "cancelled",
                }.get(result.status, "failed")
                terminal_messages = tuple(
                    message
                    for message in outcome.new_messages
                    if id(message) not in state_port.committed_message_ids
                )
                terminal_request = CommitRunBoundaryRequest(
                    commit_id=f"{result.run_id}:terminal:{state_port.run.revision}",
                    kind="terminal",
                    session_id=self.session_id,
                    run_id=result.run_id,
                    expected_run_revision=state_port.run.revision,
                    expected_session_revision=state_port.session.revision,
                    stop_reason=result.stop_reason,
                    terminal_status=terminal_status,  # type: ignore[arg-type]
                    result=result,
                    new_messages=terminal_messages,
                    workspace_effects=WorkspaceEffectsSnapshot(
                        changed=bool(result.workspace_changed),
                        affected_paths=tuple(result.affected_paths),
                    ),
                )
                finished = None
                for attempt in range(2):
                    try:
                        finished = self.state_service.commit_run_boundary(terminal_request)
                        break
                    except OSError:
                        if attempt == 1:
                            raise
                        await asyncio.sleep(0)
                if finished is None:  # pragma: no cover - defensive
                    raise RuntimeError("Terminal commit produced no receipt")
                state_port.run = finished.run
                state_port.session = finished.session
                for message, record in zip(terminal_messages, finished.committed_messages, strict=True):
                    state_port.committed_message_ids[id(message)] = record.message_id
            self.session_state = state_port.session
        if store_outcome:
            for event in events:
                payload = dict(event)
                await self.conversation.dispatch_event(payload)
            committed_messages = list(outcome.new_messages)
            self.conversation.append_messages(committed_messages)
            for message in committed_messages:
                message_id = state_port.committed_message_ids.get(id(message)) if state_port else None
                if message_id is not None:
                    _set_session_message_id(message, message_id)
            self.conversation.remember_result(result)

        if _is_terminal_outcome(outcome):
            self._submit_captured_memory_proposals(result)

        if prepared.rollback_baseline is not None:
            self._write_rollback_metadata(
                result,
                self._rollback_baseline(prepared.rollback_baseline),
            )
            if _is_terminal_outcome(outcome):
                self._discard_rollback_baseline(prepared.rollback_baseline)
        self._calibrate_context_usage(result)
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
            events=[dict(event) for event in events],
            outcome=outcome,
            snapshots={
                "context": prepared.context_refs,
                "memory": prepared.memory_refs,
                "plan": prepared.plan_refs,
                "rollback": prepared.rollback_baseline,
            },
        )
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
            "entry_ids": self.state_service.list_entry_ids(self.session_id),
            "entries": self.state_service.list_entries(self.session_id),
            "tree": self.state_service.get_session_tree(self.session_id),
            "leaf_id": (self.state_service.get_session(self.session_id).leaf_message_id if self.state_service.get_session(self.session_id) else None),
            "current_mode": self.current_mode,
            "planning_budget_profile": self.planning_budget_profile,
            "plan_summary": self.plan_summary(),
            "pending_plan_approval": self.pending_plan_approval(),
        }

    def plan_summary(self) -> dict[str, object] | None:
        state = self.workflow_plan_state()
        if not isinstance(state, dict):
            return None
        items = state.get("steps")
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
            "goal_preview": _short_text(
                (
                    state.get("definition", {}).get("summary")
                    if isinstance(state.get("definition"), dict)
                    else ""
                ),
                limit=72,
            ),
            "total_items": total,
            "done_items": done,
            "active_item_preview": _short_text(active, limit=72),
        }

    def runtime_checkpoint(self) -> dict[str, Any] | None:
        run_id = self.session_state.current_run_id
        if run_id is None:
            return None
        run = self.state_service.get_run(run_id)
        if run is None or run.checkpoint is None:
            return None
        checkpoint = run.checkpoint
        waiting = checkpoint.waiting
        phase = checkpoint.resume_point
        if waiting is not None:
            phase = {
                "tool_approval": "tool_approval",
                "plan_confirmation": "plan_approval",
                "user_input": str(waiting.payload.get("stop_reason") or "waiting_user"),
                "continuation": "continuation",
            }[waiting.kind]
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "phase": phase,
            "run_id": run.run_id,
            "approval_id": (
                waiting.request_id
                if waiting is not None and waiting.kind == "tool_approval"
                else None
            ),
            "request_id": waiting.request_id if waiting is not None else None,
            "waiting_kind": waiting.kind if waiting is not None else None,
            "run_state": dict(run.core_state),
        }

    def component_checkpoint_state(self, owner: str) -> dict[str, object] | None:
        run_id = self.session_state.current_run_id
        if run_id is None:
            return None
        run = self.state_service.get_run(run_id)
        if run is None or run.checkpoint is None:
            return None
        for component in run.checkpoint.components:
            if component.owner == owner:
                return dict(component.state)
        return None

    def _checkpoint_run_state(self, run_id: str) -> dict[str, object] | None:
        run = self.state_service.get_run(run_id)
        return dict(run.core_state) if run is not None else None

    def continuation_run_id(self) -> str:
        run_id = self.session_state.current_run_id
        if run_id is None:
            raise ValueError("No paused run to continue")
        return run_id

    def resume_run_id(self, approval_id: str) -> str:
        run_id = self.session_state.current_run_id
        run = self.state_service.get_run(run_id) if run_id is not None else None
        waiting = run.checkpoint.waiting if run is not None and run.checkpoint is not None else None
        if waiting is None or waiting.kind != "tool_approval" or waiting.request_id != approval_id:
            raise ValueError(f"Approval not found: {approval_id}")
        return run.run_id

    def pending_plan_approval(self) -> dict[str, Any] | None:
        state = self.workflow_plan_state()
        if isinstance(state, dict) and state.get("status") == "proposed":
            return dict(state)
        return None

    def current_plan_state(self) -> dict[str, Any] | None:
        run_id = self.session_state.current_run_id
        run = self.state_service.get_run(run_id) if run_id is not None else None
        if run is None:
            return None
        try:
            core = load_core_state(
                run.core_state,
                original_request=self._last_substantive_user_request()
                or "Continue the current task",
            )
        except (TypeError, ValueError):
            return None
        plan = core.task.plan
        return plan.to_dict() if plan is not None else None

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

    def core_limits(self, mode: str | None = None) -> CoreLimits:
        normalized = ensure_run_mode(mode or self.current_mode)
        return CoreLimits(
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
            self.session_state = self.state_service.update_session_mode(
                self.session_id,
                normalized,
                expected_revision=self.session_state.revision,
            )
            self.append_event(
                {
                    "type": "mode_changed",
                    "current_mode": normalized,
                }
            )
        return normalized

    def archive_plan_for_mode_switch(
        self,
        target_mode: str,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        del run_id
        normalized = ensure_run_mode(target_mode)
        current = self.current_plan_state()
        if not isinstance(current, dict):
            return None
        if current.get("status") not in {"proposed", "active"}:
            return current
        if normalized != "plan" or current.get("status") != "active":
            return current
        state = self._apply_plan_command(
            AbandonPlan(
                command_id=self._plan_command_id("abandon_mode_switch"),
                expected_revision=int(current["revision"]),
                reason="mode_switch",
            ),
            mode=self.current_mode,
        )
        self._finish_current_plan_run("plan.abandoned_for_mode_switch")
        return state

    def approve_current_plan(self, *, switch_to_build: bool = True) -> dict[str, Any] | None:
        before = self.workflow_plan_state()
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return before
        state = self._apply_plan_command(
            ApprovePlan(
                command_id=self._plan_command_id("approve"),
                expected_revision=int(before["revision"]),
            ),
            mode="plan",
        )
        if switch_to_build:
            self.set_current_mode("build")
        return state

    def reject_current_plan(self) -> dict[str, Any] | None:
        before = self.workflow_plan_state()
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return before
        state = self._apply_plan_command(
            RejectPlan(
                command_id=self._plan_command_id("reject"),
                expected_revision=int(before["revision"]),
            ),
            mode="plan",
        )
        self._finish_current_plan_run("plan.rejected")
        self.set_current_mode("plan")
        return state

    def abandon_current_plan(self) -> dict[str, Any] | None:
        before = self.current_plan_state()
        if not isinstance(before, dict):
            return before
        state = self._apply_plan_command(
            AbandonPlan(
                command_id=self._plan_command_id("abandon"),
                expected_revision=int(before["revision"]),
                reason="user_abandoned",
            ),
            mode=self.current_mode,
        )
        self._finish_current_plan_run("plan.abandoned")
        return state

    def approve_current_plan_revision(self) -> dict[str, Any] | None:
        before = self.workflow_plan_state()
        if not isinstance(before, dict) or before.get("pending_revision") is None:
            return before
        return self._apply_plan_command(
            ApprovePlanRevision(
                command_id=self._plan_command_id("approve_revision"),
                expected_revision=int(before["revision"]),
            ),
            mode=self.current_mode,
        )

    def reject_current_plan_revision(self) -> dict[str, Any] | None:
        before = self.workflow_plan_state()
        if not isinstance(before, dict) or before.get("pending_revision") is None:
            return before
        return self._apply_plan_command(
            RejectPlanRevision(
                command_id=self._plan_command_id("reject_revision"),
                expected_revision=int(before["revision"]),
            ),
            mode=self.current_mode,
        )

    def _apply_plan_command(
        self,
        command: CoreCommand,
        *,
        mode: str,
    ) -> dict[str, Any] | None:
        run_id = self.session_state.current_run_id
        run = self.state_service.get_run(run_id) if run_id is not None else None
        if run is None or run.checkpoint is None:
            raise ValueError("No active Core run owns the current Task Plan")
        core = load_core_state(
            run.core_state,
            original_request=self._last_substantive_user_request()
            or "Continue the current task",
        )
        reduction = apply_core_command(
            core,
            command,
            ReductionContext(
                run_id=run.run_id,
                mode=ensure_run_mode(mode),
                now_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            ),
        )
        result = reduction.command_results[0] if reduction.command_results else None
        if result is not None and result.status == "rejected":
            raise ValueError(f"Plan command rejected: {result.reason}")
        durable_events = tuple(
            project_core_domain_event(
                event,
                event_id=f"{run.run_id}:runtime_plan:{run.revision}:{index}",
                run_id=run.run_id,
                session_id=self.session_id,
            )
            for index, event in enumerate(reduction.events, start=1)
        )
        committed = self.state_service.commit_run_boundary(
            CommitRunBoundaryRequest(
                commit_id=(
                    f"{run.run_id}:core_command:{command.command_id}:{run.revision}"
                ),
                kind="progress",
                session_id=self.session_id,
                run_id=run.run_id,
                expected_run_revision=run.revision,
                expected_session_revision=self.session_state.revision,
                phase="tools",
                resume_point="after_tools",
                core_state=reduction.state.to_dict(),
                durable_events=durable_events,
                components=run.checkpoint.components,
                workspace=run.checkpoint.workspace,
            )
        )
        self.session_state = committed.session
        plan = reduction.state.task.plan
        return plan.to_dict() if plan is not None else None

    def _finish_current_plan_run(self, stop_reason: str) -> None:
        run_id = self.session_state.current_run_id
        run = self.state_service.get_run(run_id) if run_id is not None else None
        if run is None:
            return
        committed = self.state_service.commit_run_boundary(
            CommitRunBoundaryRequest(
                commit_id=f"{run.run_id}:plan_terminal:{run.revision}",
                kind="terminal",
                session_id=self.session_id,
                run_id=run.run_id,
                expected_run_revision=run.revision,
                expected_session_revision=self.session_state.revision,
                terminal_status="cancelled",
                stop_reason=stop_reason,
            )
        )
        self.session_state = committed.session

    def _plan_command_id(self, action: str) -> str:
        run_id = self.session_state.current_run_id or "no_run"
        run = self.state_service.get_run(run_id) if run_id != "no_run" else None
        revision = run.revision if run is not None else 0
        return f"runtime:{run_id}:{action}:{revision}"

    def close(self) -> None:
        self.conversation.clear_listeners()

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event)
        return self.state_service.append_event(self.session_id, payload)

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
        return rollback

    def _admit_prompt_memory(
        self,
        text: str,
        *,
        run_id: str | None,
        source_message_id: str | None,
        state_port: RuntimeSessionStateAdapter,
    ) -> None:
        try:
            receipt = self.memory_service.admit_user_prompt(
                text,
                session_id=self.session_id,
                run_id=run_id or "prompt",
            )
            for record in receipt.records:
                state_port.queue_durable_event(
                    {
                        "type": "memory_user_explicit_admitted",
                        "run_id": run_id,
                        "source_message_id": source_message_id,
                        "memory_id": record.id,
                        "status": record.status,
                    }
                )
        except Exception as exc:
            logger.warning("failed to admit prompt memory: %s", exc)
            state_port.queue_durable_event(
                {
                    "type": "memory_warning",
                    "operation": "prompt_memory_admission",
                    "message": str(exc),
                }
            )

    def capture_memory_proposals(
        self,
        run_id: str,
        proposals: tuple[MemoryProposal, ...],
        error: str | None,
    ) -> None:
        if proposals:
            self._pending_memory_proposals[run_id] = tuple(proposals)
        else:
            self._pending_memory_proposals.pop(run_id, None)
        if error:
            self._memory_proposal_errors[run_id] = str(error)
        else:
            self._memory_proposal_errors.pop(run_id, None)

    def _submit_captured_memory_proposals(self, result: AgentRunResult) -> None:
        proposals = self._pending_memory_proposals.pop(result.run_id, ())
        sidecar_error = self._memory_proposal_errors.pop(result.run_id, None)
        if sidecar_error:
            self.state_service.append_event(
                self.session_id,
                {
                    "type": "memory_proposal_invalid",
                    "message": sidecar_error,
                },
                run_id=result.run_id,
            )
        if not self.memory_enabled or result.status != "completed" or not proposals:
            return
        verification_passed = result.signals.verification_status == "passed"
        try:
            receipt = self.memory_service.submit_proposals(
                MemoryProposalBatch(
                    session_id=self.session_id,
                    run_id=result.run_id,
                    origin="agent_finalization",
                    verification_passed=verification_passed,
                    proposals=proposals,
                )
            )
        except Exception as exc:
            logger.warning("failed to submit memory proposals: %s", exc)
            self.state_service.append_event(
                self.session_id,
                {
                    "type": "memory_proposal_failed",
                    "message": str(exc),
                },
                run_id=result.run_id,
            )
            return
        self.state_service.append_event(
            self.session_id,
            {
                "type": "memory_proposals_submitted",
                "memory_ids": [record.id for record in receipt.records],
                "rejected": list(receipt.rejected),
            },
            run_id=result.run_id,
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
                report=(dict(self.context_service.latest_report) or None),
                actual_input_tokens=actual_input,
            )
        except Exception as exc:
            logger.warning("failed to calibrate context usage: %s", exc)

    def _write_rollback_metadata(
        self,
        result: AgentRunResult,
        baseline: GitRollbackBaseline,
    ) -> None:
        self.state_service.write_rollback_metadata(
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

    def _loop_context(
        self,
        mode: str | None = None,
        *,
        synthetic_control: dict[str, object] | None = None,
        checkpoint_phase: str = "",
    ) -> dict[str, object]:
        ensure_run_mode(mode or self.current_mode)
        values: dict[str, Any] = {
            "system_prompt": self.conversation.system_prompt,
        }
        if synthetic_control:
            values["synthetic_control"] = dict(synthetic_control)
        if checkpoint_phase:
            values["checkpoint_phase"] = checkpoint_phase
        return values

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

    def _new_context_service(self, options: SessionOptions) -> ContextService:
        return ContextService(
            workspace_dir=self.workspace_dir,
            session_id=self.session_id,
            budget_config=ContextBudgetConfig(
                context_window=options.model.context_window,
                max_output_tokens=options.model.max_tokens,
                safety_margin_tokens=min(
                    max(
                        0,
                        options.model.context_window
                        - options.model.max_tokens
                        - 128,
                    ),
                    min(
                        8192,
                        max(1024, int(options.model.context_window * 0.05)),
                    ),
                ),
            ),
            memory_recall=self.memory_service if self.memory_enabled else None,
            summarizer=None,
        )

    def _restore_context_checkpoint(self, run: Any) -> None:
        checkpoint = run.checkpoint
        if checkpoint is None:
            return
        for component in checkpoint.components:
            if component.owner == "context":
                self.context_service.restore_checkpoint_state(dict(component.state))
                return

    @staticmethod
    def _restore_tool_checkpoint(tools: Any, run: Any) -> None:
        checkpoint = run.checkpoint
        if checkpoint is None:
            return
        for component in checkpoint.components:
            if component.owner == "tools":
                tools.restore_checkpoint_state(dict(component.state))
                return

    def _restore_active_checkpoint(self) -> None:
        run_id = self.session_state.current_run_id
        if run_id is None:
            return
        run = self.state_service.get_run(run_id)
        if run is None or run.checkpoint is None:
            return
        self._restore_context_checkpoint(run)
        self._restore_rollback_checkpoint(run)

    @staticmethod
    def _ensure_workspace_recovery(workspace_status: Any) -> None:
        if workspace_status.status == "unchanged":
            return
        paths = sorted(
            {
                *workspace_status.changed_paths,
                *workspace_status.missing_paths,
            }
        )
        detail = ", ".join(paths) if paths else "workspace root or Git state"
        raise ValueError(
            "Workspace changed after checkpoint; inspect before resuming: " + detail
        )

    def _restore_rollback_checkpoint(self, run: Any) -> None:
        checkpoint = run.checkpoint
        if checkpoint is None:
            return
        for component in checkpoint.components:
            if component.owner == "rollback":
                self._rollback_baselines[run.run_id] = _rollback_baseline_from_state(
                    component.state
                )
                return

    def _capture_workspace_checkpoint(self, core_state: dict[str, object]):
        affected = core_state.get("affected_paths")
        facts = core_state.get("facts")
        if isinstance(facts, dict):
            workspace = facts.get("workspace")
            if isinstance(workspace, dict):
                affected = workspace.get("affected_paths")
        paths = [str(path) for path in affected] if isinstance(affected, list) else []
        return capture_workspace_checkpoint(self.workspace_dir, tracked_paths=paths)

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
        events = self.state_service.load_events(self.session_id, run_id=run_id)
        turn_ids = [
            int(event.get("turn_id", 0))
            for event in events
            if isinstance(event.get("turn_id"), int)
        ]
        return len(events), max(turn_ids, default=0)


def _user_message(intent: SessionRunIntent) -> UserMessage:
    content: list[TextContent | ImageContent] = [TextContent(text=intent.text)]
    for raw in intent.images:
        mime_type = "image/png"
        data = raw
        if raw.startswith("data:") and ";base64," in raw:
            header, data = raw.split(",", 1)
            declared = header.removeprefix("data:").removesuffix(";base64").strip()
            if declared:
                mime_type = declared
        content.append(ImageContent(data=data, mime_type=mime_type))
    return UserMessage(content=content)


def _rollback_baseline_state(baseline: GitRollbackBaseline) -> dict[str, object]:
    return {
        "eligible": baseline.eligible,
        "reason": baseline.reason,
        "head": baseline.head,
        "branch": baseline.branch,
        "status_before": baseline.status_before,
    }


def _rollback_baseline_from_state(state: dict[str, object]) -> GitRollbackBaseline:
    return GitRollbackBaseline(
        eligible=bool(state.get("eligible")),
        reason=_optional_text(state.get("reason")),
        head=_optional_text(state.get("head")),
        branch=_optional_text(state.get("branch")),
        status_before=str(state.get("status_before") or ""),
    )


def new_run_id() -> str:
    return f"run_{uuid4().hex[:12]}"


def _continuation_control(kind: str) -> dict[str, object] | None:
    instruction = {
        "plan_approved": (
            "The canonical Task Plan has been approved and is active. Continue the same task "
            "in build mode from the first unfinished plan item. Execute the approved plan "
            "directly: do not restate it, redesign it, or create another Task Plan, and do not "
            "call create_build_plan while this active plan exists. Use update_plan_progress only "
            "to record progress on the existing plan and close_plan for final closeout. Every "
            "active-plan update must preserve the canonical item IDs exactly as supplied in the "
            "current plan. Preserve the approved goal, scope, constraints, and completion criteria."
        ),
        "plan_rejected": (
            "The user rejected the proposed plan. Ask one concise question about what "
            "should change; do not create a replacement plan until feedback is provided."
        ),
        "plan_feedback": (
            "The latest user message is feedback on the proposed plan. Revise the same "
            "plan with propose_plan, then summarize the canonical revision."
        ),
        "plan_clarification": (
            "Continue planning from the user's clarification. Publish a decision-complete "
            "plan with propose_plan when enough information is available."
        ),
        "mode_changed": "Continue the same task using the current mode policy.",
        "automatic_continuation": "Continue the same task from the saved checkpoint.",
    }.get(kind, "")
    if not instruction:
        return None
    scope = {
        "plan_feedback": "plan_revision_only",
        "plan_clarification": "plan_revision_only",
        "plan_approved": "approved_plan_execution_only",
        "plan_rejected": "final_answer_only",
    }.get(kind, "current_mode_only")
    return {
        "id": f"synthetic_continuation_{kind}",
        "kind": kind,
        "scope": scope,
        "source": "runner",
        "instruction": instruction,
        "reason": kind,
        "expires_after_turns": 1,
    }


def _mode_policy(mode: str) -> str:
    if mode == "plan":
        return (
            "当前 mode=plan。你仍是同一个 Coding Agent，处理同一个用户任务，但本轮只允许只读调查和方案设计，禁止修改工作区。"
            "Plan 是固定的宏观工作流，遵循五阶段：理解任务 → Subagent 探索 → 主 Agent 设计 → 审查并发布 canonical plan → 框架审批和切换 Build。"
            "框架负责模式、工具边界、canonical plan 状态、审批状态和 Plan 到 Build 的切换；主 Agent 负责理解目标、拆分探索任务、"
            "决定 Subagent 的关注范围、综合证据、识别真正阻塞的问题，并设计最终方案。不得自行假设计划已获批准或切换模式。"
            "阶段一，理解任务：分离对象级任务和控制级指令。代码、行为、测试和配置目标属于对象级；“给方案”“先分析”“不要修改”等"
            "只约束交付方式，不能成为计划摘要或执行步骤。识别用户目标、约束、当前证据和真正阻塞的歧义；"
            "只有缺少会实质改变实现范围或设计的必要信息时，才提出一个具体澄清问题。"
            "阶段二，Subagent 探索：探索阶段默认使用 dispatch_exploration 派发只读 Subagent 探索仓库，并按最少必要原则选择 0 到 3 个。"
            "上下文已经充分或任务真正微小时使用 0 个；已知文件或单一范围需要确认时使用 1 个；存在两个独立调查方向时使用 2 个；"
            "只有跨模块、架构不明或风险较高时使用 3 个。多个任务必须具有不同且具体的调查范围，分别覆盖相关实现、调用链、测试或风险，"
            "不得重复搜索同一区域。主 Agent 不应先用大量 ls/read/grep/find 顺序扫描仓库，这些工具只用于报告后的局部确认和缺口补充。"
            "dispatch_exploration 的 reuse=auto 会自动复用未过期报告；只有需要查看、筛选或比较已有报告时才使用 list_exploration_agents。"
            "阶段三，主 Agent 设计：综合用户上下文、Subagent 报告和必要的定点核查，选择一个推荐实现方案。Subagent 只提供仓库事实、"
            "风险、设计约束和验证线索，主 Agent 对最终设计、影响范围、执行步骤和验证方式负责。"
            "阶段四，审查并发布：检查方案是否覆盖用户目标、当前实现、目标设计、影响范围、风险、执行步骤、完成标准和验证方式。"
            "若仍有阻塞性问题，直接询问用户；若证据充分且方案已可交给 Build 执行，必须在当前回合直接调用 propose_plan。"
            "propose_plan 用于发布初始计划，或在批准前根据用户反馈修订同一个 proposed plan；"
            "task_understanding、current_implementation、target_design、impact_scope、risks_and_open_questions、verification_plan "
            "必须分别记录任务理解、仓库证据、目标设计、影响范围、风险待确认项和验证方案；"
            "summary 只做压缩概括，不能替代这些结构化字段。"
            "items 只能描述批准后实际要执行的代码修改与验证，不能写分析需求、查看代码、撰写方案、回复用户或等待审批。"
            "普通文本方案不是可审批的 Task Plan。不要先完整展示文本草案、询问用户方向是否合适，或等待用户认可文本草案后才调用 propose_plan；"
            "运行时会在 propose_plan 成功后统一展示 canonical plan 并发起审批。"
            "阶段五，框架审批和切换 Build：propose_plan 只表示计划已提交，不表示用户已经批准。审批阶段所有条目保持 pending；"
            "发布后等待用户审查、拒绝、批准或提出修改。未经运行时确认批准，不得执行实现、推进步骤、"
            "声称已经开始实现，或承诺下一步立即修改代码。"
            "用户反馈只能用于继续规划、修改同一个 proposed plan、拒绝或等待批准。只有高置信审批命令会由运行时转换为状态变化。"
        )
    if mode == "read":
        return (
            "当前 mode=read。你仍是同一个 Coding Agent，处理同一个用户任务，但本轮只做只读探索、定位、解释、审查和状态说明。"
            "目标是回答用户当前问题，并区分代码事实、合理推断和建议；不要默认生成实施计划。"
            "不得修改工作区、运行会产生副作用的命令，不得创建、推进或完成 Task Plan，也不得继续执行未完成步骤。"
            "可以引用当前 Task Plan 作为背景并说明其状态，但它不改变 read 模式的只读边界。代码定位优先用 read/grep/find。"
        )
    return (
        "当前 mode=build。你仍是同一个 Coding Agent，处理同一个用户任务，本轮可以在权限允许范围内读取、修改、运行命令并验证。"
        "没有 current Task Plan 且任务复杂时，可以用 create_build_plan 创建简要 active 执行计划；简单任务可直接实现和验证。"
        "已有 active plan 时，其中的执行目标、完成标准和步骤是本次任务的执行契约，必须直接推进而不是重新制定方案；"
        "进度更新只提交发生变化的步骤、expected_revision、完成说明和证据引用，不得回传整份计划快照。"
        "结构调整必须作为带原因的正式 revision 提交；来自 Plan 模式的已批准计划需要用户再次确认 revision。"
        "用户的“给方案”等控制级表达不能替换执行目标。只有用户明确要求修改，或当前步骤已发生"
        "五次有效实现/验证失败，或实际代码与计划基础明显不一致时，才可用 update_plan_progress 提议 revision。"
        "精确修改优先 apply_patch，"
        "单点替换用 edit，新建或整体重写才用 write。代码定位优先用 read/grep/find，shell 主要用于测试和项目命令。"
        "执行 active plan 时尽量每完成一个主要步骤就更新状态，但中间状态更新是软约束。"
        "最终答复前必须检查当前 Task Plan 是否完成，并依据 completion criteria、实际改动和最新验证结果调用 close_plan；"
        "close_plan 只提交关闭请求、expected_revision、总结和证据引用，最终完成状态由 Core 策略判定。"
    )


def _plan_goal(plan: object) -> str:
    if not isinstance(plan, dict):
        return ""
    definition = plan.get("definition")
    if not isinstance(definition, dict):
        return ""
    return _optional_text(definition.get("summary")) or ""


def _core_state_for_run(
    original_request: str,
    plan_state: dict[str, Any] | None,
) -> CoreState:
    state = CoreState.new(original_request)
    plan = load_plan_state(plan_state)
    if plan is None:
        return state
    return replace(state, task=replace(state.task, plan=plan))


def _continuation_entry(
    checkpoint: Any,
    waiting: Any,
    messages: list[Message],
) -> ModelEntry | ToolResultEntry:
    if waiting is not None:
        return ModelEntry()
    resume_point = checkpoint.resume_point
    if resume_point in {"before_model", "after_tools"}:
        return ModelEntry()
    if resume_point in {"after_model", "before_finalization"}:
        return ModelEntry(last_assistant_message(messages))
    if resume_point == "before_tools":
        calls = unsettled_tool_calls(messages)
        if not calls:
            raise ValueError("before_tools checkpoint has no unsettled Tool calls")
        return ToolResultEntry(
            interrupted_tool_results(
                calls,
                code="runtime.tool_execution_ambiguous",
                message=(
                    "The prior Run stopped after Tool preparation; Runtime cannot "
                    "prove whether these calls executed."
                ),
            )
        )
    raise ValueError(f"Unsupported continuation checkpoint: {resume_point}")


def _is_continue_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.strip().lower().strip("。.!！?？").split())
    return normalized in _CONTINUE_REQUESTS


def _is_terminal_outcome(outcome: CoreOutcome) -> bool:
    return outcome.status != "waiting"


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


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


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


def _set_session_message_id(message: Message, message_id: str) -> None:
    metadata = getattr(message, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault("session_message_id", message_id)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "RuntimeSessionCoordinator",
    "new_run_id",
    "runtime_retry_policy",
]
