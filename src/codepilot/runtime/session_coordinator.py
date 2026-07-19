"""协调 Session 输入、命令、等待恢复、Run 执行和终态提交。"""

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
from codepilot.tools.state import InteractionResponse
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
        run_id = begun.run.run_id
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
                state=load_core_state(begun.run.core_state),
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
        if intent.kind == "user_input_response":
            return await self._prepare_interaction_resume(
                intent,
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
            limits=_continuation_limits(
                self.core_limits(mode),
                resumed_run.core_state,
                enabled=intent.kind == "automatic_continuation",
                request_id=waiting.request_id if waiting is not None else "",
            ),
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

    async def _prepare_interaction_resume(
        self,
        intent: SessionContinuationIntent,
        *,
        run_id: str,
        model: ModelDescriptor,
        tools: Any | None,
    ) -> PreparedAgentRun:
        recovery = self.state_service.inspect_recovery(
            RecoveryRequest(
                session_id=self.session_id,
                run_id=run_id,
                expected_waiting_kind="user_input",
            )
        )
        if recovery.bundle is None or recovery.bundle.run.checkpoint is None:
            raise ValueError(f"Run is not recoverable: {run_id}")
        if tools is None:
            raise ValueError("User input recovery requires a Tool port")
        self._restore_tool_checkpoint(tools, recovery.bundle.run)
        waiting = recovery.bundle.run.checkpoint.waiting
        if waiting is None:
            raise ValueError("User input checkpoint has no waiting state")
        payload = dict(waiting.payload)
        response = InteractionResponse(
            interaction_id=str(payload.get("interaction_id") or ""),
            request_fingerprint=str(payload.get("request_fingerprint") or ""),
            session_id=self.session_id,
            tool_call_id=str(payload.get("tool_call_id") or waiting.request_id),
            tool_name=str(payload.get("tool_name") or "request_user_input"),
            registration_id=str(payload.get("registration_id") or ""),
            answers={"answer": intent.text},
        )
        prepared_resume = tools.prepare_resume(response)
        resumed_run = self.state_service.resume_run(
            ResumeRunRequest(
                session_id=self.session_id,
                run_id=run_id,
                checkpoint_id=recovery.bundle.run.checkpoint.checkpoint_id,
                request_id=waiting.request_id,
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
        self._restore_context_checkpoint(recovery.bundle.run)
        self._restore_rollback_checkpoint(recovery.bundle.run)
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
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=CoreRunInput(
                session_id=self.session_id,
                run_id=run_id,
                entry=ToolResultEntry((result,)),
                messages=tuple(messages),
                state=resumed_run.core_state,
                model=model,
                mode=self.current_mode,
                limits=self.core_limits(),
                context_seed=self._loop_context(),
            ),
            context_port=self.context_service,
            state_port=state_port,
            plan_refs={"plan_state": self.active_plan_state()},
            rollback_baseline=self._rollback_baseline_ref(run_id),
        )

    async def _commit_run(
        self,
        prepared: PreparedAgentRun,
        outcome: CoreOutcome,
        result: AgentRunResult,
        *,
        events: tuple[dict[str, object], ...] = (),
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
                    "cancelled": "cancelled",
                }[outcome.status]
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
        for event in events:
            payload = dict(event)
            try:
                await self.conversation.dispatch_event(payload)
            except Exception as exc:
                logger.warning("failed to project committed runtime event: %s", exc)
        committed_messages = list(outcome.new_messages)
        self.conversation.append_messages(committed_messages)
        for message in committed_messages:
            message_id = state_port.committed_message_ids.get(id(message)) if state_port else None
            if message_id is not None:
                _set_session_message_id(message, message_id)
        self.conversation.remember_result(result)

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
        await self._run_post_commit_effects(prepared, outcome, result)
        return record

    async def _run_post_commit_effects(
        self,
        prepared: PreparedAgentRun,
        outcome: CoreOutcome,
        result: AgentRunResult,
    ) -> None:
        """Run non-authoritative effects after the Sessions commit has succeeded."""

        terminal = _is_terminal_outcome(outcome)
        if terminal:
            try:
                self._submit_captured_memory_proposals(result)
            except Exception as exc:
                logger.warning("failed to finalize memory proposals: %s", exc)

        rollback_ref = prepared.rollback_baseline
        if rollback_ref is not None:
            try:
                self._write_rollback_metadata(
                    result,
                    self._rollback_baseline(rollback_ref),
                )
            except Exception as exc:
                logger.warning("failed to write rollback metadata: %s", exc)
            finally:
                if terminal:
                    self._discard_rollback_baseline(rollback_ref)

        self._calibrate_context_usage(result)
        try:
            prompt_text = _prompt_text(prepared)
            await self._run_lifecycle_hooks(
                text=prompt_text,
                is_continue=_is_continue_text(prompt_text),
                hooks=self.after_prompt_hooks,
            )
        except Exception as exc:
            logger.warning("failed to run after-prompt hook: %s", exc)

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
            "current_plan": self.current_plan_state(),
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
            core = load_core_state(run.core_state)
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
        core = load_core_state(run.core_state)
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
                workspace_dir=self.workspace_dir,
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
        normalized_mode = ensure_run_mode(mode or self.current_mode)
        values: dict[str, Any] = {
            "system_prompt": self.conversation.system_prompt,
            "mode_policy": _mode_policy(normalized_mode),
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


def _continuation_limits(
    limits: CoreLimits,
    raw_state: object,
    *,
    enabled: bool,
    request_id: str,
) -> CoreLimits:
    if not enabled:
        return limits
    state = load_core_state(raw_state)
    counters = state.facts.counters
    per_turn = limits.max_tool_calls_per_turn
    if request_id.endswith("run.max_tool_calls_per_turn"):
        per_turn = None
    return replace(
        limits,
        max_model_turns=limits.max_model_turns + counters.model_turns,
        max_tool_iterations=limits.max_tool_iterations + counters.tool_iterations,
        max_tool_calls=(
            None
            if limits.max_tool_calls is None
            else limits.max_tool_calls + counters.tool_calls
        ),
        max_tool_calls_per_turn=per_turn,
        repeated_tool_call_limit=(
            limits.repeated_tool_call_limit
            + state.facts.loop_guards.repeated_tool_calls
        ),
    )


def _continuation_control(kind: str) -> dict[str, object] | None:
    instruction = {
        "plan_approved": (
            "canonical Task Plan 已获批准并处于 active 状态。从第一个未完成步骤继续执行，"
            "不要复述、重新设计或创建第二份计划。保留现有 step_id；仅用 "
            "update_plan_progress 更新进度。最终答复前确保所有步骤均已标记 completed；"
            "close_plan 仅为兼容接口，不是完成任务的必要条件。"
        ),
        "plan_rejected": (
            "用户拒绝了 proposed plan，但尚未给出修改方向。只询问一个会实质影响方案的具体问题；"
            "收到反馈前不要创建替代计划。"
        ),
        "plan_feedback": (
            "最新用户消息是对当前 proposed plan 的修改意见。保留未被否定的内容，提交完整修订版；"
            "本轮必须成功调用 propose_plan，不能只输出文字方案。"
        ),
        "plan_clarification": (
            "把用户补充信息合并到当前规划。证据充分后在本轮调用 propose_plan 提交可执行方案。"
        ),
        "mode_changed": "保持同一个工程目标，严格按当前 mode policy 继续。",
        "automatic_continuation": "从 Runtime 保存的 checkpoint 继续同一任务，不重复已经提交的动作。",
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
            "当前 mode=plan。只允许只读调查、澄清和方案设计，禁止修改工作区、执行实现或自行切换模式。"
            "先识别用户真正的软件目标、约束和阻塞性歧义，再读取足以支撑设计的仓库事实。"
            "已知文件或局部问题直接用 read/grep/find；只有调查开放、跨模块或可拆成独立问题时才调用 dispatch_exploration，"
            "任务范围必须互不重复并优先 reuse=auto。Subagent 只提供证据，最终设计由主 Agent 综合。"
            "仅当缺失信息会实质改变实现范围或架构时，才用 request_user_input 提出一个具体问题。"
            "证据充分后必须在当前回合调用 propose_plan；普通文本方案不是提交。计划需覆盖当前实现、目标设计、影响范围、"
            "风险、完成标准和验证方式，items 只写批准后要执行的实现或验证工作。"
            "propose_plan 成功只表示等待用户审查，不表示已获批准；提交后不得开始实现。"
        )
    if mode == "read":
        return (
            "当前 mode=read。只做只读探索、定位、解释、审查和状态说明；不得修改工作区、运行有副作用的命令，"
            "也不得创建、推进或关闭 Task Plan。用 read/grep/find 获取必要证据，区分已观察事实、合理推断和建议，"
            "然后直接回答当前问题。"
        )
    return (
        "当前 mode=build。可以在权限范围内读取、修改、运行项目命令并验证。修改前读取相关实现，优先使用专用文件工具，"
        "command 用于 argv 形式的项目命令，只有需要 Shell 语法时才用 bash。"
        "存在 active canonical Task Plan 时，直接执行第一个未完成步骤，不得创建第二份计划；进度变化用 "
        "update_plan_progress 提交并保留现有 step_id。没有计划且任务确实复杂时才用 "
        "create_build_plan，简单任务直接完成。"
        "若新证据使原计划基础失效，按工具协议提交 revision，不能在文本中悄悄改变范围。"
        "完成实现后运行与风险相称的验证；存在计划时，最终答复前确保所有步骤都已通过 "
        "update_plan_progress 标记 completed。Task 与 Plan 是否完成由 Core 根据结构化状态判定。"
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
