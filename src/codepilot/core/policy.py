"""根据 Core 状态和运行限制决定下一步模型、工具、等待或终止动作。"""

from __future__ import annotations

from dataclasses import dataclass, field

from codepilot.protocols import TextContent, ToolCall

from .contracts import (
    CallModel,
    CoreDecision,
    CoreDirective,
    CoreLimits,
    CoreReason,
    CoreWait,
    ExecuteTools,
    ModelPurpose,
    Terminate,
    Wait,
)
from .observations import (
    CancellationObservation,
    CoreObservation,
    ModelObservation,
    ToolBatchObservation,
    UserInputObservation,
)
from .plan import RunMode, ensure_run_mode
from .state import CoreState, assess_core_state


@dataclass(frozen=True)
class PolicyContext:
    """供策略判断当前状态的只读上下文。"""
    mode: RunMode
    limits: CoreLimits = field(default_factory=CoreLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ensure_run_mode(self.mode))
        if not isinstance(self.limits, CoreLimits):
            raise TypeError("limits must be CoreLimits")


class CorePolicy:
    """根据状态和限制选择下一步 Core 指令。"""
    @staticmethod
    def decide(
        state: CoreState,
        observation: CoreObservation,
        context: PolicyContext,
    ) -> CoreDecision:
        """根据当前状态、观察和限制选择下一步指令。"""
        if state.task.status == "satisfied":
            return Terminate("completed", "task.completed")
        if state.task.status == "abandoned":
            return Terminate("cancelled", "run.cancelled")

        fatal = state.facts.failures.latest
        if (
            fatal is not None
            and not fatal.recoverable
            and fatal.code != "run.cancelled"
        ):
            return Terminate(
                "failed",
                fatal.code,
                fatal.message,
                fatal.evidence_refs,
            )

        external_wait = _external_wait(observation)
        if external_wait is not None:
            return external_wait

        if isinstance(observation, CancellationObservation) or _has_tool_status(
            observation, "cancelled"
        ):
            return Terminate("cancelled", "run.cancelled")

        if context.mode == "read" and state.facts.workspace.changed:
            return Terminate("failed", "read.workspace_changed")

        plan = state.task.plan
        if plan is not None and plan.status == "proposed":
            if (
                isinstance(observation, UserInputObservation)
                and observation.text != state.task.current_goal
            ):
                return _call_model("replan", "plan.feedback_received")
            return Wait(
                CoreWait(
                    "plan_confirmation",
                    plan.plan_id,
                    CoreReason("plan.confirmation_required", recoverable=True),
                    {"plan_id": plan.plan_id, "revision": plan.revision},
                )
            )
        if (
            plan is not None
            and plan.status == "active"
            and plan.pending_revision is not None
        ):
            return Wait(
                CoreWait(
                    "plan_confirmation",
                    plan.plan_id,
                    CoreReason("plan.revision_confirmation_required", recoverable=True),
                    {
                        "plan_id": plan.plan_id,
                        "revision": plan.revision,
                        "confirmation": "revision",
                    },
                )
            )

        budget_wait = _budget_wait(state, observation, context)
        if budget_wait is not None:
            return budget_wait

        if isinstance(observation, ModelObservation):
            calls = _tool_calls(observation)
            if calls:
                return ExecuteTools(
                    calls,
                    CoreReason("model.requested_tools", recoverable=True),
                )

        latest_failure = state.facts.failures.latest
        if (
            latest_failure is not None
            and latest_failure.recoverable
            and _needs_recovery(state, observation)
        ):
            if any(item.kind == "replan_required" for item in state.task.blockers):
                return _call_model(
                    "replan",
                    "replan.required",
                    latest_failure.evidence_refs,
                )
            if (
                state.facts.failures.count_for(latest_failure.code)
                <= context.limits.max_recovery_attempts
            ):
                return _call_model(
                    "recovery",
                    "recovery.required",
                    latest_failure.evidence_refs,
                )
            return Terminate(
                "failed",
                "core.recovery_exhausted",
                evidence_refs=latest_failure.evidence_refs,
            )

        if any(item.kind == "replan_required" for item in state.task.blockers):
            evidence = tuple(
                ref
                for item in state.task.blockers
                if item.kind == "replan_required"
                for ref in item.evidence_refs
            )
            return _call_model("replan", "replan.required", evidence)

        assessment = assess_core_state(state)
        if assessment.status == "needs_verification":
            return _call_model(
                "verification",
                "verification.required",
                state.facts.workspace.evidence_refs,
            )

        if plan is not None and plan.status == "active":
            all_steps_completed = all(
                step.status == "completed" for step in plan.steps
            )
            if plan.close_request is not None and all_steps_completed:
                if state.task.blockers:
                    return _call_model(
                        "recovery",
                        "task.blocked",
                        tuple(
                            ref
                            for item in state.task.blockers
                            for ref in item.evidence_refs
                        ),
                    )
                if _is_final_response_candidate(observation):
                    return Terminate("completed", "plan.completed")
                return _call_model("final_response", "final_response.required")
            if plan.close_request is not None or all_steps_completed:
                return _call_model("plan_closeout", "plan.closeout_required")
            if _is_final_candidate(observation):
                return _call_model("reasoning", "plan.progress_required")

        if _is_final_candidate(observation):
            if context.mode == "read" and state.facts.workspace.changed:
                return Terminate("failed", "read.workspace_changed")
            if state.task.blockers:
                return _call_model(
                    "recovery",
                    "task.blocked",
                    tuple(
                        ref
                        for item in state.task.blockers
                        for ref in item.evidence_refs
                    ),
                )
            return Terminate("completed", "task.completed")

        return _call_model("reasoning", "reasoning.continue")


def _external_wait(observation: CoreObservation) -> Wait | None:
    if not isinstance(observation, ToolBatchObservation):
        return None
    for result in observation.results:
        if result.error is not None and result.error.code == "tool.recovery.ambiguous":
            return Wait(
                CoreWait(
                    "user_input",
                    f"recovery:{result.tool_call_id}",
                    CoreReason(
                        "tool.recovery.ambiguous",
                        result.error.message,
                        source="tools",
                        recoverable=True,
                        evidence_refs=(result.tool_call_id,),
                    ),
                    {
                        "tool_call_id": result.tool_call_id,
                        "tool_name": result.tool_name,
                    },
                )
            )
        if result.status == "approval_required" and result.approval is not None:
            approval = result.approval
            return Wait(
                CoreWait(
                    "tool_approval",
                    approval.approval_id,
                    CoreReason(
                        "tool.approval_required",
                        approval.reason,
                        source="tools",
                        recoverable=True,
                        evidence_refs=(result.tool_call_id,),
                    ),
                    {
                        "tool_call_id": result.tool_call_id,
                        "tool_name": result.tool_name,
                        "risk": approval.risk,
                    },
                )
            )
        if result.status == "user_input_required" and result.interaction is not None:
            request_id = str(
                result.interaction.get("request_id") or result.tool_call_id
            )
            return Wait(
                CoreWait(
                    "user_input",
                    request_id,
                    CoreReason(
                        "tool.user_input_required",
                        source="tools",
                        recoverable=True,
                        evidence_refs=(result.tool_call_id,),
                    ),
                    dict(result.interaction),
                )
            )
    return None


def _budget_wait(
    state: CoreState,
    observation: CoreObservation,
    context: PolicyContext,
) -> Wait | None:
    counters = state.facts.counters
    reason = None
    calls = (
        _tool_calls(observation) if isinstance(observation, ModelObservation) else ()
    )
    final_candidate = _is_final_candidate(observation) and not state.task.blockers
    if (
        context.limits.max_tool_calls_per_turn is not None
        and len(calls) > context.limits.max_tool_calls_per_turn
    ):
        reason = "run.max_tool_calls_per_turn"
    elif counters.model_turns > context.limits.max_model_turns or (
        counters.model_turns >= context.limits.max_model_turns
        and isinstance(observation, (ModelObservation, ToolBatchObservation))
        and not calls
        and not final_candidate
    ):
        reason = "run.max_model_turns"
    elif counters.tool_iterations > context.limits.max_tool_iterations:
        reason = "run.max_tool_iterations"
    elif (
        context.limits.max_tool_calls is not None
        and counters.tool_calls > context.limits.max_tool_calls
    ):
        reason = "run.max_tool_calls"
    elif (
        state.facts.loop_guards.repeated_tool_calls
        > context.limits.repeated_tool_call_limit
    ):
        reason = "run.repeated_tool_call"
    if reason is None:
        return None
    return Wait(
        CoreWait(
            "continuation",
            f"continuation:{reason}",
            CoreReason(reason, recoverable=True),
        )
    )


def _tool_calls(observation: ModelObservation) -> tuple[ToolCall, ...]:
    if observation.message is None:
        return ()
    return tuple(
        block for block in observation.message.content if isinstance(block, ToolCall)
    )


def _is_final_candidate(observation: CoreObservation) -> bool:
    if (
        not isinstance(observation, ModelObservation)
        or observation.status != "completed"
    ):
        return False
    if observation.message is None or _tool_calls(observation):
        return False
    return any(
        isinstance(block, TextContent) and bool(block.text.strip())
        for block in observation.message.content
    )


def _is_final_response_candidate(observation: CoreObservation) -> bool:
    return (
        isinstance(observation, ModelObservation)
        and observation.purpose == "final_response"
        and _is_final_candidate(observation)
    )


def _has_tool_status(observation: CoreObservation, status: str) -> bool:
    return isinstance(observation, ToolBatchObservation) and any(
        result.status == status for result in observation.results
    )


def _needs_recovery(state: CoreState, observation: CoreObservation) -> bool:
    if any(
        item.kind in {"verification_failed", "tool_unavailable", "replan_required"}
        for item in state.task.blockers
    ):
        return True
    if isinstance(observation, ModelObservation):
        if observation.status == "failed" or observation.message is None:
            return True
        return not _tool_calls(observation) and not any(
            isinstance(block, TextContent) and bool(block.text.strip())
            for block in observation.message.content
        )
    if isinstance(observation, ToolBatchObservation):
        return any(
            result.error is not None
            or (
                hasattr(result.data.get("verification"), "get")
                and result.data["verification"].get("status") == "failed"
            )
            for result in observation.results
        )
    return False


def _call_model(
    purpose: ModelPurpose,
    reason_code: str,
    evidence_refs: tuple[str, ...] = (),
) -> CallModel:
    return CallModel(
        purpose=purpose,
        directive=CoreDirective(
            code=f"core.{purpose}",
            evidence_refs=evidence_refs,
        ),
        reason=CoreReason(
            reason_code,
            recoverable=True,
            evidence_refs=evidence_refs,
        ),
    )


__all__ = [
    "CorePolicy",
    "PolicyContext",
]
