from __future__ import annotations

"""Readable task controller used by the agent loop.

The controller does not try to be a planner, verifier, recovery engine, or
rollback system.  It only keeps a short task checklist aligned with tool
evidence so the loop can decide whether to continue, wait, or stop.
"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import cast
from uuid import uuid4

from codepilot.protocols import Message, TaskSummary, TextContent, ToolResultMessage, UserMessage

from ..state import RunState
from .contracts import (
    TaskMode,
    TaskPlanningState,
    ensure_task_mode,
    task_planning_state_from_mapping,
)
from .modes import policy_for_mode
from .planner import PlannedTaskStep
from .state import (
    CompletionCheck,
    ExecutionDecision,
    TASK_STEP_KINDS,
    TaskState,
    TaskStep,
    TaskStepKind,
    TaskStepStatus,
)
from .tools import COMPLETE_TASK_STEP_TOOL, TASK_UPDATE_TOOL


AgentMessage = Message

_MAX_STEPS = 6
_MAX_STEP_TITLE_CHARS = 100
class TaskController:
    def initialize(
        self,
        prompts: Iterable[AgentMessage],
        *,
        proposed_steps: Iterable[object] | None = None,
        goal: str | None = None,
        acceptance_criteria: Iterable[str] | None = None,
        constraints: Iterable[str] | None = None,
        mode: TaskMode | str = "build",
        planning: TaskPlanningState | Mapping[str, object] | None = None,
        max_replans_per_run: int | None = None,
        task_state: Mapping[str, object] | None = None,
    ) -> TaskState:
        del acceptance_criteria, constraints, max_replans_per_run

        selected_mode = ensure_task_mode(mode)
        planning_state = _planning_state(planning, selected_mode)
        recovered = (
            build_task_state_from_payload(prompts, task_state, mode=selected_mode)
            if task_state is not None
            else None
        )
        if recovered is not None:
            recovered.planning = planning_state if planning is not None else recovered.planning
            return recovered

        task_goal = _compact(goal, limit=1200) or _goal_from_prompts(prompts)
        raw_steps = list(proposed_steps or ())
        if not raw_steps:
            raw_steps = [policy_for_mode(selected_mode).default_step_title]
        steps = _normalize_steps(raw_steps)
        task = TaskState(
            task_id=f"task_{uuid4().hex[:12]}",
            goal=task_goal or "完成当前请求",
            steps=steps,
            mode=selected_mode,
            planning=planning_state,
        )
        task.advance()
        return task

    def after_tool_results(
        self,
        task: TaskState,
        run: RunState,
        results: list[ToolResultMessage],
    ) -> ExecutionDecision:
        if not results:
            return self._decision(task, "continue", "no_tool_results")

        refs = evidence_refs(results)
        step = task.current_step() or task.advance()

        if any(result.status == "approval_required" for result in results):
            if step is not None:
                step.block("等待工具审批", evidence_refs=refs)
            task.phase = "waiting"
            task.recent_error_code = "approval_required"
            return self._decision(task, "wait_approval", "approval_required")

        if any(result.status == "cancelled" for result in results):
            if step is not None:
                step.block("工具执行已取消", evidence_refs=refs)
            task.phase = "waiting"
            task.recent_error_code = "cancelled"
            return self._decision(task, "stop", "cancelled")

        if any(result.status == "denied" for result in results):
            if step is not None:
                step.block("工具执行被拒绝", evidence_refs=refs)
            task.phase = "waiting"
            task.recent_error_code = "permission_denied"
            return self._decision(task, "stop", "permission_denied")

        if self._apply_task_control_signal(task, results):
            return self._continue_after_tool(task, "task_control")

        if any(is_tool_unavailable(result) for result in results):
            if step is not None:
                step.block("工具不可用", evidence_refs=refs)
            task.phase = "waiting"
            task.recent_error_code = "tool_not_found"
            return self._decision(task, "stop", "tool_unavailable")

        if has_failed_verification(results):
            if step is not None:
                step.record_failure(verification_failure_note(results), evidence_refs=refs)
            task.phase = "acting"
            task.next_action = repair_next_action(results)
            task.recent_error_code = "verification_failed"
            return self._decision(task, "continue", "verification_failed")

        if has_non_verification_error(results):
            if step is not None:
                step.record_failure("工具执行失败，等待模型根据错误继续调整", evidence_refs=refs)
            task.phase = "acting"
            task.recent_error_code = first_error_code(results) or "tool_error"
            return self._decision(task, "continue", "tool_error")

        if has_passed_verification(results):
            self._complete_current_and_verification_step(task, refs)
            task.recent_error_code = None
            return self._continue_after_tool(task, "verification_passed")

        if step is not None:
            step.add_evidence(refs)

        if any(result.workspace_changed for result in results):
            task.phase = "verifying"
            task.next_action = "运行最相关的测试或检查"
            task.recent_error_code = None
            return self._decision(task, "continue", "workspace_changed")

        if step is not None and _step_can_finish_after_read(task, step, results):
            step.complete(summary="已完成只读分析", evidence_refs=refs)
            task.advance()

        task.phase = "acting" if task.current_step_id else "finished"
        task.recent_error_code = None
        return self._continue_after_tool(task, "tool_results")

    def check_completion(self, task: TaskState, run: RunState) -> CompletionCheck:
        blocked = task.blocked_step_titles()
        if blocked:
            return self._record_completion(
                task,
                CompletionCheck(
                    satisfied=False,
                    reason="blocked_steps",
                    missing=blocked,
                    can_continue=False,
                ),
            )

        if run.workspace_changed and not run.fresh_verification_passed:
            can_continue = task.completion_prompt_count == 0
            if can_continue:
                task.completion_prompt_count += 1
            return self._record_completion(
                task,
                CompletionCheck(
                    satisfied=False,
                    reason="modified_without_fresh_verification",
                    missing=["fresh_verification"],
                    can_continue=can_continue,
                ),
            )

        open_steps = task.open_steps()
        if open_steps and task.recent_error_code is None:
            step = task.current_step() or task.advance()
            if step is not None:
                step.complete(
                    summary="模型已给出最终答复",
                    evidence_refs=["model:final_answer"],
                )
                task.advance()
            open_steps = task.open_steps()

        if open_steps:
            return self._record_completion(
                task,
                CompletionCheck(
                    satisfied=False,
                    reason="incomplete_steps",
                    missing=[step.title for step in open_steps],
                    can_continue=False,
                ),
            )

        task.mark_finished()
        return self._record_completion(
            task,
            CompletionCheck(satisfied=True, reason="all_steps_completed"),
        )

    def completion_steering(self, check: CompletionCheck) -> UserMessage:
        text = (
            "工作区已经发生修改，但还没有针对最新状态的成功验证。"
            "请运行最相关的测试或检查；如果当前环境无法验证，请说明原因和剩余风险。"
        )
        return UserMessage(
            content=[TextContent(text=text)],
            metadata={
                "task_completion_gate": {
                    "reason": check.reason,
                    "missing": list(check.missing),
                }
            },
        )

    def render_context(self, task: TaskState) -> str:
        lines = [
            "## Current Task",
            f"Goal: {task.goal}",
            f"Mode: {task.mode}",
            f"Phase: {task.phase}",
            "",
            "Steps:",
        ]
        for step in task.steps:
            note = f" note={step.note}" if step.note else ""
            lines.append(f"- [{step.status}] {step.title}{note}")
            if step.acceptance:
                lines.append(f"  acceptance: {step.acceptance}")
            if step.verification_hint:
                lines.append(f"  verify: {step.verification_hint}")
        if task.next_action:
            lines.extend(["", f"Next action: {task.next_action}"])
        return "\n".join(lines)

    def summarize(self, task: TaskState) -> TaskSummary:
        return TaskSummary(
            task_id=task.task_id,
            goal=task.goal,
            completed_steps=task.completed_step_titles(),
            pending_steps=task.pending_step_titles(),
            blocked_steps=task.blocked_step_titles(),
            next_action=task.next_action,
            completion_satisfied=task.completion_satisfied,
            completion_reason=task.completion_reason,
            attempts=[],
            change_sets=[],
            replans=[],
            control_signal=self.control_signal(task),
            step_details={
                step.title: {
                    "id": step.id,
                    "kind": step.kind,
                    "status": step.status,
                    "acceptance": step.acceptance,
                    "verification_hint": step.verification_hint,
                    "summary": step.summary,
                    "evidence_refs": list(step.evidence_refs),
                    "failure_count": step.failure_count,
                }
                for step in task.steps
            },
        )

    def event_payload(self, task: TaskState) -> dict[str, object]:
        return {
            "task_id": task.task_id,
            "goal": task.goal,
            "mode": task.mode,
            "phase": task.phase,
            "current_step_id": task.current_step_id,
            "next_action": task.next_action,
            "completion_satisfied": task.completion_satisfied,
            "completion_reason": task.completion_reason,
            "steps": [asdict(step) for step in task.steps],
            "planning": task.planning.to_signal(),
        }

    def control_signal(self, task: TaskState) -> dict[str, object]:
        current = task.current_step()
        return {
            "task_id": task.task_id,
            "mode": task.mode,
            "planning": task.planning.to_signal(),
            "phase": task.phase,
            "current_step_id": current.id if current else None,
            "current_step_title": current.title if current else None,
            "current_step_acceptance": current.acceptance if current else None,
            "current_step_verification_hint": current.verification_hint if current else None,
            "next_action": task.next_action,
            "recent_error_code": task.recent_error_code,
            "last_decision": task.last_decision,
            "completion_satisfied": task.completion_satisfied,
            "completion_reason": task.completion_reason,
        }

    def _apply_task_control_signal(
        self,
        task: TaskState,
        results: list[ToolResultMessage],
    ) -> bool:
        for result in results:
            payload = complete_step_payload(result)
            if payload is not None:
                step = task.current_step() or task.advance()
                if step is None:
                    return True
                refs = _payload_refs(payload)
                if not refs:
                    step.note = "task_control rejected: missing_evidence"
                    task.recent_error_code = "task_control_rejected"
                    return True
                step.complete(
                    summary=_compact(payload.get("summary"), limit=240) or "步骤已完成",
                    evidence_refs=refs,
                )
                task.advance()
                task.recent_error_code = None
                return True

            payload = task_update_payload(result)
            if payload is None:
                continue
            step = _step_by_id(task, _compact(payload.get("step_id"), limit=80)) or task.current_step()
            if step is None:
                return True
            refs = _payload_refs(payload)
            status = payload.get("proposed_status")
            summary = _compact(payload.get("summary"), limit=240)
            if status == "completed":
                if not refs:
                    step.note = "task_control rejected: missing_evidence"
                    task.recent_error_code = "task_control_rejected"
                    return True
                step.complete(summary=summary or "步骤已完成", evidence_refs=refs)
                task.advance()
            elif status == "blocked":
                step.block(summary or "步骤被标记为阻塞", evidence_refs=refs)
                task.phase = "waiting"
                task.recent_error_code = "task_step_blocked"
            else:
                step.start()
                step.note = summary or None
                step.add_evidence(refs)
            return True
        return False

    def _complete_current_and_verification_step(
        self,
        task: TaskState,
        refs: list[str],
    ) -> None:
        step = task.current_step() or task.advance()
        if step is None:
            task.mark_finished()
            return
        step.complete(summary="验证通过", evidence_refs=refs)
        next_step = task.advance()
        if next_step is not None and is_verification_step(next_step):
            next_step.complete(summary="验证通过", evidence_refs=refs)
            task.advance()

    def _continue_after_tool(self, task: TaskState, reason: str) -> ExecutionDecision:
        return self._decision(task, "continue", reason)

    def _record_completion(
        self,
        task: TaskState,
        check: CompletionCheck,
    ) -> CompletionCheck:
        task.completion_satisfied = check.satisfied
        task.completion_reason = check.reason
        if check.satisfied:
            task.mark_finished()
        return check

    def _decision(
        self,
        task: TaskState,
        action: str,
        reason: str,
    ) -> ExecutionDecision:
        task.last_decision = action
        return ExecutionDecision(action=action, reason=reason, next_action=task.next_action)  # type: ignore[arg-type]


def build_task_state_from_payload(
    prompts: Iterable[AgentMessage],
    task_state: Mapping[str, object],
    *,
    mode: TaskMode | str | None = None,
) -> TaskState | None:
    raw_steps = _raw_steps_from_task_state(task_state)
    if not raw_steps:
        return None

    selected_mode = _safe_task_mode(task_state.get("current_mode") or mode or "build")
    steps = _normalize_steps(raw_steps)
    if not steps:
        return None

    current_step_id = _compact(task_state.get("current_step_id"), limit=80)
    if current_step_id and not any(step.id == current_step_id for step in steps):
        current_step_id = ""
    task = TaskState(
        task_id=_compact(task_state.get("task_id"), limit=120) or f"task_{uuid4().hex[:12]}",
        goal=_goal_from_task_state(task_state, prompts),
        steps=steps,
        mode=selected_mode,
        planning=(
            task_planning_state_from_mapping(task_state.get("planning"))
            if isinstance(task_state.get("planning"), Mapping)
            else TaskPlanningState(phase="recovered", source="recovered")
        ),
        current_step_id=current_step_id or None,
        phase="acting",
        next_action=_compact(task_state.get("next_action"), limit=240) or None,
        completion_reason=_compact(task_state.get("blocked_reason"), limit=120),
    )
    current = task.current_step()
    if current is None:
        task.advance()
    elif current.status == "pending":
        current.start()
        task.next_action = current.title
    if task.blocked_steps():
        task.phase = "waiting"
    return task


def evidence_refs(results: list[ToolResultMessage]) -> list[str]:
    refs: list[str] = []
    for result in results:
        if result.tool_call_id:
            refs.append(f"tool:{result.tool_call_id}")
        if result.approval_id:
            refs.append(f"approval:{result.approval_id}")
        if result.verification and result.tool_call_id:
            refs.append(f"verification:{result.tool_call_id}")
        refs.extend(f"file:{path}" for path in result.affected_paths)
    return list(dict.fromkeys(refs))


def first_error_code(results: list[ToolResultMessage]) -> str | None:
    return next((result.error_code for result in results if result.error_code), None)


def has_failed_verification(results: list[ToolResultMessage]) -> bool:
    return any(_verification_status(result) == "failed" for result in results)


def has_passed_verification(results: list[ToolResultMessage]) -> bool:
    return any(_verification_status(result) == "passed" for result in results)


def has_non_verification_error(results: list[ToolResultMessage]) -> bool:
    return any(
        (result.status != "success" or result.is_error)
        and _verification_status(result) is None
        for result in results
    )


def is_tool_unavailable(result: ToolResultMessage) -> bool:
    if result.error_code == "tool_not_found":
        return True
    if result.status == "success" and not result.is_error:
        return False
    text = " ".join(
        block.text for block in result.content if isinstance(block, TextContent)
    ).lower()
    return text.startswith("tool ") and " not found" in text


def is_verification_step(step: TaskStep) -> bool:
    text = f"{step.kind} {step.title} {step.verification_hint or ''}".lower()
    return any(marker in text for marker in ("verify", "test", "pytest", "验证", "测试", "检查"))


def verification_failure_note(results: list[ToolResultMessage]) -> str:
    detail = verification_failure_detail(results)
    return f"验证失败，需要修复：{detail}" if detail else "验证失败，需要修复"


def verification_failure_detail(results: list[ToolResultMessage]) -> str:
    for result in results:
        verification = result.verification
        if not isinstance(verification, Mapping) or verification.get("status") != "failed":
            continue
        parts = [
            text
            for text in (
                _compact(verification.get("command"), limit=160),
                _compact(verification.get("summary"), limit=220),
            )
            if text
        ]
        exit_code = verification.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            parts.append(f"exit_code={exit_code}")
        return "；".join(parts)
    return ""


def repair_next_action(results: list[ToolResultMessage]) -> str:
    detail = verification_failure_detail(results)
    if detail:
        return f"根据验证失败证据修复实现：{detail}"
    return "根据验证失败证据定位根因并完成最小修复"


def complete_step_payload(result: ToolResultMessage) -> Mapping[str, object] | None:
    return _task_control_payload(result, tool_name=COMPLETE_TASK_STEP_TOOL, action="complete_step")


def task_update_payload(result: ToolResultMessage) -> Mapping[str, object] | None:
    return _task_control_payload(result, tool_name=TASK_UPDATE_TOOL, action="update_step")


def _task_control_payload(
    result: ToolResultMessage,
    *,
    tool_name: str,
    action: str,
) -> Mapping[str, object] | None:
    if result.tool_name != tool_name:
        return None
    payload = result.metadata.get("task_control")
    if not isinstance(payload, Mapping):
        return None
    if payload.get("action") != action or payload.get("valid") is False:
        return None
    return payload


def _normalize_steps(raw_steps: Iterable[object]) -> list[TaskStep]:
    steps: list[TaskStep] = []
    seen: set[str] = set()
    for raw in raw_steps:
        title, kind, acceptance, verification_hint, status, step_id = _step_fields(raw, len(steps))
        if not title or title in seen:
            continue
        seen.add(title)
        steps.append(
            TaskStep(
                id=step_id or f"step_{len(steps) + 1}",
                title=title,
                kind=kind,
                status=status,
                acceptance=acceptance,
                verification_hint=verification_hint,
            )
        )
        if len(steps) >= _MAX_STEPS:
            break
    if not steps:
        steps.append(TaskStep(id="step_1", title="完成当前请求"))
    return steps


def _step_fields(
    raw: object,
    index: int,
) -> tuple[str, TaskStepKind, str | None, str | None, TaskStepStatus, str | None]:
    if isinstance(raw, PlannedTaskStep):
        return (
            raw.title,
            raw.kind,
            raw.acceptance,
            raw.verification_hint,
            "pending",
            None,
        )
    if isinstance(raw, Mapping):
        return (
            _compact(raw.get("title") or raw.get("description"), limit=_MAX_STEP_TITLE_CHARS),
            _step_kind(raw.get("kind")),
            _optional_text(raw.get("acceptance")),
            _optional_text(raw.get("verification_hint")),
            _step_status(raw.get("status")),
            _compact(raw.get("id"), limit=80) or f"step_{index + 1}",
        )
    return (
        _compact(raw, limit=_MAX_STEP_TITLE_CHARS),
        _infer_step_kind(raw),
        None,
        None,
        "pending",
        None,
    )


def _raw_steps_from_task_state(task_state: Mapping[str, object]) -> list[object]:
    raw_steps = task_state.get("steps")
    if isinstance(raw_steps, list) and raw_steps:
        return raw_steps
    for key in ("approved_plan", "proposed_plan"):
        plan = task_state.get(key)
        if isinstance(plan, Mapping) and isinstance(plan.get("steps"), list):
            return list(plan["steps"])
    return []


def _goal_from_prompts(prompts: Iterable[AgentMessage]) -> str:
    for message in reversed(list(prompts)):
        if not isinstance(message, UserMessage):
            continue
        if isinstance(message.content, str):
            return _compact(message.content, limit=1200) or "完成当前请求"
        text = "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        )
        return _compact(text, limit=1200) or "完成当前请求"
    return "完成当前请求"


def _goal_from_task_state(
    task_state: Mapping[str, object],
    prompts: Iterable[AgentMessage],
) -> str:
    raw_goal = task_state.get("goal")
    if isinstance(raw_goal, Mapping):
        raw_goal = raw_goal.get("value")
    return (
        _compact(raw_goal, limit=1200)
        or _compact(task_state.get("raw_user_request"), limit=1200)
        or _goal_from_prompts(prompts)
    )


def _planning_state(
    planning: TaskPlanningState | Mapping[str, object] | None,
    mode: TaskMode,
) -> TaskPlanningState:
    if isinstance(planning, TaskPlanningState):
        return planning
    if isinstance(planning, Mapping):
        return task_planning_state_from_mapping(planning)
    return TaskPlanningState(phase="execution" if mode == "plan" else "none", source="default")


def _step_by_id(task: TaskState, step_id: str) -> TaskStep | None:
    return next((step for step in task.steps if step.id == step_id), None)


def _payload_refs(payload: Mapping[str, object]) -> list[str]:
    raw_refs = payload.get("evidence_refs")
    if not isinstance(raw_refs, list | tuple):
        return []
    return [text for item in raw_refs if (text := _compact(item, limit=160))]


def _step_can_finish_after_read(
    task: TaskState,
    step: TaskStep,
    results: list[ToolResultMessage],
) -> bool:
    if task.mode in {"read", "plan"}:
        return True
    if step.kind in {"read", "plan", "summarize"}:
        return True
    return False


def _verification_status(result: ToolResultMessage) -> str | None:
    verification = result.verification
    if not isinstance(verification, Mapping):
        return None
    status = verification.get("status")
    return status if status in {"passed", "failed", "cancelled", "unknown"} else None


def _infer_step_kind(value: object) -> TaskStepKind:
    text = _compact(value, limit=160).lower()
    if any(token in text for token in ("pytest", "test", "验证", "测试", "检查")):
        return "verify"
    if any(token in text for token in ("修改", "修复", "实现", "edit", "fix")):
        return "edit"
    if any(token in text for token in ("阅读", "分析", "查找", "inspect", "read")):
        return "read"
    if any(token in text for token in ("计划", "plan")):
        return "plan"
    if any(token in text for token in ("总结", "summarize")):
        return "summarize"
    return "other"


def _step_kind(value: object) -> TaskStepKind:
    text = _compact(value, limit=40)
    return cast(TaskStepKind, text) if text in TASK_STEP_KINDS else "other"


def _step_status(value: object) -> TaskStepStatus:
    text = _compact(value, limit=40)
    if text in {"pending", "in_progress", "completed", "blocked"}:
        return cast(TaskStepStatus, text)
    return "pending"


def _safe_task_mode(value: object) -> TaskMode:
    try:
        return ensure_task_mode(value)
    except ValueError:
        return "build"


def _optional_text(value: object) -> str | None:
    return _compact(value, limit=240) or None


def _compact(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


__all__ = [
    "TaskController",
    "build_task_state_from_payload",
    "complete_step_payload",
    "evidence_refs",
    "first_error_code",
    "has_failed_verification",
    "has_non_verification_error",
    "has_passed_verification",
    "is_tool_unavailable",
    "is_verification_step",
    "repair_next_action",
    "task_update_payload",
    "verification_failure_note",
]
