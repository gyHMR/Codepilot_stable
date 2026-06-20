from __future__ import annotations

"""Deterministic task feedback controller used by AgentLoop.

The controller keeps the first version intentionally small: the model still
decides semantic actions, while this module binds task progress to runtime
evidence such as tool results, file changes, verification, and approvals.
"""

import uuid
from dataclasses import asdict
from typing import Iterable, Mapping

from codepilot.protocols import TaskSummary, TextContent, ToolResultMessage, UserMessage

from .run_state import RunState
from .task_state import CompletionCheck, ExecutionDecision, TaskState, TaskStep
from .types import AgentMessage


_MAX_STEPS = 6
_MAX_STEP_TITLE_CHARS = 80
_READ_TOOL_MARKERS = ("read", "grep", "find", "glob", "ls", "search", "status", "codegraph")
_WRITE_TOOL_MARKERS = ("write", "edit", "patch", "apply")


class TaskController:
    """Maintain a lightweight, evidence-bound task state for one run."""

    def initialize(
        self,
        prompts: Iterable[AgentMessage],
        *,
        proposed_steps: Iterable[str] | None = None,
        acceptance_criteria: Iterable[str] | None = None,
        constraints: Iterable[str] | None = None,
        recovered_task: Mapping[str, object] | None = None,
    ) -> TaskState:
        if recovered_task is not None:
            recovered = self._from_recovered_task(prompts, recovered_task)
            if recovered is not None:
                return recovered
        goal = _goal_from_prompts(prompts)
        steps = self._normalize_steps(proposed_steps or ["完成当前请求"])
        task = TaskState(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            goal=goal,
            constraints=[item.strip() for item in constraints or [] if item.strip()],
            acceptance_criteria=[
                item.strip() for item in acceptance_criteria or [] if item.strip()
            ],
            steps=steps,
            current_step_id=steps[0].id if steps else None,
            phase="acting",
            next_action=steps[0].title if steps else None,
        )
        if steps:
            steps[0].status = "in_progress"
        return task

    def _from_recovered_task(
        self,
        prompts: Iterable[AgentMessage],
        recovered_task: Mapping[str, object],
    ) -> TaskState | None:
        progress = recovered_task.get("task_progress")
        if not isinstance(progress, Mapping):
            return None
        goal = str(recovered_task.get("goal") or _goal_from_prompts(prompts)).strip()
        if not goal:
            goal = _goal_from_prompts(prompts)
        steps: list[TaskStep] = []
        seen: set[str] = set()

        def add_steps(raw: object, status: str) -> None:
            if not isinstance(raw, list):
                return
            for value in raw:
                title = " ".join(str(value).strip().split())
                if not title or title in seen:
                    continue
                seen.add(title)
                steps.append(
                    TaskStep(
                        id=f"step_{len(steps) + 1}",
                        title=title[:_MAX_STEP_TITLE_CHARS],
                        status=status,  # type: ignore[arg-type]
                    )
                )
                if len(steps) >= _MAX_STEPS:
                    return

        add_steps(progress.get("completed_steps"), "completed")
        add_steps(progress.get("blocked_steps"), "blocked")
        add_steps(progress.get("pending_steps"), "pending")
        if not steps:
            return None
        current = next(
            (step for step in steps if step.status == "pending"),
            None,
        )
        if current is not None:
            current.status = "in_progress"
        next_action = recovered_task.get("next_action")
        task = TaskState(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            goal=goal,
            steps=steps,
            current_step_id=current.id if current else None,
            phase="acting" if current else "waiting",
            next_action=str(next_action) if isinstance(next_action, str) and next_action else (
                current.title if current else None
            ),
            completion_satisfied=bool(progress.get("completion_satisfied", False)),
            completion_reason=str(progress.get("completion_reason", "")),
        )
        return task

    def after_tool_results(
        self,
        task: TaskState,
        run: RunState,
        results: list[ToolResultMessage],
    ) -> ExecutionDecision:
        if not results:
            return ExecutionDecision("continue", "no_tool_results", task.next_action)

        if any(result.status == "cancelled" for result in results):
            task.phase = "waiting"
            return ExecutionDecision("stop", "cancelled", task.next_action)

        if any(result.status == "approval_required" for result in results):
            self._block_current_step(task, "等待工具审批")
            task.phase = "waiting"
            return ExecutionDecision("wait_approval", "approval_required", task.next_action)

        if any(result.status == "denied" for result in results):
            self._block_current_step(task, "工具权限拒绝")
            return ExecutionDecision("replan", "permission_denied", task.next_action)

        if self._has_failed_verification(results):
            step = self._current_step(task)
            if step is not None:
                step.failure_count += 1
                step.status = "in_progress"
                step.note = "验证失败，需要修复"
                step.evidence_refs.extend(_evidence_refs(results))
                if step.failure_count >= 2:
                    if task.replan_count >= task.max_replans_per_run:
                        step.status = "blocked"
                        step.note = "连续失败且已达到重新规划上限"
                        task.phase = "waiting"
                        task.next_action = "报告连续失败并等待用户指示"
                        return ExecutionDecision(
                            "stop",
                            "replan_limit_exceeded",
                            task.next_action,
                        )
                    self._replan_after_failure(task, results)
                    return ExecutionDecision(
                        "replan",
                        "repeated_step_failure",
                        task.next_action,
                    )
            return ExecutionDecision("repair", "verification_failed", task.next_action)

        for result in results:
            if self._result_has_progress(result):
                step = self._current_step(task)
                if step is not None:
                    step.status = "completed"
                    step.note = None
                    step.evidence_refs.extend(_evidence_refs([result]))
                    self._advance(task)

        if all(step.status == "completed" for step in task.steps):
            task.phase = "finished"
            return ExecutionDecision("finish", "all_steps_completed")

        task.phase = "verifying" if run.workspace_changed else "acting"
        return ExecutionDecision("continue", "next_step", task.next_action)

    def check_completion(self, task: TaskState, run: RunState) -> CompletionCheck:
        if any(step.status == "blocked" for step in task.steps):
            check = CompletionCheck(
                satisfied=False,
                reason="blocked_steps",
                missing=["unblocked_steps"],
                can_continue=False,
            )
            self._record_completion(task, check)
            return check

        if run.workspace_changed and not run.fresh_verification_passed:
            can_continue = task.completion_prompt_count == 0
            check = CompletionCheck(
                satisfied=False,
                reason="modified_without_fresh_verification",
                missing=["fresh_verification"],
                can_continue=can_continue,
                unverified=not can_continue,
            )
            if can_continue:
                task.completion_prompt_count += 1
            self._record_completion(task, check)
            return check

        incomplete = [
            step.title
            for step in task.steps
            if step.status not in {"completed", "blocked"}
        ]
        if incomplete and not run.workspace_changed:
            for step in task.steps:
                if step.status == "in_progress":
                    step.status = "completed"
                    if not step.evidence_refs:
                        step.evidence_refs.append("model:final_answer")
            incomplete = [
                step.title
                for step in task.steps
                if step.status not in {"completed", "blocked"}
            ]

        if incomplete:
            check = CompletionCheck(
                satisfied=False,
                reason="incomplete_steps",
                missing=incomplete,
                can_continue=False,
            )
            self._record_completion(task, check)
            return check

        check = CompletionCheck(satisfied=True, reason="all_steps_completed")
        self._record_completion(task, check)
        task.phase = "finished"
        return check

    def completion_steering(self, check: CompletionCheck) -> UserMessage:
        text = (
            "工作区已经发生修改，但当前没有与最新工作区状态一致的成功验证。\n"
            "请运行最相关的测试或检查；如果环境无法验证，请明确记录原因和剩余风险。"
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
            f"Phase: {task.phase}",
            "",
            "Steps:",
        ]
        for step in task.steps:
            evidence = (
                f" evidence={', '.join(step.evidence_refs[-3:])}"
                if step.evidence_refs
                else ""
            )
            note = f" note={step.note}" if step.note else ""
            lines.append(f"- [{step.status}] {step.title}{evidence}{note}")
        current = self._current_step(task)
        if current is not None:
            lines.append("")
            lines.append(f"Current step: {current.title}")
        if task.next_action:
            lines.append(f"Next action: {task.next_action}")
        if task.constraints:
            lines.append("Constraints:")
            lines.extend(f"- {item}" for item in task.constraints)
        return "\n".join(lines)

    def summarize(self, task: TaskState) -> TaskSummary:
        return TaskSummary(
            task_id=task.task_id,
            goal=task.goal,
            completed_steps=[
                step.title for step in task.steps if step.status == "completed"
            ],
            pending_steps=[
                step.title for step in task.steps if step.status in {"pending", "in_progress"}
            ],
            blocked_steps=[
                step.title for step in task.steps if step.status == "blocked"
            ],
            next_action=task.next_action,
            completion_satisfied=task.completion_satisfied,
            completion_reason=task.completion_reason,
        )

    def event_payload(self, task: TaskState) -> dict[str, object]:
        return asdict(task)

    def _normalize_steps(self, raw_steps: Iterable[str]) -> list[TaskStep]:
        seen: set[str] = set()
        steps: list[TaskStep] = []
        for raw in raw_steps:
            title = " ".join(str(raw).strip().split())
            if not title or title in seen:
                continue
            seen.add(title)
            steps.append(
                TaskStep(
                    id=f"step_{len(steps) + 1}",
                    title=title[:_MAX_STEP_TITLE_CHARS],
                )
            )
            if len(steps) >= _MAX_STEPS:
                break
        if not steps:
            steps.append(TaskStep(id="step_1", title="完成当前请求"))
        return steps

    def _current_step(self, task: TaskState) -> TaskStep | None:
        if task.current_step_id is None:
            return None
        return next(
            (step for step in task.steps if step.id == task.current_step_id),
            None,
        )

    def _advance(self, task: TaskState) -> None:
        for step in task.steps:
            if step.status in {"pending", "in_progress"}:
                step.status = "in_progress"
                task.current_step_id = step.id
                task.next_action = step.title
                return
        task.current_step_id = None
        task.next_action = None

    def _block_current_step(self, task: TaskState, note: str) -> None:
        step = self._current_step(task)
        if step is not None:
            step.status = "blocked"
            step.note = note

    def _replan_after_failure(
        self,
        task: TaskState,
        results: list[ToolResultMessage],
    ) -> None:
        task.replan_count += 1
        current = self._current_step(task)
        if current is None:
            return
        current.title = "根据最新失败证据调整方案"
        current.status = "in_progress"
        current.failure_count = 0
        current.note = "保留已完成步骤，局部替换当前和待办步骤"
        current.evidence_refs.extend(_evidence_refs(results))
        current_index = task.steps.index(current)
        task.steps = [
            *task.steps[: current_index + 1],
            TaskStep(id=f"step_{current_index + 2}", title="重新运行相关验证"),
        ]
        task.current_step_id = current.id
        task.next_action = current.title
        task.phase = "acting"

    def _has_failed_verification(self, results: list[ToolResultMessage]) -> bool:
        return any(
            isinstance(result.verification, dict)
            and result.verification.get("status") == "failed"
            for result in results
        )

    def _result_has_progress(self, result: ToolResultMessage) -> bool:
        if result.status != "success":
            return False
        if isinstance(result.verification, dict):
            return result.verification.get("status") == "passed"
        if result.workspace_changed is True:
            return True
        name = result.tool_name.lower()
        if any(marker in name for marker in _READ_TOOL_MARKERS):
            return True
        if any(marker in name for marker in _WRITE_TOOL_MARKERS):
            return result.workspace_changed is True
        return False

    def _record_completion(self, task: TaskState, check: CompletionCheck) -> None:
        task.completion_satisfied = check.satisfied
        task.completion_reason = check.reason


def _goal_from_prompts(prompts: Iterable[AgentMessage]) -> str:
    for message in reversed(list(prompts)):
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                return message.content.strip() or "完成当前请求"
            text = "".join(
                block.text for block in message.content if isinstance(block, TextContent)
            ).strip()
            return text or "完成当前请求"
    return "继续当前任务"


def _evidence_refs(results: list[ToolResultMessage]) -> list[str]:
    refs: list[str] = []
    for result in results:
        if result.tool_call_id:
            refs.append(f"tool:{result.tool_call_id}")
        if result.approval_id:
            refs.append(f"approval:{result.approval_id}")
        if isinstance(result.verification, dict) and result.tool_call_id:
            refs.append(f"verification:{result.tool_call_id}")
        for path in result.affected_paths:
            refs.append(f"file:{path}")
    return refs


__all__ = ["TaskController"]
