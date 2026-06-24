from __future__ import annotations

"""AgentLoop 使用的确定性任务反馈控制器。

本模块的第一版刻意保持精简：模型仍负责决定语义层面的动作，
而此模块将任务进度绑定到运行时证据上（工具结果、文件变更、验证结果、审批状态等）。
"""

import uuid
from dataclasses import asdict
from typing import Iterable, Mapping

from codepilot.protocols import TaskSummary, TextContent, ToolResultMessage, UserMessage

from .run_state import RunState
from .task_planner import PlannedTaskStep
from .task_tools import COMPLETE_TASK_STEP_TOOL
from .task_state import (
    AttemptRecord,
    ChangeSet,
    CompletionCheck,
    ExecutionDecision,
    ReplanRecord,
    TaskState,
    TaskStep,
)
from .types import AgentMessage


_MAX_STEPS = 6                   # 单个任务最大步骤数
_MAX_STEP_TITLE_CHARS = 80       # 步骤标题最大字符数
_READ_TOOL_MARKERS = ("read", "grep", "find", "glob", "ls", "search", "status", "codegraph")  # 只读工具名称标记
_WRITE_TOOL_MARKERS = ("write", "edit", "patch", "apply")  # 写入工具名称标记


class TaskController:
    """任务控制器：为一次运行维护轻量级的、与证据绑定的任务状态。"""

    def initialize(
        self,
        prompts: Iterable[AgentMessage],
        *,
        proposed_steps: Iterable[object] | None = None,
        goal: str | None = None,
        acceptance_criteria: Iterable[str] | None = None,
        constraints: Iterable[str] | None = None,
        recovered_task: Mapping[str, object] | None = None,
    ) -> TaskState:
        """初始化任务状态：从用户消息中提取目标，生成初始步骤。"""
        if recovered_task is not None:
            recovered = self._from_recovered_task(prompts, recovered_task)
            if recovered is not None:
                return recovered
        goal = (goal or _goal_from_prompts(prompts)).strip() or "完成当前请求"
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
        """从恢复的任务记忆中重建 TaskState（用于会话恢复场景）。"""
        progress = recovered_task.get("task_progress")
        if not isinstance(progress, Mapping):
            return None
        goal = str(recovered_task.get("goal") or _goal_from_prompts(prompts)).strip()
        if not goal:
            goal = _goal_from_prompts(prompts)
        steps: list[TaskStep] = []
        seen: set[str] = set()
        details = progress.get("step_details")
        step_details = details if isinstance(details, Mapping) else {}

        def add_steps(raw: object, status: str) -> None:
            if not isinstance(raw, list):
                return
            for value in raw:
                title = " ".join(str(value).strip().split())
                if not title or title in seen:
                    continue
                seen.add(title)
                raw_detail = step_details.get(title)
                detail = raw_detail if isinstance(raw_detail, Mapping) else {}
                steps.append(
                    TaskStep(
                        id=f"step_{len(steps) + 1}",
                        title=title[:_MAX_STEP_TITLE_CHARS],
                        status=status,  # type: ignore[arg-type]
                        kind=str(detail.get("kind") or "other"),
                        acceptance=(
                            str(detail.get("acceptance"))
                            if detail.get("acceptance") is not None
                            else None
                        ),
                        verification_hint=(
                            str(detail.get("verification_hint"))
                            if detail.get("verification_hint") is not None
                            else None
                        ),
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
        """工具执行后更新任务状态并返回执行决策（继续/修复/重新规划/停止等）。"""
        if not results:
            return self._decision(task, "continue", "no_tool_results")

        attempt = self._record_attempt(task, results)
        self._record_change_sets(task, attempt, results)

        if any(result.status == "cancelled" for result in results):
            task.phase = "waiting"
            task.recent_failure_type = "cancelled"
            return self._decision(task, "stop", "cancelled")

        if any(result.status == "approval_required" for result in results):
            self._block_current_step(
                task,
                "等待工具审批",
                evidence_refs=_evidence_refs(results),
            )
            task.phase = "waiting"
            task.recent_error_code = "approval_required"
            task.recent_failure_type = "approval_required"
            return self._decision(task, "wait_approval", "approval_required")

        if any(result.status == "denied" for result in results):
            self._block_current_step(
                task,
                "工具权限拒绝",
                evidence_refs=_evidence_refs(results),
            )
            task.recent_error_code = _first_error_code(results) or "permission_denied"
            task.recent_failure_type = "permission_denied"
            return self._decision(task, "replan", "permission_denied")

        unavailable = [
            result for result in results if _is_tool_unavailable(result)
        ]
        if unavailable:
            step = self._current_step(task)
            if step is not None:
                step.status = "blocked"
                step.note = "工具不可用"
                step.evidence_refs.extend(_evidence_refs(unavailable))
            task.phase = "waiting"
            task.next_action = "报告工具不可用并等待用户指示"
            task.recent_error_code = "tool_not_found"
            task.recent_failure_type = "tool_unavailable"
            return self._decision(task, "stop", "tool_unavailable")

        if self._has_non_verification_error(results):
            step = self._current_step(task)
            task.action_intent = "debug_failure"
            task.recent_error_code = _first_error_code(results) or "tool_error"
            task.recent_failure_type = "tool_error"
            if step is not None:
                step.failure_count += 1
                step.status = "in_progress"
                step.note = "工具执行失败，需要修复或调整方案"
                step.evidence_refs.extend(_evidence_refs(results))
            return self._decision(task, "repair", "tool_error")

        if self._has_failed_verification(results):
            step = self._current_step(task)
            task.action_intent = "debug_failure"
            task.recent_error_code = "verification_failed"
            task.recent_failure_type = "verification_failed"
            self._mark_latest_changes_failed(task, results)
            if step is not None:
                step.failure_count += 1
                step.status = "in_progress"
                step.note = "验证失败，需要修复"
                step.evidence_refs.extend(_evidence_refs(results))
                if step.failure_count >= 2:
                    if task.change_sets:
                        self._mark_rollback_required(task)
                        self._record_replan(
                            task,
                            trigger="verification_failed",
                            evidence_refs=_evidence_refs(results),
                            requires_revert=True,
                        )
                        task.phase = "waiting"
                        task.next_action = "报告可能需要撤销的变更并等待用户确认"
                        return self._decision(
                            task,
                            "propose_revert",
                            "repeated_failure_after_change",
                        )
                    if task.replan_count >= task.max_replans_per_run:
                        step.status = "blocked"
                        step.note = "连续失败且已达到重新规划上限"
                        task.phase = "waiting"
                        task.next_action = "报告连续失败并等待用户指示"
                        return self._decision(task, "stop", "replan_limit_exceeded")
                    self._replan_after_failure(task, results)
                    return self._decision(task, "replan", "repeated_step_failure")
            return self._decision(task, "repair", "verification_failed")

        if self._has_passed_verification(results):
            self._complete_verified_steps(task, results)
            self._mark_latest_changes_verified(task, results)
        elif self._has_complete_step_signal(results):
            self._complete_current_step_from_signal(task, results)
        else:
            for result in results:
                self._update_progress_from_result(task, result)

        if all(step.status == "completed" for step in task.steps):
            task.phase = "finished"
            return self._decision(task, "finish", "all_steps_completed")

        task.phase = (
            "verifying"
            if run.workspace_changed and not run.fresh_verification_passed
            else "acting"
        )
        return self._decision(task, "continue", "next_step")

    def check_completion(self, task: TaskState, run: RunState) -> CompletionCheck:
        """检查任务是否完成：验证阻塞步骤、工作区变更、未完成步骤等条件。"""
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
        if incomplete and not run.workspace_changed and task.recent_error_code is None:
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
        """生成完成引导消息：当工作区已修改但未通过验证时，提示 Agent 运行验证。"""
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
        """将任务状态渲染为 Markdown 格式的上下文文本（注入系统提示词）。"""
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
            progress = (
                f" progress={step.progress_state}"
                if step.progress_state != "none"
                else ""
            )
            note = f" note={step.note}" if step.note else ""
            lines.append(f"- [{step.status}] {step.title}{progress}{evidence}{note}")
            detail_parts = [f"Kind: {step.kind}"]
            if step.acceptance:
                detail_parts.append(f"Acceptance: {step.acceptance}")
            if step.verification_hint:
                detail_parts.append(f"Verification hint: {step.verification_hint}")
            if step.summary:
                detail_parts.append(f"Summary: {step.summary}")
            if detail_parts != ["Kind: other"]:
                lines.append("  - " + "; ".join(detail_parts))
        current = self._current_step(task)
        if current is not None:
            lines.append("")
            lines.append(f"Current step: {current.title}")
            if current.acceptance:
                lines.append(f"Acceptance: {current.acceptance}")
            if current.verification_hint:
                lines.append(f"Verification hint: {current.verification_hint}")
            lines.append(
                "When this step's acceptance criteria are satisfied, call "
                f"`{COMPLETE_TASK_STEP_TOOL}` with a short evidence-backed summary."
            )
        if task.next_action:
            lines.append(f"Next action: {task.next_action}")
        if task.action_intent:
            lines.append(f"Action intent: {task.action_intent}")
        if task.recent_error_code:
            lines.append(f"Recent error: {task.recent_error_code}")
        if task.rollback_required:
            lines.append(
                "Rollback required: " + ", ".join(task.rollback_targets)
            )
        if task.constraints:
            lines.append("Constraints:")
            lines.extend(f"- {item}" for item in task.constraints)
        return "\n".join(lines)

    def summarize(self, task: TaskState) -> TaskSummary:
        """生成任务摘要：包含已完成、待处理、阻塞步骤和完成状态。"""
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
            attempts=[asdict(item) for item in task.attempts],
            change_sets=[asdict(item) for item in task.change_sets],
            replans=[asdict(item) for item in task.replans],
            control_signal=self.control_signal(task),
            step_details={
                step.title: {
                    "kind": step.kind,
                    "acceptance": step.acceptance,
                    "verification_hint": step.verification_hint,
                    "summary": step.summary,
                }
                for step in task.steps
            },
        )

    def event_payload(self, task: TaskState) -> dict[str, object]:
        """将任务状态转换为字典格式，用于事件上报。"""
        return asdict(task)

    def control_signal(self, task: TaskState) -> dict[str, object]:
        """输出给上下文与记忆模块的轻量任务控制信号。"""
        current = self._current_step(task)
        return {
            "task_id": task.task_id,
            "phase": task.phase,
            "current_step_id": current.id if current else None,
            "current_step_title": current.title if current else None,
            "current_step_acceptance": current.acceptance if current else None,
            "current_step_verification_hint": (
                current.verification_hint if current else None
            ),
            "next_action": task.next_action,
            "action_intent": task.action_intent,
            "current_attempt_id": (
                task.attempts[-1].attempt_id if task.attempts else None
            ),
            "recent_failure_type": task.recent_failure_type,
            "recent_error_code": task.recent_error_code,
            "rollback_required": task.rollback_required,
            "rollback_targets": list(task.rollback_targets),
            "last_decision": task.last_decision,
        }

    def _normalize_steps(self, raw_steps: Iterable[object]) -> list[TaskStep]:
        """规范化步骤列表：去重、截断标题、限制最大步骤数。"""
        seen: set[str] = set()
        steps: list[TaskStep] = []
        for raw in raw_steps:
            title, kind, acceptance, verification_hint = _step_fields(raw)
            if not title or title in seen:
                continue
            seen.add(title)
            steps.append(
                TaskStep(
                    id=f"step_{len(steps) + 1}",
                    title=title[:_MAX_STEP_TITLE_CHARS],
                    kind=kind,
                    acceptance=acceptance,
                    verification_hint=verification_hint,
                )
            )
            if len(steps) >= _MAX_STEPS:
                break
        if not steps:
            steps.append(TaskStep(id="step_1", title="完成当前请求"))
        return steps

    def _current_step(self, task: TaskState) -> TaskStep | None:
        """获取当前正在执行的步骤。"""
        if task.current_step_id is None:
            return None
        return next(
            (step for step in task.steps if step.id == task.current_step_id),
            None,
        )

    def _advance(self, task: TaskState) -> None:
        """推进到下一个待处理步骤。"""
        for step in task.steps:
            if step.status in {"pending", "in_progress"}:
                step.status = "in_progress"
                task.current_step_id = step.id
                task.next_action = step.title
                return
        task.current_step_id = None
        task.next_action = None

    def _block_current_step(
        self,
        task: TaskState,
        note: str,
        *,
        evidence_refs: list[str] | None = None,
    ) -> None:
        """将当前步骤标记为阻塞状态。"""
        step = self._current_step(task)
        if step is not None:
            step.status = "blocked"
            step.note = note
            for ref in evidence_refs or []:
                if ref not in step.evidence_refs:
                    step.evidence_refs.append(ref)

    def _replan_after_failure(
        self,
        task: TaskState,
        results: list[ToolResultMessage],
    ) -> None:
        """失败后重新规划：保留已完成步骤，替换当前和后续步骤。"""
        task.replan_count += 1
        current = self._current_step(task)
        if current is None:
            return
        self._record_replan(
            task,
            trigger="verification_failed",
            evidence_refs=_evidence_refs(results),
            requires_revert=False,
        )
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
        """检查工具结果中是否包含失败的验证。"""
        return any(
            isinstance(result.verification, dict)
            and result.verification.get("status") == "failed"
            for result in results
        )

    def _has_non_verification_error(self, results: list[ToolResultMessage]) -> bool:
        return any(
            (result.status != "success" or result.is_error)
            and not isinstance(result.verification, dict)
            for result in results
        )

    def _has_passed_verification(self, results: list[ToolResultMessage]) -> bool:
        return any(
            isinstance(result.verification, dict)
            and result.verification.get("status") == "passed"
            for result in results
        )

    def _has_complete_step_signal(self, results: list[ToolResultMessage]) -> bool:
        return any(_complete_step_payload(result) is not None for result in results)

    def _record_attempt(
        self,
        task: TaskState,
        results: list[ToolResultMessage],
    ) -> AttemptRecord:
        intent = _infer_action_intent(results)
        current = self._current_step(task)
        attempt = AttemptRecord(
            attempt_id=f"attempt_{uuid.uuid4().hex[:12]}",
            step_id=current.id if current else None,
            action_intent=intent,
            tool_call_ids=[
                result.tool_call_id
                for result in results
                if result.tool_call_id
            ],
            evidence_refs=_evidence_refs(results),
            status="failed" if any(result.is_error for result in results) else "succeeded",
            failure_type=_first_error_code(results),
            failure_reason=_first_error_code(results),
        )
        task.attempts.append(attempt)
        task.action_intent = intent
        return attempt

    def _record_change_sets(
        self,
        task: TaskState,
        attempt: AttemptRecord,
        results: list[ToolResultMessage],
    ) -> None:
        for result in results:
            evidence = result.metadata.get("change_evidence")
            if not isinstance(evidence, dict):
                continue
            affected = [
                str(path)
                for path in evidence.get("affected_paths", result.affected_paths)
                if isinstance(path, str)
            ]
            if not affected:
                continue
            task.change_sets.append(
                ChangeSet(
                    change_id=f"change_{uuid.uuid4().hex[:12]}",
                    attempt_id=attempt.attempt_id,
                    step_id=attempt.step_id,
                    affected_paths=affected,
                    before_hashes={
                        str(key): str(value)
                        for key, value in evidence.get("before_hashes", {}).items()
                    } if isinstance(evidence.get("before_hashes"), dict) else {},
                    after_hashes={
                        str(key): str(value)
                        for key, value in evidence.get("after_hashes", {}).items()
                    } if isinstance(evidence.get("after_hashes"), dict) else {},
                    tool_call_ids=[result.tool_call_id] if result.tool_call_id else [],
                    diff_summary=result.diff_summary,
                    status="pending",
                )
            )

    def _update_progress_from_result(
        self,
        task: TaskState,
        result: ToolResultMessage,
    ) -> None:
        if result.status != "success":
            return
        step = self._current_step(task)
        if step is None:
            return
        step.evidence_refs.extend(_evidence_refs([result]))
        if result.workspace_changed is True:
            step.status = "in_progress"
            step.progress_state = "changed"
            step.note = "已产生文件变更，等待验证"
            task.phase = "verifying"
            task.action_intent = "edit_file"
            task.recent_error_code = None
            task.recent_failure_type = None
            return
        name = result.tool_name.lower()
        if any(marker in name for marker in _READ_TOOL_MARKERS):
            step.status = "in_progress"
            step.progress_state = "evidence_collected"
            task.action_intent = "read_context"
            task.recent_error_code = None
            task.recent_failure_type = None

    def _complete_verified_steps(
        self,
        task: TaskState,
        results: list[ToolResultMessage],
    ) -> None:
        refs = _evidence_refs(results)
        step = self._current_step(task)
        if step is None:
            step = next(
                (item for item in task.steps if item.status in {"pending", "in_progress"}),
                None,
            )
        if step is None:
            task.current_step_id = None
            task.next_action = None
            task.phase = "finished"
            return
        step.status = "completed"
        step.progress_state = "verified"
        step.note = None
        step.summary = "验证通过"
        for ref in refs:
            if ref not in step.evidence_refs:
                step.evidence_refs.append(ref)
        next_step = self._next_incomplete_step_after(task, step)
        if next_step is not None and _is_verification_step(next_step):
            next_step.status = "completed"
            next_step.progress_state = "verified"
            next_step.note = None
            next_step.summary = "验证通过"
            for ref in refs:
                if ref not in next_step.evidence_refs:
                    next_step.evidence_refs.append(ref)
        task.action_intent = "run_verification"
        task.recent_error_code = None
        task.recent_failure_type = None
        self._advance(task)
        task.phase = "finished" if task.current_step_id is None else "acting"

    def _complete_current_step_from_signal(
        self,
        task: TaskState,
        results: list[ToolResultMessage],
    ) -> None:
        step = self._current_step(task)
        if step is None:
            return
        refs = _evidence_refs(results)
        summaries: list[str] = []
        for result in results:
            payload = _complete_step_payload(result)
            if payload is None:
                continue
            summary = payload.get("summary")
            if isinstance(summary, str) and summary.strip():
                summaries.append(summary.strip())
            raw_refs = payload.get("evidence_refs")
            if isinstance(raw_refs, list):
                refs.extend(
                    str(item)
                    for item in raw_refs
                    if isinstance(item, str) and item.strip()
                )
        step.status = "completed"
        step.note = None
        step.summary = summaries[-1] if summaries else "步骤已完成"
        for ref in refs:
            if ref not in step.evidence_refs:
                step.evidence_refs.append(ref)
        task.recent_error_code = None
        task.recent_failure_type = None
        self._advance(task)
        task.phase = "finished" if task.current_step_id is None else "acting"

    def _mark_latest_changes_failed(
        self,
        task: TaskState,
        results: list[ToolResultMessage],
    ) -> None:
        refs = _evidence_refs(results)
        for change in task.change_sets:
            if change.status in {"pending", "verified"}:
                change.status = "failed"
                change.verification_refs.extend(refs)

    def _mark_latest_changes_verified(
        self,
        task: TaskState,
        results: list[ToolResultMessage],
    ) -> None:
        refs = _evidence_refs(results)
        for change in task.change_sets:
            if change.status in {"pending", "failed"}:
                change.status = "verified"
                change.verification_refs.extend(refs)

    def _mark_rollback_required(self, task: TaskState) -> None:
        targets: list[str] = []
        for change in task.change_sets:
            if change.status in {"pending", "failed"}:
                change.status = "revert_required"
                targets.extend(change.affected_paths)
        task.rollback_required = True
        task.rollback_targets = sorted(set(targets))

    def _record_replan(
        self,
        task: TaskState,
        *,
        trigger: str,
        evidence_refs: list[str],
        requires_revert: bool,
    ) -> None:
        current = self._current_step(task)
        task.replans.append(
            ReplanRecord(
                replan_id=f"replan_{uuid.uuid4().hex[:12]}",
                trigger=trigger,
                failed_attempt_id=(
                    task.attempts[-1].attempt_id if task.attempts else None
                ),
                abandoned_strategy=current.title if current else None,
                new_strategy="根据最新失败证据调整方案",
                evidence_refs=list(evidence_refs),
                requires_revert=requires_revert,
                rollback_targets=list(task.rollback_targets),
            )
        )

    def _decision(
        self,
        task: TaskState,
        action: str,
        reason: str,
    ) -> ExecutionDecision:
        task.last_decision = action
        return ExecutionDecision(action, reason, task.next_action)  # type: ignore[arg-type]

    def _result_has_progress(self, result: ToolResultMessage) -> bool:
        """判断工具结果是否代表实质进展（成功、验证通过或工作区变更）。"""
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
        """将完成检查结果记录到任务状态中。"""
        task.completion_satisfied = check.satisfied
        task.completion_reason = check.reason

    def _next_incomplete_step_after(
        self,
        task: TaskState,
        step: TaskStep,
    ) -> TaskStep | None:
        try:
            start = task.steps.index(step) + 1
        except ValueError:
            return None
        return next(
            (
                item
                for item in task.steps[start:]
                if item.status in {"pending", "in_progress"}
            ),
            None,
        )


def _goal_from_prompts(prompts: Iterable[AgentMessage]) -> str:
    """从用户消息中提取任务目标（取最后一条用户消息的文本内容）。"""
    for message in reversed(list(prompts)):
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                return message.content.strip() or "完成当前请求"
            text = "".join(
                block.text for block in message.content if isinstance(block, TextContent)
            ).strip()
            return text or "完成当前请求"
    return "继续当前任务"


def _step_fields(raw: object) -> tuple[str, str, str | None, str | None]:
    """Extract normalized step fields from string/dict/PlannedTaskStep-like input."""

    if isinstance(raw, PlannedTaskStep):
        return (
            _compact(raw.title, limit=_MAX_STEP_TITLE_CHARS),
            _compact(raw.kind, limit=40) or "other",
            _optional_text(raw.acceptance),
            _optional_text(raw.verification_hint),
        )
    if isinstance(raw, Mapping):
        return (
            _compact(raw.get("title"), limit=_MAX_STEP_TITLE_CHARS),
            _compact(raw.get("kind"), limit=40) or "other",
            _optional_text(raw.get("acceptance")),
            _optional_text(raw.get("verification_hint")),
        )
    return _compact(raw, limit=_MAX_STEP_TITLE_CHARS), "other", None, None


def _is_verification_step(step: TaskStep) -> bool:
    if step.kind == "verify":
        return True
    text = f"{step.title} {step.verification_hint or ''}".lower()
    markers = ("验证", "测试", "检查", "verify", "test", "pytest")
    return any(marker in text for marker in markers)


def _complete_step_payload(result: ToolResultMessage) -> Mapping[str, object] | None:
    if result.tool_name != COMPLETE_TASK_STEP_TOOL:
        return None
    payload = result.metadata.get("task_control")
    if not isinstance(payload, Mapping):
        return None
    if payload.get("action") != "complete_step":
        return None
    if payload.get("valid") is False:
        return None
    return payload


def _optional_text(value: object) -> str | None:
    text = _compact(value, limit=240)
    return text or None


def _compact(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:limit]


def _evidence_refs(results: list[ToolResultMessage]) -> list[str]:
    """从工具结果中提取证据引用列表（工具调用ID、审批ID、验证ID、文件路径）。"""
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


def _is_tool_unavailable(result: ToolResultMessage) -> bool:
    if result.status != "error" and not result.is_error:
        return False
    if result.error_code == "tool_not_found":
        return True
    text = " ".join(
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    ).lower()
    return text.startswith("tool ") and " not found" in text


def _first_error_code(results: list[ToolResultMessage]) -> str | None:
    return next(
        (
            result.error_code
            for result in results
            if result.error_code
        ),
        None,
    )


def _infer_action_intent(results: list[ToolResultMessage]) -> str:
    if any(_complete_step_payload(result) is not None for result in results):
        return "complete_step"
    if any(
        isinstance(result.verification, dict)
        and result.verification.get("status") == "failed"
        for result in results
    ):
        return "debug_failure"
    if any(isinstance(result.verification, dict) for result in results):
        return "run_verification"
    if any(result.workspace_changed is True for result in results):
        return "edit_file"
    names = " ".join(result.tool_name.lower() for result in results)
    if any(marker in names for marker in _READ_TOOL_MARKERS):
        return "read_context"
    if any(marker in names for marker in _WRITE_TOOL_MARKERS):
        return "edit_file"
    return "tool_action"


__all__ = ["TaskController"]
