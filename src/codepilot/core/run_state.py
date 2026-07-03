from __future__ import annotations

# 新手导读：RunState 保存一次 Agent run 中的计数器、工作区变化和验证状态。
# 关注点：它不是持久化存储，只是主循环运行时判断重复调用和完成条件的轻量状态。

"""一次 Agent Run 执行期间收集的机械性执行事实。

RunState 刻意保持"机械性"：它只记录本次运行中发生了什么，
而不决定用户的任务是否在语义上已完成。
TaskController 将这些事实与 TaskState 结合使用，做出完成判断。
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import cast

from codepilot.protocols import (
    AgentRunCounters,
    AgentRunResult,
    AgentRunStatus,
    AgentRunStopReason,
    AssistantMessage,
    ErrorInfo,
    RunVerification,
    RunVerificationStatus,
    TaskSummary,
    ToolCall,
    ToolResultMessage,
)

from .types import AgentMessage


def new_run_id() -> str:
    """生成一个新的 Run ID（格式：run_ + 12位UUID hex）。"""
    return f"run_{uuid.uuid4().hex[:12]}"


@dataclass
class RunState:
    """Run 状态：记录一次运行过程中的执行事实（计数器、受影响路径、验证结果等）。"""
    run_id: str
    session_id: str | None
    counters: AgentRunCounters = field(default_factory=AgentRunCounters)  # 运行计数器
    affected_paths: set[str] = field(default_factory=set)  # 受影响的文件路径集合
    workspace_changed: bool = False  # 工作区是否被修改
    verification: list[RunVerification] = field(default_factory=list)  # 验证结果列表
    fresh_verification_passed: bool = False  # 最新一次验证是否通过
    last_tool_fingerprint: str | None = None  # 上一次工具调用的指纹（用于检测重复调用）
    repeated_tool_calls: int = 0  # 连续重复工具调用次数

    def has_repeated_call(
        self,
        tool_calls: list[ToolCall],
        *,
        limit: int,
    ) -> bool:
        """检测是否存在连续重复的工具调用（基于 JSON 指纹比对）。"""
        if limit <= 0:
            return False
        for tool_call in tool_calls:
            fingerprint = json.dumps(
                [tool_call.name, tool_call.arguments],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if fingerprint == self.last_tool_fingerprint:
                self.repeated_tool_calls += 1
            else:
                self.last_tool_fingerprint = fingerprint
                self.repeated_tool_calls = 1
            if self.repeated_tool_calls > limit:
                return True
        return False

    def collect_tool_results(self, results: list[ToolResultMessage]) -> None:
        """收集工具执行结果：更新计数器、受影响路径和验证状态。"""
        self.counters.tool_calls += len(results)
        for result in results:
            self.affected_paths.update(result.affected_paths)
            if result.workspace_changed:
                self.workspace_changed = True
                self.fresh_verification_passed = False
            if result.verification:
                verification = result.verification
                status = _verification_status(verification.get("status"))
                self.verification.append(
                    RunVerification(
                        tool_call_id=result.tool_call_id,
                        tool_name=result.tool_name,
                        status=status,
                        command=_optional_str(verification.get("command")),
                        exit_code=_optional_int(verification.get("exit_code")),
                        summary=str(verification.get("summary", "")),
                    )
                )
                self.fresh_verification_passed = status == "passed"

    def result(
        self,
        *,
        status: AgentRunStatus,
        stop_reason: AgentRunStopReason,
        messages: list[AgentMessage],
        final_message: AssistantMessage | None,
        error: ErrorInfo | None = None,
        task: TaskSummary | None = None,
    ) -> AgentRunResult:
        """构建并返回最终的 AgentRunResult 结构化结果。"""
        return AgentRunResult(
            run_id=self.run_id,
            session_id=self.session_id,
            status=status,
            stop_reason=stop_reason,
            counters=self.counters,
            messages=list(messages),
            final_message=final_message,
            error=error,
            affected_paths=sorted(self.affected_paths),
            workspace_changed=self.workspace_changed,
            verification=list(self.verification),
            task=task,
        )


def _verification_status(value: object) -> RunVerificationStatus:
    """将原始值转换为验证状态枚举，无法识别时返回 "unknown"。"""
    if value in {"passed", "failed", "cancelled", "unknown"}:
        return cast(RunVerificationStatus, value)
    return "unknown"


def _optional_str(value: object) -> str | None:
    """安全提取可选字符串：值为 str 则返回，否则返回 None。"""
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    """安全提取可选整数：值为 int（排除 bool）则返回，否则返回 None。"""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["RunState", "new_run_id"]
