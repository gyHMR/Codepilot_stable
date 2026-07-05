from __future__ import annotations

# 新手导读：这里仅放任务控制工具的协议名和识别逻辑，不放可执行工具对象。
# 关注点：core 解释 complete_task_step 的语义；tools 层负责创建和执行 AgentTool。

"""Task-control tool signal names understood by core."""

from codepilot.protocols import TASK_CONTROL_COMPLETE_TOOL


COMPLETE_TASK_STEP_TOOL = TASK_CONTROL_COMPLETE_TOOL


__all__ = [
    "COMPLETE_TASK_STEP_TOOL",
]
