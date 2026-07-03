# Codepilot 任务规划控制设计

本文档记录当前 Codepilot 的任务规划控制实现。阅读时请以代码为准，核心代码集中在：

```text
src/codepilot/core/task_control/
  contracts.py    # task_mode、planning budget、discovery report、planning state
  modes.py        # read/edit/plan 的模式策略
  discovery.py    # plan 模式的只读事实发现
  planner.py      # 基于上下文和 discovery report 生成计划
  bootstrap.py    # plan 模式启动编排：恢复、discovery、synthesis
  controller.py   # 任务控制 facade：初始化、工具后更新、完成门控、摘要
  state.py        # TaskState、TaskStep、AttemptRecord、ChangeSet、ReplanRecord
  evidence.py     # 从工具结果提取证据、动作意图、错误信息
  verifier.py     # 解释 verification 工具结果
  replanner.py    # 失败后的修复和重规划辅助规则
  stop.py         # 完成门控和 completion steering
  tools.py        # complete_task_step 运行时管理工具
```

任务控制不是一个庞大的 FSM。它更像一条运行期反馈控制链：

```text
用户请求
  -> 选择 task_mode
  -> 初始化任务状态
  -> 每轮模型调用前注入任务上下文
  -> 工具执行
  -> 工具事实 + 运行事实更新 TaskState
  -> 得到下一步决策
  -> 完成门控判断是否可以结束
  -> 摘要、事件、恢复投影、审计记录
```

设计目标是让 Agent 仍然负责语义判断和代码行动，但 Runtime 给它提供边界、证据和完成门槛。

## 1. 总体设计理念

### 1.1 模型负责语义，框架负责边界

模型负责：

- 理解用户到底想做什么。
- 决定下一步读什么、改什么、跑什么验证。
- 根据失败日志推理根因。
- 判断某一步验收标准是否已经满足。

框架负责：

- 记录当前任务目标、步骤、阶段和证据。
- 根据工具结果判断是否真的发生了读取、修改、验证、失败或权限阻塞。
- 在工作区修改后要求新鲜验证。
- 检测连续失败、重规划上限和可能需要回滚的变更。
- 把状态写入事件、审计和恢复投影。

所以任务控制的边界是：它约束 Agent，不替 Agent 写业务决策。

### 1.2 唯一用户级行为口径是 task_mode

当前用户可见模式只有：

```text
read | edit | plan
```

`planning_budget_profile` 只是 plan discovery 的预算调优参数，不是新模式。

### 1.3 证据比声明更重要

任务状态不以“模型说完成了”为唯一依据，而是绑定到工具结果：

- `ToolResultMessage.tool_call_id`
- verification status
- workspace changed
- affected paths
- change evidence
- `complete_task_step` 的 task_control 元数据

模型可以总结“我完成了”，但完成门控仍会检查是否有阻塞步骤、未完成步骤和修改后的新鲜验证。

## 2. 用户请求进入后，先确定模式

入口来自 CLI、RPC、Web 或 runtime service。模式最终进入 `AgentLoopConfig.task_mode`。

配置来源大致是：

```text
CLI/RPC/UserInput override
  -> CreateAgentSessionOptions.task_mode
  -> .codepilot/settings.json task_mode
  -> RuntimeDefaults.task_mode = "edit"
```

相关实现：

- `src/codepilot/runtime/config.py`
- `src/codepilot/runtime/types.py`
- `src/codepilot/sessions/session.py`
- `src/codepilot/core/types.py`
- `src/codepilot/core/task_control/modes.py`

### 2.1 read 模式

`read` 表示只读分析与回答。

策略来自 `policy_for_mode("read")`：

- `read_only=True`
- 不要求 planner
- 默认步骤是“只读分析并回答当前请求”
- task context 明确提示不要修改文件或运行有状态变化的命令

runtime 还会把 `read` 映射到只读权限：

- `read_only_mode=True`
- `tool_permission_mode="read-only"`
- 如果用户显式传入 `task_mode=read` 但又传 `workspace-write`，配置解析会报错

### 2.2 edit 模式

`edit` 是默认模式，适合简单开发任务。

特点：

- 不触发 plan discovery。
- 不调用 `TaskPlanner`。
- 由 `TaskController.initialize()` 创建一个轻量默认步骤。
- 后续仍然走工具事实、验证事实、完成门控。

这意味着 edit 不是“无任务控制”，而是“不预先复杂规划”。

### 2.3 plan 模式

`plan` 适合复杂任务。

特点：

- 必须经过 `PlanningBootstrap`。
- 默认复用当前 run 的主模型，不新增独立 planning model。
- 先做只读 discovery，再 synthesis 计划。
- 执行期仍回到普通 ReAct 工具循环。

预算由 `planning_budget_profile` 控制，默认 `balanced`：

```text
conservative: 2 model rounds, 6 read-only tool calls, 6000 estimated tokens, 30s
balanced:     4 model rounds, 12 read-only tool calls, 12000 estimated tokens, 60s
wide:         6 model rounds, 20 read-only tool calls, 20000 estimated tokens, 120s
```

## 3. AgentLoop 初始化：决定是否规划

主入口在 `src/codepilot/core/agent_loop.py`。

进入 `_run_loop()` 后会创建：

```text
LLMStreamRunner
ToolCallCoordinator
TaskController
TaskModePolicy
```

然后根据模式决定初始化方式：

```text
read/edit:
  TaskPlanningState(phase="none", source="default")
  TaskController.initialize(... proposed_steps=None ...)

plan:
  PlanningBootstrap.run(...)
  得到 TaskPlanDraft 和 TaskPlanningState
  TaskController.initialize(... proposed_steps=plan.steps, planning=planning_state ...)
```

初始化完成后，AgentLoop 会发出：

```text
task_plan_created
```

并确保工具列表里有 runtime managed 工具：

```text
complete_task_step
```

这个工具给模型一个显式动作：当当前步骤验收标准满足时，模型可以调用它完成步骤。但它不能绕过修改后的验证要求。

## 4. plan 模式第一阶段：只读事实发现

实现位置：`task_control/discovery.py`。

`PlanningDiscovery` 是一个 scratch ReAct loop：

```text
当前会话历史 + 本轮用户输入
  -> scratch AgentContext
  -> 只暴露 metadata.read_only=True 的工具
  -> 使用 LLMStreamRunner 调模型
  -> 使用 ToolCallCoordinator 执行只读工具
  -> 得到 PlanningDiscoveryReport
```

关键边界：

- discovery 的 assistant/tool messages 不写回主 `current_context.messages`。
- 工具只取 read-only metadata；没有 read-only 元数据的工具默认不可见。
- discovery 仍复用主模型、主 context prepare/transform、API key 和工具协调器。
- 事件仍正常上报，所以审计能看到 discovery 做过什么。

discovery 的停止原因来自模型和预算共同作用：

```text
sufficient_evidence
budget_exhausted
model_error
tool_error
invalid_json
no_read_only_tools
```

输出结构是 `PlanningDiscoveryReport`：

```python
PlanningDiscoveryReport(
    status="completed|failed|budget_exhausted|skipped",
    facts=(...),
    relevant_files=(...),
    risks=(...),
    verification_hints=(...),
    open_questions=(...),
    evidence_refs=(...),
    budget=PlanningBudgetUsage(...),
)
```

对应事件：

```text
planning_discovery_started
planning_discovery_step
planning_discovery_completed
```

## 5. plan 模式第二阶段：计划生成

实现位置：

- `task_control/bootstrap.py`
- `task_control/planner.py`

`PlanningBootstrap` 在 discovery 后发起 synthesis：

```text
PlanningDiscoveryReport
  -> TaskPlanner.generate(... discovery_report=...)
  -> TaskPlanDraft(goal, steps, source)
  -> TaskPlanningState(phase="execution", source=..., budget=..., discovery=...)
```

`TaskPlanner` 要求模型输出 JSON，不要输出 Markdown。它会解析：

```json
{
  "goal": "...",
  "steps": [
    {
      "title": "...",
      "kind": "investigate|edit|verify|summarize|other",
      "acceptance": "...",
      "verification_hint": "..."
    }
  ]
}
```

计划最多保留 6 步，步骤会去重、截断和校验。

计划来源：

```text
llm                 # 直接由 planner 生成
llm_with_discovery  # discovery completed 后由 planner 生成
fallback            # planner 失败时的降级计划
recovered           # 从恢复投影继续执行
default             # read/edit 的默认轻量任务
```

如果 discovery 成功但 planner 解析失败，系统不会直接失败，而是创建 fallback 单步计划，并把失败原因写入：

```text
TaskPlanningState.fallback_reason
```

对应事件：

```text
planning_synthesis_started
planning_synthesis_completed
task_plan_created
```

`task_plan_created.plan.planSource` 是 wire 字段，值从 `task.planning.source` 派生，不再维护第二份状态。

## 6. TaskState：运行期任务状态

实现位置：`task_control/state.py`。

`TaskState` 是一次 run 内的任务状态：

```python
TaskState(
    task_id=...,
    goal=...,
    steps=[TaskStep(...)],
    mode="read|edit|plan",
    planning=TaskPlanningState(...),
    current_step_id=...,
    phase="understanding|acting|verifying|waiting|finished",
    next_action=...,
    replan_count=...,
    max_replans_per_run=2,
    completion_satisfied=False,
    completion_reason="",
    attempts=[AttemptRecord(...)],
    change_sets=[ChangeSet(...)],
    replans=[ReplanRecord(...)],
)
```

### 6.1 TaskStep

每个步骤包含：

```text
id
title
status: pending | in_progress | completed | blocked
kind: investigate | edit | verify | summarize | other
acceptance
verification_hint
summary
evidence_refs
failure_count
note
progress_state: none | evidence_collected | changed | verified
```

步骤不是静态 Markdown 清单。它会随着工具事实更新：

- 读取成功：`progress_state=evidence_collected`
- 产生文件变更：`progress_state=changed`
- 验证通过：`progress_state=verified`
- 工具失败或验证失败：记录 failure_count 和 evidence_refs
- 权限或工具不可用：进入 blocked 或 waiting

### 6.2 AttemptRecord

每批工具结果会记录一次 attempt：

```text
attempt_id
step_id
action_intent
tool_call_ids
evidence_refs
status
failure_type
failure_reason
```

它回答“这一步为什么变成现在这样”。

### 6.3 ChangeSet

如果工具结果携带 `metadata.change_evidence`，TaskController 会生成 ChangeSet：

```text
affected_paths
before_hashes
after_hashes
diff_summary
status: pending | verified | failed | revert_required
verification_refs
```

这为连续验证失败后的 `propose_revert` 提供证据。

## 7. 每轮模型调用前：把任务状态注入上下文

在每次调用模型前，AgentLoop 会更新：

```python
current_context.current_task = task_controller.render_context(task)
current_context.task_signal = task_controller.control_signal(task)
```

`render_context()` 会注入：

- 当前目标
- 当前 mode
- 当前 phase
- mode guidance
- planning discovery facts
- relevant files
- risks
- verification hints
- open questions
- 步骤列表
- 当前步骤的 acceptance 和 verification_hint
- next_action
- recent_error_code
- rollback_required

示例：

```markdown
## Current Task
Goal: 实现两阶段 plan
Mode: plan
Phase: acting
Mode guidance: Plan mode: follow the generated plan step by step...

Planning facts:
- TaskController renders task context

Relevant files:
- src/codepilot/core/task_control/controller.py

Steps:
- [in_progress] 更新 TaskController 上下文 progress=evidence_collected evidence=tool:read_1
  - Kind: edit; Acceptance: 上下文包含 discovery facts; Verification hint: python -m pytest ...

Current step: 更新 TaskController 上下文
When this step's acceptance criteria are satisfied, call `complete_task_step` ...
```

`control_signal()` 是给上下文、恢复和审计用的轻量结构，其中包含统一 planning 节点：

```json
{
  "mode": "plan",
  "planning": {
    "phase": "execution",
    "source": "llm_with_discovery",
    "budget": {...},
    "discovery": {...},
    "fallback_reason": null
  },
  "current_step_title": "...",
  "next_action": "...",
  "recent_failure_type": "...",
  "rollback_required": false
}
```

## 8. 模型怎么决定做什么

框架不直接替模型选择业务动作。模型看到：

- 用户请求
- 历史消息
- 系统提示词
- 当前任务上下文
- 可用工具
- 上一轮工具结果

然后模型按 ReAct 方式输出：

```text
普通回答
或
ToolCall(...)
```

任务控制影响模型决策的方式是“提示和约束”：

- 当前步骤告诉模型此刻应该聚焦哪里。
- acceptance 和 verification_hint 告诉模型怎样算完成。
- discovery facts 降低 plan 模式瞎猜概率。
- next_action 和 recent_failure_type 给失败后的修复方向。
- read mode guidance 禁止修改。
- completion steering 会在修改后缺少验证时提醒模型继续验证。

真正执行工具前，仍由权限和工具层决定能否执行。

## 9. 工具执行后：事实进入任务控制

AgentLoop 执行工具后会做三件事：

```text
1. RunState.collect_tool_results(tool_results)
2. TaskController.after_tool_results(task, run_state, tool_results)
3. 发出 task_step_updated 和 task_decision 事件
```

`RunState` 负责运行事实，例如：

- 工具调用次数
- workspace 是否变化
- 是否有新鲜验证通过
- 重复调用情况

`TaskController` 负责任务语义状态，例如：

- 当前步骤是否有进展
- 是否验证失败
- 是否需要修复
- 是否需要重规划
- 是否应该等待用户
- 是否进入完成阶段

### 9.1 工具结果决策优先级

`after_tool_results()` 的核心优先级是：

```text
无工具结果
  -> continue

cancelled
  -> stop

approval_required
  -> block current step, phase=waiting, wait_approval

denied
  -> block current step, replan(permission_denied)

tool unavailable
  -> block current step, phase=waiting, stop(tool_unavailable)

non-verification tool error
  -> record failure, repair(tool_error)

verification failed
  -> record failure, mark latest changes failed
  -> failure_count < 2: repair
  -> failure_count >= 2 and has change sets: propose_revert
  -> failure_count >= 2 and replan budget available: replan
  -> replan budget exhausted: stop

verification passed
  -> complete verified step, mark change sets verified

complete_task_step signal
  -> complete current step

ordinary success
  -> update progress from read/change facts
```

最后如果所有步骤完成：

```text
finish(all_steps_completed)
```

否则：

```text
continue(next_step)
```

### 9.2 什么算进展

`_result_has_progress()` 和 `_update_progress_from_result()` 只做事实判断：

- `verification.status == "passed"` 是验证进展。
- `workspace_changed is True` 是修改进展。
- read/search/find/glob 类工具成功是调查进展。
- write/edit 类工具只有在 workspace changed 时才算实质进展。

框架不判断业务语义是否“优雅”，只判断工具事实是否支持任务推进。

## 10. 验证失败后怎么重试

验证解释在 `task_control/verifier.py`。

如果工具结果带：

```json
{
  "verification": {
    "status": "failed",
    "command": "...",
    "exit_code": 1,
    "summary": "..."
  }
}
```

TaskController 会：

1. 记录当前步骤失败。
2. 记录失败命令和摘要。
3. 标记最近 ChangeSet 为 `failed`。
4. 设置 `next_action`，提示模型读取失败断言和相关调用链。
5. 返回 `repair` 或 `replan`。

第一次验证失败通常是：

```text
decision=repair
```

同一步失败次数达到 2 时：

- 如果已有 ChangeSet，优先 `propose_revert`。
- 如果没有 ChangeSet 且还没到 `max_task_replans_per_run`，执行局部 replan。
- 如果 replan 次数耗尽，当前步骤 blocked，等待用户。

当前实现里的 replan 是局部的：

```text
保留已完成步骤
把当前步骤改成“根据最新失败证据调整方案”
追加“重新运行相关验证”
保留失败 evidence_refs
```

这是学习项目里的轻量重规划，不是复杂的计划树。

## 11. 是否继续、是否结束

工具后决策只解决“这一批工具结果之后怎么走”。当一轮模型没有更多工具调用后，AgentLoop 会进入完成门控：

```python
completion = task_controller.check_completion(task, run_state)
```

完成门控在 `task_control/stop.py`。

判断顺序：

```text
1. 有 blocked steps
   -> not satisfied, reason=blocked_steps, can_continue=False

2. workspace changed 但没有 fresh_verification_passed
   -> not satisfied, reason=modified_without_fresh_verification
   -> 第一次 can_continue=True，注入 completion steering
   -> 第二次 unverified=True，停止并返回未验证风险

3. 还有 pending 或 in_progress steps
   -> not satisfied, reason=incomplete_steps

4. 无阻塞、无未完成、验证满足
   -> satisfied=True, reason=all_steps_completed
```

如果缺少新鲜验证但还允许继续，AgentLoop 会注入一条用户形态的 steering message：

```text
工作区已经发生修改，但当前没有与最新工作区状态一致的成功验证。
请运行最相关的测试或检查；如果环境无法验证，请明确记录原因和剩余风险。
```

这就是“完成门控”：模型可以想结束，但框架会在关键证据缺失时把它拉回验证。

## 12. 任务怎么存储和恢复

任务恢复在 `src/codepilot/sessions/history/task_recovery.py`。

它不是长期记忆，也不会沉淀项目事实。它只保存当前 session 未完成任务的恢复投影。

### 12.1 何时开始保存

Session 在 run 开始前调用：

```text
TaskRecoveryStore.begin_task(text, run_id)
```

如果新请求 goal 和当前投影相同，会刷新 `source_run_id` 和 `updated_at`，不丢弃已有进度。

### 12.2 Run 结束后保存什么

Run 结束后，`TaskRecoveryStore.update_from_result()` 会从 `TaskSummary.control_signal` 生成 projection：

```json
{
  "goal": "...",
  "task_mode": "plan",
  "planning": {
    "phase": "execution",
    "source": "llm_with_discovery",
    "budget": {...},
    "discovery": {...},
    "fallback_reason": null
  },
  "task_progress": {
    "completed_steps": [...],
    "pending_steps": [...],
    "blocked_steps": [...],
    "completion_satisfied": false,
    "completion_reason": "...",
    "step_details": {...}
  },
  "next_action": "...",
  "source_run_id": "...",
  "created_at": "...",
  "updated_at": "..."
}
```

注意：旧的顶层 `plan_source` 不再保存。计划来源只从 `planning.source` 派生。

### 12.3 下次如何恢复

恢复规则：

```text
有 task_progress:
  -> TaskController.build_task_state_from_recovery_projection()
  -> planning.phase="recovered", planning.source="recovered"
  -> 不重新 discovery，不重新 synthesis

有 planning.discovery 但无 task_progress:
  -> PlanningBootstrap 复用 discovery report
  -> 重新 synthesis

只有 goal:
  -> 重新 discovery

projection 已完成:
  -> active_projection() 返回 None，不恢复
```

如果用户在恢复前显式切换 `/mode plan`，当前 session mode 优先于 projection mode。

## 13. 事件和审计

任务控制相关事件包括：

```text
planning_discovery_started
planning_discovery_step
planning_discovery_completed
planning_synthesis_started
planning_synthesis_completed
task_plan_created
task_step_updated
task_decision
completion_checked
task_recovery_updated
task_recovery_warning
```

事件 payload 里使用统一 planning 节点：

```json
{
  "mode": "plan",
  "planning": {
    "phase": "...",
    "source": "...",
    "budget": {...},
    "discovery": {...},
    "fallback_reason": "..."
  }
}
```

审计报告在 `src/codepilot/observability/audit.py` 中汇总：

- planning phase
- plan source
- discovery status
- facts count
- relevant files
- risks
- verification hints
- evidence refs
- budget usage
- fallback reason

这样后续可以解释“为什么生成这个计划，计划依据是什么，失败后为什么继续或停止”。

## 14. 与其他模块的边界

### 14.1 与 AgentLoop

AgentLoop 是执行编排者：

- 创建 TaskController。
- 在 plan 模式调用 PlanningBootstrap。
- 每轮模型调用前注入 task context。
- 工具执行后把结果交给 TaskController。
- 根据 TaskController 和 run_decisions 的结果继续、停止或等待。

TaskController 不直接调用模型，也不直接执行工具。python -m codepilot.evaluation experiment planning --eval-id exp-planning --repeat 2

### 14.2 与 ToolCallCoordinator

ToolCallCoordinator 负责工具执行、权限、before/after hooks 和事件。

TaskController 只消费工具结果：

```text
success | error | denied | approval_required | cancelled
```

它不能绕过权限，也不能因为计划需要某个动作就自动放行。

### 14.3 与 RunState

RunState 记录运行事实：

- workspace 是否改变
- 是否有新鲜验证通过
- 工具调用计数
- 模型尝试次数

TaskState 记录任务语义：

- goal
- steps
- current step
- phase
- next_action
- attempts/change_sets/replans

TaskController 读取 RunState，但不复制 RunState。

### 14.4 与记忆系统

当前任务恢复投影不是长期记忆。

本轮设计暂时只支持任务恢复和审计，不自动把项目事实写入记忆。未来可以考虑沉淀失败经验、用户纠正，但不要把 discovery facts 自动当项目事实长期保存。

## 15. 典型流程

### 15.1 read 模式

```text
用户: 帮我解释这个模块怎么工作
task_mode=read
runtime 强制 read-only 权限
TaskController 创建默认只读步骤
模型读取文件、搜索代码
complete_task_step 或最终回答
CompletionGate 检查无阻塞、无未完成步骤
结束并生成 TaskSummary
```

### 15.2 edit 模式

```text
用户: 修复一个小 bug
task_mode=edit
不调用 planner
TaskController 创建“完成当前请求”步骤
模型读取代码
模型修改文件
RunState.workspace_changed=True
CompletionGate 要求 fresh verification
模型运行测试
verification passed
TaskController 标记步骤 verified/completed
结束
```

### 15.3 plan 模式

```text
用户: 重构任务控制逻辑
task_mode=plan
PlanningBootstrap 检查恢复投影
无 active recovery
PlanningDiscovery 用只读工具收集 facts/relevant_files/risks
TaskPlanner 基于 discovery report 生成最多 6 步计划
TaskController 初始化 TaskState
AgentLoop 进入普通 ReAct 执行
每步根据工具事实更新状态
验证失败时 repair 或 replan
完成门控通过后结束
恢复投影和审计记录 planning 节点
```

### 15.4 验证连续失败

```text
edit 产生 ChangeSet
pytest failed
decision=repair
模型修复后再跑 pytest
pytest failed again
failure_count >= 2 且有 ChangeSet
decision=propose_revert
task.phase=waiting
next_action=报告可能需要撤销的变更并等待用户确认
```

如果没有 ChangeSet，且未达到 replan 上限：

```text
decision=replan
当前步骤改为“根据最新失败证据调整方案”
追加“重新运行相关验证”
继续执行
```

## 16. 当前实现的学习型边界

当前设计故意保持轻量：

- 没有复杂 FSM 框架。
- 没有多 planner 模型。
- 没有计划树、依赖图或优先级队列。
- plan discovery 是 scratch，不污染主对话历史。
- 重规划是局部替换，不重建整棵任务图。
- 完成门控只检查关键事实，不做业务正确性判定。

它的价值在于给 Coding Agent 提供一条可解释的执行证据链：

```text
为什么做这一步
依据哪些事实生成计划
工具实际做了什么
验证为什么失败或通过
失败后为什么修复、重规划或建议回滚
为什么现在可以结束或不能结束
```

后续扩展可以围绕这个契约继续做，而不是新增第二套任务控制口径。
