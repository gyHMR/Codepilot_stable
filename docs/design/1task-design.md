# Codepilot 任务规划模块设计与实现

## 1. 为什么需要任务规划模块？

在 Coding Agent 的实际使用中，我们发现了一个核心问题：**Agent 无法知道任务进行到哪一步了**。

想象这个场景：
- 用户说："修复这个 bug 并运行测试"
- Agent 修改了代码
- Agent 说："已经修复完成"
- 但实际上没有运行测试

问题出在哪里？**模型停止调用工具 ≠ 任务完成**。

模型可能因为以下原因停止：
- 认为已经完成（但其实没有验证）
- 遇到困难，不知道下一步做什么
- 权限被拒绝，无法继续
- 上下文太长，忘记了原始目标

旧方案的问题是：**完全依赖模型自己判断是否完成**，没有外部验证机制。

## 2. 旧方案的问题

### 2.1 没有显式目标和步骤状态

系统无法回答：
- 用户最终想完成什么？
- 当前正在处理哪一步？
- 哪些步骤已经完成？
- 哪些步骤被阻塞？
- 下一步为什么是这个操作？

### 2.2 完成判断完全依赖模型

```python
# 旧方案：模型不返回 ToolCall 时就认为完成
if not assistant.tool_calls:
    result.status = "completed"
    result.stop_reason = "final_answer"
```

但模型没有继续调用工具，可能是因为：
- 修改代码后没有验证（最常见的问题）
- 测试失败后直接总结
- 权限拒绝导致关键动作未执行
- 工作区变化使旧验证结果失效
- 用户明确要求的某个子目标仍未完成

### 2.3 工具结果没有统一转化为任务反馈

工具结果只是作为消息返回给模型：

```
ToolResult → 进入消息历史 → 模型自行理解
```

Runtime 没有统一判断：
- 当前步骤是否完成？
- 是否需要修复？
- 是否应该局部重新规划？
- 是否可以结束任务？

### 2.4 失败后没有自动恢复机制

测试失败后，模型可能会：
- 忽略失败，继续下一步
- 重复尝试同样的失败方案
- 放弃任务，直接报告失败

没有机制确保：**失败 → 修复 → 重新验证** 的闭环。

## 3. 新设计的核心思想

新设计遵循一个核心原则：**模型负责语义，Runtime 负责边界**。

### 3.1 分离职责

```
模型擅长：
├── 理解用户意图
├── 拆解任务
├── 判断代码含义
├── 提出修改方式
└── 根据失败日志形成修复方案

Runtime 擅长：
├── 判断工具是否真实执行
├── 判断权限是否允许
├── 检查工作区是否变化
├── 判断验证是否成功且仍然新鲜
├── 检测重复失败和预算耗尽
└── 执行完成门槛
```

### 3.2 证据绑定的动态计划

任务步骤不是一次性 Markdown 清单，而是**随工具执行动态更新**的状态机：

```
步骤完成 ≠ 模型声称完成
步骤完成 = 绑定工具结果、文件变化或验证证据
```

### 3.3 完成必须有证据

模型准备结束时，Runtime 必须检查：
- 修改后是否有验证？
- 验证是否成功？
- 验证是否对应当前工作区版本？
- 是否有未完成的步骤？

```
模型停止调用工具 ≠ 任务自动完成
```

## 4. 架构设计

### 4.1 核心组件

```python
# 数据结构
src/codepilot/core/task_state.py
├── TaskStep          # 单个步骤
├── TaskState         # 任务状态
├── ExecutionDecision # 执行决策
└── CompletionCheck   # 完成检查

# 控制器
src/codepilot/core/task_controller.py
└── TaskController    # 任务控制器（核心逻辑）
```

### 4.2 执行流程

```
用户请求
    ↓
初始化 TaskState
    ↓
ContextCompiler 注入任务上下文
    ↓
LLM 决定下一步动作
    ↓
权限检查 + 工具执行
    ↓
收集 ToolResult 证据
    ↓
TaskController 更新任务状态
    ↓
执行决策（继续/修复/重新规划/等待/完成/停止）
    ↓
CompletionGate 检查是否可以结束
    ↓
生成最终结果
```

## 5. 核心实现细节

### 5.1 TaskStep：带证据的步骤

```python
@dataclass
class TaskStep:
    id: str                          # 步骤唯一标识
    title: str                       # 步骤标题
    status: TaskStepStatus = "pending"  # pending/in_progress/completed/blocked
    evidence_refs: list[str] = field(default_factory=list)  # 证据引用
    failure_count: int = 0           # 失败次数
    note: str | None = None          # 备注信息
```

**关键点**：
- `evidence_refs` 绑定实际的工具调用 ID、文件路径、验证 ID
- `failure_count` 用于触发重新规划
- `note` 记录阻塞原因或最新发现

### 5.2 TaskState：任务全局状态

```python
@dataclass
class TaskState:
    task_id: str                     # 任务唯一标识
    goal: str                        # 任务目标
    constraints: list[str]           # 约束条件
    acceptance_criteria: list[str]   # 验收标准
    steps: list[TaskStep]            # 任务步骤列表
    current_step_id: str | None      # 当前正在执行的步骤 ID
    phase: TaskPhase                 # understanding/acting/verifying/waiting/finished
    next_action: str | None          # 下一步动作描述
    replan_count: int = 0            # 已重新规划次数
    max_replans_per_run: int = 2     # 单次运行最大重新规划次数
    completion_satisfied: bool = False  # 任务是否满足完成条件
    completion_reason: str = ""      # 完成/未完成原因
```

**设计亮点**：
- `TaskState` 只保存任务语义状态，不复制 `RunState` 的执行计数器
- `phase` 跟踪任务当前阶段（理解/执行/验证/等待/完成）
- `replan_count` 限制重新规划次数，防止无限循环

### 5.3 TaskController：核心控制逻辑

```python
class TaskController:
    def initialize(
        self,
        prompts,
        *,
        proposed_steps=None,
        task_recovery_projection=None,
    ) -> TaskState:
        """初始化任务状态"""
        goal = _goal_from_prompts(prompts)  # 从用户消息提取目标
        steps = self._normalize_steps(proposed_steps or ["完成当前请求"])
        # 归一化：去重、截断、限制数量
        return TaskState(goal=goal, steps=steps, ...)
    
    def after_tool_results(self, task, run, results) -> ExecutionDecision:
        """工具执行后更新任务状态并返回决策"""
        # 1. 检查取消、审批、权限拒绝
        # 2. 检查验证失败
        # 3. 检查是否有进展
        # 4. 返回决策（continue/repair/replan/stop/finish）
    
    def check_completion(self, task, run) -> CompletionCheck:
        """检查任务是否完成"""
        # 1. 检查是否有阻塞步骤
        # 2. 检查修改后是否有新鲜验证
        # 3. 检查是否有未完成步骤
        # 4. 返回完成检查结果
```

### 5.4 步骤归一化

```python
def _normalize_steps(self, raw_steps: Iterable[str]) -> list[TaskStep]:
    seen: set[str] = set()
    steps: list[TaskStep] = []
    for raw in raw_steps:
        title = " ".join(str(raw).strip().split())
        if not title or title in seen:  # 去重
            continue
        seen.add(title)
        steps.append(TaskStep(
            id=f"step_{len(steps) + 1}",
            title=title[:80],  # 截断标题
        ))
        if len(steps) >= 6:  # 最多 6 步
            break
    if not steps:
        steps.append(TaskStep(id="step_1", title="完成当前请求"))
    return steps
```

**关键点**：
- 最多 6 个步骤（保持轻量）
- 自动去重
- 标题限长 80 字符
- 空步骤生成默认步骤

### 5.5 工具结果到任务反馈

```python
def after_tool_results(self, task, run, results) -> ExecutionDecision:
    # 1. 取消 → 停止
    if any(result.status == "cancelled" for result in results):
        return ExecutionDecision("stop", "cancelled")
    
    # 2. 需要审批 → 等待
    if any(result.status == "approval_required" for result in results):
        self._block_current_step(task, "等待工具审批")
        return ExecutionDecision("wait_approval", "approval_required")
    
    # 3. 权限拒绝 → 重新规划
    if any(result.status == "denied" for result in results):
        self._block_current_step(task, "工具权限拒绝")
        return ExecutionDecision("replan", "permission_denied")
    
    # 4. 验证失败 → 修复或重新规划
    if self._has_failed_verification(results):
        step = self._current_step(task)
        step.failure_count += 1
        if step.failure_count >= 2:
            # 连续失败 2 次 → 重新规划
            self._replan_after_failure(task, results)
            return ExecutionDecision("replan", "repeated_step_failure")
        return ExecutionDecision("repair", "verification_failed")
    
    # 5. 有进展 → 推进步骤
    for result in results:
        if self._result_has_progress(result):
            step = self._current_step(task)
            step.status = "completed"
            step.evidence_refs.extend(_evidence_refs([result]))
            self._advance(task)
    
    # 6. 所有步骤完成 → 完成
    if all(step.status == "completed" for step in task.steps):
        return ExecutionDecision("finish", "all_steps_completed")
    
    return ExecutionDecision("continue", "next_step")
```

### 5.6 判断工具结果是否有进展

```python
def _result_has_progress(self, result: ToolResultMessage) -> bool:
    if result.status != "success":
        return False
    
    # 验证通过 → 有进展
    if isinstance(result.verification, dict):
        return result.verification.get("status") == "passed"
    
    # 工作区变化 → 有进展
    if result.workspace_changed is True:
        return True
    
    # 只读工具成功 → 有进展（理解类步骤）
    name = result.tool_name.lower()
    if any(marker in name for marker in ("read", "grep", "find", "glob")):
        return True
    
    # 写入工具无变化 → 无进展
    if any(marker in name for marker in ("write", "edit")):
        return result.workspace_changed is True
    
    return False
```

**设计亮点**：
- 不猜测代码语义，只判断执行事实
- 不同类型工具有不同的进展判断标准
- 验证结果是最高优先级的进展信号

### 5.7 CompletionGate：完成门槛

```python
def check_completion(self, task, run) -> CompletionCheck:
    # 1. 有阻塞步骤 → 不能完成
    if any(step.status == "blocked" for step in task.steps):
        return CompletionCheck(
            satisfied=False,
            reason="blocked_steps",
            missing=["unblocked_steps"],
            can_continue=False,
        )
    
    # 2. 修改后无新鲜验证 → 不能完成
    if run.workspace_changed and not run.fresh_verification_passed:
        can_continue = task.completion_prompt_count == 0
        return CompletionCheck(
            satisfied=False,
            reason="modified_without_fresh_verification",
            missing=["fresh_verification"],
            can_continue=can_continue,
            unverified=not can_continue,
        )
    
    # 3. 有未完成步骤 → 不能完成
    incomplete = [step.title for step in task.steps 
                  if step.status not in {"completed", "blocked"}]
    if incomplete:
        return CompletionCheck(
            satisfied=False,
            reason="incomplete_steps",
            missing=incomplete,
            can_continue=False,
        )
    
    # 4. 所有条件满足 → 可以完成
    return CompletionCheck(satisfied=True, reason="all_steps_completed")
```

**关键设计**：
- 修改后必须有新鲜验证（防止"改了代码但没测试"）
- `can_continue` 控制是否可以继续尝试
- `completion_prompt_count` 防止重复提示

### 5.8 重新规划机制

```python
def _replan_after_failure(self, task, results):
    """失败后重新规划：保留已完成步骤，替换当前和后续步骤"""
    task.replan_count += 1
    current = self._current_step(task)
    
    # 保留已完成步骤，替换当前步骤
    current.title = "根据最新失败证据调整方案"
    current.status = "in_progress"
    current.failure_count = 0
    current.evidence_refs.extend(_evidence_refs(results))
    
    # 只保留当前步骤和新步骤
    current_index = task.steps.index(current)
    task.steps = [
        *task.steps[:current_index + 1],  # 保留已完成的
        TaskStep(id=f"step_{current_index + 2}", title="重新运行相关验证"),
    ]
```

**设计亮点**：
- 局部重新规划，不丢弃已完成的步骤
- 保留失败证据，用于后续分析
- 限制重新规划次数（默认 2 次），防止无限循环

### 5.9 会话恢复支持

```python
def build_task_state_from_recovery_projection(
    prompts,
    task_recovery_projection,
) -> TaskState | None:
    """从恢复的 task recovery projection 中重建 TaskState"""
    progress = task_recovery_projection.get("task_progress")
    if not isinstance(progress, Mapping):
        return None
    
    steps: list[TaskStep] = []
    
    # 重建已完成步骤
    for title in progress.get("completed_steps", []):
        steps.append(TaskStep(id=f"step_{len(steps)+1}", title=title, status="completed"))
    
    # 重建阻塞步骤
    for title in progress.get("blocked_steps", []):
        steps.append(TaskStep(id=f"step_{len(steps)+1}", title=title, status="blocked"))
    
    # 重建待处理步骤
    for title in progress.get("pending_steps", []):
        steps.append(TaskStep(id=f"step_{len(steps)+1}", title=title, status="pending"))
    
    return TaskState(goal=goal, steps=steps, ...)
```

**关键点**：
- 从 `TaskMemory` 投影重建任务状态
- 保留已完成步骤的证据
- 支持跨会话恢复

## 6. 与 AgentLoop 的集成

### 6.1 初始化

```python
async def _run_loop(...):
    task_controller = TaskController()
    task = task_controller.initialize(
        current_context.messages,
        task_recovery_projection=current_context.task_recovery_projection,
    )
    # 发送任务计划创建事件
    await emitter.emit({"type": "task_plan_created", "task": ...})
```

### 6.2 每轮模型调用前

```python
# 注入任务上下文到系统提示词
current_context.current_task = task_controller.render_context(task)
```

渲染后的上下文：

```markdown
## Current Task
Goal: 修复配置加载失败并运行相关测试
Phase: acting

Steps:
- [completed] 定位配置加载调用链
- [in_progress] 修改错误处理
- [pending] 运行相关测试

Current step: 修改错误处理
Next action: 检查 config.py 中异常转换逻辑
Constraints:
- 保持现有公共入口一致
```

### 6.3 工具执行后

```python
# 收集工具结果
state.collect_tool_results(tool_results)

# 更新任务状态并获取决策
decision = task_controller.after_tool_results(task, state, tool_results)

# 发送任务状态更新事件
await emitter.emit({"type": "task_step_updated", "task": ...})
await emitter.emit({"type": "task_decision", "decision": ...})

# 处理决策
if decision.action == "stop":
    return await _stop_with_error(...)
```

### 6.4 模型准备结束时

```python
# 检查是否可以完成
completion = task_controller.check_completion(task, state)

# 发送完成检查事件
await emitter.emit({"type": "completion_checked", "completion": ...})

# 如果不能完成但可以继续，注入引导消息
if not completion.satisfied and completion.can_continue:
    pending_messages = [task_controller.completion_steering(completion)]
    continue
```

引导消息示例：

```
工作区已经发生修改，但当前没有与最新工作区状态一致的成功验证。
请运行最相关的测试或检查；如果环境无法验证，请明确记录原因和剩余风险。
```

### 6.5 Run 结束时

```python
return await _finish_run(
    emitter,
    state.result(
        ...,
        task=task_controller.summarize(task),  # 保存任务摘要
    ),
)
```

## 7. 设计亮点总结

### 7.1 证据绑定的动态计划

```
传统方案：一次性生成计划 → 执行 → 结束
新方案：初始计划 → 每步更新 → 绑定证据 → 动态调整
```

计划不是静态的 Markdown 清单，而是随工具执行动态更新的状态机。

### 7.2 模型行动与 Harness 完成判断分离

```python
# 模型负责语义
"我应该读取这个文件" → 模型决定
"我应该修改这行代码" → 模型决定
"我应该运行测试" → 模型决定

# Runtime 负责边界
"文件是否真的被读取了" → Runtime 判断
"代码是否真的被修改了" → Runtime 判断
"测试是否真的通过了" → Runtime 判断
"任务是否真的完成了" → Runtime 判断
```

### 7.3 完成必须有证据

```python
# 不允许的情况：
# 1. 修改代码后没有验证
# 2. 验证失败后直接结束
# 3. 有未完成步骤就报告完成

# 强制要求：
# 修改后必须有新鲜验证
# 验证必须对应当前工作区版本
# 所有步骤必须完成或明确阻塞
```

### 7.4 局部重新规划

```python
# 不是清空整个计划
# 而是保留已完成步骤，替换当前和后续步骤

# 触发条件：
# 1. 同一步连续失败 2 次
# 2. 关键前提被工具证据否定
# 3. 权限拒绝使原路径不可执行

# 限制：
# 最多重新规划 2 次
# 超过后停止并报告
```

### 7.5 不猜测代码语义

```python
# Runtime 可以判断：
"文件是否修改" → 检查 workspace_changed
"命令是否成功" → 检查 exit_code
"测试是否通过" → 检查 verification.status

# Runtime 不直接判断：
"根因是否完全正确" → 模型和测试共同判断
"实现是否优雅" → 模型判断
"业务逻辑是否满足" → 外部测试判断
```

### 7.6 会话恢复支持

```python
# 从 TaskMemory 投影重建任务状态
# 保留已完成步骤的证据
# 恢复 next_action 和 blocked 原因
# 支持跨会话继续任务
```

## 8. 决策规则详解

### 8.1 决策优先级

```python
def decide(task, run, tool_results):
    # 1. 最高优先级：取消
    if has_cancelled(tool_results):
        return stop("cancelled")
    
    # 2. 需要审批
    if needs_approval(tool_results):
        return wait_approval("approval_required")
    
    # 3. 预算耗尽
    if run_budget_exhausted(run):
        return stop("budget_exhausted")
    
    # 4. 验证失败
    if verification_failed(tool_results):
        if current_step_failed_repeatedly(task):
            return replan("repeated_step_failure")
        return repair("verification_failed")
    
    # 5. 当前步骤完成
    if current_step_completed(task):
        advance_to_next_step(task)
    
    # 6. 所有步骤完成
    if all_steps_completed(task):
        return finish("all_steps_completed")
    
    # 7. 继续执行
    return continue_("next_step")
```

### 8.2 各决策的含义

| 决策 | 含义 | 触发条件 |
|------|------|----------|
| `continue` | 继续执行下一步 | 工具成功，还有未完成步骤 |
| `repair` | 修复当前步骤 | 验证失败（第一次） |
| `replan` | 重新规划 | 连续失败 2 次，或权限拒绝 |
| `wait_approval` | 等待用户审批 | 工具需要审批 |
| `finish` | 完成任务 | 所有步骤完成，验证通过 |
| `stop` | 停止执行 | 预算耗尽，或重新规划超限 |

## 9. 与其他模块的边界

### 9.1 与 RunState

```
RunState（执行事实）：
├── 调用次数
├── 文件变化
├── 验证结果
└── 重复调用

TaskState（任务语义）：
├── 目标
├── 步骤
├── 当前进度
└── 下一步
```

`TaskController` 读取 `RunState`，但不复制其计数器。

### 9.2 与 TaskMemory

```
TaskState（当前 Run 内的活跃状态）
    ↓ Run 结束时投影
TaskMemory（跨 Run 恢复所需的任务摘要）
```

投影内容：
- goal
- constraints
- confirmed_findings
- blocked_on
- next_action

### 9.3 与 ContextCompiler

`ContextCompiler` 负责把任务状态组织给模型：
- 当前任务目标
- 当前步骤
- 最新工具证据
- 活跃文件
- 相关记忆

`TaskController` 不直接拼接 System Prompt。

### 9.4 与工具权限

权限层决定某个动作能否执行，`TaskController` 只消费结果：
- `allow` 后执行成功
- `denied`
- `approval_required`
- `cancelled`

它不能绕过权限，也不能因为计划需要某一步就自动放行。

## 10. 学习建议

1. **从 task_state.py 开始**：理解数据结构是理解整个系统的基础
2. **重点阅读 task_controller.py**：这是任务规划的核心逻辑
3. **理解决策规则**：`after_tool_results` 的决策逻辑是核心
4. **看测试用例**：test_task_planning.py 展示了各种使用场景
5. **查看 agent_loop.py**：理解任务规划如何与主循环集成

## 11. 后续扩展方向

当前设计有意保持简单，后续可以扩展：
- 子步骤树
- 步骤依赖图
- 优先级队列
- 负责人
- 预计时间
- 百分比进度
- 自动回滚
- 多 Agent 协作

但核心的**证据绑定 + 完成门槛 + 局部重新规划**设计是稳定的。

## 12. 实际效果示例

### 12.1 正常修复流程

```
用户：修复 divide() 除零错误并运行测试

TaskState:
1. 定位 divide 实现
2. 修改除零处理
3. 运行相关测试

read 成功 → step 1 completed

edit 成功，workspace_changed=true → step 2 completed

pytest passed → step 3 completed

CompletionGate:
修改存在 + 最新验证通过 + 无阻塞步骤
→ finish
```

### 12.2 验证失败后修复

```
edit 成功 → 进入验证

pytest failed → 当前验证步骤 failure_count=1 → decision=repair

模型读取失败日志并再次 edit → 旧验证失效

pytest passed → verification fresh → finish
```

### 12.3 重复失败后重新规划

```
同一验证步骤连续失败两次
→ decision=replan
→ 保留已经完成的定位步骤
→ 替换当前修改和验证步骤

再次失败且达到 replan 上限
→ stop
→ 保存 TaskMemory.next_action 和失败证据
```

### 12.4 权限阻塞

```
模型请求未知 Shell 命令
→ PermissionPolicy 返回 approval_required
→ TaskState.phase=waiting
→ RunStatus=waiting_approval

用户批准 → 恢复原 ToolCall → 更新当前步骤
```
