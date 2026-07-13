# Core 重构目标设计

## 文档状态

本文是 Codepilot Core 重构的目标设计规格，描述目标边界、状态模型、决策协议、执行循环、计划治理、错误与事件语义、迁移顺序和验收标准。

本文描述目标设计，不表示当前代码已经实现。重构期间，现有 `1core-design.md` 继续用于理解旧设计背景；涉及 Core 重构目标时，以本文为准。重构完成后，应合并仍然有效的内容并删除重复说明，避免长期保留两套 Core 设计口径。

本文建立在当前 Sessions v2 和 Runtime 重构方向上，不重新设计 Context 与 Memory。Core 重构完成并稳定后，再以本文确定的契约为前提讨论 Context 和 Memory。

## 1. 重构动机

当前 Core 已经不只是模型与工具之间的简单循环。它同时承担了：

- Run 机械计数与重复调用限制。
- Plan 的创建、更新、审批和完成处理。
- 模型重试与退避。
- Tool Approval 的恢复。
- Context preflight 和未闭合 ToolCall 修补。
- Workspace、Verification 和失败信号提取。
- Boundary 提交及 Tools checkpoint 收集。
- Event envelope、序号和审计事件记录。
- 最终答复门控与合成控制提示。

这些职责分散在 `runner.py`、`RunState`、`RunGuard`、`PlanStateManager`、Plan Tool、Runtime Coordinator 和 Sessions Adapter 中。同一个语义经常存在多个判断入口或多个事实来源，例如：

- `status + stop_reason` 共同推断等待类型。
- Plan 同时存在于 Core、Runtime Manager、Tool Service 和 Sessions `core_state`。
- 完成判断同时存在于模型最终答复、RunGuard、Plan closeout 和 runner 分支。
- 模型重试、deadline、cancellation 和资源释放跨越 Core 与 Runtime。
- Boundary 提交由 Core 直接了解 Tools recovery state 和 Sessions 恢复细节。

继续在现有 runner 上添加条件，会使 Core 从“任务推进内核”变成第二个 Agent 或第二个 Runtime。因此本次重构的重点不是增加更多控制能力，而是缩小 Core 的权力并建立单一决策链。

## 2. Core 的定位

Core 是一次 Agent Run 内的确定性任务推进内核。

> Core 维护任务语义状态和可观察运行事实，消费模型、工具与用户观察，应用确定性策略，并产生下一步决策。

Core 负责：

- 维护当前任务目标、阻塞、计划和执行事实。
- 将模型、工具、用户输入和外部命令归一化为 Observation。
- 通过 Reducer 产生新的 CoreState 和领域事件。
- 判断下一步调用模型、执行工具、等待还是终止。
- 判断当前是否需要恢复、验证、重规划或完成。
- 在稳定边界提交 Core 状态、消息增量和领域事件。
- 保证已经持久化的 ToolCall 最终得到 ToolResult。

Core 不负责：

- 代替模型理解代码语义、选择文件或编写修复方案。
- 管理 Runtime task、deadline、资源范围或强制取消。
- 管理模型 provider retry/backoff 和连接生命周期。
- 执行 Sessions prepare/recovery/terminal commit。
- 解释或持久化 Tools、Context、Workspace component checkpoint。
- 选择 Context 分层内容、执行压缩或访问 Memory。
- 分配 Runtime event ID、时间戳和序号。
- 判断工具权限、安全策略、副作用幂等和回滚方式。

## 3. 与其他模块的边界

| 模块 | 拥有的职责 | Core 与其交互方式 |
|---|---|---|
| Model/LLM | 模型调用、provider 错误归一化、重试与退避 | Core 通过 ModelPort 请求一次模型动作 |
| Tools | 注册、权限、审批、执行、副作用、工具恢复 | Core 通过 ToolPort 执行已允许的调用并消费 ToolResult |
| Context | 上下文分层、选择、压缩、证据投影 | Core 只提供 purpose、directive 和不透明 seed |
| Memory | 写入、治理、召回、冲突与重复管理 | Core 不直接依赖；由 Context 或 Runtime 组合使用 |
| Sessions | Session/Run/Checkpoint 的持久化权威 | Runtime Boundary Adapter 将 CoreBoundary 转换为提交请求 |
| Observability | Trace、审计和评测投影 | 消费已提交事件和结果，不参与 Core 决策 |
| Runtime | Run 准备、执行资源、取消、恢复、终态提交 | 调用一次 `run_core(input, ports)` 并处理其结果 |
| Interfaces | 用户输入、等待交互、结果展示 | 只通过 Runtime，不直接调用 Core 内部对象 |

依赖方向保持为：

```text
protocols -> llm/tools -> core -> sessions/observability
                                     -> extensions -> runtime -> interfaces
```

Core 不允许导入 `sessions`、`observability`、`extensions`、`runtime` 或 `interfaces`。

## 4. 总体结构

```text
Runtime
  -> run_core(CoreRunInput, CorePorts)
       -> Core Driver
            -> normalize Observation
            -> Core Reducer
            -> Core Policy
            -> CoreDecision
            -> ModelPort / ToolPort / wait / terminate
            -> CoreBoundary
  <- CoreOutcome
  -> Waiting 或 Terminal Commit
```

`CoreDecision` 是 Core 内部协议。Runtime 不逐条解释 Decision，也不执行 Agent Loop 分支。否则任务推进逻辑会从 Core 迁移到 Runtime，形成第二套控制器。

## 5. 对外契约

### 5.1 CoreRunInput

```text
CoreRunInput
├── run_id
├── entry: ModelEntry | ToolResultEntry
├── messages
├── state: CoreState | serialized CoreState
├── mode: read | build | plan
├── model: ModelDescriptor
├── limits: CoreLimits
└── context_seed: opaque
```

约束：

- Prompt、Continuation 和 Resume 最终都转换为统一 entry。
- Tool Approval 由 Runtime/Tools 完成恢复，Core 只接收最终 ToolResultEntry。
- 不再同时传递 `user_prompt` 与 UserMessage。
- 不再分别传递 `run_state` 与 `plan_state`。
- 不包含 ApprovalResponse、retry policy、deadline、event sequence 或 Runtime phase。

### 5.2 CorePorts

```text
CorePorts
├── model: ModelPort
├── tools: ToolPort | None
├── context: ContextPort
├── boundary: BoundaryPort
├── live_events: LiveEventSink | None
└── cancellation: CancellationProbe | None
```

Ports 是一次调用可使用的能力，不代表其生命周期归 Core 所有。Runtime 创建并释放具体资源。

`live_events` 只用于非权威实时反馈。CoreDomainEvent 不通过它持久化，而是随 CoreBoundary 原子提交。

为了让 `before_tools` 真正位于副作用之前，目标 ToolPort 需要区分无外部副作用的 batch preparation 与实际 execution：

```text
prepare_batch(tool_calls) -> prepared batch / approval challenge / rejected results
execute_prepared(batch_id) -> final ToolResults
```

Preparation 可以建立稳定 tool attempt 和内存恢复状态，但不能修改 workspace 或访问具有外部副作用的能力。Runtime Boundary Adapter 从同一个 ToolPort 读取不透明 checkpoint；CoreBoundary 不承载 Tools 私有恢复结构。

### 5.3 CoreBoundary

```text
CoreBoundary
├── kind
├── state: CoreState
├── new_messages
├── domain_events
└── wait: CoreWait | None
```

Boundary kind 只有：

```text
before_model
after_model
before_tools
after_tools
waiting
before_terminal
```

CoreBoundary 不包含：

- Tool recovery checkpoint。
- Context checkpoint。
- Workspace checkpoint。
- Sessions revision、commit ID 或 receipt。
- Runtime lifecycle phase。

Runtime Boundary Adapter 负责收集这些组件状态，并将通用 `waiting` 和 `before_terminal` 映射到 Sessions 的 WaitingState、phase 与 resume point。

### 5.4 CoreOutcome

```text
CoreOutcome
├── status: completed | waiting | failed | cancelled
├── reason: CoreReason
├── state: CoreState
├── new_messages
├── final_message
├── wait: CoreWait | None
├── usage
└── error
```

Outcome 不重复保存 counters、signals、plan、workspace effects 和 verification 等可从 CoreState 派生的字段。Runtime 可以为兼容接口构造投影。

### 5.5 CoreWait

```text
CoreWait
├── kind: tool_approval | user_input | plan_confirmation | continuation
├── reason: CoreReason
├── request_id
└── payload
```

Runtime 不再通过 `status + stop_reason` 推断等待类型。只有 Waiting Boundary 提交成功后，Core 才能返回 waiting outcome。

### 5.6 外部状态命令

计划批准、拒绝和放弃等外部操作通过窄入口处理：

```text
apply_core_command(state, command, context)
  -> CoreReduction(state, events, command_result)
```

Runtime 负责提交返回结果，但不能自行修改 PlanState。

## 6. CoreState

CoreState 是 Run-scoped 状态。一次 Run 可以跨 waiting/resume 持续存在；进入 terminal 后不再继续变化。

```text
CoreState
├── schema_version
├── task: TaskState
└── facts: RunFacts
```

CoreState 不保存消息、Context、Memory、Runtime phase、deadline、事件 envelope、usage 或组件 checkpoint。

### 6.1 TaskState

```text
TaskState
├── original_request
├── current_goal
├── status: active | blocked | satisfied | abandoned
├── plan: PlanState | None
└── blockers: tuple[TaskBlocker, ...]
```

`original_request` 是本 Run 的不可变任务语义快照，不替代 Messages 的会话内容权威。`current_goal` 可以在用户明确改变约束后更新，但每次变化必须来自 UserInputObservation。

TaskBlocker 只表达阻止任务推进的语义条件：

```text
user_input_required
tool_unavailable
verification_failed
plan_incomplete
replan_required
```

每个 blocker 包含 reason、evidence refs 和 recoverable。审批等待使用 CoreWait，不伪装成普通 blocker。

存在可由 Agent 自主处理的 recoverable blocker 时，Task 仍可保持 active，由 Recovery/ReplanPolicy 消解；只有必须依赖外部输入、关键能力永久缺失或自主路径耗尽时，Task status 才进入 blocked。这样不会因为一次验证失败同时表达“需要修复”和“禁止继续”。

### 6.2 RunFacts

```text
RunFacts
├── counters
├── workspace
├── verification
├── failures
├── loop_guards
└── observation_ledger
```

`workspace` 至少包含：

- 当前 workspace revision。
- 是否发生变化。
- 受影响路径。
- 变化证据引用。

`verification` 至少包含：

- `none | unknown | passed | failed | stale | unavailable`。
- verified revision。
- attempted checks。
- evidence refs。
- 无法验证的结构化原因。

工作区发生变化时 revision 增加，旧验证变为 stale。验证只有在对应当前 revision 时才是新鲜证据。同一并发 ToolBatch 内的修改与验证默认不建立先后关系，因此验证不自动视为对该次修改有效，除非 ToolResult 提供明确顺序。

`observation_ledger` 记录已应用的模型 turn、tool call 和 command ID，用于 Boundary 重试与恢复时幂等。它只在 Run 内存在，受 CoreLimits 限制。

### 6.3 CoreAssessment

以下状态由 State 派生，不单独持久化：

```text
active
blocked
needs_verification
needs_replan
ready_to_finish
satisfied
```

现有 `RunSignalsSummary` 改为兼容投影，不能继续作为第二份事实。

## 7. Observation 与 Reducer

### 7.1 Observation

```text
CoreObservation
├── ModelObservation
├── ToolBatchObservation
├── UserInputObservation
├── CoreCommandObservation
└── CancellationObservation
```

模型和工具的可预期失败也属于 Observation：

- Tool 参数或执行失败进入 ToolResult。
- 权限拒绝进入最终 ToolResult。
- provider 自身重试耗尽后可以形成 ModelObservation(status=failed)。
- Plan 命令校验失败形成 CommandResult(status=rejected)。

Tool preparation 返回的 approval challenge 是等待观察，不是最终 ToolResult。Core 在 waiting Boundary 中保存明确的 CoreWait，Runtime/Tools 在 Resume 时完成 approve/deny；批准后产生实际执行结果，拒绝后产生 denied ToolResult，再以 ToolResultEntry 恢复 Core。ToolCall 可以跨 waiting 暂时未闭合，但在 Run 终止前必须得到最终 ToolResult。

未知 Python 异常、契约错误和基础设施错误不转换为普通 Observation。

### 7.2 Reducer

```text
reduce_observation(state, observation, context)
  -> CoreReduction(state, events, command_results)
```

Reducer 必须：

- 纯函数、确定性、无 I/O。
- 不读取系统时间，由 ReductionContext 注入 now。
- 通过 observation ID 保证重复应用幂等。
- 只记录事实，不决定下一步动作。

ToolBatch 按固定顺序归约：

1. 去重。
2. 更新 counters。
3. 记录 effects。
4. 更新 workspace revision。
5. 更新 verification。
6. 记录 failure。
7. 应用 ToolResult 中的 CoreCommand。
8. 更新 loop guard facts。
9. 产生 CoreDomainEvent。

Reducer 消费 ToolResult 的结构化语义，例如 `effects`、`data.verification`、`data.output_quality`、`data.core_command` 和 error kind。不得依赖工具名称或解析自然语言输出。

所有持久化状态变化都必须经过 Reducer 模块。Policy Decision 引起的状态变化通过同模块的 `apply_decision()` 完成，Driver 不直接 `replace()` CoreState。

## 8. Policy 与 Decision

```text
CorePolicy.decide(state, latest_observation, policy_context)
  -> CoreDecision
```

Policy 是纯函数，不调用 Model、Tool、Context、Sessions 或 Event Sink。

CoreDecision 只有四类：

```text
CallModel
ExecuteTools
Wait
Terminate
```

### 8.1 CallModel

CallModel 使用结构化 purpose：

```text
reasoning
recovery
verification
replan
plan_publish
plan_closeout
final_response
```

Decision 携带 CoreDirective：code、constraints 和 evidence refs。Core 不生成大段自然语言控制提示；Context/Prompt 层负责将 directive 渲染成适合模型的语言。

### 8.2 ExecuteTools

ExecuteTools 包含模型已经提出的 ToolCall 和本轮工具目录快照标识。Policy 只判断这些调用当前是否允许推进，不替模型生成具体工具参数。

当预算、重复调用或其他策略拒绝执行已经持久化的 ToolCall 时，Core 必须生成 synthetic interrupted ToolResult 并提交，不能留下孤立 ToolCall。

### 8.3 Wait 与 Terminate

Wait 必须携带结构化 CoreWait。Terminate 只表达：

```text
completed
failed
cancelled
```

Runtime 仍负责 terminal commit 和资源释放。

### 8.4 固定优先级

Policy 按以下顺序判断：

1. 已记录的 fatal task condition。真正的 Core invariant violation 直接抛出异常。
2. 已存在的外部等待条件。
3. 取消或 Tool interruption。
4. Plan confirmation。
5. hard budget 与 loop guard。
6. recoverable error。
7. replan。
8. verification。
9. plan closeout。
10. completion。
11. 默认 reasoning。

高优先级条件命中后不继续执行低优先级分支。该顺序必须通过表驱动测试固定。

## 9. Core Driver

Driver 只负责固定流程和 Port 协作，不包含任务策略。

### 9.1 模型动作

```text
apply CallModel
  -> commit before_model
  -> ContextPort.prepare
  -> ModelPort.call
  -> append AssistantMessage
  -> reduce ModelObservation
  -> commit after_model
  -> Policy.decide
```

### 9.2 工具动作

```text
apply ExecuteTools
  -> ToolPort.prepare_batch（无外部副作用）
  -> approval challenge 时转入 Wait
  -> ready 时 commit before_tools
  -> ToolPort.execute_prepared
  -> reduce ToolBatchObservation
  -> append final ToolResultMessages
  -> commit after_tools
  -> Policy.decide
```

`before_tools` 提交失败时，ToolPort 的实际 execution 不得被调用。Preparation 已建立的内存状态由 Runtime 释放，不能产生 workspace 或外部副作用。

### 9.3 等待与终止

```text
apply Wait
  -> commit waiting
  -> return CoreOutcome(waiting)

apply Terminate
  -> commit before_terminal
  -> return CoreOutcome(terminal)
  -> Runtime terminal commit
```

Waiting commit 失败时不得向调用者报告 waiting。Before-terminal commit 不是 Sessions terminal commit；最终 RunResult 仍由 Runtime 提交。

### 9.4 Journals

Driver 内部使用两个轻量 journal：

- MessageJournal：维护全部消息视图和尚未提交的消息增量。
- CoreEventJournal：维护尚未随 Boundary 提交的领域事件。

Boundary 成功后才能标记相应 delta 已提交。实时模型增量和工具进度不进入这些 journal。

### 9.5 异常处理

Driver 不使用捕获所有异常的总兜底。声明过的模型失败和 ToolResult 是 Observation；以下错误穿透到 Runtime：

- Boundary 失败。
- CoreContractError。
- CoreInvariantError。
- 未知 Port 异常。
- 强制 task cancellation。

Core 在安全点发现协作式取消时，可以归一化为 CancellationObservation 并由 Policy 返回 Terminate(cancelled)。模型、工具或 Boundary 调用进行中发生的强制取消直接由 Runtime 收尾。

ToolPort 必须把协作式取消归一化为 interrupted ToolResult。若进程在结果提交前崩溃，Sessions 保留最近的非终态 `before_tools` checkpoint，后续由 Tools 对账恢复；进程崩溃本身不能制造一个带孤立 ToolCall 的虚假 terminal Run。

## 10. PlanState 与 CoreCommand

PlanState 只存在于 `TaskState.plan`，Core 是唯一语义所有者。Sessions 透明保存，Runtime 和 Tool 只能提交命令。

```text
PlanState
├── plan_id
├── origin: plan_mode | build_mode
├── status: proposed | active | completed | rejected | abandoned
├── revision
├── definition
├── steps
├── pending_revision
└── close_request
```

Plan 不重复保存 original request 和 current goal。

PlanStep 状态只有：

```text
pending | in_progress | completed
```

第一版延续“最多一个 in_progress step”的约束。任务阻塞放入 TaskBlocker，不增加 step blocked 状态。

Completed step 保存 completion note 和 evidence refs。Reducer 只接受指向本 Run 已知 Observation/ToolResult 的 evidence ID；对于无法外部验证的纯分析步骤可以只保存 completion note，但 Plan 整体关闭仍需满足 CompletionPolicy 的证据要求。

### 10.1 命令

模型或 Plan Tool 可以产生：

```text
SubmitPlan(definition)
UpdatePlanProgress(expected_revision, updates)
ProposePlanRevision(expected_revision, reason, replacement)
RequestPlanClose(expected_revision, summary, evidence_refs)
```

外部用户操作产生：

```text
ApprovePlan
RejectPlan
ApprovePlanRevision
RejectPlanRevision
AbandonPlan
```

普通进度更新使用 delta，不提交完整 PlanSnapshot。初始计划和正式 revision 可以携带完整 definition，由 Reducer 分配稳定 plan/step ID 并增加 revision。

### 10.2 模式规则

- read：拒绝计划变更。
- plan：SubmitPlan 创建 proposed plan，随后等待用户确认。
- build：SubmitPlan 创建轻量 active plan，可以直接执行。

模型不能自行声明是否需要审批。Plan approval 后 Core 激活计划并产生领域事件；是否将 Session mode 从 plan 切换为 build 由 Runtime 应用策略处理。

Plan mode 下，Core Policy 只允许只读或计划相关工具。工具是否具有 workspace mutation 风险由 Tool metadata 和 Tools Security 提供，Core 不通过工具名称猜测。

### 10.3 修订规则

- 单次普通失败不触发 replan。
- Build 轻量计划在达到 ReplanPolicy 阈值后可以自动修订。
- 用户批准过的计划只能生成 pending revision，不能静默替换。
- revision 应尽量保留仍然有效的 completed steps 和证据。
- expected_revision 不匹配时命令被拒绝，不能覆盖新计划。

### 10.4 关闭规则

RequestPlanClose 只是关闭申请：

1. Reducer 记录 close request。
2. CompletionPolicy 检查步骤、阻塞和验证。
3. 缺少验证时要求验证。
4. 验证失败时进入恢复或重规划。
5. 条件满足后才将 Plan 和 Task 标记完成。

模型可以申请完成，但不能强制完成。

## 11. Recovery、Replan 与 Completion

### 11.1 RecoveryPolicy

Recovery 表示目标和计划仍然有效，只需修复执行问题。

可恢复情况包括：

- 普通 Tool 错误、参数错误和格式错误。
- 风险操作被拒绝但存在替代路径。
- 验证失败。
- 输出截断、空回复。
- Tool 不可用但存在其他能力。
- Plan step 的早期失败。

不可作为模型恢复的问题包括：

- State/schema 损坏。
- Boundary 和 workspace recovery 失败。
- Port 注册或身份不一致。
- deadline 和强制 cancellation。
- recovery budget 已耗尽且没有新证据。

Recovery 同时具有总预算和按 reason 的预算，避免一种失败耗尽全部执行次数。

### 11.2 ReplanPolicy

Replan 表示目标仍然有效，但当前路径已经不再成立。

触发条件包括：

- 同一 Plan step 多次有效失败。
- 关键能力不可用。
- 新证据推翻计划假设。
- 用户改变约束。
- 当前计划无法满足 completion criteria。
- Recovery 多次执行但没有进展。

一次 typo、单次测试失败、等待审批、网络抖动或输出截断不触发 replan。

### 11.3 CompletionPolicy

没有 ToolCall 的 AssistantMessage 只是 final candidate，不等于完成。

read mode 完成要求：

- 存在非空用户可见答复。
- 没有 workspace change。
- 没有 blocker 或 fatal failure。

build mode 未发生修改时完成要求：

- 存在非空用户可见答复。
- 没有 blocker、失败验证或未关闭的 active plan。

build mode 发生修改时完成要求：

- 当前 workspace revision 存在 fresh passed verification；或者
- 存在结构化 VerificationUnavailable，记录 reason、attempted checks 和 evidence refs，并在最终答复明确未验证风险。

Verification stale/unknown 时先要求验证；failed 时先恢复，超过阈值后重规划，再根据可操作性等待用户或失败。

Plan 完成额外要求：

- 所有 steps completed。
- 没有 blockers。
- 没有 failed/stale verification。
- 工作区修改后有 fresh pass 或结构化 unavailable。

## 12. 错误模型

### 12.1 任务失败

Tool failure、verification failure、用户拒绝、空响应和声明过的模型不可用是可观察任务结果，进入 Reducer 和 Policy。

### 12.2 Core 异常

第一版只建立两个主要异常：

```text
CoreContractError
CoreInvariantError
```

缺少必需 Port、非法状态、未知 Decision、revision 不变量损坏和未闭合 ToolCall 都属于此类，不返回伪装成业务失败的 CoreOutcome。

### 12.3 Runtime/基础设施异常

Boundary、Sessions、workspace recovery、deadline、资源初始化与释放、未知 Python 异常由 Runtime 处理。Runtime 决定 failed/cancelled，尝试 terminal commit，记录错误并释放资源。

### 12.4 CoreReason

```text
CoreReason
├── code
├── message
├── source
├── recoverable
├── evidence_refs
└── details
```

Policy 依赖稳定 code，例如 `tool.execution_failed`、`verification.failed`、`plan.revision_conflict` 和 `core.recovery_exhausted`。Runtime 将其映射为外部兼容的 stop reason，不解析 message。

## 13. 事件模型

### 13.1 CoreDomainEvent

Reducer 根据状态变化产生领域事件，例如 PlanProposed、PlanActivated、BlockerAdded、RecoveryRequested、VerificationRecorded 和 TaskSatisfied。

CoreDomainEvent 不包含 run/session ID、时间戳和 sequence，必须随 CoreBoundary 和 CoreState 原子提交。

### 13.2 RuntimeEvent

Run started/waiting/resumed/finished、deadline、资源错误和 boundary receipt 由 Runtime 产生并添加 envelope。

### 13.3 LiveEvent

模型 delta、Tool started/progress 等事件用于实时反馈。Live sink 失败不应终止 Agent；Runtime 可以记录丢失诊断。需要持久化的操作事件由唯一 Event Adapter 排队到下一次 Boundary，不能由 Core 和 Tool 各生成一份不同事实。

Events 不参与恢复。Messages 与 Sessions RunState 中的 CoreState 仍然是恢复权威。

## 14. 与 Sessions v2 和 Runtime 设计的对齐

### 14.1 Sessions

Sessions 继续保持：

- SessionState -> RunState -> Checkpoint 两层权威结构。
- Messages 是内容权威。
- Events 不是恢复权威。
- `commit_run_boundary` 是唯一持久化入口。
- Sessions 不解释 CoreState 和组件 checkpoint。

本次兼容范围只覆盖重构开始时当前 Sessions v2 中已保存的旧 Core payload，使 staged migration 可以恢复已有非终态 Run。它不恢复更早的历史 Sessions API、旧目录结构或 Sessions v1 文件。读取旧 Core payload 后，下一次成功 Boundary 只写新 schema。

### 14.2 Runtime

沿用当前 Runtime 的 `RunEnvironment`、`RunResourceScope`、`RunExecutor`、`RunCoordinator` 和 `RuntimeLifecycle` 方向，不重新建立第二套执行框架。

现有 Runtime 设计中的以下内容需要随本文校正：

- CorePorts 不直接暴露 MemoryPort。
- CoreOutcome 不携带组件 checkpoint。
- Runtime 不解释 CoreDecision。
- 通用 Core waiting Boundary 由 Runtime Adapter 映射到 Sessions waiting 类型。
- Plan 不再由 Runtime PlanStateManager 管理。

## 15. 目标文件布局

```text
src/codepilot/core/
├── __init__.py
├── contracts.py
├── state.py
├── plan.py
├── commands.py
├── observations.py
├── reducer.py
├── policy.py
├── events.py
├── errors.py
├── driver.py
├── model_step.py
├── tool_step.py
└── tool_adapters/
```

文件职责：

| 文件 | 目标职责 |
|---|---|
| contracts.py | Core 对外输入、Ports、Boundary、Wait、Outcome 和 Decision value objects |
| state.py | CoreState、TaskState、RunFacts 和派生 assessment |
| plan.py | Plan 数据结构与静态校验，不包含状态转换方法 |
| commands.py | 模型、Tool 和外部用户可提交的 CoreCommand |
| observations.py | Observation 数据结构与 Port 结果归一化协议 |
| reducer.py | 唯一状态转换入口 |
| policy.py | Assessment、优先级和 CoreDecision |
| events.py | 无 Runtime envelope 的领域事件 |
| errors.py | Core 契约与不变量异常 |
| driver.py | 固定执行循环、journals 和 Boundary 调用 |
| model_step.py | 单次模型动作和 ModelObservation 转换 |
| tool_step.py | ToolBatch 执行、ToolObservation 和消息投影 |
| tool_adapters/ | 将 Plan/Interaction Tool 输入转换为 CoreCommand |

第一版不建立通用状态机、Policy plugin、Manager/Service 层或复杂依赖注入容器。只有单文件确实难以维护时再拆分子包。

## 16. 现有实现迁移映射

| 当前实现 | 目标 |
|---|---|
| runner 主循环 | 移入 driver.py；runner 暂时保留兼容 facade |
| RunState | 替换为 CoreState；RunSignalsSummary 变为投影 |
| PlanState 状态转换方法 | 移入 Reducer |
| PlanStateManager | 删除 |
| RunGuard | 拆入 Recovery/Completion Policy 后删除 |
| Core 模型 retry/backoff | 移入 LLM Adapter 或 Runtime |
| Core Approval resume | 移入 Runtime/Tools，Core 接收 ToolResultEntry |
| context_preflight | 由 ToolCall settlement 不变量替代后删除 |
| BoundaryCommitter 收集 Tool checkpoint | 移入 Runtime Boundary Adapter |
| Core EventRecorder envelope | 移入 Runtime Event Adapter |
| Plan Tool 直接写状态 | 改为产生 CoreCommand |
| Runtime approve/reject 直接改 Plan | 改为 apply_core_command |

## 17. 分阶段迁移顺序

### 阶段 A：行为基线

补齐 Boundary 顺序、等待恢复、审批拒绝、验证失败、提交失败、ToolCall 闭合和当前 Sessions v2 Core payload 恢复测试。不改变生产行为。

### 阶段 B：纯 Core Kernel

实现 State、Command、Observation、Reducer、Policy、Error 和 DomainEvent。只接纯单元测试，不接入 runner。

### 阶段 C：新契约与 Runtime Adapter

引入 CoreRunInput、CorePorts、CoreBoundary、CoreOutcome、状态加载器和 legacy outcome/event 投影。更新 Runtime Boundary/Error/Event Adapter。

### 阶段 D：Core Driver

实现 `run_core()`，使用 fake Ports 验证全部时序和故障路径。`run_agent_loop()` 暂时作为兼容 facade。

### 阶段 E：Plan 命令化

迁移 Plan Tool 和外部 approve/reject，删除 PlanStateManager 和直接状态写入。

### 阶段 F：切换调用方

依次切换主 Gateway、Continuation/Resume 和 Subagent。每次只保留一条实际执行路径，不通过长期 feature flag 维持双状态机。

### 阶段 G：删除旧路径

删除旧 runner 分支、RunGuard、旧 RunState、Core retry、Core Approval resume、Context preflight 和冗余 outcome 字段。更新 Core、Runtime 和 Sessions 设计文档中的旧契约。

每个阶段必须先更新调用者和契约测试，再删除无调用者代码。

## 18. 架构不变量

1. CoreState 只能通过 Reducer 模块改变。
2. 下一步只能由 CorePolicy 产生。
3. 相同输入必须得到相同 Reduction 和 Decision。
4. Plan 只存在于 TaskState.plan。
5. Runtime 不解释 CoreDecision。
6. before_tools 成功提交后才能执行工具。
7. 每个持久化 ToolCall 必须得到 ToolResult。
8. waiting Boundary 成功后才能返回 waiting。
9. 工作区修改后，完成必须有当前 revision 的验证证据或结构化 unavailable。
10. 基础设施异常不能交给模型修复。
11. Event 不能成为恢复权威。
12. Core 不读取 Context/Memory 内部结构。
13. Core 不拥有 Port 和 Runtime resource 生命周期。
14. Core 不反向依赖上层模块。

## 19. 测试与验收矩阵

| 层级 | 必须验证的行为 |
|---|---|
| State | schema round-trip、当前旧 payload 迁移、workspace revision、verification stale |
| Reducer | Observation 幂等、重复 ToolResult、批处理顺序、非法转换拒绝 |
| Plan | mode、revision、approval、pending revision、close request、单一 in-progress |
| Policy | 固定优先级、完成条件、恢复预算、replan 阈值、外部等待 |
| Driver | Boundary 顺序、消息/事件 delta、Port 调用顺序、ToolCall settlement |
| Runtime | 异常归一化、资源释放、deadline、模型重试、mode switch |
| Sessions | 每个恢复点、commit 幂等、当前旧 Core payload 读取、新 schema 单写 |
| Events | Domain/Runtime/Live 分层、live sink 隔离、审计投影兼容 |
| E2E | Read 回答、Build 修改验证、Approval Resume、Plan 审批与修订 |
| Regression | Gateway、CLI、Subagent、Observability、Evaluation 和全量测试 |

必须进行的故障注入：

- before_tools 提交失败时 ToolPort 未调用。
- after_tools 提交失败时恢复不会重复已确认副作用。
- waiting 提交失败时不报告 waiting。
- live event sink 失败时 Agent 继续。
- durable event 提交失败时停止推进。
- provider retry 耗尽后形成声明过的模型失败结果。
- 旧 revision PlanCommand 被拒绝。
- workspace 在验证后再次变化时验证变 stale。
- approved plan revision 必须等待用户确认。
- Runtime 强制取消后仍释放全部 Run 资源。

## 20. 重构完成标准

以下条件全部满足时，Core 重构才算完成：

- `PlanStateManager`、`RunGuard` 和 Core 模型重试已移除。
- CoreRunInput 不再包含 approval decision、deadline、retry policy 和事件序号。
- Runtime 不直接修改 PlanState，不解释 CoreDecision。
- 当前 Sessions v2 的旧 Core payload 可以按约定读取并升级，新提交只写新 schema。
- Gateway、CLI 和 Subagent 的外部行为保持兼容。
- Observability/Evaluation 不依赖已删除的 Core 内部字段。
- 新增测试、类型/格式检查和全量测试通过。
- 不存在长期 feature flag、双状态机、Plan 双写或两套执行入口。
- Core、Runtime、Sessions 三份设计文档的契约已经同步。

## 21. 第一版明确不实现

- 通用工作流/FSM 框架。
- Plan tree、依赖图和多 planner 模型。
- Policy plugin 系统。
- Core 内自动回滚和 Git 策略。
- Core 直接治理 Context 或 Memory。
- Event Sourcing 或通过事件重放恢复状态。
- 分布式执行、租约和多 Runtime 写入。
- 通用 Effect 抽象和事务管理器。
- 为所有旧历史 Session 格式提供长期兼容。

## 22. 最终目标

重构完成后，一次 Agent Run 的任务推进只有一条语义链：

```text
Observation
  -> Reducer
  -> CoreState
  -> Policy
  -> CoreDecision
  -> Driver 调用 Port
  -> 新 Observation
```

可靠执行只有一条外部链：

```text
Runtime prepare
  -> run_core
  -> CoreBoundary
  -> Runtime Boundary Adapter
  -> Sessions commit_run_boundary
  -> CoreOutcome
  -> Runtime waiting/terminal commit
  -> resource release
```

Core 负责“根据已知事实，任务下一步应该如何推进”；Runtime 负责“可靠地执行并持久化这次推进”；Model、Tools、Context、Memory 和 Sessions 各自保持独立权威，任何模块都不通过第二套状态或调用入口重新实现 Core。
