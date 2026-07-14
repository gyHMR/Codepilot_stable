# Runtime 审计权威模型与词汇基线

## 1. 基线状态

本文是 `docs/design/8-end-to-end-chain-audit-plan.md` 阶段 0 的第一项产物，冻结后续 12 条链路共同使用的业务事实所有权、ID 语义、状态词汇和跨层映射边界。

- 基线日期：2026-07-14。
- 事实来源：当前工作树中的真实源码和 `docs/design/1core-design.md` 至 `7runtime-design.md`。
- 审计范围：运行系统；`evaluation` 内部实现不纳入。
- 基线性质：定义后续审计的判断标准，不代表当前所有实现已经满足该标准。
- 修改规则：后续发现冲突时，先记录证据和目标所有者，不通过新增兼容层保留两套口径。

当前工作树包含未提交的重构文件。CodeGraph 用于定位符号和调用关系；当索引结果与未提交文件的磁盘内容不一致时，以显式 UTF-8 读取到的当前磁盘内容为准。

## 2. 术语解释

本文区分四类对象：

| 类别 | 含义 | 示例 |
|---|---|---|
| 语义权威 | 唯一可以创建或推进某项业务事实的模块 | Core 推进 Task/Plan；Tools 推进 ToolAttempt |
| 持久化权威 | 唯一决定可恢复状态的模块 | Sessions 保存 RunState、Message、Checkpoint |
| 执行权威 | 当前进程中唯一控制 task、deadline、取消和 release 的模块 | Runtime |
| 投影 | 从权威对象单向派生、供其他边界消费的只读表示 | CoreOutcome -> AgentRunResult -> RuntimeFrame |

“存在多个状态对象”不等于“双重权威”。不同语义层可以拥有不同状态，但必须存在单向、显式、穷尽的映射；投影不能反向修改源状态。

## 3. 权威模型冻结

### 3.1 任务语义与运行事实

| 业务事实 | 唯一所有者 | 当前权威对象/入口 | 持久化方式 | 允许的消费者 | 禁止行为 |
|---|---|---|---|---|---|
| 当前任务、目标、阻塞、完成状态 | Core | `CoreState.task`，`core/reducer.py` | Sessions 透明保存 `CoreState.to_dict()` | Runtime/Context/Interface 只读投影 | Runtime、Tools、Interface 直接替换 TaskState |
| Plan、Plan revision、Plan step | Core | `PlanState`，CoreCommand -> `apply_core_command()` | 仅位于 `core_state.task.plan` | Runtime/Tools 只提交命令 | 独立 Plan store、Runtime 直接推进 Plan |
| 模型/工具/用户观察后的事实 | Core Reducer | `reduce_observation()` | 随 CoreBoundary 提交 | Policy、Observability | Driver/Policy 绕过 Reducer 修改事实 |
| Core 下一步决策 | Core Policy | `CorePolicy.decide()` | 不单独持久化 | Core Driver | Runtime 解释或执行 CoreDecision 分支 |
| Session 身份和导航 | Sessions | `SessionState`、`SessionStateService` | `session.json` | Runtime/Interface View | Runtime Registry 充当持久化 Session 事实 |
| Run 生命周期、revision、终态 | Sessions | `RunState`、`commit_run_boundary()` | `run.json` | Runtime/Interface View | Runtime 保存第二份可恢复 RunState |
| 对话内容和消息链 | Sessions | `MessageRecord`、Session leaf | `messages.jsonl` | Context、Runtime Conversation 投影 | Checkpoint 或 Runtime 内存消息替代 Message 权威 |
| 稳定恢复边界 | Sessions | `RunCheckpoint`、`ComponentCheckpoint`、`WorkspaceCheckpoint` | `run.json` | Runtime Recovery | Event replay 或 Python 调用栈作为恢复依据 |
| task、deadline、取消、收敛、release | Runtime | `RunResourceScope`、`RunCancellationToken`、`RuntimeLifecycle` | 不持久化 | RuntimeFrame/Trace 只读投影 | 将 Runtime execution state 写入 Sessions |

设计证据：

- `docs/design/1core-design.md:35`：Core 是确定性任务推进内核。
- `docs/design/1core-design.md:505`：PlanState 只存在于 Core TaskState。
- `docs/design/6sessions-design.md:5`：Sessions 是 Session、Run、Message、Checkpoint 的持久化权威。
- `docs/design/6sessions-design.md:29`：持久化运行权威为 CoreState 和 Sessions RunState。
- `docs/design/7runtime-design.md:56`：Runtime Execution State 只存在于进程内。

### 3.2 能力域权威

| 能力域 | 唯一所有者 | 当前权威对象/入口 | Sessions 角色 | Core 角色 | Runtime 角色 |
|---|---|---|---|---|---|
| LLM Provider 请求和流式事件 | LLM | `ModelPort`、标准 LLM events；具体 Provider adapter | 无 | 消费标准 ModelPort | 装配 model、处理 Runtime retry wrapper 和资源 |
| Tool 注册和目录 | Tools | `ToolRegistry`、`ToolCatalogSnapshot` | 无 | 只读 catalog，产生 ToolExecutionRequest | 装配和注入 Tool port |
| ToolAttempt 和 ToolResult | Tools | `ToolRuntime`、`ToolAttemptRecord`、`ToolResult` | 不透明保存 tools component | 消费最终 ToolResult | 恢复/调用 Tool control port，不重写 attempt |
| Approval/Interaction | Tools | `ApprovalChallenge`、`InteractionRequest`、Tool state store | WaitingState 只保存公共恢复字段 | 只表达 CoreWait | 将用户 Action 路由到原 Tool attempt |
| 工具权限和副作用声明 | Tools | `PermissionEngine`、`ToolAccessRequest`、`ToolEffect` | 只保存结果/checkpoint | 保存任务相关事实投影 | 只传递 permission mode/deadline |
| Context 选择、预算、压缩 | Context | `ContextService`、内部 `ContextState`、`ContextCheckpointState` | 不透明保存 context component | 只消费 `PreparedModelContext` | 注入 port、收集 checkpoint |
| 长期 Memory | Memory | `MemoryService`、`MemoryRecord`、User/Project repositories | 不解释 Memory | 不直接依赖 Memory | Terminal Commit 后提交 proposal；Context 通过 recall port 读取 |
| Workspace 恢复快照 | Sessions + Runtime 校验 | `WorkspaceCheckpoint`、`sessions/workspace.py` | 保存 checkpoint | 只保存 workspace 任务事实 | 恢复前校验 |
| Rollback 策略和 baseline | `sessions/rollback/` | Rollback service 与 Git baseline | 保存引用/metadata | 不参与 | 命令编排和结果投影 |

Context 和 Memory 位于 `sessions/` 命名空间只是源码聚合，不表示 `SessionStateService` 拥有其领域策略。

### 3.3 事件与界面权威

| 对象 | 所有者 | 当前入口 | 持久化 | 不变量 |
|---|---|---|---|---|
| Core Domain Event | Core | `CoreDomainEvent` | 随 CoreBoundary 原子提交 | 不含 Runtime envelope，不参与恢复 |
| Runtime event envelope | Runtime | `project_core_domain_event()`、`RuntimeSessionStateAdapter.queue_durable_event()` | durable event 交给 Sessions | 只增加关联字段，不改领域语义 |
| Boundary receipt/system event | Sessions | `SessionStateService.append_event()` | `events.jsonl` | Event 缺失不能改变 RunState |
| Live Event | Runtime/Core port | `live_events`/event sink | 默认不持久化 | 丢失或 sink 失败不能终止 Run |
| Trace/summary | Observability | Recorder/Trace projection | 审计产物 | 不能反向推进业务状态 |
| CLI/Web/DingTalk 展示 | Interface | Runtime Action/Frame | 不拥有业务事实 | 不直接读取内部状态后重新推断状态机 |

## 4. 单向投影链

以下链路被冻结为合法的单向派生关系：

```text
CoreState
  -> CoreContextView / Runtime public view
  -> Interface rendering

ToolAttemptRecord
  -> ToolResult
  -> ToolResultMessage
  -> Context evidence / model conversation

CoreBoundary
  -> CommitRunBoundaryRequest
  -> RunState + RunCheckpoint + MessageRecord + durable events

CoreOutcome
  -> AgentRunResult
  -> RuntimeFrame

ContextState + Messages + MemoryRecallResult
  -> ProjectionPlan
  -> PreparedModelContext

MemoryRecord(active only)
  -> RecalledMemory
  -> Context L3 item
```

后续审计中，若发现箭头反向写回、绕开中间权威对象或出现另一条并行派生链，按 `AUTH`、`BOUNDARY` 或 `DUP` 记录。

## 5. ID 词汇冻结

### 5.1 主链路 ID

| ID | 语义范围 | 当前生成/接纳位置 | 唯一性与幂等规则 | 禁止行为 |
|---|---|---|---|---|
| `session_id` | 一个持久化会话 | `sessions/service.py::new_session_id()`；Sessions create/open 接纳 | 在 Session 生命周期内稳定 | 重开 Session 时重新生成 |
| `request_id` | 一次用户 Prompt Action | `PromptSubmitted` 默认生成；`begin_run()` 校验和去重 | 调用方重试必须复用；相同 ID 不同输入冲突 | 在 Runtime 重试中重新生成 |
| `run_id` | 一个可跨 waiting/resume 的 Run | Runtime `new_run_id()`；Sessions `begin_run()` 接纳 | Resume/Continuation 复用原 ID | 每次恢复创建新 Run |
| `message_id` | 一条持久化 MessageRecord | 初始消息由 `(session_id, request_id)` 确定性派生；边界消息由 `(commit_id, index)` 派生 | 同一逻辑消息重试 ID 和内容相同 | Boundary 重试随机生成新消息 ID |
| `observation_id` | 一个 Run 内的一次 Core Observation | Core Driver 入口常量或 run-local sequence | 在单个 CoreState ledger 内唯一；重复归约无效 | 依赖全局唯一或跨 Run 复用 ledger |
| `tool_call_id` | 模型提出的一次工具调用 | 模型/协议层生成，Core/Tools 原样透传 | 从 AssistantMessage 到 ToolResultMessage 保持不变 | ToolRuntime 重新命名 |
| `attempt_id` | ToolRuntime 对一次 ToolCall 的执行尝试 | `attempt_id_for()` = `session_id:run_id:tool_call_id` | 同一请求恢复得到同一 attempt | Approval resume 创建第二 attempt |
| `approval_id` | 一次 ApprovalChallenge | Tools `build_approval_challenge()` | 必须同时校验 fingerprint、Session、Run、ToolCall 和过期时间 | 只凭 approval_id 执行副作用 |
| `commit_id` | 一次逻辑 Sessions boundary commit | Runtime Boundary Adapter；Terminal Commit 也由 Runtime 生成 | 相同 ID + 相同 digest 返回原 receipt；不同内容冲突 | Sessions 在重试时替换 commit_id |
| `checkpoint_id` | 一个成功持久化的恢复边界 | begin checkpoint 由 Sessions 创建；边界 checkpoint 由 commit_id 确定性派生 | 只引用成功提交的 checkpoint | Runtime 自行制造可恢复 checkpoint |
| `event_id` | 一个 durable/live 事件 envelope | Core/Runtime adapter 为领域事件生成；Sessions 为系统事件补齐 | 相同 ID 不同内容冲突 | 用 event_id 代替业务对象 ID |

### 5.2 扩展关联 ID

| ID | 所有者 | 生成规则 |
|---|---|---|
| `plan_id` | Core | `plan:{command_id}`，由 Reducer 分配 |
| `step_id` | Core | `{plan_id}:step:{index}`，由 Reducer 分配 |
| `interaction_id` | Tools | Tools 创建，响应必须回到原 ToolAttempt |
| `grant_id` | Tools Security | ApprovalGrant 创建，绑定 approval fingerprint 和 scope |
| `registration_id` | Tools Registry | 标识模型看到并实际执行的工具注册版本 |
| `memory_id` | Memory | MemoryService/Repository 创建，编辑生成新版本记录时使用新 ID |
| `projection_ref` | Context | 标识一次 PreparedModelContext 投影，不是持久化业务 ID |

### 5.3 当前实现中的 ID 生成分工

当前存在多个合法生成位置，但它们必须按语义分工：

- Runtime 生成 Action/Run/Commit 和 Runtime/Core event envelope ID。
- Sessions 生成或确定性派生 Message、Checkpoint、receipt/system event ID。
- Tools 生成 Attempt、Approval、Interaction、Grant ID。
- Core 生成 Observation、Plan、Step ID。
- Context/Memory 只生成自身领域 ID，不生成 Session/Run/Tool ID。

`begin_run()` 在缺少 `run_id` 时提供 request-scoped fallback。目标主链路仍以 Runtime 提供 `run_id` 为准；后续链路 2、3 要验证生产调用是否依赖该 fallback，不能因此形成第二个 Run ID 所有者。

## 6. 状态词汇冻结

### 6.1 Core

| 状态域 | 权威枚举 |
|---|---|
| Task | `active / blocked / satisfied / abandoned` |
| TaskBlocker | `user_input_required / tool_unavailable / verification_failed / plan_incomplete / replan_required` |
| Verification | `none / unknown / passed / failed / stale / unavailable` |
| Assessment（派生） | `active / blocked / needs_verification / needs_replan / ready_to_finish / satisfied` |
| Outcome | `completed / waiting / failed / cancelled` |
| Boundary | `before_model / after_model / before_tools / after_tools / waiting / before_terminal` |
| Wait | `tool_approval / user_input / plan_confirmation / continuation` |
| Plan | `proposed / active / completed / rejected / abandoned` |
| PlanStep | `pending / in_progress / completed` |
| RunMode | `read / plan / build` |

Core Assessment 是从 CoreState 派生的只读判断，不单独持久化。

### 6.2 Sessions

| 状态域 | 权威枚举 |
|---|---|
| SessionKind | `primary / subagent` |
| RunStatus | `created / running / waiting / completed / failed / cancelled` |
| RunPhase | `received / model / tools / finalizing / finished` |
| CommitKind | `progress / waiting / terminal` |
| ResumePoint | `before_model / after_model / before_tools / after_tools / before_finalization` |
| WaitingKind | `tool_approval / user_input / plan_confirmation / continuation` |
| CheckpointOwner | `tools / context / rollback` |
| RecoveryStatus | `ready / needs_validation / blocked / not_found` |
| WorkspaceRecoveryStatus | `unchanged / changed / missing / unknown` |

### 6.3 Runtime 与公开投影

| 状态域 | 权威/投影枚举 | 说明 |
|---|---|---|
| RuntimeExecutionState | `new / preparing / executing / waiting / resuming / cancelling / finalizing / terminal / released` | 仅进程内执行控制 |
| TerminalOutcome | `completed / failed / cancelled` | Runtime terminal 状态的附属结果 |
| AgentRunStatus | `running / completed / failed / aborted / waiting_approval / waiting_user` | `protocols` 对外兼容词汇，是 CoreOutcome 的投影 |
| AgentRunStopReason | `final_answer / max_iterations / model_error / aborted / approval_required / approval_denied / plan_approval_required / plan_clarification_required / plan_incomplete / repeated_tool_call / tool_call_limit / tool_unavailable / completion_blocked / missing_tool_port / missing_approval_decision / internal_error / deadline_exceeded / runtime_error` | Interface 稳定词汇，不回写 CoreReason |

`aborted`、`waiting_approval`、`waiting_user` 只允许出现在公开投影，不得写回 CoreOutcome 或 Sessions RunStatus。

### 6.4 Tools

| 状态域 | 权威枚举 |
|---|---|
| ToolAttemptState | `received / validating / resolving_access / awaiting_approval / awaiting_input / queued / running / succeeded / failed / denied / timed_out / cancelled / interrupted` |
| ToolStatus | `success / error / denied / approval_required / user_input_required / cancelled / timed_out / interrupted` |

ToolAttemptState 描述执行生命周期；ToolStatus 描述 Core/消息可消费的结果。二者不是同一状态机，不允许混用字段。

### 6.5 Context 与 Memory

| 状态域 | 权威枚举 |
|---|---|
| Context Layer | `l0 / l1 / l2 / l3 / l4` |
| Retention | `required / protected / budgeted / discard_first` |
| Context internal pressure | `normal / tight / critical / overflow` |
| MemoryStatus | `candidate / active / disabled / superseded / deleted` |
| MemoryScope | `user / project` |
| MemoryType | `profile / feedback / project / experience / reference` |

当前 `protocols.context.ContextPressureLevel` 的公开报告词汇只有 `normal / tight / critical`，而 Context 内部权威词汇包含 `overflow`。基线冻结内部 Context 词汇为四态；跨层报告必须有显式映射或同步协议，留待链路 9 审计，不能靠类型忽略或字符串透传。

## 7. 跨层映射冻结

### 7.1 CoreBoundary -> Sessions

当前唯一映射函数为 `runtime/session_state_adapter.py::_target_boundary_state()`：

| CoreBoundary | Sessions commit kind | Run phase | Resume point |
|---|---|---|---|
| `before_model` | `progress` | `model` | `before_model` |
| `after_model` | `progress` | `model` | `after_model` |
| `before_tools` | `progress` | `tools` | `before_tools` |
| `after_tools` | `progress` | `tools` | `after_tools` |
| `waiting(tool_approval)` | `waiting` | `tools` | `before_tools` |
| `waiting(user_input)` | `waiting` | `model` | `after_model` |
| `waiting(plan_confirmation)` | `waiting` | `model` | `after_model` |
| `waiting(continuation)` | `waiting` | `model` | `after_model` |
| `before_terminal` | `progress` | `finalizing` | `before_finalization` |

Terminal Commit 不由 CoreBoundary 直接表示。Core 返回 terminal CoreOutcome 后，Runtime 构造 `kind=terminal` 的 `CommitRunBoundaryRequest`。

### 7.2 CoreOutcome -> Runtime/Sessions/Interface

| CoreOutcome.status | AgentRunStatus | Runtime terminal outcome | Sessions terminal status |
|---|---|---|---|
| `completed` | `completed` | `completed` | `completed` |
| `waiting` + tool approval | `waiting_approval` | 无 | 不终态化，RunStatus=`waiting` |
| `waiting` + 其他 wait | `waiting_user` | 无 | 不终态化，RunStatus=`waiting` |
| `failed` | `failed` | `failed` | `failed` |
| `cancelled` | `aborted` | `cancelled` | `cancelled` |

当前映射入口：

- `runtime/contracts.py::external_status()`。
- `runtime/contracts.py::terminal_outcome_for_status()`。
- `runtime/coordinator.py::_agent_result_from_outcome()`。
- `runtime/session_coordinator.py::_commit_run()`。

CoreReason 到 AgentRunStopReason 只允许通过 `external_stop_reason()`。该函数当前对未列明的非 Runtime reason 回落为 `completion_blocked`；链路 3、6 必须检查该 fallback 是否掩盖未知错误，但其他模块不得再建立第二个 reason 映射。

### 7.3 ToolAttemptState -> ToolResult.status

语义映射冻结为：

| ToolAttemptState | ToolResult.status | 说明 |
|---|---|---|
| `awaiting_approval` | `approval_required` | 挂起结果，不是终态 |
| `awaiting_input` | `user_input_required` | 挂起结果，不是终态 |
| `succeeded` | `success` | 终态 |
| `failed` | `error` | 终态 |
| `denied` | `denied` | 终态 |
| `timed_out` | `timed_out` | 终态 |
| `cancelled` | `cancelled` | 终态 |
| `interrupted` | `interrupted` | 终态 |
| `received / validating / resolving_access / queued / running` | 无最终 ToolResult | 进行中状态 |

当前映射分散在 `ToolRuntime._settle()`、审批/交互结果构造函数和失败辅助函数中，`tools/state.py::transition()` 本身只执行 dataclass replace，不校验状态图。链路 4、5 必须验证状态图和映射是否集中、穷尽且有表驱动测试；在此之前不得由 Core/Runtime 补充第二套 Tool 状态判断。

### 7.4 ToolResult -> ToolResultMessage

唯一标准投影入口是 `tools/results.py::to_tool_result_message()`：

- 保持 `tool_call_id` 和 `tool_name`。
- 将 effect 投影为 `affected_paths/workspace_changed`。
- 将 registration、output validation、content trust 和 timing 放入 metadata。
- `user_input_required` 不允许投影成最终 ToolResultMessage。
- Core 只追加最终结果消息；挂起通过 CoreWait/Sessions WaitingState 表达。

### 7.5 CoreDomainEvent -> Durable Event

唯一标准 Core event envelope 入口是 `runtime/contracts.py::project_core_domain_event()`：

- Core 提供 `kind/payload/evidence_refs`。
- Runtime 添加 `event_id/run_id/session_id/type`。
- Sessions `append_event()` 做 JSON 归一化、脱敏、ID 冲突校验和持久化。
- receipt/system event 可以由 Sessions 产生，但不得复制 Core 领域事件语义。

## 8. 持久化与内存边界

| 对象 | 必须持久化 | 只能内存存在 | 备注 |
|---|---|---|---|
| SessionState/RunState | 是 | 否 | Sessions 权威 |
| MessageRecord | 是 | Runtime 可持有投影 | 内容权威在 Sessions |
| CoreState | 随 Boundary 保存 | Driver 持有未提交版本 | 只有成功 commit 可恢复 |
| RunCheckpoint/WaitingState | 是 | 否 | waiting 必须具备 checkpoint |
| Tool pending attempt | 作为 opaque tools component | ToolRuntime 可持有活动对象 | Sessions 不解释 state |
| Context checkpoint | 作为 opaque context component | ContextService 持有活动状态 | Sessions 不解释 state |
| RuntimeLifecycle/ResourceScope | 否 | 是 | 进程退出后不恢复 |
| Live event queue/token delta | 否 | 是 | 丢失不影响权威状态 |
| Durable event | 是 | Adapter 可暂存到下一 Boundary | 不是恢复权威 |
| MemoryRecord | Memory repository | Service 可缓存 | 不进入 SessionState |
| Rollback metadata/baseline ref | 是 | Runtime 可缓存具体 baseline | 与 recovery checkpoint 语义分开 |

## 9. 后续链路必须遵守的不变量

1. 所有 CoreState 变化经过 Reducer 或 Core command reducer。
2. Sessions 是 Run 生命周期、Message、Checkpoint 和 revision 的唯一持久化写入者。
3. Runtime execution state 永不写入 Sessions 充当恢复事实。
4. Prompt retry 复用 request_id；Resume/Continuation 复用 run_id。
5. ToolCall 贯穿为同一个 tool_call_id，Approval resume 复用 attempt_id。
6. Progress/Waiting/Terminal commit 使用稳定 commit_id 和 expected revision。
7. Message/Checkpoint/receipt event 在 Boundary 重试时确定性派生。
8. CoreBoundary 不携带 Tools/Context/Rollback 私有 checkpoint。
9. Waiting Commit 成功后才能展示 waiting；Terminal Commit 成功后才能展示成功终态。
10. Event、Trace、Runtime Registry 和 Interface View 都不能成为恢复权威。
11. Context、Memory、Tools 和 Rollback 的领域策略不进入 SessionStateService。
12. 任何公开投影只允许单向派生，不允许反向修改源状态。

## 10. 本步骤校验结论

### 10.1 已与当前代码一致

- CoreState、Session/Run/Checkpoint、RuntimeLifecycle、ToolAttempt/ToolResult、ContextService/ContextState、MemoryService/MemoryRecord 的所有权与目标设计总体一致。
- `commit_run_boundary()` 是当前 Sessions 的统一 Run 边界写入口。
- CoreBoundary 到 Sessions phase/resume point 已有单一映射函数。
- CoreOutcome 到外部状态、终态和 stop reason 已集中在 `runtime/contracts.py`。
- Attempt、Approval、Message、Checkpoint 和 Event ID 均存在明确生成或接纳位置。
- Context 和 Memory 的新公开面分别集中在 `ContextService` 与 `MemoryService`。

### 10.2 后续必须验证的边界

以下内容不在本步骤修改，在对应链路中以测试和调用图判断：

1. ToolAttempt 状态转移没有在 `transition()` 中定义合法状态图，当前约束主要分散在 ToolRuntime 调用顺序中。
2. Context 内部 pressure 为四态，公共 `protocols.context` 报告为三态，需要显式映射或协议统一。
3. Runtime 同时在 progress/waiting adapter 和 terminal coordinator 中生成 commit_id，需要验证是否仍是一套提交语义。
4. Sessions 为缺失 run_id 提供 fallback，需要确认生产主链路从不依赖第二个生成口径。
5. `external_stop_reason()` 的默认 `completion_blocked` 可能隐藏未登记 reason，需在普通 Run/终止链路做穷尽性检查。
6. Core observation_id 只保证 Run 内唯一，需验证所有恢复入口不会生成与 ledger 冲突的 ID。
7. Event ID 分别由 Core/Runtime/Sessions 按事件层级生成，需验证相同领域事件不会通过两条路径重复持久化。

这些是已定位的审计边界，不是已经确认的缺陷。确认缺陷后统一写入 `docs/audit/runtime/issues.md`，避免在基线文档中提前下结论。

## 11. 变更约束

后续阶段修改任何状态、ID 或映射前，必须同步检查本文。允许的变更只有两类：

1. 修复实现，使代码符合本基线。
2. 经明确设计决策修改本基线，再同步修改所有调用者、持久化协议、测试和文档。

禁止通过兼容 alias、双写字段、旧枚举 fallback 或自然语言解析同时保留新旧口径。
