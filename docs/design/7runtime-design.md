# Runtime 执行编排与恢复设计

## 文档状态

本文定义 Codepilot Runtime v2 的目标边界、执行协议、恢复时序、文件布局、迁移顺序和验收标准，作为 Runtime 重构的唯一目标参考。

本文描述目标设计，不表示所有目标代码已经实现。重构完成后不保留旧 Runtime 调用入口、旧 Sessions 写入口或旧 Session 文件兼容读取。

第一版不引入通用依赖注入容器、事件溯源系统、分布式事务、独立调度器或 Effect 抽象。Prompt 和 Resume 使用同一条 Runtime 执行路径，Sessions 是持久化事实的唯一写入者，Core 决定任务语义，Runtime 只编排执行。

## 1. Runtime 的定位

Runtime 是一次 Agent Run 的执行编排层。它创建和管理 Run 执行环境，驱动 Core，调度 Model 和 Tool，处理取消、超时、异常、资源释放和 Checkpoint 恢复，并将内部事件转换为外部 RuntimeFrame。

Runtime 负责：

- 将用户 Action 转换为 Prompt、Resume 或 Cancel 操作。
- 创建和管理一次 Run 的执行环境。
- 调用 Core Agent Loop。
- 提供 Model、Tool、Context、Memory 等能力接口。
- 处理并发、取消、超时和任务收敛。
- 编排 Progress、Waiting 和 Terminal Commit。
- 在恢复前加载并校验 Checkpoint 与 Workspace。
- 将 Core Event 转换为实时 RuntimeFrame。
- 确保 Run 级资源最终释放。

Runtime 不负责：

- 任务语义状态机、Plan 推进和完成条件。
- 决定模型下一步调用什么工具。
- 选择或治理模型上下文。
- Memory admission、retrieval 或长期记忆治理。
- Tool 的权限、副作用和安全策略。
- 保存 Session Message、Event、Run State 或 Checkpoint。
- 直接操作 Sessions Repository。

依赖方向保持为：

```text
protocols -> llm/tools -> core -> sessions/observability
                                     -> extensions -> runtime -> interfaces
```

Runtime 可以依赖 Sessions 的领域服务和 Core 的执行协议，但不能反向把 Runtime 的执行状态写入 Sessions。

## 2. 三类状态模型

必须区分三个状态来源：

```text
Runtime Execution State  当前进程内的执行控制状态
Core Task State          任务语义、Plan、完成条件和任务组件状态
Sessions RunState        持久化事实和恢复依据
```

### 2.1 Runtime Execution State

Runtime 状态只存在于当前进程内，不作为独立事实持久化：

```text
new
  -> preparing
  -> executing
  -> waiting
  -> resuming
  -> finalizing
  -> terminal
  -> released
```

取消和超时通过：

```text
executing -> cancelling -> finalizing
```

规则：

- terminal 必须携带 completed、failed 或 cancelled 结果。
- released 只表示 Run 级资源已经释放，不写入 Sessions。
- waiting -> resuming -> executing，不能绕过恢复准备直接执行。
- 只有 Waiting Commit 成功后才能进入 Runtime waiting。
- 只有 Terminal Commit 成功后才能进入 Runtime terminal。
- 同一个 Run 只能有一个活动执行尝试。

### 2.2 Core Task State

Core Task State 由 Core 管理，包含任务语义状态、Plan、完成判断和组件状态。Runtime 只能接收快照或 Delta，不能解释或推进这些语义。

### 2.3 Sessions RunState

Sessions RunState 是持久化事实，至少包含：

- Run 身份和 Session 归属。
- 当前持久化状态和 revision。
- 最近一个稳定 Checkpoint。
- 已提交消息游标和组件快照。
- 等待或终态信息。
- 恢复前需要校验的 Workspace 状态。

Sessions RunState 是恢复权威；Runtime 内存状态和未提交的 Core 状态不能作为恢复依据。

## 3. RunEnvironment 与资源范围

第一版只引入三个轻量概念：

```text
RunEnvironment
RunEnvironmentFactory
RunResourceScope
```

### 3.1 RunEnvironment

RunEnvironment 是一次执行尝试的能力集合，不是传递给 Core 的完整对象：

```text
RunEnvironment
├── session_id
├── run_id
├── trigger: prompt | resume | continuation
├── model
├── tools
├── context
├── memory
├── cancellation
├── deadline
├── event_sink
├── state_port
└── resources
```

Session 级资源包括 Model 配置、Tool Registry、Memory Service、Context Service 和会话投影。Run 级资源包括 run_id、Core 输入、取消令牌、deadline、事件流、StatePort、Tool task handles 和临时资源。

### 3.2 RunResourceScope

RunResourceScope 是 Run 级资源的唯一拥有者，负责：

- 追踪 Core、Model、Tool 子任务。
- 传播取消信号。
- 管理 deadline timer。
- 等待任务收敛或执行强制取消。
- 关闭 Model/Tool 连接和临时资源。
- 幂等执行 release。

它不负责释放 Session 级服务，也不负责 Sessions 持久化。

## 4. Runtime 执行协议

### 4.1 外部 Action

Runtime 对外接收统一 Action：

```text
RunAction
├── session_id
├── request_id
├── kind: prompt | resume | cancel
├── run_id?
└── input?
```

request_id 是用户操作的幂等标识。Prompt 重试时必须复用同一个 request_id，避免在 begin_run 前崩溃后重复保存用户消息。

### 4.2 AgentLoopInput

Prompt 和 Resume 使用同一个输入协议：

```text
AgentLoopInput
├── entry: prompt | resume
├── user_message?
├── recovery_snapshot?
└── runtime_options
```

Resume 不创建新的 Run，而是使用原 run_id 和最新成功 Checkpoint 创建新的 RunEnvironment。

### 4.3 AgentLoopPorts

Core 不接收完整 RunEnvironment，只接收最小能力集合：

```text
AgentLoopPorts
├── model: ModelPort
├── tools: ToolPort
├── context: ContextPort
├── memory: MemoryPort
├── state: TaskStatePort
├── events: CoreEventSink
└── cancellation: CancellationToken
```

这些是能力接口，不是具体 Service 或 Repository 的暴露。

### 4.4 AgentLoopOutcome

所有执行出口都返回统一结果：

```text
AgentLoopOutcome
├── status: completed | waiting | failed | cancelled
├── termination_reason?
├── final_message?
├── durable_events
├── task_state_snapshot?
├── checkpoint?
└── error?
```

waiting 不是终态，只能触发 Waiting Commit。completed、failed 和 cancelled 才能触发 Terminal Commit。

## 5. RunExecutor

Runtime 只有一个执行入口：

RunExecutor.execute(environment, input) -> AgentLoopOutcome

RunExecutor 负责：

- 构造 AgentLoopPorts。
- 调用 Core Agent Loop。
- 转发实时 Core Event。
- 管理 Run 级取消和 deadline。
- 捕获并归一化 Core、Model、Tool 异常。
- 等待 Core 和子任务收敛。
- 返回 AgentLoopOutcome。

RunExecutor 不负责：

- 创建或结束 Run。
- 调用 Sessions Repository。
- 写 Message、Event 或 Checkpoint。
- 判断 Plan 是否完成。
- 选择 Context 或治理 Memory。
- 释放 Session 级资源。

## 6. Sessions Commit Protocol

Runtime 与 Sessions 之间只使用以下领域接口：

```text
open_session(session_id) -> SessionSnapshot
begin_run(request) -> RunStart
load_recovery(run_id) -> RecoverySnapshot
commit_run_boundary(request) -> CommitReceipt
```

commit_run_boundary 是唯一的公开持久化提交入口。它通过一次原子操作保存：

- durable events；
- Messages、leaf 和 fork 变化；
- Plan/Context 等组件快照；
- Workspace Checkpoint；
- Run State 和 revision。

提交类型只有三种：

```text
progress
waiting
terminal
```

它们共享同一个实现，不建立三套持久化流程。

### 6.1 幂等和并发

每个提交必须带有：

```text
commit_id
expected_revision
```

相同 commit_id 的重试返回原 CommitReceipt，不重复写入。过期 expected_revision 必须被拒绝，不能覆盖新状态。Terminal 状态成功提交后不可被后续提交覆盖。

### 6.2 Boundary 事件

Core Event 可以标识持久化意图：

```text
CoreEvent
├── kind
├── durability: live | durable
├── boundary: none | progress | waiting
├── payload
└── checkpoint_state?
```

Token 增量、调试信息和临时进度只走实时 Event Sink。消息、Tool 调用结果、Approval、Run 状态和 Checkpoint 等领域事实进入下一次 Commit。

当 Core 到达稳定 Progress Boundary 时，Coordinator 必须等待 CommitReceipt 后，才允许继续执行可能产生副作用的下一段工作。

Session 内的 pending Tool Attempt 只能通过 `components.tools` 恢复，不建立独立 `tool_state.json`。Runtime 创建新的 ToolRuntime 后，将 opaque Tools component 交回 ToolPort 恢复；Session/Project Approval Grant 由 Tools Security 的工作区级 Store 独立管理。

## 7. 取消、超时、失败与释放

所有非正常结束都经过同一条管线：

```text
取消 / 超时 / 异常
  -> request_termination(reason)
  -> cancelling
  -> 停止新任务并收敛已有任务
  -> AgentLoopOutcome
  -> Terminal Commit
  -> 最终 RuntimeFrame
  -> RunResourceScope.release()
  -> released
```

终止原因至少包括：

```text
user_cancelled
deadline_exceeded
core_error
model_error
tool_error
workspace_changed
runtime_error
```

用户取消归类为 cancelled/user_cancelled。超时归类为 failed/deadline_exceeded。取消、超时和异常通过同一个终止仲裁器决定唯一生效的终止意图。

取消先进行协作式传播，再在 grace period 后强制取消仍未收敛的任务。无论 Outcome 如何，release 都必须在 finally 中幂等调用。

Terminal Commit 失败时不得发送成功最终 Frame。Runtime 可以使用同一 commit_id 有限重试；资源仍然必须释放，后续启动流程从最后成功边界继续恢复。

## 8. 主链路与恢复时序

### 8.1 正常执行

```text
Action(prompt)
  -> Gateway
  -> Controller
  -> Coordinator.prepare
  -> Sessions.begin_run
  -> RunEnvironmentFactory
  -> RunExecutor.execute
  -> Core / Model / Tool
  -> Progress Commit（可重复）
  -> AgentLoopOutcome
  -> Terminal Commit
  -> final RuntimeFrame
  -> resource release
```

Gateway 不直接调用 Core，不创建 Core task，也不处理 Terminal Commit。

### 8.2 Waiting 与 Resume

```text
Core waiting Outcome
  -> Waiting Commit
  -> Runtime waiting
  -> 释放当前执行尝试资源

Action(resume)
  -> load_recovery
  -> validate Workspace
  -> begin_run(resume)
  -> 创建新的 RunEnvironment
  -> RunExecutor.execute(entry=resume)
```

Resume 使用原 run_id，只从最后一次成功的 Checkpoint 恢复。Workspace 校验失败时不得启动 Core。

### 8.3 崩溃恢复

| 崩溃位置 | 恢复行为 |
|---|---|
| begin_run 前 | 使用相同 request_id 重试 Action |
| Progress Commit 前 | 从上一个成功边界恢复 |
| Progress Commit 成功后 | 从最新边界恢复 |
| Waiting Commit 成功后 | 保持 waiting，等待 Resume |
| Terminal Commit 成功后 | 不再执行，直接返回最终结果 |
| Workspace 已变化 | 阻止恢复并要求验证 |
| Tool 副作用后未提交 | 使用稳定 tool_call_id 由 Tool 层保证重试幂等 |

Runtime 不恢复未提交的 Python 调用栈、流式半成品或进程内对象。

## 9. 目标文件布局

Runtime 目标布局保持轻量，不为每个概念建立独立框架：

```text
src/codepilot/
├── core/
├── tools/
├── sessions/
│   ├── contracts.py
│   ├── service.py
│   ├── repository.py
│   ├── filesystem.py
│   ├── serde.py
│   ├── recovery.py
│   ├── workspace.py
│   ├── context/
│   ├── memory/
│   └── rollback/
├── runtime/
│   ├── __init__.py
│   ├── contracts.py
│   ├── actions.py
│   ├── gateway.py
│   ├── controller.py
│   ├── coordinator.py
│   ├── environment.py
│   ├── executor.py
│   ├── lifecycle.py
│   ├── frames.py
│   ├── approvals.py
│   └── tool_adapters/
└── interfaces/
```

### 9.1 Runtime 文件职责

| 文件 | 应包含 | 不应包含 |
|---|---|---|
| contracts.py | Action、Outcome、Environment、Runtime 状态协议 | 具体执行和持久化 |
| actions.py | Prompt、Resume、Cancel 输入转换 | Core 循环和 Plan 规则 |
| gateway.py | 外部入口、Session 并发、Frame 输出 | Core task、Repository 写入 |
| controller.py | Action 到 Coordinator 请求的转换 | 消息、事件和 Checkpoint 写入 |
| coordinator.py | Prepare、Recovery、Commit、Termination 编排 | Context/Memory 具体治理 |
| environment.py | EnvironmentFactory 和资源组装 | Agent Loop 和文件持久化 |
| executor.py | 唯一 Core 执行入口和异常归一化 | Sessions Commit 和任务完成判断 |
| lifecycle.py | Runtime 状态、终止仲裁、资源释放 | Sessions RunState 和 Plan 状态 |
| frames.py | RuntimeFrame 生成 | 持久化 Event |
| approvals.py | Approval Action 的 Runtime 转换 | Approval 事实存储 |
| tool_adapters/ | Tool Port 到具体工具的适配 | Tool 副作用策略 |

第一阶段可以暂时保留现有 session_controller.py 和 session_coordinator.py 文件名，但逻辑必须遵守本表职责；迁移完成后只保留一套公开入口，不同时导出新旧名称。

### 9.2 Sessions 与 Workspace 边界

Context、Memory、Rollback 和 Workspace 放在 sessions/ 下只是源码聚合，不表示它们都由 Session State 管理。

第一版继续使用中立的 sessions/workspace.py，不建立独立 workspace/ 包。它只提供路径边界、文件 Hash、变更描述和低级 Git 查询，不拥有工具权限、Context 治理、Rollback 策略或新的事实存储。

tools/sandbox.py 仍负责工具执行时的安全和权限策略；sessions/rollback/ 仍负责回滚判断和流程。二者可以复用 sessions/workspace.py 的中立能力，但不能互相接管职责。

## 10. 分阶段迁移顺序

### 阶段 A：冻结协议与契约测试

建立 Runtime 状态、Input/Outcome、Commit 幂等、终态语义和资源释放测试。此阶段不增加兼容包装层。

### 阶段 B：统一 Core 入口

将 Prompt 和 Resume 收敛到统一 AgentLoop.run(input, ports)，统一 Event、Boundary 和 Outcome 协议。

### 阶段 C：建立 Environment 与 ResourceScope

完成 Session/Run 资源划分、CancellationToken、deadline、task tracking 和幂等 release。

### 阶段 D：提取唯一 RunExecutor

把 Gateway 中的 Core task、事件转发、异常、取消和超时逻辑迁入 Executor。Gateway 只调用 Controller/Coordinator。

### 阶段 E：收敛 Coordinator

将大型旧 Coordinator 拆为 Prepare、Recovery、Commit、Termination 和 Release 编排。Context、Memory、Rollback 和 Workspace 逻辑回到各自模块。

### 阶段 F：接入统一 Commit Protocol

所有消息、领域事件、组件快照、Checkpoint 和 Run State 通过 commit_run_boundary 保存。接入 Progress、Waiting、Terminal 三种边界，并处理幂等和 revision 冲突。

### 阶段 G：接入完整终止流程

接入 User Cancel、Deadline、Core/Model/Tool Failure、终止仲裁、协作式/强制取消和 Terminal Commit 失败处理。

### 阶段 H：接入恢复并删除旧路径

验证 Prompt 重试、Progress 恢复、Waiting Resume、Terminal 不重复执行和 Workspace Changed。最后删除旧的 Gateway Core 调用、RuntimeSessionRecords、直接 Repository 写入口和历史兼容读取。

每个阶段都必须先更新调用者和契约测试，再删除无调用者的旧代码。阶段结束后项目只能保留一套调用口径。

## 11. 删除清单与禁止回归

重构完成后不得存在：

- Gateway 直接调用 Core Agent Loop。
- Gateway 自己创建或取消 Core task。
- Runtime 直接写 Sessions Repository。
- Coordinator 直接追加 Message、Event 或 Checkpoint。
- RuntimeSessionRecords 或等价的第二份 Run 事实。
- Prompt 和 Resume 两套执行编排。
- 多套 Terminal Commit 或重复 Result 写入。
- Runtime 自己实现 Context 选择或 Memory Admission。
- Sessions 解释 Tool 副作用或提供 Effect 抽象。
- 旧 Session 文件格式和旧 API 的兼容读取。

## 12. 测试与验收矩阵

### 协议与状态

- Runtime 状态只按允许的转移变化。
- Waiting、Terminal 和 Released 不互相混淆。
- Prompt/Resume 使用同一 Executor。
- 同一个 Run 不能并发执行两个活动尝试。

### Commit 与恢复

- 相同 commit_id 重试不重复写入。
- 过期 revision 被拒绝。
- 消息和最终结果不会重复持久化。
- 每个稳定边界都有可恢复 Checkpoint。
- Workspace 改变会阻止恢复。
- Terminal Commit 成功后不会再次执行 Run。

### 终止与资源

- Cancel、Timeout、Core/Model/Tool Failure 走同一终止管线。
- 取消可以传播到所有 Run 子任务。
- 不响应取消的任务最终不会阻塞资源释放。
- release 重复调用不会产生二次清理错误。
- Terminal Commit 失败时不会错误发送成功 Frame。

### 主链路

- CLI Prompt 可以完成完整 Model/Tool/Message/Terminal 流程。
- Tool Approval 和 Plan Approval 不绕过 Runtime/Tools 规定的接口。
- Waiting 后 Resume 使用原 run_id 并从正确 Checkpoint 继续。
- 进程崩溃、上下文压缩和 Workspace 变化场景有明确结果。

## 13. 第一版明确不实现

以下能力留待后续迭代，不应提前进入 Runtime 核心：

- 通用事务管理器或分布式锁服务。
- Event Sourcing 和通过重放 Event 恢复 Run State。
- 独立 Effect 抽象。
- 多级 Checkpoint 策略和自动压缩策略。
- 跨 Session 分布式执行和租约协议。
- 通用 Port Registry 或复杂依赖注入容器。
- 独立 Runtime 调度器和后台工作队列。

## 14. 最终目标

重构完成后，Coding Agent 的一次运行应当只有以下事实流：

```text
Action
  -> Runtime Coordinator
  -> RunExecutor
  -> Core / Model / Tool
  -> Boundary Outcome
  -> SessionStateService.commit_run_boundary
  -> RuntimeFrame
```

Runtime 负责把执行可靠地推进到下一个边界，Core 负责决定任务如何推进，Tools 负责工具行为和副作用，Context/Memory 负责各自治理，Sessions 负责保存和恢复已经提交的事实。任何层都不能通过第二套接口重新实现另一层的职责。
