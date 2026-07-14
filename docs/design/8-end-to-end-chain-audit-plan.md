# Codepilot 端到端主链路清理排查实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan one chain at a time. Do not dispatch parallel implementation work across chains that share Core, Sessions, Runtime, or Tools contracts.

**目标：** 以 12 条端到端链路为单位，清理模块重构后产生的双重口径、契约不兼容、重复实现、边界倒置、真实运行故障和 Core 可读性问题，最终收敛为一套可恢复、可验证、可追踪的运行主干。

**方法：** 先冻结权威模型、ID、状态枚举和持久化边界，再按依赖顺序逐条完成“现状取证 -> 契约审计 -> 黄金场景 -> 修复 -> 删除旧路径 -> 验收”。每条链路只允许一个业务事实所有者，其他模块只能通过明确的 Port、DTO 或只读投影消费该事实。

**技术栈：** Python 3.10+、pytest、CodeGraph、AST/import 边界测试、Sessions 文件持久化、Runtime/Core/Tools/Context/Memory 现有公开契约。

---

## 1. 文档状态与范围

本文基于 2026-07-14 当前工作树，而不是仅基于 Git HEAD。当前工作树正在进行 Core、Runtime、Context、Memory、Tools 和 Sessions 的分模块重构，因此审计时必须以磁盘上的实际代码为准，并保留用户已有修改。

本轮纳入：

- `src/codepilot/protocols/`
- `src/codepilot/llm/`
- `src/codepilot/tools/`
- `src/codepilot/core/`
- `src/codepilot/sessions/`
- `src/codepilot/observability/`
- `src/codepilot/extensions/`
- `src/codepilot/runtime/`
- `src/codepilot/interfaces/`
- 与上述链路直接相关的 `test/` 和 `docs/design/`

本轮不纳入：

- `src/codepilot/evaluation/` 的内部设计与实现审计。
- Benchmark 内容、结果和调参；不得把 `benchmarks/` 产物提交到 Git。
- 与 12 条链路无关的界面美化、功能扩展或基础设施重写。
- 通用 DI 容器、事件溯源、分布式调度器、通用事务框架等新架构。

发生设计不确定性时，不直接用代码替代决策。按以下顺序处理：

1. 本文已经冻结的权威关系和审计规则。
2. `docs/design/1core-design.md` 至 `7runtime-design.md` 中明确标记的目标边界。
3. 当前代码只用于描述现状，不自动升级为目标设计。
4. README 只作为用户文档证据，不能覆盖目标设计。
5. 若两个有效设计仍冲突，在 `docs/audit/runtime/decisions.md` 记录决策项，确认后再修改代码。

## 2. 当前事实底稿

### 2.1 当前源码分层

当前源码已经具备目标层级：

```text
protocols -> llm/tools -> core -> sessions/observability
                                   -> extensions -> runtime -> interfaces
```

目录存在并不代表边界已经成立。审计需要同时检查：

- import 依赖是否符合方向。
- 业务事实是否仍由下游模块重新维护。
- 相同能力是否存在多个公开入口。
- DTO、状态枚举和错误码是否在边界处发生隐式翻译。
- 设计文档、README、测试和实际文件是否一致。

### 2.2 当前普通 Run 的实际主调用链

基于当前代码，普通 Prompt 的实际主干是：

```text
Interface
  -> RuntimeGateway.dispatch()
  -> RuntimeGateway._dispatch_action()
  -> RuntimeGateway._run_prompt()
  -> SessionController.runs.prepare()
  -> RunCoordinator.prepare()
  -> RuntimeSessionCoordinator._prepare_run()
  -> SessionStateService.begin_run()
  -> RunCoordinator.create_environment()
  -> RuntimeGateway._execute_prepared_run()
  -> RunCoordinator.execute()
  -> RunExecutor.execute()
  -> RunExecutor._execute_core()
  -> core.run_core()
       -> ContextPreparationPort.prepare()
       -> ModelPort.stream()
       -> Core Reducer / Policy
       -> ToolExecutionPort（按决策进入）
       -> BoundaryPort.commit()
  -> RuntimeSessionStateAdapter.commit()
  -> SessionStateService.commit_run_boundary()
  -> RunCoordinator.commit()
  -> RuntimeSessionCoordinator._commit_run()
  -> RuntimeFrame
```

这条链路是后续所有审计的锚点。任何新的普通 Run 入口、Prompt 专用执行循环、绕过 `run_core()` 的模型或工具调用，都视为高优先级问题。

### 2.3 已确认的起始风险信号

以下只是规划阶段已经确认的风险信号，不替代逐链路最终结论：

1. `README.md` 仍引用已经不存在的 `runtime/assembly.py`、`sessions/controller.py`、`sessions/session.py`、`core/loop.py`、`core/task_control/` 等路径，文档口径已经落后于当前代码。
2. `runtime/session_coordinator.py` 超过 1500 行，同时涉及 Run 准备、恢复、Context、Memory、Plan、Rollback、终态提交和生命周期 Hook，是跨领域逻辑重新聚集的首要审计点。
3. `RuntimeGateway._execute_prepared_run()` 当前处理 ActiveRun、取消、资源收敛、Terminal Commit、release 和 Frame 投影；需要核对这些职责是否应由 Gateway、Coordinator、Executor 或 ResourceScope 分别拥有。
4. `RuntimeSessionCoordinator._commit_run()` 同时处理 Terminal Commit、Conversation 投影、Memory proposal、Rollback metadata、Context 校准和 Hook，需要验证是否存在提交后副作用失败、重复执行或职责混合。
5. `core/reducer.py`、`core/contracts.py`、`core/state.py`、`core/plan.py` 均较大。文件大小本身不是错误，但需要检查是否混入了多个状态机、重复校验或跨层翻译。
6. 当前模块边界基线测试可通过：

```text
python -m pytest \
  test/test_ports_boundary.py \
  test/test_runtime_boundary_contracts.py \
  test/test_project_layout.py \
  test/test_runtime_core_entry.py \
  test/test_context_memory_architecture.py -q

30 passed
```

## 3. 冻结唯一权威模型

### 3.1 业务事实所有权

| 业务事实 | 唯一语义所有者 | 持久化/执行角色 | 允许的投影 | 禁止行为 |
|---|---|---|---|---|
| 当前目标、任务状态、Plan、阻塞、完成原因 | Core | Sessions 透明保存 `core_state` | Runtime/Interface 只读 View | Runtime、Tools 或 Interface 自行推进 Plan/Task |
| 模型/工具观察后的任务事实 | Core Reducer | 随 CoreBoundary 提交 | Observability 读取事件 | Driver、Policy、Runtime 直接修改 CoreState |
| Session、Run、Message、WaitingState、Checkpoint、revision | Sessions | Sessions Repository | Runtime/Interface View | Runtime 维护第二份可恢复 Run 事实 |
| 当前进程的 task、deadline、取消、收敛、release | Runtime | 只在进程内存在 | RuntimeFrame/Trace | 写入 Sessions 充当恢复权威 |
| ToolRegistration、Catalog、ToolAttempt、ApprovalChallenge、ToolResult | Tools | Tools checkpoint 由 Sessions 不透明保存 | Core 消费最终 ToolResult；Interface 展示 Approval | Core/Runtime 重新实现工具权限、attempt 或结果语义 |
| 工具副作用声明和执行权限 | Tools | ToolResult effects；Workspace 实际状态 | Core 保存任务相关只读事实投影 | Core 按工具名称猜测副作用 |
| Workspace recovery checkpoint | Sessions 保存，Runtime 校验 | Sessions checkpoint + `sessions/workspace.py` 中立能力 | Interface 展示差异 | 将 checkpoint 当作 rollback baseline 或权限授权 |
| Rollback baseline、预览和回滚策略 | `sessions/rollback/` | Sessions 保存引用或 metadata | Runtime 命令编排、Interface 展示 | Tools、Core 或 SessionStateService 重写回滚策略 |
| Context 分层、投影、预算、压缩、Context checkpoint | Context | checkpoint 由 Sessions 不透明保存 | Core 只消费 `PreparedModelContext` | Runtime/Core 自行拼 Prompt 或选择历史 |
| 长期 Memory 记录、准入、召回、冲突和生命周期 | Memory | Memory Repository | Context 召回 Active；Runtime 提交 Candidate | Sessions/Core 直接读写 Memory 文件或实现准入 |
| Provider 请求、流式事件、重试分类、模型能力 | LLM/Runtime Model Adapter | Provider 自身资源 | Core 消费标准 ModelPort 事件 | Core 解析 Provider 私有事件或实现 retry/backoff |
| Core Domain Event | Core | 随 Boundary 原子提交 | Runtime 加 envelope，Observability/Interface 投影 | Interface 事件反向修改业务状态 |
| Runtime Event envelope、live/durable 分流 | Runtime | durable 交给 Sessions，live 只内存传播 | Observability、CLI、Web、DingTalk | Live Event 充当恢复或完成依据 |
| 展示数据和交互格式 | Interface | 不拥有业务事实 | CLI/Web/DingTalk 各自渲染 | Interface 读取内部对象后自行推断状态机 |

### 3.2 投影不等于第二份权威

以下对象可以同时存在，但必须有单向派生关系：

```text
CoreState -> Runtime public view -> Interface rendering
ToolAttemptRecord -> ToolResult -> ToolResultMessage
CoreBoundary -> CommitRunBoundaryRequest -> RunCheckpoint
CoreOutcome -> AgentRunResult -> RuntimeFrame
Context internal report -> Context command view
MemoryRecord -> Context L3 projection
```

审计时不能只因为字段相似就强行合并。允许保留不同语义层的状态枚举，但必须满足：

- 有唯一转换函数。
- 转换是单向的。
- 映射穷尽所有枚举值。
- 不通过自然语言、`status + reason` 或默认分支猜测。
- 有表驱动测试固定映射。

## 4. 冻结 ID、状态与边界词汇

### 4.1 ID 贯穿表

| ID | 生成/接纳方 | 必须贯穿的链路 | 幂等用途 |
|---|---|---|---|
| `session_id` | Runtime 打开请求，Sessions 接纳并持久化 | Interface -> Runtime -> Sessions -> Events | Session 重开与界面路由 |
| `request_id` | Interface/Runtime Action | Prompt -> `begin_run` | Prompt 重试不重复创建 Run/UserMessage |
| `run_id` | Runtime 生成，Sessions 接纳为权威 Run 身份 | Runtime -> Core -> Tools/Context/Events -> Sessions | Resume/Continuation 必须复用原 Run |
| `message_id` | Sessions 按提交确定性生成 | MessageRecord -> Context/Conversation | Boundary 重试不重复消息 |
| `observation_id` | Core | Model/Tool/Command Observation -> Reducer ledger | 重复归约不改变 CoreState |
| `tool_call_id` | 模型协议 | LLM -> Core -> Tools -> ToolResult -> Message/Event | 同一调用必须最终闭合一次 |
| `attempt_id` | Tools 按执行请求确定性生成 | Tool prepare/approval/execute/resume | 防止审批恢复后重复执行 |
| `approval_id` | Tools | ToolAttempt -> RuntimeFrame -> Interface decision -> Tools resume | 决策只能作用于原挑战 |
| `commit_id` | Runtime Boundary Adapter | CoreBoundary -> Sessions commit | 相同提交重试返回同一 receipt |
| `checkpoint_id` | Sessions 按 commit 派生 | RunState -> Recovery | 只从最近成功边界恢复 |
| `event_id` | Core/Runtime/Sessions 按事件层级生成 | Durable/Live -> Recorder/Interface | 重放去重与链路关联 |

每条链路报告必须包含一张“ID 生成点、透传字段、校验点、丢失点、重新生成点”表。发现中途重新生成业务 ID 时，除明确的边界 envelope ID 外，默认按 P1 处理。

### 4.2 当前必须对齐的状态枚举

| 领域 | 当前权威枚举 |
|---|---|
| Core Task | `active / blocked / satisfied / abandoned` |
| Core Outcome | `completed / waiting / failed / cancelled` |
| Core Boundary | `before_model / after_model / before_tools / after_tools / waiting / before_terminal` |
| Core Wait | `tool_approval / user_input / plan_confirmation / continuation` |
| Plan | `proposed / active / completed / rejected / abandoned` |
| Plan Step | `pending / in_progress / completed` |
| Sessions Run | `created / running / waiting / completed / failed / cancelled` |
| Sessions Phase | `received / model / tools / finalizing / finished` |
| Sessions Commit | `progress / waiting / terminal` |
| Resume Point | `before_model / after_model / before_tools / after_tools / before_finalization` |
| Runtime Lifecycle | `new / preparing / executing / waiting / resuming / cancelling / finalizing / terminal / released` |
| Tool Attempt | `received / validating / resolving_access / awaiting_approval / awaiting_input / queued / running / succeeded / failed / denied / timed_out / cancelled / interrupted` |
| Tool Result | `success / error / denied / approval_required / user_input_required / cancelled / timed_out / interrupted` |

重点不是把所有枚举合并成一个，而是建立以下显式映射：

```text
ToolAttemptState -> ToolResult.status
ToolResult.status -> Core Observation / Task facts
CoreWait.kind -> Sessions WaitingState.kind
CoreBoundary.kind -> Sessions commit kind / phase / resume_point
CoreOutcome.status -> Sessions terminal_status / AgentRunResult.status / RuntimeFrame
Runtime termination reason -> CoreReason / stop_reason / external error code
```

## 5. 每条链路统一审计模板

每条链路必须完成以下 12 项，不允许只做模块代码阅读：

1. **入口与出口**：唯一公开入口、最终返回对象、调用者和消费者是否明确。
2. **输入输出契约**：相邻节点是否使用同一 DTO/Protocol，是否存在字段丢失、重命名或隐式默认值。
3. **ID 贯穿**：`session_id/run_id/request_id/tool_call_id/attempt_id/approval_id/commit_id` 的生成、透传和校验是否完整。
4. **状态机映射**：每个状态只由权威模块推进，跨层转换是否显式且穷尽。
5. **持久化边界**：哪些事实必须在继续副作用前提交，哪些只能是内存态，哪些只是展示投影。
6. **失败语义**：预期失败、未知异常、取消、超时、拒绝和恢复阻塞如何映射，是否发生错误吞噬或错误终态化。
7. **幂等与并发**：重试、重复请求、重复审批、双击、重连、崩溃恢复和并发 Action 是否只产生一次事实和副作用。
8. **唯一权威**：是否存在双重口径、相互覆盖的状态或通过自然语言推断结构化事实。
9. **重复实现**：是否对完全相同的功能写了多个方法、多个 wrapper 或多条执行路径。
10. **边界倒置**：Core 定义的 Port/协议是否被其他层绕开，其他层是否自行实现 Core 的决策、Reducer 或状态更新。
11. **真实运行风险**：在网络抖动、进程退出、磁盘失败、界面断连、工具不响应、工作区外部变化时是否会出错或泄漏资源。
12. **可读性与文档一致性**：调用链是否一眼可追踪，文件职责是否集中，README/设计/测试/代码是否使用同一名称和路径。

### 5.1 重复实现判定规则

下列情况判定为需要清理：

- 两个函数接收相同业务输入、产生相同业务结果，却分别维护状态或错误语义。
- Prompt、Resume、Continuation、Subagent 各自复制执行循环，而不是复用同一入口。
- 多个模块分别解析同一 `status/reason/metadata` 并得到同一业务判断。
- Adapter 除字段翻译外，还包含策略判断、状态推进或持久化。
- 为保留旧 API 建立无调用者兼容层，导致新旧路径同时存在。

下列情况可以保留，但要写清边界：

- 纯 DTO 转换函数。
- 面向不同 Interface 的无状态渲染函数。
- 同一权威事实的只读 View。
- 不同语义层之间有穷尽测试的显式状态映射。

精确重复先用 AST 规范化函数体筛查，语义重复再用 CodeGraph 查看调用者、被调用者和状态副作用。工具结果只作为候选，最终以业务语义和调用路径判断，不按代码相似度自动删除。

### 5.2 问题分类与严重级别

| 类型 | 含义 |
|---|---|
| `AUTH` | 双重权威或状态被多个模块推进 |
| `CONTRACT` | DTO、字段、枚举或错误码不兼容 |
| `DUP` | 相同能力或调用路径重复实现 |
| `BOUNDARY` | 依赖倒置、绕过 Port、跨层实现业务规则 |
| `PERSIST` | 提交、revision、checkpoint、恢复或幂等错误 |
| `LIFECYCLE` | 取消、超时、并发、资源释放或后台任务泄漏 |
| `BEHAVIOR` | 真实使用中会产生错误结果或副作用 |
| `READABILITY` | 职责混合、调用链晦涩、过度抽象或大文件失控 |
| `DOC` | 设计、README、测试和代码口径漂移 |

| 级别 | 判定标准 |
|---|---|
| P0 | 数据损坏、安全绕过、重复外部副作用、不可恢复的持久化错误 |
| P1 | 主链路错误终态、重复模型/工具调用、取消泄漏、恢复执行错误 |
| P2 | 双重口径、重复实现、边界倒置、重要异常语义不一致 |
| P3 | 可读性、命名、文档漂移或低风险冗余 |

每个问题记录必须包含：链路、类型、级别、设计证据、代码证据、触发场景、当前所有者、目标所有者、删除项、测试和是否需要用户决策。

## 6. 十二条链路的审计清单

### 6.1 链路总表

| # | 链路 | 当前主要入口/文件 | 首要权威 | 最小黄金场景 |
|---|---|---|---|---|
| 1 | 启动与能力装配 | `interfaces/*/main.py`、`runtime/config.py`、`runtime/builder.py`、`llm/registry.py`、`extensions/`、`tools/registry.py` | Runtime 负责装配；各能力模块拥有自身语义 | 配置加载后 Provider、Tools、MCP、Skill 可见，Session 关闭后资源释放 |
| 2 | Session 生命周期 | `RuntimeGateway.open_session/close`、`runtime/registry.py`、`sessions/service.py` | Sessions 持久化；Runtime 管理进程内实例 | 创建 -> Prompt -> 关闭 -> 重开 -> 历史一致 -> 删除无越界 |
| 3 | 普通 Run / 模型调用 | `RuntimeGateway.dispatch` -> `RunCoordinator` -> `RunExecutor` -> `run_core` | Core 任务语义，Sessions Run 事实，Runtime 执行 | 用户文本 -> 模型文本 -> Terminal Commit -> completed Frame |
| 4 | 工具执行 | `core/tool_step.py`、`tools/runtime.py`、`tools/registry.py` | Tools | ToolCall -> prepare -> before_tools -> execute -> ToolResult -> 模型总结 |
| 5 | 审批与用户交互 | `ToolRuntime`、`RuntimeSessionCoordinator._prepare_resume`、Gateway approval Action | Tools challenge/attempt；Sessions waiting/checkpoint | 挂起 -> 进程重启 -> 批准 -> 原 attempt 仅执行一次 |
| 6 | 取消、超时与释放 | `runtime/environment.py`、`executor.py`、`gateway.py`、`lifecycle.py` | Runtime | 工具/模型执行中取消与 deadline，终态唯一且无残留任务 |
| 7 | 崩溃恢复与继续 | `sessions/service.py`、`session_state_adapter.py`、`session_coordinator.py` | Sessions 恢复事实；Runtime 编排 | `after_model` 崩溃恢复不重复模型；`before_tools` 不盲目重放副作用 |
| 8 | Workspace 检查点与回滚 | `sessions/workspace.py`、`sessions/rollback/`、`runtime/commands.py` | Sessions checkpoint；Rollback 策略模块 | Run 修改文件 -> preview -> rollback -> 文件与 Run metadata 对齐 |
| 9 | Context 治理 | `sessions/context/`、`core/model_step.py` | Context | 历史/任务/工具证据/Memory -> 预算 -> 压缩 -> PreparedModelContext -> checkpoint |
| 10 | Memory 管理 | `sessions/memory/`、Runtime terminal proposal | Memory | 明确记忆/自动 Candidate -> 新 Session 召回 -> L3 注入，不污染当前任务事实 |
| 11 | Plan 与 Subagent | `core/plan.py`、`core/commands.py`、`runtime/subagents/`、特殊 Tool adapter | Core Plan；Runtime Subagent 调度 | Plan 确认 -> 只读 Subagent -> 部分结果 -> 主 Run 合并，禁止写工具 |
| 12 | 事件、可观测性与界面投影 | `core/events.py`、`runtime/contracts.py`、`observability/`、CLI/Web/DingTalk | Core 事件语义；Runtime envelope；Sessions durable | 同一 Run 的 durable/live 事件可关联，SSE 重放不改变权威状态 |

### 6.2 链路 1：启动与能力装配

重点检查：

- `SessionOpenIntent -> load_runtime_config() -> build_runtime_session()` 是否只有一条装配路径。
- Provider Registry 是否 Session 隔离，是否存在 import 时隐式注册或全局污染。
- Tool Registry 的 builtin/caller/skill/extension/MCP 覆盖规则是否唯一。
- MCP 和 Skill 延迟加载失败是否明确降级，关闭 Session 时 transport 是否释放。
- CLI、Web、DingTalk 是否通过同一 Runtime Builder，而不是各自拼装能力。
- README 和设计文档必须改为当前真实文件名。

主要证据测试：

```text
test/test_workspace_model_config.py
test/test_llm_registry.py
test/test_builtin_registrations_v2.py
test/test_tool_extensions_v2.py
test/test_mcp_client_v2.py
test/test_skill_packages.py
test/test_dingtalk_entrypoint.py
```

### 6.3 链路 2：Session 生命周期

重点检查：

- RuntimeSession Registry 与 Sessions `SessionState` 是否严格区分进程内对象和持久化事实。
- `open/create/reopen/close/delete/fork/switch` 是否都通过公开 Runtime/Sessions 接口。
- 一个 Session 同时只能有一个活动 Run；并发 Prompt 必须稳定拒绝。
- Close 只释放资源，不修改已经提交的 Run 事实。
- Delete 的磁盘边界、关联 Run/Message/Event/Memory 处理必须明确，不能误删 Workspace。
- 重开 Session 时 model、mode、plan、context checkpoint 的来源优先级唯一。

主要证据测试：

```text
test/test_sessions_state_v2.py
test/test_sessions_persistence_v2.py
test/test_runtime_sessions_state_v2.py
test/test_command_router.py
test/test_web_history.py
test/test_web_service.py
```

### 6.4 链路 3：普通 Run / 模型调用

这是第一条完整审计和修复链路。

重点检查：

- Prompt 只能通过 `RuntimeGateway -> SessionController.runs -> RunExecutor -> run_core`。
- `CoreRunInput` 必须是 Prompt/Resume/Continuation 的统一输入，不保留旧 Run DTO。
- Context 在每次模型调用前物化，Core 不自行拼 Prompt。
- Provider retry 只在 Runtime Model Adapter/LLM 层，Core 只消费标准失败事件。
- Reducer 是所有 CoreState 变化的唯一入口；Driver 只编排。
- `before_model/after_model/before_terminal/terminal` 的提交先后必须可证明。
- Terminal Commit 失败不能返回成功 Frame。

黄金场景断言：

```text
同一 request_id 只创建一个 Run 和一条 UserMessage
同一 run_id 贯穿 Core、LLM correlation、Boundary、Result 和 Event
模型只调用一次
最终 AssistantMessage 只持久化一次
RunState == completed, phase == finished, checkpoint is None
Session current_run_id is None
```

主要证据测试：

```text
test/test_core_driver.py
test/test_core_reducer.py
test/test_core_policy.py
test/test_runtime_core_entry.py
test/test_runtime_contracts.py
test/test_runtime_sessions_state_v2.py
test/test_web_runtime_integration.py
```

### 6.5 链路 4：工具执行

重点检查：

- Model `ToolCall` 到 `ToolExecutionRequest` 不丢失 `run_id/session_id/tool_call_id/registration_id`。
- `prepare_batch()` 无外部副作用；`before_tools` 成功后才能 `execute_prepared()`。
- 参数解析、Schema、AccessResolver、PermissionEngine、Handler、Output Codec 顺序唯一。
- ToolAttemptState 与 ToolResult.status 有一张穷尽映射表。
- 同一 batch 的并发、barrier、顺序和 interrupted 结果语义一致。
- 所有已持久化 ToolCall 在等待或终态前得到最终 ToolResult，不能留下孤立调用。
- Core 特殊 Plan/Interaction adapter 只翻译命令，不维护第二份状态。

主要证据测试：

```text
test/test_tool_contracts_v2.py
test/test_tool_argument_parsing_v2.py
test/test_tool_execution_v2.py
test/test_tool_runtime_v2.py
test/test_tool_security_state_v2.py
test/test_tool_special_adapters_v2.py
test/test_core_driver.py
```

### 6.6 链路 5：审批与用户交互

重点检查：

- ApprovalChallenge 和 InteractionRequest 的完整记录只属于 Tools。
- Sessions WaitingState 只保存公共恢复字段，不复制完整 challenge。
- Waiting Commit 成功后才能向 Interface 展示暂停。
- Resume 使用原 `run_id/tool_call_id/attempt_id`，不能创建第二个 ToolAttempt。
- 批准、拒绝、重复决策、错误 Session、过期 challenge 都有确定结果。
- `prepare_resume()` 必须在副作用执行前把 prepared tool checkpoint 原子保存。
- 进程在批准后、执行前、执行后、after_tools 前崩溃时都有明确恢复策略。

主要证据测试：

```text
test/test_runtime_tool_resume_atomic.py
test/test_runtime_sessions_state_v2.py
test/test_tool_security_state_v2.py
test/test_tool_special_adapters_v2.py
test/test_cli_refactor.py
test/test_dingtalk_contract.py
test/test_web_service.py
```

### 6.7 链路 6：取消、超时与资源释放

重点检查：

- 用户取消、deadline、Interface stream 取消和内部异常进入同一个终止仲裁。
- Runtime Lifecycle 的每个转移只在一个组件实现。
- 取消先协作传播，再强制取消；Model/Tool 子任务和 timer 都由 RunResourceScope 跟踪。
- 进入 Terminal Commit 后取消仲裁关闭，Commit 和 release 不可被 Interface 断连打断。
- `release()` 幂等，重复调用不报错，不释放 Session 级服务。
- timeout 必须映射为 failed/deadline_exceeded，用户取消映射为 cancelled。
- 终态、Frame、RunState 和 ErrorInfo 不出现互相矛盾的状态。

主要证据测试：

```text
test/test_runtime_environment.py
test/test_runtime_contracts.py
test/test_runtime_sessions_state_v2.py
test/test_tool_execution_v2.py
test/test_model_port_v2.py
test/test_web_service.py
```

### 6.8 链路 7：崩溃恢复与继续执行

重点检查：

- 恢复依据只来自最后一次成功的 Sessions checkpoint。
- `inspect_recovery -> workspace validation -> component restore -> resume_run -> execute` 顺序唯一。
- `after_model` 恢复不能重复模型调用。
- `after_tools` 恢复不能重复工具执行。
- `before_tools` 后执行结果未知时不得自动重放有副作用工具。
- 未提交 Python 栈、半个流式响应和 Runtime 内存对象不能被伪装成可恢复状态。
- schema 升级只发生在读取边界，新写入只使用当前 CoreState schema。

故障注入点：

```text
begin_run 前
progress commit 前/后
after_model commit 后
before_tools commit 后
tool execution 后、after_tools commit 前
waiting commit 后
terminal commit 前/后
```

主要证据测试：

```text
test/test_sessions_persistence_v2.py
test/test_sessions_state_v2.py
test/test_runtime_sessions_state_v2.py
test/test_runtime_tool_resume_atomic.py
test/test_context_checkpoint.py
```

### 6.9 链路 8：Workspace 检查点与回滚

重点检查：

- Workspace checkpoint、Tool effects、Core workspace facts、Rollback baseline 四者语义分开。
- Run 前 baseline 的 clean/dirty 限制和远程入口限制一致。
- `affected_paths` 从 ToolResult 到 Core、Result、Rollback metadata 不丢失。
- 外部修改、Git HEAD 变化、文件缺失时必须阻止自动回滚或恢复。
- rollback/revert 只作用于该 Run 的受控路径，不触碰 `.codepilot/` 或用户后续改动。
- 回滚后文件事实、Run metadata、Context freshness 和界面结果重新对齐。

主要证据测试：

```text
test/test_rollback_v2.py
test/test_runtime_sessions_state_v2.py
test/test_command_router.py
test/test_dingtalk_contract.py
```

### 6.10 链路 9：Context 治理

重点检查：

- Core 只提交类型化 `ContextPrepareRequest`，Context 返回 `PreparedModelContext`。
- L0-L4、Required/Protected/Budgeted/DiscardFirst 是 Context 内部唯一规则。
- L2 Tool evidence 和 L4 ToolResultMessage 使用同一 ProjectionPlan 和来源引用。
- token 估算、预算选择、compaction 和最终硬校验没有多套实现。
- 压缩失败、Memory recall 失败、artifact 缺失必须安全降级。
- Context checkpoint 只保存最小恢复状态，不复制 Sessions Message 或长期 Memory。
- Runtime 只注入 Context port 和收集 checkpoint，不自行选择内容。

主要证据测试：

```text
test/test_context_contracts.py
test/test_context_layers.py
test/test_context_projection.py
test/test_context_budget.py
test/test_context_compaction.py
test/test_context_checkpoint.py
test/test_context_governance.py
test/test_runtime_context.py
```

### 6.11 链路 10：Memory 沉淀与管理

重点检查：

- 明确记忆、自动 proposal、candidate、active、disabled、superseded、deleted 的边界唯一。
- 自动 proposal 只在 Terminal Commit 成功后提交；失败不能回滚已完成 Run。
- 当前任务状态、Context summary、工具日志和敏感信息不得进入长期 Memory。
- User/Project scope、冲突、重复、编辑、审批和 purge 只由 MemoryService/Repository 实现。
- Context 只通过 MemoryRecallPort 读取 Active 记录。
- 新 Session 召回结果可追溯到 memory_id，且受 Context 预算控制。

主要证据测试：

```text
test/test_memory_admission.py
test/test_memory_repository.py
test/test_memory_recall.py
test/test_memory_management.py
test/test_memory_migration.py
test/test_runtime_context_memory_integration.py
test/test_context_memory_contracts.py
```

### 6.12 链路 11：Plan 与 Subagent

重点检查：

- Plan 只存在于 `CoreState.task.plan`，Runtime、Tools、Sessions 不维护第二份 Plan 事实。
- 所有 Plan 更新通过 CoreCommand -> Reducer，revision 和 evidence 校验唯一。
- Plan mode 的确认、修订、拒绝、放弃和 closeout 不依赖 Interface 自行推断。
- Subagent 只读工具白名单、预算、取消和结果缓存边界明确。
- 主 Run 与 Subagent 不共享可写 ToolRuntime，不允许 Subagent 修改 Workspace。
- Subagent partial/timeout/error 结果进入主 Run 的方式唯一，不直接写主 Run 的 CoreState。
- 主 Run 和 Subagent 是否复用 RunExecutor 必须与 Runtime 设计一致。

主要证据测试：

```text
test/test_core_plan_commands.py
test/test_plan_protocol_rewrite.py
test/test_plan_subagents.py
test/test_tool_special_adapters_v2.py
test/test_command_router.py
```

### 6.13 链路 12：事件、可观测性与界面投影

重点检查：

- Core Domain Event、Runtime envelope、Durable Event、Live Event 使用不同职责和明确映射。
- live event 丢失不能影响 RunState、Checkpoint 或最终结果。
- durable event 必须随边界提交，不由 Event Sink 直接写 Sessions。
- event_id、session_id、run_id、tool_call_id、turn_id 可用于完整关联。
- CLI、Web、DingTalk 只消费 Runtime Action/Frame，不读取 Core/Sessions 内部对象后重新推断。
- SSE 重放、断线重连和重复 event 不改变权威状态。
- Observability/Trace 只从规范事件生成，不定义第二套 Tool/Run 词汇。

主要证据测试：

```text
test/test_observability_v2.py
test/test_runtime_contracts.py
test/test_web_events.py
test/test_web_runtime_integration.py
test/test_cli_refactor.py
test/test_dingtalk_contract.py
```

## 7. 推荐执行顺序与阶段门

### 阶段 0：冻结系统契约

- [x] 创建 `docs/audit/runtime/00-authority-and-vocabulary.md`，复制并校验本文第 3、4 节的权威、ID、状态和映射。
- [ ] 创建 `docs/audit/runtime/issues.md`，使用统一问题记录格式。
- [ ] 创建 `docs/audit/runtime/decisions.md`，只记录需要用户确认的设计冲突。
- [ ] 运行架构边界基线测试并保存命令和结果摘要。
- [ ] 用 CodeGraph 输出当前普通 Run、Tool、Approval、Recovery 四条主调用链。
- [ ] 对 `src/codepilot` 做 import 方向扫描和精确重复函数候选扫描。
- [ ] 冻结公开入口清单；后续阶段不得新增第二条兼容路径。

阶段门：权威事实、ID、状态映射和公开入口全部有唯一书面口径；未决问题已被显式列出，没有靠实现者自行猜测的空白。

### 阶段 1：普通 Run 主干（链路 3）

- [ ] 建立 `docs/audit/runtime/03-normal-run.md` 的 as-is 调用图。
- [ ] 逐边界核对 `CoreRunInput/CorePorts/CoreBoundary/CoreOutcome`。
- [ ] 检查 Gateway、Coordinator、Executor、SessionCoordinator 的职责是否与 Runtime 设计一致。
- [ ] 补齐文本模型完成的链路级黄金测试。
- [ ] 对 Terminal Commit 失败、Provider stream 失败、重复 request_id 做故障测试。
- [ ] 修复时先迁移调用者，再删除旧入口和兼容投影。
- [ ] 更新 README 的主链路和真实路径。

阶段门：普通 Run 只有一条执行路径；模型、消息、终态提交均不重复；失败不会返回成功 Frame。

### 阶段 2：工具执行（链路 4）

- [ ] 建立 ToolCall 到 ToolResult 的 ID 和状态映射表。
- [ ] 验证 `prepare_batch` 与 `execute_prepared` 的副作用边界。
- [ ] 审计 Registry、Codec、Security、Execution、Result 的重复校验和错误转换。
- [ ] 补齐 before_tools commit 失败、batch barrier、stale registration 黄金场景。
- [ ] 删除绕过 ToolRuntime 的执行方法和旧结果类型。

阶段门：所有工具通过唯一 ToolRuntime；已持久化 ToolCall 最终闭合；副作用前一定存在成功边界提交。

### 阶段 3：审批与用户交互（链路 5）

- [ ] 画出 approval/interaction 的 suspend、persist、restart、resume 时序。
- [ ] 核对 WaitingState、Tool checkpoint、ApprovalChallenge 的所有权。
- [ ] 运行批准、拒绝、重复批准、错误 Session、进程重启场景。
- [ ] 验证 prepared resume checkpoint 先于实际工具副作用提交。
- [ ] 删除 Runtime/Interface 中对 ToolAttempt 的重复状态维护。

阶段门：审批恢复使用原 attempt，副作用最多一次，Waiting 和 Resume 可跨进程完成。

### 阶段 4：取消、超时与释放（链路 6）

- [ ] 明确 Gateway、Executor、RunResourceScope、Lifecycle 的唯一职责。
- [ ] 建立所有终止源到最终状态/原因的映射表。
- [ ] 注入模型不响应、工具不响应、Interface 断连、Terminal Commit 延迟。
- [ ] 验证资源收敛、release 幂等和 ActiveRun 清理。
- [ ] 删除重复 `try/finally` 清理和多处终态仲裁。

阶段门：任何终止源都只生成一个终态；后台 task、timer、transport 和 ActiveRun 无残留。

### 阶段 5：崩溃恢复与回滚（链路 7、8）

- [ ] 先审计恢复，再审计 rollback；二者不得混为同一 checkpoint。
- [ ] 对每个 CoreBoundary 注入进程退出并验证恢复动作。
- [ ] 验证 `after_model/after_tools` 不重复调用，`before_tools` 不盲目重放。
- [ ] 验证 workspace changed/missing/unknown 全部分支。
- [ ] 验证 rollback 对外部改动、dirty baseline 和受控路径的保护。

阶段门：每个稳定边界都有唯一恢复语义；恢复和回滚均不会覆盖用户后续修改。

### 阶段 6：Context 与 Memory（链路 9、10）

- [ ] 先冻结 Context 请求/响应和 checkpoint，再审计 Memory recall/admission。
- [ ] 检查 L2/L4 统一投影、token 预算、compaction 和 fallback。
- [ ] 检查 Memory proposal 必须发生在 Terminal Commit 成功后。
- [ ] 运行 artifact 缺失、compaction 失败、recall 失败、memory write 失败场景。
- [ ] 删除 Runtime 中的 Context 选择和 Memory 准入逻辑。

阶段门：Context 和 Memory 各自只有一个 Service 语义入口；失败可降级且不破坏已提交 Run。

### 阶段 7：Plan 与 Subagent（链路 11）

- [ ] 核对所有 Plan 命令是否经过 Reducer。
- [ ] 检查 Plan snapshot、revision、closeout 和 Interface 命令是否存在第二套规则。
- [ ] 验证 Subagent 只读工具、资源限制、取消和 partial result。
- [ ] 核对主 Run 与 Subagent 对 RunExecutor 的复用是否一致。
- [ ] 删除 Runtime/Tool 中直接修改 PlanState 的方法。

阶段门：Plan 只有 Core 权威；Subagent 无写副作用；主 Run 合并结果不绕过 Core。

### 阶段 8：启动、Session 与事件投影收口（链路 1、2、12）

- [ ] 在下游契约稳定后审计所有 Interface 的启动和装配入口。
- [ ] 验证 Session create/reopen/close/delete/fork/switch 的完整生命周期。
- [ ] 对齐 CLI/Web/DingTalk 的 Action/Frame 和错误语义。
- [ ] 验证 durable/live event、SSE replay、Trace 和审计投影。
- [ ] 清理 README、设计文档、导出面和废弃模块引用。

阶段门：所有 Interface 只依赖 Runtime 公开面；文档、代码、测试和实际路径一致；12 条黄金场景全部可单独运行。

## 8. Core 专项结构审计

Core 除参与链路 3、4、11 外，还要单独完成文件布局和可读性审计。

### 8.1 目标文件职责

| 文件 | 允许职责 | 重点禁止 |
|---|---|---|
| `contracts.py` | Core 对外 Port、Input、Boundary、Outcome、Decision DTO | Runtime/Sessions DTO、具体实现、持久化逻辑 |
| `state.py` | CoreState 及事实对象、序列化和受控 schema 升级 | I/O、策略决策、Runtime 生命周期 |
| `observations.py` | 模型/工具/用户/命令/取消的输入事实 | 下一步动作判断 |
| `reducer.py` | 纯状态归约、命令应用、领域事件 | Port 调用、I/O、隐藏的策略分支 |
| `policy.py` | 从 State/Observation 产生 Decision | 直接修改 State、调用 Model/Tool |
| `driver.py` | 固定 reduce-decide-execute-boundary 循环 | Plan 规则、权限规则、Provider retry、Sessions commit 实现 |
| `model_step.py` | 一次 Context prepare + 一次 ModelPort action | 多轮循环、Provider 私有重试、Session 操作 |
| `tool_step.py` | Core/Tools DTO 翻译、一次 batch prepare/execute | ToolAttempt/Approval 权威、权限和副作用策略 |
| `transcript.py` | 从消息序列派生未闭合 ToolCall/最后消息 | 持久化或第二份消息状态 |
| `plan.py` | Plan 值对象、验证和纯领域操作 | Runtime/Interface 命令编排 |
| `commands.py` | CoreCommand DTO 和解析 | 直接更新外部状态 |
| `events.py` | CoreDomainEvent | Runtime envelope 和 Interface 格式 |
| `tool_adapters/` | 特殊工具到 CoreCommand/Interaction 的窄翻译 | 自己维护 Plan、等待或 ToolAttempt 状态 |

### 8.2 Core 调用逻辑验收

目标调用逻辑必须可以压缩为：

```text
run_core
  -> normalize Observation
  -> reduce_observation
  -> CorePolicy.decide
  -> CallModel | ExecuteTools | Wait | Terminate
  -> apply_decision / reduce result
  -> commit CoreBoundary
  -> CoreOutcome
```

以下信号触发拆分或删除评审，但不是机械行数规则：

- 一个文件同时拥有两个以上独立状态机。
- 一个函数同时做 DTO 转换、策略判断、I/O 和持久化。
- 相同状态校验出现在三个以上文件。
- Driver 中出现工具名、Session phase、Provider 类型或 Interface 分支。
- Reducer/Policy 需要读取时间、文件、网络或可变全局状态。
- Adapter 产生新的业务事实，而不是翻译已有事实。
- 文件超过约 800 行或函数超过约 80 行时，必须说明为何仍是单一职责；不能仅因行数拆分。

### 8.3 Core 接口被外层重写的检查法

对 Core 的每个公开入口执行以下检查：

1. 用 CodeGraph 列出全部调用者。
2. 查找调用者是否在调用前后复制 Core 的状态判断。
3. 查找外层是否直接构造或替换 CoreState/PlanState。
4. 查找外层是否解析 Core message/reason 来推断结构化状态。
5. 查找是否存在第二个 loop、runner、controller 或 completion gate。
6. 保留纯 Adapter；删除带业务规则的重复实现。

## 9. 黄金场景与故障注入策略

### 9.1 链路级测试命名

保持当前 `test/` 平铺约定，新增测试使用：

```text
test/test_chain_startup_assembly.py
test/test_chain_session_lifecycle.py
test/test_chain_normal_run.py
test/test_chain_tool_execution.py
test/test_chain_approval_resume.py
test/test_chain_termination.py
test/test_chain_crash_recovery.py
test/test_chain_workspace_rollback.py
test/test_chain_context_governance.py
test/test_chain_memory_lifecycle.py
test/test_chain_plan_subagent.py
test/test_chain_event_projection.py
```

新增链路测试优先复用现有 fixture、fake ModelPort、ToolRegistration、SessionStateService 和 RuntimeGateway，不复制模块单测已经验证的内部细节。

### 9.2 每个黄金场景必须验证

- 最终用户可见结果。
- 权威状态和持久化文件。
- 所有关键 ID 的连续性。
- 模型/工具/提交调用次数。
- durable event 与 live event 的差异。
- 资源释放和 ActiveRun 清理。
- 重复执行同一请求后的幂等结果。

### 9.3 必须覆盖的真实故障

| 故障 | 主要链路 | 必须结果 |
|---|---|---|
| Provider 网络断开/stream incomplete | 3、6 | 结构化失败，不伪造完成消息 |
| MCP 启动或调用失败 | 1、4 | 能力不可用或 ToolResult 失败，Session 可关闭 |
| Tool handler 超时或忽略取消 | 4、6 | 最终收敛，cleanup 执行，attempt 终态明确 |
| Boundary/Terminal Commit I/O 失败 | 3、5、7 | 不继续副作用，不返回成功 Frame，可按 commit_id 重试 |
| 进程在各 checkpoint 后退出 | 5、7 | 从最后成功边界恢复，不重复副作用 |
| Interface 在 finalizing 时断连 | 6、12 | Terminal Commit 与 release 完成后再传播取消 |
| 重复 Prompt/Approval/Resume | 2、3、5、7 | 单 Run、单 Message、单 Attempt、单副作用 |
| Workspace 被外部修改 | 7、8 | 阻止恢复/回滚并给出结构化原因 |
| Context compaction 或 recall 失败 | 9、10 | 安全降级，不破坏 Run |
| Memory 自动写入失败 | 10 | 已完成 Run 不回滚 |
| Subagent 部分超时 | 11 | 返回 partial result，主 Run 可继续或明确失败 |
| SSE 重连和重复 event | 12 | 展示可重放，权威状态不变 |

## 10. 审计产物与记录格式

每条链路只维护一份报告，避免报告本身形成第二套口径：

```text
docs/audit/runtime/
├── 00-authority-and-vocabulary.md
├── issues.md
├── decisions.md
├── 01-startup-assembly.md
├── 02-session-lifecycle.md
├── 03-normal-run.md
├── 04-tool-execution.md
├── 05-approval-interaction.md
├── 06-termination-release.md
├── 07-crash-recovery.md
├── 08-workspace-rollback.md
├── 09-context-governance.md
├── 10-memory-lifecycle.md
├── 11-plan-subagent.md
└── 12-events-interfaces.md
```

每份链路报告固定包含：

```text
1. 范围与非范围
2. As-is 调用图（带真实 symbol/path）
3. 权威事实与边界 DTO
4. ID 贯穿表
5. 状态/错误映射表
6. 持久化与 checkpoint 表
7. 重复实现和边界倒置清单
8. 真实故障场景
9. 修复顺序和删除清单
10. 测试命令与结果
11. 未决设计问题
12. 链路完成结论
```

原始 trace、临时 AST 扫描结果、测试 workspace 和故障注入文件写入 `.codepilot/audit/` 或 pytest 临时目录，不提交敏感数据、运行产物或 benchmark。

## 11. 单条链路的标准执行步骤

后续每次只选择一条链路，严格按以下顺序执行：

1. **读取目标设计**：只读取与该链路直接相关的设计章节。
2. **CodeGraph 取证**：列出入口 symbol、调用路径、核心 DTO 和所有调用者。
3. **建立 as-is 图**：只描述当前代码，不提前写目标方案。
4. **填写 12 项审计模板**：所有结论必须有文件/符号/测试证据。
5. **运行现有测试**：记录当前通过和失败，不先修改断言来适配实现。
6. **添加黄金场景失败测试**：先证明真实链路缺口或边界冲突。
7. **确认目标权威**：若设计冲突，先进入 decision gate。
8. **最小修复**：一次只迁移一个事实或一个入口，不做无关重构。
9. **删除旧路径**：调用者迁移完成后立即删除重复方法、类型、兼容导出和文档引用。
10. **专项验证**：运行该链路的模块测试、链路黄金测试和架构边界测试。
11. **全局回归**：运行完整 pytest、compileall 和 `git diff --check`。
12. **关闭报告**：更新问题状态、删除清单和最终调用图，再进入下一链路。

建议验证命令：

```text
python -m compileall -q src/codepilot
python -m pytest test/test_chain_normal_run.py -q
python -m pytest test/test_core_driver.py test/test_core_reducer.py test/test_core_policy.py test/test_runtime_core_entry.py test/test_runtime_contracts.py test/test_runtime_sessions_state_v2.py test/test_web_runtime_integration.py -q
python -m pytest test/test_ports_boundary.py test/test_runtime_boundary_contracts.py -q
python -m pytest test -q
git diff --check
```

上面是首次执行链路 3 的精确命令。后续链路使用第 6 节列出的现有测试文件，并运行第 9.1 节对应的链路黄金测试文件。

## 12. 变更纪律

- 不同时修改两条共享状态模型的链路。
- 不为了短期通过测试增加兼容 wrapper、双写字段或 fallback 旧路径。
- 不把设计目标和当前实现混写；报告必须区分 as-is 与 target。
- 不在 Interface、Runtime 或 Adapter 中修补 Core/Tools/Context/Memory 的领域规则。
- 不把 Runtime 内存状态持久化为新的恢复权威。
- 不依赖日志文本、最终回答文本或错误消息字符串推断状态。
- 不以大规模文件重排代替边界修复；先收敛权威和调用入口，再判断是否拆文件。
- 不改动或提交 benchmark。
- 每次删除旧路径前，用 CodeGraph/`rg` 确认没有调用者和文档引用。
- 每条链路结束时，README 和设计文档同步更新，不把文档修复集中拖到最后。

## 13. 全部排查完成标准

只有同时满足以下条件，12 条链路清理才算完成：

1. 每个业务事实有唯一所有者，其他层只有 Port/DTO/只读投影。
2. 普通 Prompt、Resume、Continuation 和 Subagent 的执行入口关系清晰，不存在重复主循环。
3. 所有关键 ID 从入口贯穿到持久化、事件和界面，重复请求不会生成第二份事实。
4. Core、Sessions、Runtime、Tools 的状态枚举有显式、穷尽、可测试映射。
5. Progress、Waiting、Terminal Commit 和各 resume point 具有明确幂等语义。
6. 工具审批、取消、超时、崩溃恢复和 rollback 不重复副作用。
7. Context 和 Memory 不复制 Session/Run/Task 事实，失败可以安全降级。
8. Plan 只有 Core 权威，Subagent 不获得写 Workspace 的能力。
9. Durable Event、Live Event、Trace 和 Interface 投影不反向控制业务状态。
10. 完全相同的能力只保留一个实现；Adapter 只做翻译。
11. Core 文件职责和调用链符合第 8 节，无法解释职责的大文件已拆分或删减。
12. README、设计文档、测试名称、公开导出和实际源码路径一致。
13. 12 个链路黄金场景和关键故障注入测试全部通过。
14. `python -m compileall -q src/codepilot`、`python -m pytest test -q`、`git diff --check` 全部通过。
15. Git 变更中不包含 benchmark、临时 trace、测试 workspace、密钥或敏感数据。

## 14. 首次执行建议

下一步只执行“阶段 0 + 阶段 1”，不要同时进入 Tool、Context 或 Memory 修复：

```text
冻结 authority/ID/state
  -> 固定普通 Run as-is 调用图
  -> 新增最小文本完成黄金场景
  -> 审计 Gateway/Coordinator/Executor/SessionCoordinator 职责
  -> 收敛 Terminal Commit 和 Frame 返回
  -> 删除旧入口
  -> 完成链路 3 报告与回归
```

普通 Run 主干稳定后，工具、审批、取消和恢复才有可靠的共同承载面。否则后续每条链路都会继续在不稳定的 Runtime/Sessions 边界上各自增加补丁。
