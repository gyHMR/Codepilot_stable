# Codepilot V2 运行主线导读

这份文档按一次真实 coding agent run 来读 Codepilot。它不是旧文件索引，也不是
抽象分层口号，而是告诉新手：一个用户请求从哪里进入、跨层传递什么信息、哪些
模块负责思考、哪些模块负责保存证据。

当前 V2 主脊柱是：

```text
UserAction
  -> RuntimeGateway.dispatch()
  -> SessionController.prepare_run() / prepare_resume() / apply_command()
  -> core.run_agent_loop() / resume_agent_loop()
  -> SessionController.commit_run()
  -> RuntimeFrame
```

依赖方向保持：

```text
protocols <- llm/tools <- core <- sessions/observability <- extensions <- runtime <- interfaces
```

配套文档：

| 文档 | 用途 |
|---|---|
| `docs/design/AGENT_LOOP_CONTRACTS_V2.md` | 完整 V2 契约、目标模块形态和落地检查点 |
| `docs/design/ARCHITECTURE_BOUNDARIES.md` | 当前各层职责、允许接口和禁止跨层对象 |
| `docs/design/V2_COMPLETION_AUDIT.md` | 判断 V2 重构是否真正完成的逐项证据矩阵 |
| `docs/design/NEXT_CONVERSATION_SUMMARY.md` | 下一次接续重构时的状态摘要和验证记录 |

读代码时建议沿运行方向看，而不是从底层类型一层层倒推。

## 1. 接口层：只把用户动作翻译成 UserAction

入口文件：

| 场景 | 先看文件 |
|---|---|
| CLI 普通运行 | `src/codepilot/interfaces/cli/main.py`、`runner.py` |
| CLI 交互 shell | `src/codepilot/interfaces/cli/shell.py` |
| CLI 审批展示 | `src/codepilot/interfaces/cli/approval.py` |
| DingTalk 远程消息 | `src/codepilot/interfaces/dingtalk/bridge.py` |
| Evaluation | `src/codepilot/evaluation/runner.py`、`cli.py` |

接口层只做两件事：

1. 把用户输入转为 `PromptSubmitted`、`CommandSubmitted`、`ApprovalDecided`、`RunCancelled`。
2. 把 `RuntimeFrame` 渲染成终端、RPC JSONL、DingTalk 回复或评测记录。

接口层不应该知道 `AgentSession`、`Agent`、`ToolRuntime`、store、memory writer。
最终回答来自 `RunFinishedFrame`，命令输出来自 `CommandFinishedFrame`，审批提示来自
`ApprovalRequiredFrame`。

## 2. Runtime 层：应用门面和 live registry

核心文件：

| 文件 | 作用 |
|---|---|
| `src/codepilot/runtime/gateway.py` | `RuntimeGateway.open_session()` / `dispatch()` 主入口 |
| `src/codepilot/runtime/session_opening.py` | `SessionOpenIntent`、`SessionRef`、`AppSessionView` |
| `src/codepilot/runtime/actions.py` | `UserAction` 和 `RuntimeFrame` |
| `src/codepilot/runtime/sessions.py` | session registry、active run registry |
| `src/codepilot/runtime/approvals.py` | approval transaction registry |
| `src/codepilot/runtime/session_opening.py` / `views.py` | `SessionOpenIntent`、`AppSessionView`、`SessionStatus`、`CommandDescriptor` 等只读 DTO |
| `src/codepilot/runtime/configuration.py` | config explain 等配置视图 |
| `src/codepilot/runtime/assembly.py` | 装配模型、工具、扩展、hooks，并创建 `SessionController` |
| `src/codepilot/runtime/assembly_types.py` | `RuntimeAssembly`、诊断和能力目录等 runtime 内部装配记录 |
| `src/codepilot/runtime/assembly.py` | 配置、模型、工具、prompt、hook 的具体装配 |

Runtime 的职责是应用级调度：

```text
RuntimeGateway.dispatch(PromptSubmitted)
  -> controller.prepare_run()
  -> core.run_agent_loop()
  -> controller.commit_run()
  -> ProgressFrame / RunFinishedFrame / FailedFrame
```

审批路径是：

```text
ToolInterruption
  -> ApprovalRequiredFrame
  -> ApprovalDecided
  -> resume_agent_loop()
  -> ToolPort.resume()
```

Runtime 可以持有 live `SessionController`、active task、pending approval、assembly，
但 public surface 不返回 live object。旧 `RuntimeService` 已删除，旧
`submit_turn()` / `approve()` / `execute_command()` / `list_commands()` /
`list_pending_approvals()` / run getter / live getter 均不再作为入口。
`RuntimeGateway` 只暴露 `open_session()`、`dispatch()`、`describe()`、`close()`、
`close_all()`；命令目录和 pending approvals 通过 `describe()` 的应用视图读取。
runtime 装配层不再创建或返回旧 `AgentSession` live object；sessions 内部承载已命名为 `SessionRuntime`。

## 3. Session 层：会话事实源和 run 生命周期

核心文件：

| 文件 | 作用 |
|---|---|
| `src/codepilot/sessions/controller.py` | `SessionController.prepare_run()` / `commit_run()` |
| `src/codepilot/sessions/contracts.py` | `SessionRunIntent`、`PreparedAgentRun`、`SessionRunRecord` |
| `src/codepilot/sessions/session.py` | session-owned store、memory、context、rollback 的内部承载 |
| `src/codepilot/sessions/conversation_state.py` | session-owned messages、model、tools、listeners、last result |
| `src/codepilot/sessions/commands.py` | slash command 语义 |
| `src/codepilot/sessions/command_state.py` | slash command 需要的 session-owned action 和只读 view |
| `src/codepilot/sessions/history/task_recovery.py` | 任务恢复投影 |
| `src/codepilot/sessions/context/governor.py` | 每次模型调用前的上下文治理 |
| `src/codepilot/sessions/memory/*` | 记忆召回和沉淀 |
| `src/codepilot/sessions/history/git_rollback.py` | Git rollback baseline、plan、apply |

`prepare_run()` 做会话侧前置准备：

- 生成 run id。
- 执行 before hooks。
- prompt memory admission。
- task recovery begin / projection。
- context freshness steering。
- rollback baseline。
- 构造 `AgentLoopInput`，并把本次 run 的 `ContextPort` 放入 `PreparedAgentRun`。

`commit_run()` 做会话侧收尾：

- 写 run result、events、messages。
- 写 rollback metadata。
- 更新 task recovery。
- finalize memory。
- finalize context governor。
- 执行 after hooks。

Session 可以持有 store、memory、context governor、conversation state，但只向 runtime
返回 `SessionRunRecord` 和 snapshot。

## 4. Core 层：一次 agent loop 的执行引擎

核心文件：

| 文件 | 作用 |
|---|---|
| `src/codepilot/core/contracts.py` | `AgentLoopInput`、`AgentLoopPorts`、`AgentLoopOutcome` |
| `src/codepilot/core/loop.py` | `run_agent_loop()` / `resume_agent_loop()` |
| `src/codepilot/core/model_turn.py` | 构造 `LLMRequest`，消费 `ModelPort.stream()` |
| `src/codepilot/core/tool_turn.py` | 执行一轮工具调用，转换 `ToolObservation` |
| `src/codepilot/core/stopping.py` | 完成结果、重复工具调用、工具上限等停止判断 |
| `src/codepilot/core/task_runtime.py` | V2 task-control adapter |
| `src/codepilot/core/task_control/*` | 任务步骤、规则、恢复、计划状态 |

Core 不 import sessions/runtime/interfaces，也不写 `.codepilot`。它只回答一个问题：
给定上下文、模型端口、工具端口和限制，如何跑完一次 agent loop？

主循环可以按这四段读：

```text
model_turn
  -> assistant message / tool calls
tool_turn
  -> tool observations / interruption
stopping
  -> final answer / max limit / repeated call / approval wait
task_runtime
  -> task context injection / task summary
```

V2 core outcome 是唯一输出。任务控制、审批等待、工具上限、重试、workspace effects、
verification 都通过 `AgentLoopOutcome` 返回给 session。

旧 core loop island 已删除，新手阅读主线从 `src/codepilot/core/loop.py` 开始。

## 5. LLM 层：模型端口和 provider 适配

核心文件：

| 文件 | 作用 |
|---|---|
| `src/codepilot/llm/ports.py` | `ModelPort`、`LLMRequest`、`LLMEvent`、`ProviderModelPort` |
| `src/codepilot/llm/api_registry.py` | provider 分发 |
| `src/codepilot/llm/providers/*` | Anthropic / OpenAI-compatible 等 provider |
| `src/codepilot/llm/event_stream.py` | 旧 provider stream 到统一 assistant message |

Core 只依赖 `ModelPort.stream(request)`。provider registry、API key、stream 函数、
message conversion 都被 `ProviderModelPort` 包起来。LLM 层不接收 session id 之外的
应用对象，也不管理 agent run。

## 6. Tools 层：工具端口、权限、审批、安全执行

核心文件：

| 文件 | 作用 |
|---|---|
| `src/codepilot/tools/ports.py` | `ToolPort`、`ToolInvocation`、`ToolObservation`、`ToolInterruption` |
| `src/codepilot/tools/execution.py` | `ToolRuntime` 权限、schema、审批、执行、安全结果 |
| `src/codepilot/tools/policy.py` | 权限决策 |
| `src/codepilot/tools/approval.py` | approval provider / deferred approval |
| `src/codepilot/tools/result_safety.py` | secret、PII、prompt injection 防护 |
| `src/codepilot/tools/builtins/*` | 文件、搜索、shell、workspace status |

V2 工具调用链：

```text
core.tool_turn
  -> ToolPort.execute(ToolInvocation)
  -> ToolObservation(success/error/approval_required)
```

审批不再由 runtime 手工拼工具结果。`ToolObservation(status="approval_required")`
会携带 `ToolInterruption`，runtime 只登记 transaction；用户决策后 runtime 进入
`resume_agent_loop()`，core 在恢复循环中调用 `ToolPort.resume(ToolResumeDecision)`。
审批后的工具结果作为正常 outcome evidence 返回，不再由 session reconciliation
helper 事后合并。

## 7. Context / Memory / Task / Rollback 的位置

这些设计理念保留，但职责位置要清楚：

| 设计 | 主要位置 | 跨层输出 |
|---|---|---|
| 上下文治理 | `sessions/context/governor.py` | `ContextPort.prepare()` 返回模型调用上下文和 report |
| 压力感知裁剪 | `sessions/context/policy.py`、`projector.py` | `context_prepared` 事件和 context report |
| memory 召回 | `sessions/memory/retriever.py` | prepared context 中的 memory blocks |
| memory 沉淀 | `sessions/memory/writer.py` | run commit 后 memory events |
| task recovery | `sessions/history/task_recovery.py` | `task_strategy.recovery_projection` 和 `AgentRunResult.task` |
| task control | `core/task_runtime.py`、`core/task_control/*` | `TaskSummary` |
| rollback | `sessions/history/git_rollback.py` | rollback metadata、rollback command record |

Core 可以执行 task-control，但不能直接读写 session task recovery store。Session 负责
把 `TaskSummary` 映射成下一次 run 的 recovery projection。

## 8. Slash Command 主线

命令不进模型：

```text
CommandSubmitted
  -> RuntimeGateway.dispatch()
  -> SessionController.apply_command()
  -> sessions.commands.apply_session_command()
  -> sessions.commands
  -> CommandFinishedFrame
```

`/mode`、`/fork`、`/switch`、`/memory`、`/rollback` 都应通过 session command intent
表达。Interface 只渲染命令结果，不反查 session 内部对象。

## 9. Observability

核心文件：

| 文件 | 作用 |
|---|---|
| `src/codepilot/observability/events.py` | 归一化事件 |
| `src/codepilot/observability/trace.py` | 构建 run trace |
| `src/codepilot/observability/summary.py` | 构建 run report |
| `src/codepilot/observability/audit.py` | 只读 audit bundle |

Observability 不参与主执行决策。它消费 `RuntimeFrame`、`SessionRunRecord`、
`AgentLoopOutcome.events`、run files，输出报告、trace、audit bundle。

## 10. 推荐阅读顺序

新手建议按这个顺序读：

```text
src/codepilot/runtime/actions.py
src/codepilot/runtime/gateway.py
src/codepilot/sessions/contracts.py
src/codepilot/sessions/controller.py
src/codepilot/core/contracts.py
src/codepilot/core/loop.py
src/codepilot/core/model_turn.py
src/codepilot/core/tool_turn.py
src/codepilot/core/stopping.py
src/codepilot/core/task_runtime.py
src/codepilot/llm/ports.py
src/codepilot/tools/ports.py
src/codepilot/sessions/session.py
src/codepilot/sessions/context/governor.py
src/codepilot/sessions/history/task_recovery.py
src/codepilot/sessions/history/git_rollback.py
```

带问题读时：

| 我想知道 | 先看 |
|---|---|
| 用户输入怎么进入主线 | `runtime/actions.py`、`runtime/gateway.py` |
| 一次 prompt 怎么执行 | `sessions/controller.py`、`core/loop.py` |
| 模型请求怎么构造 | `core/model_turn.py`、`llm/ports.py` |
| 工具怎么执行和审批 | `core/tool_turn.py`、`tools/ports.py`、`tools/execution.py` |
| 任务状态怎么进入 prompt | `core/task_runtime.py`、`core/task_control/controller.py` |
| 上下文为什么变短 | `sessions/context/governor.py`、`policy.py`、`projector.py` |
| memory 怎么召回和沉淀 | `sessions/memory/retriever.py`、`writer.py` |
| run 证据怎么保存 | `sessions/controller.py`、`sessions/persistence/run_store.py` |
| rollback 怎么判断安全 | `sessions/history/git_rollback.py` |
| 命令怎么执行 | `sessions/commands.py`、`sessions/command_state.py` |

## 11. 当前迁移边界

已迁到 V2 主线：

- RuntimeGateway / UserAction / RuntimeFrame。
- SessionController prepare/commit。
- Core loop contracts、model/tool ports、approval resume。
- Task context injection、task summary、task recovery writeback。
- Tool approval interruption。
- Core retry、tool events、重复工具调用和工具上限 gate。
- Plan-mode task strategy 的 planning budget 和外部步骤输入。
- Final verification grace、completion gate、post-tool task decision 已进入 V2 `core/stopping.py` / `core/loop.py`。
- Provider stream/complete、模型 capability 过滤和 vision capability 错误已进入 V2 `llm.ports.ProviderModelPort`。
- command-created derived session 通过 `SessionController.stage_derived_session()` /
  `claim_derived_controller()` 在 sessions/runtime 边界交接。

已删除的 legacy 参考：

- 旧 core loop island：`agent_loop.py`、`llm_runner.py`、`tool_coordinator.py`、`run_decisions.py`。
- 旧模型规划 bootstrap/discovery 执行器。
- 旧 approval result 事后拼接模块：`sessions.run_reconciliation`。

后续重构不应重新依赖旧 Agent 外观；有价值行为应进入 V2 `model_turn`、`tool_turn`、`stopping`、`task_runtime` 或端口实现。
