# Codepilot Agent Loop Contracts V2

本文档定义 Codepilot V2 的目标层级契约。它不是当前代码接口清单，
也不是给旧实现补一层新名字。这里的接口、模块形态和依赖关系都从
一次 coding agent run 的职责流反推：

```text
UserAction
  -> RuntimeGateway.dispatch()
  -> SessionController.prepare_run() / prepare_resume() / apply_command()
  -> core.run_agent_loop() / resume_agent_loop()
  -> ModelPort / ToolPort
  -> SessionController.commit_run()
  -> RuntimeFrame
```

核心原则：

- 先问每层在 agent loop 中负责回答什么问题，再定义它暴露什么接口。
- 上层只提交意图，下层只返回职责范围内的结果或快照。
- 跨层传递意图、能力、执行结果和快照，不传递 live object。
- 保留 context governance、压力感知裁剪、memory、task recovery、rollback、
  approval、安全策略和 observability 的设计理念。
- 删除为了兼容旧调用链而存在的别名、薄包装、live getter 和绕路查询。
- 模块形态按职责聚合，不按文件大小拆分，也不按当前实现倒推。

## 1. 依赖方向

目标依赖方向：

```text
protocols <- llm/tools <- core <- sessions/observability <- extensions <- runtime <- interfaces
```

含义：

- 右侧可以调用左侧，左侧不能 import 或了解右侧。
- `protocols` 是最底层协议语言，不包含执行逻辑。
- `llm` 和 `tools` 是 core 的能力端口，不知道 session/runtime/interface。
- `core` 只执行一次 agent loop，不知道持久化、memory 文件、CLI 或 DingTalk。
- `sessions` 拥有会话事实和生命周期副作用，调用 core 但不依赖 runtime。
- `observability` 只读消费执行证据，不参与主链路决策。
- `extensions` 加载能力包，由 runtime 装配，不参与 agent loop 判断。
- `runtime` 是应用门面，可以持有 live session 和 assembly，但不能向 interface
  暴露 live object。
- `interfaces` 只做输入适配和输出渲染。

依赖允许关系：

| 层级 | 可以依赖 | 禁止依赖 | 原因 |
|---|---|---|---|
| `protocols` | 标准库 | 所有 `codepilot.*` 业务层 | 协议必须是无方向的共同语言 |
| `llm` | `protocols` | `core`、`sessions`、`runtime`、`interfaces` | 模型层只适配 provider |
| `tools` | `protocols` | `core`、`sessions`、`runtime`、`interfaces` | 工具安全边界不能知道调用方 |
| `core` | `protocols`、`llm`/`tools` ports | `sessions`、`runtime`、`interfaces` | core 是一次 run 的执行引擎 |
| `sessions` | `protocols`、`core`、`llm`/`tools` 协议视图 | `runtime`、`interfaces` | session 是会话语义层，不是应用门面 |
| `observability` | `protocols`、`core`/`sessions` 结果 DTO | `interfaces`，以及会改变执行的模块 | observability 是只读证据视图 |
| `extensions` | `protocols`、工具/命令协议 | `core`、`sessions`、`interfaces` | extension 生产能力，不驱动 run |
| `runtime` | `sessions`、`extensions`、`observability`、`llm`/`tools` 装配 | `interfaces` | runtime 是被 interface 调用的应用服务 |
| `interfaces` | `runtime` public contract | `core`、`sessions`、`tools`、`llm` 内部 | interface 不理解 agent 内部 |

## 2. 跨层信息

V2 只允许五类信息跨层：

| 类别 | 例子 | 说明 |
|---|---|---|
| Identity | `session_id`、`run_id`、`approval_id`、`tool_call_id` | 用于关联，不携带行为 |
| Intent | prompt、command、approval decision、cancel request | 上层表达要做什么 |
| Capability | model descriptor、tool schema、command descriptor、hook descriptor | 下层声明能做什么 |
| Execution | model event、tool observation、agent event、loop outcome | 执行过程中产生的事实 |
| Snapshot | session view、run record、context report、audit bundle | 只读状态，不暴露 live object |

默认禁止跨层：

- `SessionRuntime`、`ToolRuntime`、`Agent` 等 live object。
- store、memory writer、context governor、rollback manager。
- 可执行 tool adapter。
- terminal、renderer、DingTalk SDK client。
- provider client 或 HTTP session。

## 3. 主调用链契约

### 3.1 Prompt run

```text
interfaces
  PromptSubmitted
        |
        v
runtime
  dispatch() validates app state, registers active run
        |
        v
sessions
  prepare_run() builds AgentLoopInput and ContextPort
        |
        v
core
  run_agent_loop() coordinates model turn, tool turn, stop decision
        |                         |
        |                         +--> tools.ToolPort
        +----------------------------> llm.ModelPort
        |
        v
sessions
  commit_run() persists transcript, memory, recovery, rollback metadata
        |
        v
runtime
  RunFinishedFrame / ProgressFrame / FailedFrame
        |
        v
interfaces
  render frames
```

### 3.2 Command

```text
CommandSubmitted
  -> RuntimeGateway.dispatch()
  -> SessionController.apply_command()
  -> SessionCommandRecord
  -> CommandFinishedFrame
```

命令不进入模型。`/mode`、`/memory`、`/context`、`/rollback`、`/fork`、
`/switch` 都是 session 语义动作，interface 不直接调用 session 子系统。

### 3.3 Approval

```text
ToolPort.execute()
  -> ToolObservation(status="approval_required", interruption=ToolInterruption)
  -> AgentLoopOutcome(status="waiting_approval")
  -> RuntimeGateway records approval transaction
  -> ApprovalRequiredFrame
  -> ApprovalDecided
  -> SessionController.prepare_resume()
  -> core.resume_agent_loop()
  -> ToolPort.resume()
  -> SessionController.commit_run()
```

职责划分：

- Runtime 管理用户审批事务。
- Tools 负责审批后的安全执行。
- Session 准备恢复上下文。
- Core 决定恢复后的 loop 如何继续。
- Interface 只展示审批并提交用户决定。

## 4. Layer Contracts

### 4.1 `protocols`

回答的问题：各层用什么共同语言描述消息、工具、事件、命令和错误？

职责边界：

- 定义跨层 DTO、枚举、轻量校验和序列化友好的结构。
- 不执行模型、工具、持久化、命令或渲染。
- 不引用任何上层对象。

暴露给其他层：

```python
ContentBlock
TextContent
ImageContent
ThinkingContent
Message
UserMessage
AssistantMessage
ToolCall
ToolResultMessage
Tool
ToolMetadata
ToolResult
ToolResultStatus
ToolRiskLevel
AgentEvent
RuntimeEventType
AgentEventSink
ensure_runtime_event_type
AgentRunStatus
AgentRunStopReason
AgentRunCounters
TaskSummary
RunVerification
RegisteredCommand
SessionCommandContext
SessionLifecycleContext
ToolHookContextSnapshot
BeforeToolCallContext
AfterToolCallContext
ContextReport
ErrorInfo
LLMErrorInfo
```

目标模块形态：

```text
protocols/
  messages.py      # 对话消息、content block、tool call
  tools.py         # 模型可见 tool schema、tool result、metadata
  events.py        # agent/model/tool event 的稳定事件形态
  runs.py          # counters、usage、verification、workspace effects
  commands.py      # extension command 和 lifecycle hook DTO
  tool_hooks.py    # tool hook context snapshot
  context.py       # context report 和只读证据 DTO
  errors.py        # 稳定 error code 和异常基类
```

推理依据：这些概念都被多个层级共享，但没有执行权；它们应该停留在协议层。
provider 内部类型、ToolRuntime 内部类型、registry、adapter、runtime live object
都不能进入 `protocols`。

### 4.2 `llm`

回答的问题：给定统一模型请求，如何调用具体 provider 并返回统一模型事件？

职责边界：

- 管理模型目录、provider registry、API key/resource 配置。
- 把 `LLMRequest` 转成 provider API 调用。
- 把 provider stream 转成 `LLMEvent`。
- 提供 token/context 估算。
- 不判断 agent 是否完成，不执行工具，不保存 session。

暴露给 core/runtime 装配：

```python
class ModelPort:
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]: ...

@dataclass(frozen=True)
class LLMCorrelation:
    run_id: str = ""
    session_id: str = ""

@dataclass(frozen=True)
class LLMRequest:
    model: ModelDescriptor
    messages: tuple[Message, ...]
    system_prompt: str
    tools: tuple[Tool, ...]
    options: LLMOptions
    correlation: LLMCorrelation
```

返回：

```python
LLMEvent.started
LLMEvent.text_delta
LLMEvent.reasoning_delta
LLMEvent.tool_call_delta
LLMEvent.completed(message, usage)
LLMEvent.failed(error)
```

目标模块形态：

```text
llm/
  ports.py             # ModelPort、LLMRequest、LLMEvent
  provider_types.py    # provider bridge callback 签名，供 runtime/session 装配配置引用
  adapters.py          # ProviderModelPort 等，把 provider registry 适配成 ModelPort
  models.py            # ModelDescriptor、capabilities、catalog
  api_registry.py      # provider 注册与选择
  event_stream.py      # provider 内部流式事件聚合
  env_api_keys.py      # provider 默认环境变量名和读取
  errors.py            # provider 异常到 LLMErrorInfo 的分类
  overflow.py          # token/context 估算
  providers/
    __init__.py        # 空包根；不聚合导出，不触发注册
    register_builtins.py  # 显式注册内置 provider，不在 import 时执行
    openai_compatible.py
    anthropic.py
    _common.py
```

推理依据：core 需要的是模型能力端口，不需要知道 provider registry、HTTP、
鉴权、重试细节。`ports.py` 只描述端口契约；把现有 provider registry 接到
`ModelPort` 的具体桥接代码放在 `adapters.py`。runtime/session 装配配置如需
接收旧 provider registry 的注入回调，只引用 `provider_types.py` 中的签名，
不引用具体 adapter。provider 差异应被 llm 层吞掉。`codepilot.llm` 顶层不再
转发 protocol DTO 或 provider registry 函数，也不在 import 时自动注册 provider。
`codepilot.llm.providers` 包根同样不是 provider 门面；需要 provider 函数时导入
具体模块，需要启用内置 provider 时由 runtime assembly 显式调用
`register_builtin_api_providers()`。

### 4.3 `tools`

回答的问题：模型提出一个工具调用意图后，如何安全、可审批、可审计地执行？

职责边界：

- 维护工具定义和工具目录。
- 参数 schema 校验。
- 权限策略、风险分类、审批请求。
- 执行工具并收集结果。
- 对结果做 secret/PII/prompt-injection 防护。
- 不决定 agent loop 是否结束，不管理 session 生命周期。

暴露给 core/runtime 装配：

```python
class ToolPort:
    def catalog(self) -> ToolCatalogView: ...
    async def execute(self, invocation: ToolInvocation) -> ToolObservation: ...
    async def resume(self, decision: ToolResumeDecision) -> ToolObservation: ...

@dataclass(frozen=True)
class ToolCatalogView:
    tools: tuple[Tool, ...]

@dataclass(frozen=True)
class ToolPolicyContext:
    session_id: str | None
    metadata: Mapping[str, object]

@dataclass(frozen=True)
class ToolInvocation:
    run_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, object]
    source: ToolInvocationSource
    policy_context: ToolPolicyContext
    context: ToolHookContextSnapshot | None = None
```

返回：

```python
@dataclass(frozen=True)
class ToolObservation:
    tool_call_id: str
    name: str
    status: "success" | "error" | "approval_required"
    content: list[ContentBlock]
    affected_paths: tuple[str, ...]
    workspace_changed: bool
    verification: tuple[RunVerification, ...]
    interruption: ToolInterruption | None
    metadata: dict[str, object]
```

目标模块形态：

```text
tools/
  ports.py          # ToolPort、ToolInvocation、ToolObservation、ToolInterruption
  adapters.py       # ToolRuntimePort，把 ToolRuntime 安全管线适配成 ToolPort
  contracts.py      # tool authoring 和 ToolRuntime 内部 request/result 类型
  registry.py       # 工具实例和 metadata 的登记表
  metadata.py       # 内置工具 metadata 和外部工具保守推断
  execution.py      # ToolRuntime 安全执行管线
  policy.py         # permission/risk 决策
  approval.py       # approval provider、deferred approval
  argument_schema.py # 工具参数 JSON schema 子集校验
  result_safety.py  # 工具输出脱敏、prompt-injection 标记和可信度
  workspace_safety.py # 工作区路径边界和文件状态快照
  shell_safety.py   # shell 命令分类、环境过滤、输出截断
  builtins/
    __init__.py     # create_builtin_tools 聚合工厂，不扩展成杂乱门面
    files.py
    search.py
    shell.py
    task_control.py
    workspace_status.py
```

推理依据：工具层是安全边界。无论调用者是 core、测试还是未来的其他入口，
工具执行都必须经过同一条安全管线。`ports.py` 保持为 core 可消费的纯端口契约；
`adapters.py` 负责把 ToolRuntime 的安全执行管线桥接为 `ToolPort`。`codepilot.tools`
顶层只暴露 tool authoring 和 runtime 装配能力，不暴露 `ToolPort` 或 `ToolRuntimePort`。

### 4.4 `core`

回答的问题：给定准备好的上下文和能力端口，如何跑完一次 agent loop？

职责边界：

- 组织 model turn。
- 解析 assistant message 和 tool calls。
- 组织 tool turn。
- 处理 task-control 信号。
- 做停止判断、重复调用判断、工具上限判断、retry。
- 返回唯一结果 `AgentLoopOutcome`。
- 不知道 session 文件、memory 写入、rollback metadata、runtime approval registry。

暴露给 sessions：

```python
async def run_agent_loop(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
) -> AgentLoopOutcome: ...

async def resume_agent_loop(
    input: AgentResumeInput,
    ports: AgentLoopPorts,
) -> AgentLoopOutcome: ...
```

核心 DTO：

```python
@dataclass(frozen=True)
class PreparedContext(Mapping[str, object]):
    values: Mapping[str, object]

@dataclass(frozen=True)
class AgentLoopInput:
    run_id: str
    correlation: RunCorrelation
    messages: list[Message]
    context: PreparedContext
    model: ModelDescriptor
    tools: list[Tool]
    task_strategy: TaskStrategy
    limits: AgentLoopLimits
    retry_policy: RetryPolicy

@dataclass(frozen=True)
class TaskStrategy:
    enabled: bool = False
    mode: TaskMode = "edit"
    goal: str | None = None
    steps: tuple[object, ...] = ()
    planning: TaskPlanningState | None = None
    planning_budget_profile: PlanningBudgetProfile = "balanced"
    max_replans_per_run: int | None = None
    recovery_projection: dict[str, object] | None = None

@dataclass(frozen=True)
class RetryPolicy:
    enabled: bool = False
    max_retries: int = 0
    base_delay_ms: int = 0

EventSink = Callable[[AgentEvent], None]

@dataclass(frozen=True)
class AgentLoopPorts:
    model: ModelPort
    tools: ToolPort
    context: ContextPort | None
    events: EventSink | None
```

`PreparedContext` 是 sessions 交给 core 的只读 loop 上下文快照，至少可包含
`system_prompt`、`session_id`，并允许 core 在单次 run 内注入
`current_task` / `task_control_signal` 这类轻量控制信号。它不是
`sessions.context.ContextGovernor` 的 live object，也不是 runtime session state。

`EventSink` 是 core loop 的同步事件出口，用于把已经补齐 run/turn 元数据的
`AgentEvent` 推给外层记录或渲染；它不是 session/runtime 的 live object，也不承担
异步事件总线职责。

输出：

```python
@dataclass(frozen=True)
class AgentLoopOutcome:
    run_id: str
    status: "completed" | "waiting_approval" | "waiting_user" | "failed" | "cancelled"
    stop_reason: str
    new_messages: list[Message]
    final_message: AssistantMessage | None
    interruptions: list[ToolInterruption]
    counters: AgentRunCounters
    usage: Usage | None
    verification: list[RunVerification]
    workspace_effects: WorkspaceEffects
    events: list[AgentEvent]
    task: TaskSummary | None
```

目标模块形态：

```text
core/
  contracts.py      # AgentLoopInput、AgentLoopOutcome、AgentLoopPorts
  loop.py           # run_agent_loop、resume_agent_loop 主控制流
  model_step.py     # 构造 LLMRequest，消费 ModelPort，整理 LLM 消息
  tool_step.py      # 调 ToolPort，生成 tool observations
  state.py          # 单次 run 内部状态
  task/             # 任务状态、计划、步骤、控制信号语义
```

推理依据：core 的自然分段不是按旧类拆，而是按 agent loop 的四个动作拆：
模型回合、工具回合、停止判断、任务控制。

### 4.5 `sessions`

回答的问题：一个会话如何把持久状态准备成一次 run，又如何把 run 结果沉淀回会话？

职责边界：

- 管理 transcript、history、run store。
- 管理 context governance、压力感知裁剪、context ledger。
- 管理 memory 召回和沉淀。
- 管理 task recovery。
- 管理 rollback baseline、preview、apply metadata。
- 执行 lifecycle hooks。
- 把会话事实转换为 `AgentLoopInput`，把 `AgentLoopOutcome` 转换为 `SessionRunRecord`。
- 不管理 runtime active task，不渲染用户输出，不执行工具安全管线。

暴露给 runtime：

```python
class SessionController:
    def describe(self) -> SessionView: ...
    async def prepare_run(self, intent: SessionRunIntent) -> PreparedAgentRun: ...
    async def prepare_resume(self, intent: SessionResumeIntent) -> PreparedAgentRun: ...
    async def commit_run(
        self,
        prepared: PreparedAgentRun,
        outcome: AgentLoopOutcome,
    ) -> SessionRunRecord: ...
    async def apply_command(self, intent: SessionCommandIntent) -> SessionCommandRecord: ...
    def subscribe(self, listener: SessionEventListener) -> Unsubscribe: ...
    def close(self) -> None: ...
```

`prepare_run()` 产物：

```python
@dataclass(frozen=True)
class PreparedAgentRun:
    run_id: str
    session_id: str
    loop_input: AgentLoopInput
    resume_input: AgentResumeInput | None
    context_port: ContextPort | None
    input_messages: list[Message]
    rollback_baseline: RollbackBaselineRef | None
    context_refs: dict[str, object]
    memory_refs: dict[str, object]
    recovery_refs: dict[str, object]
```

`rollback_baseline` 是公开 ref，不是 `sessions/history` 内部的 git baseline 对象：

```python
@dataclass(frozen=True)
class RollbackBaselineRef:
    session_id: str
    run_id: str
    kind: Literal["rollback_baseline_ref"] = "rollback_baseline_ref"
```

推理依据：rollback baseline 的真实内容只供 session commit 阶段写入审计元数据；
runtime 只需要把 `PreparedAgentRun` 原样带回 `commit_run()`，不应该看见 history
内部实现类型。

`commit_run()` 产物：

```python
@dataclass(frozen=True)
class SessionRunRecord:
    run_id: str
    session_id: str
    status: AgentLoopStatus
    stop_reason: str
    new_messages: list[Message]
    final_text: str
    events: list[AgentEvent]
    outcome: AgentLoopOutcome | None
    snapshots: dict[str, object]
```

命令产物：

```python
@dataclass(frozen=True)
class SessionCommandRecord:
    session_id: str
    command: str
    handled: bool
    output_lines: tuple[str, ...]
    switched_session_id: str | None
    data: dict[str, object]
```

目标模块形态：

```text
sessions/
  contracts.py        # Session intent、PreparedAgentRun、record、view
  controller.py       # runtime-facing SessionController
  prepare.py          # session-owned state aggregate、prepare run/resume、run 前副作用
  commit.py           # outcome 写回、memory/task recovery/rollback 收尾
  conversation.py     # transcript state 和事件订阅
  commands.py         # command router 及命令需要的 session-owned action/view
  storage.py          # layout、serde、SessionStore、RunStore、repository bootstrap
  context/
    governor.py
    policy.py
    projector.py
    snapshot.py
  memory/
    retriever.py
    writer.py
    store.py
  history/
    branching.py
    checkpoint.py
    task_recovery.py
    git_rollback.py
```

推理依据：session 的核心不是“调用 agent”，而是维护会话事实。prepare/commit
把会话语义和 core 执行隔开，使 context、memory、rollback 可以独立演进。

### 4.6 `observability`

回答的问题：一次 run 的事实如何被整理成可解释、可审计、可评测的证据？

职责边界：

- 归一化事件。
- 从 `SessionRunRecord`、`AgentLoopOutcome.events` 和 audit files 构建 trace/report。
- 给 evaluation 和人类阅读提供只读证据。
- 不改变 run 决策，不 patch core，不读取 session live object。

暴露给 runtime/evaluation：

```python
build_run_trace(record: SessionRunRecord, events: Iterable[AgentEvent]) -> RunTrace
build_run_report(record: SessionRunRecord, events: Iterable[AgentEvent]) -> RunReport
load_audit_bundle(ref: AuditBundleRef) -> AuditBundle
```

目标模块形态：

```text
observability/
  events.py       # event normalization
  trace.py        # timeline / call graph / evidence trace
  summary.py      # human-readable report
  audit.py        # read-only audit bundle loader
  recorder.py     # optional sink for structured evidence
```

推理依据：observability 是证据视图层。它可以理解 run record 和 events，
但不能成为主执行链路的控制器。

### 4.7 `extensions`

回答的问题：外部能力如何被加载成 Codepilot 可以消费的工具、命令和 hooks？

职责边界：

- 加载 skills。
- 加载 MCP 工具。
- 加载 extension commands。
- 加载 lifecycle hooks 和 tool hooks。
- 产出能力包和 diagnostics。
- 不参与 agent loop 决策，不持有 session 生命周期。

暴露给 runtime：

```python
load_extensions(request: ExtensionLoadRequest) -> ExtensionBundle
load_mcp_tools(request: MCPLoadRequest) -> ToolBundle
load_skill_commands(request: SkillLoadRequest) -> CommandBundle
```

返回：

```python
@dataclass(frozen=True)
class ExtensionBundle:
    tools: tuple[AgentTool, ...]
    commands: tuple[RegisteredCommand, ...]
    lifecycle_hooks: tuple[LifecycleHook, ...]
    tool_hooks: tuple[ToolHook, ...]
    diagnostics: tuple[ExtensionDiagnostic, ...]
```

目标模块形态：

```text
extensions/
  loader.py       # 聚合 extension loading
  skills.py       # skill manifest / command loading
  mcp.py          # MCP capability loading
  commands.py     # extension command adaptation
  hooks.py        # lifecycle/tool hook adaptation
  diagnostics.py
```

推理依据：extensions 只生产能力。能力何时被使用由 runtime/session/core 决定。

### 4.8 `runtime`

回答的问题：用户动作如何进入应用，如何找到会话、装配能力、管理 active run 和审批？

职责边界：

- 打开/关闭 session。
- 持有 session registry。
- 持有 active run registry。
- 持有 pending approval registry。
- 装配 model/tool/extensions/session controller。
- 把 `UserAction` 转成 session/core 调用。
- 把 session/core 结果转成 `RuntimeFrame`。
- 不向 interface 暴露 live session、assembly、tool runtime 或 store。

暴露给 interfaces/evaluation：

```python
class RuntimeGateway:
    def open_session(self, intent: SessionOpenIntent) -> SessionRef: ...
    async def dispatch(
        self,
        session_id: str,
        action: UserAction,
    ) -> AsyncIterator[RuntimeFrame]: ...
    def describe(self, session_id: str) -> AppSessionView: ...
    def close(self, session_id: str) -> None: ...
    async def close_all(self) -> None: ...
```

用户动作：

```python
UserAction =
  PromptSubmitted
  CommandSubmitted
  ApprovalDecided
  RunCancelled
```

输出 frame：

```python
RuntimeFrame =
  ProgressFrame
  ApprovalRequiredFrame
  RunFinishedFrame
  CommandFinishedFrame
  CancelledFrame
  FailedFrame
```

目标模块形态：

```text
runtime/
  gateway.py          # RuntimeGateway 和 dispatch 主入口
  actions.py          # UserAction / RuntimeFrame
  opening.py          # SessionOpenIntent / SessionRef / AppSessionView
  sessions.py         # session registry / active run registry
  approvals.py        # approval transaction registry
  assembly.py         # RuntimeAssemblyIntent、配置解析、模型/工具/prompt/hook 装配
  views.py            # SessionStatus、CommandDescriptor、builtin command view
  configuration.py    # config explain / model resolution views
```

推理依据：runtime 是应用服务门面。它可以知道 live objects 如何被装配，
但它暴露给 interface 的只能是应用用例和只读视图。

### 4.9 `interfaces`

回答的问题：不同用户入口如何把输入变成 `UserAction`，把 `RuntimeFrame` 展示给人？

职责边界：

- CLI 参数、REPL 文本、RPC JSONL、DingTalk 消息解析。
- 用户输入标准化成 `UserAction`。
- 渲染 progress/result/error/approval/command frame。
- 处理界面本地交互体验。
- 不理解 agent loop、memory、tool runtime、session store。

暴露给外部入口：

```python
run_cli(...)
run_interactive(...)
run_rpc(...)
DingTalkBridge.handle_message(...)
DingTalkBridge.iter_replies(...)
```

只允许调用 runtime：

```python
runtime.open_session(SessionOpenIntent(...))
runtime.dispatch(session_id, UserAction)
runtime.describe(session_id)
runtime.close(session_id)
runtime.close_all()
```

目标模块形态：

```text
interfaces/
  cli/
    main.py       # parser / process entrypoint
    runner.py     # run modes: print, interactive, rpc
    shell.py      # REPL input loop
    approval.py   # approval rendering and user decision parsing
    renderer.py   # RuntimeFrame -> terminal output
    startup.py    # startup view from AppSessionView
  dingtalk/
    bridge.py     # inbound message -> UserAction, frames -> replies
    schemas.py
    transport.py
  rpc/
    jsonl.py      # optional: JSONL protocol adapter if separated later
```

推理依据：interface 是适配层。它可以丰富交互体验，但不能补查内部状态来完成渲染。

### 4.10 `evaluation`

回答的问题：如何离线运行 benchmark 并用证据判断 agent 是否完成任务？

职责边界：

- 加载 benchmark。
- 通过 `RuntimeGateway` 驱动真实 run。
- 收集 `RuntimeFrame`、`SessionRunRecord`、observability report。
- 计算指标。
- 不 patch core，不直接读取 session internals，不绕过 runtime。

暴露给 CLI/测试：

```python
EvaluationRunner.run_case(case: EvaluationCase) -> EvaluationResult
load_benchmark(path: Path) -> BenchmarkSuite
score_evidence(evidence: EvalEvidence) -> EvalScore
```

目标模块形态：

```text
evaluation/
  schema.py
  loader.py
  runner.py
  evidence.py
  scorer.py
  cli.py
```

推理依据：evaluation 是用户入口的一种自动化形态，但它更关心证据和评分。
它应该驱动应用门面，而不是钻进 core/session 内部。

## 5. 不应暴露的旧口径

这些名字即使当前实现里一度方便，也不应成为 V2 契约：

Runtime 旧入口：

- `submit_turn()`
- `execute_command()`
- `approve()`
- `cancel()`
- `continue_session()`

Runtime live getter：

- `get_session()`
- `get_assembly()`
- `get_latest_assistant_message()`
- `get_session_state()`
- `get_memory_state()`
- `get_context_report()`

Session/Core 旧 facade：

- `AgentSession.run(text)`
- `Agent.run(text)`
- `SessionRuntime.run()`
- `continue_after_tool_approval()`
- `execute_approved_tool_call()`
- `replace_tool_result_message()`

判断标准：

1. 如果接口返回 live object，删除或改成 snapshot。
2. 如果接口只是旧方法别名，删除。
3. 如果接口让上层理解下层内部结构，改成 intent 或 command。
4. 如果接口只服务一个历史绕路，合并到主脊柱。

## 6. 目标阅读顺序

新手按一次 run 阅读代码时，推荐顺序是：

```text
runtime/actions.py
runtime/gateway.py
sessions/contracts.py
sessions/controller.py
sessions/prepare.py
sessions/commit.py
core/contracts.py
core/loop.py
core/model_step.py
core/tool_step.py
llm/ports.py
llm/adapter.py
tools/ports.py
tools/adapter.py
tools/engine.py
sessions/context/governor.py
sessions/memory/retriever.py
sessions/history/task_recovery.py
sessions/history/git_rollback.py
observability/trace.py
```

这条阅读路线体现的是职责顺序，不是文件依赖的唯一合法顺序。

## 7. 测试契约

边界测试必须覆盖：

- `interfaces` 不 import `core`、`sessions`、`tools`、`llm`。
- `runtime` public surface 只暴露应用门面和只读 DTO。
- `sessions` 不 import `runtime` 或 `interfaces`。
- `core` 不 import `sessions`、`runtime`、`interfaces`。
- `tools` 不 import `runtime` 或 `interfaces`。
- contracts 不包含 live object 字段。

流程测试必须覆盖：

- `PromptSubmitted` 产生 `ProgressFrame` 和 `RunFinishedFrame`。
- `CommandSubmitted` 产生 `CommandFinishedFrame`。
- `ToolObservation(status="approval_required")` 产生 `ApprovalRequiredFrame`。
- `ApprovalDecided` 走 `resume_agent_loop()`，并由 `ToolPort.resume()` 执行审批后的工具。
- `RunCancelled` 产生 `CancelledFrame`。
- `prepare_run()` 执行 context/memory/task recovery/rollback baseline 准备。
- `commit_run()` 持久化 result/events/messages/memory/task recovery/rollback metadata。

验证命令：

```bash
python -m pytest -q
```

## 8. 接口判断准则

新增、保留或删除接口时，按这些问题判断：

1. 这个接口是否表达 agent run 的一个必要阶段？
2. 这个接口传递的是 intent、capability、execution、snapshot，还是 live object？
3. 上层是否必须知道这个职责，还是只是当前实现调用方便？
4. 换掉该层内部实现后，上层是否需要跟着修改？
5. 新手能否从接口名看出它位于一次 run 的哪个阶段？

如果一个接口只是把当前实现的内部 helper 暴露出来，它不进入 V2 契约。

## 9. 当前代码和本文档的关系

本文档是目标契约，不是由当前代码反推的说明书。当前代码可以作为落地进度的
证据，但不能作为决定接口是否合理的理由。

当代码和本文档冲突时，按下面顺序处理：

1. 如果冲突来自旧兼容入口，以本文档为准，删除旧入口。
2. 如果冲突来自文档过度理想化，先回到 agent loop 职责重新推理，再修改文档。
3. 如果冲突来自必要的安全、审计或持久化设计，保留设计理念，但调整代码表达，
   让它仍然沿 V2 主脊柱可读。

最终成功标准不是“所有旧测试都被兼容”，而是一次 coding agent run 可以沿
`UserAction -> RuntimeGateway.dispatch() -> SessionController -> core.run_agent_loop()
-> ModelPort/ToolPort -> SessionRunRecord -> RuntimeFrame` 顺畅阅读。
