# Codepilot 统一工具系统设计

## 1. 文档状态

- 状态：已确认设计，待分阶段实施
- 适用范围：`protocols`、`tools`、`core`、`sessions`、`extensions`、`runtime`、`interfaces`
- 核心原则：先稳定工具协议和单工具执行闭环，再逐步加入安全、审批、超时、取消、并发、恢复和扩展能力

本文定义 Codepilot 工具系统的目标边界、统一协议、执行状态机、安全策略和迁移方案。它描述的是目标架构，不代表当前代码已经全部实现。

## 2. 设计目标

工具系统是 Coding Agent 的本地行动边界。模型只能产生工具调用意图，任何工具实现都不能绕过 `ToolRuntime` 直接执行。

统一约束如下：

```text
Core 产生 ToolCall
  -> ToolRuntime 统一处理
  -> Registry materialize
  -> 输入校验
  -> 资源解析与权限判断
  -> 审批、调度、超时与取消控制
  -> ToolHandler 执行
  -> 输出校验、副作用核验与结果防护
  -> ToolResult 返回 Core
```

设计目标：

1. 所有内置、Plan、子代理、Interaction、Extension、Skill 和 MCP 工具进入同一条执行管线。
2. 输入和输出都有明确 Schema/Codec，并在执行前后分别验证。
3. 权限、审批、超时、取消、基础并发和恢复由 `ToolRuntime` 统一管理。
4. 工具执行产生的副作用、错误、耗时和审计信息使用结构化协议表达。
5. 工具定义、模型可见描述、执行实现和运行策略相互分离。
6. 保持依赖方向：`protocols -> tools -> core -> sessions/observability -> extensions -> runtime -> interfaces`。

## 3. 工具系统边界

### 3.1 工具系统负责

- 工具定义与注册。
- 工具目录快照和版本身份管理。
- 输入参数解析、Schema 校验和类型转换。
- 工具调用 materialize 和 handler 调度。
- 权限、风险和资源访问判断。
- 安全审批状态管理。
- timeout、取消、基础并发和资源清理。
- 副作用预测、上报、核验和记录。
- 输出 Schema 校验、结果脱敏、可信度标记和大小控制。
- 返回唯一、标准化的 `ToolResult`。

### 3.2 工具系统不负责

- 决定 Agent 下一步任务。
- 生成、修改或批准任务计划的业务语义。
- 判断任务是否完成。
- 选择哪些内容进入模型上下文。
- 判断哪些信息写入长期记忆。
- 管理完整会话历史。
- 直接与 CLI、Web、钉钉或用户交互。
- 决定 ToolResult 如何影响 Core 的后续推理。

### 3.3 特殊工具的边界

Plan、子代理和人机交互能力仍可以表现为模型可见工具，但其业务 handler 归所属模块所有：

- Plan handler 归 `core`。
- Interaction workflow 归 `core`。
- Subagent handler 归 `runtime`。
- MCP adapter 归 `extensions`。

这些 handler 只能通过注册协议进入 `ToolRuntime`。工具系统只理解通用 Spec、Policy、Codec、Handler 和 Result，不理解 PlanState、任务完成、记忆或 UI 语义。

## 4. 工具分类模型

工具采用“行为类别 + 来源 + 显式策略”的三轴模型。

### 4.1 行为类别

```text
filesystem   文件读取、写入、编辑和补丁
search       内容搜索和路径搜索
command      Bash、PowerShell 和进程执行
delegation   子代理派发、查询和取消
plan         计划提议、更新和关闭
interaction  请求用户提供选择、确认或补充信息
external     网络服务、MCP 和外部系统操作
```

### 4.2 工具来源

```text
builtin
caller
skill
extension
mcp
```

来源和类别不能混为一谈。例如 MCP 搜索工具的类别是 `search`，来源是 `mcp`。

### 4.3 显式策略

类别只提供默认策略，单个工具必须声明精确策略：

- `allowed_modes`
- `declared_effects`
- `required_permissions`
- `base_risk`
- `approval`
- `timeout`
- `concurrency`
- `output_limits`

类别不能代替实际副作用、风险、审批或并发判断。

## 5. 总体架构

```text
Model Provider
  -> ToolCall
  -> Core
  -> ToolExecutionRequest
  -> ToolRuntime
      -> ToolRegistry.materialize
      -> input_codec.decode
      -> access_resolver.resolve
      -> PermissionEngine.decide
      -> ApprovalStore / InteractionStore
      -> Scheduler
      -> ToolHandler
      -> output_codec.encode
      -> EffectAuditor
      -> ToolOutputRenderer
      -> ToolResultGuard
      -> ToolStateStore
  -> ToolResult
  -> Core 投影为 ToolResultMessage
```

跨层只保留以下稳定对象：

- `ToolSpec`
- `ToolExecutionRequest`
- `ToolResult`
- `ToolProgressEvent`
- `ApprovalChallenge/Decision`
- `InteractionRequest/Response`

当前 `ToolCallRequest -> ToolResult -> ToolObservation -> ToolResultMessage` 的多层近重复结构应逐步收敛。

## 6. 模型可见协议

### 6.1 ToolSpec

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object] | None
    schema_version: int = 1
```

约束：

- `name` 全局唯一，格式为 `[A-Za-z][A-Za-z0-9_-]{0,63}`。
- `description` 仅描述模型应如何使用工具。
- `input_schema` 在 handler 启动前验证。
- `output_schema` 验证工具的结构化领域输出；仅无法提供可信 Schema 的外部适配器允许为 `None`。
- Schema 以 input/output codec 为唯一真值，Registry 从 codec 生成 ToolSpec 投影，禁止调用方分别提交两份可能不一致的 Schema。
- Spec 是不可变快照。
- description、Schema 或 Policy 的变化都会改变 registration identity。

### 6.2 ToolExecutionRequest

```python
@dataclass(frozen=True)
class ToolExecutionRequest:
    run_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, object]
    mode: ToolMode
    registration_id: str
    idempotency_key: str | None = None
    deadline_at_ms: int | None = None
```

Request 不得携带：

- System Prompt。
- 完整消息历史。
- Memory。
- PlanState。
- CLI/Web 对象。
- ApprovalProvider。
- SessionRuntime。
- 可直接执行的 handler。

业务 handler 需要的状态通过构造时注入的窄接口获得，不能把完整 runtime 当作 service locator 塞进 request。`registration_id` 必须来自产生本次模型请求的 Catalog snapshot，不能由模型或外部调用方自行提供。

## 7. 注册协议

### 7.1 ToolRegistration

```python
@dataclass(frozen=True)
class ToolRegistration:
    version: str
    implementation_version: str
    spec: ToolSpec
    category: ToolCategory
    source: ToolSource
    owner: str
    policy: ToolPolicy
    input_codec: ToolCodec[object]
    output_codec: ToolCodec[object]
    handler: ToolHandler
    renderer: ToolOutputRenderer
    access_resolver: ToolAccessResolver
```

`ToolRegistration` 是待注册定义，不携带可信 registration ID。Registry 校验定义后生成内部 `MaterializedTool`，并为其分配 registration ID。

外部 owner 通过受控 builder 创建 Registration：builder 从 codec 生成 ToolSpec 中的 Schema；Registry 再做深度一致性校验，不一致时拒绝注册。调用方不能独立维护 Spec Schema 与 codec Schema。

registration identity 至少绑定：

- owner、tool name、显式 version 和 implementation_version。
- description、input/output Schema、category、source 和完整 Policy。
- codec、renderer、resolver 与 handler 的显式实现版本。

不能 hash Python callable、对象地址或 `repr()`。替换、卸载后恢复、Extension/MCP 重连都会产生新的 revision 和 registration ID。

`owner` 示例：

```text
codepilot.builtin
codepilot.core.plan
codepilot.runtime.subagent
extension:<extension-id>
skill:<skill-id>
mcp:<server-id>
caller:<client-id>
```

### 7.2 Opaque handler

包含 handler 的 MaterializedTool 只允许 Registry 和 ToolRuntime 持有。公共目录只能返回：

```python
@dataclass(frozen=True)
class ToolCatalogEntry:
    spec: ToolSpec
    category: ToolCategory
    source: ToolSource
    policy: ToolPolicyView
    registration_id: str
    version: str
```

生产代码不能通过 `registry.get(name).execute(...)` 绕过 Runtime。

- `MaterializedTool` 不从 `tools.__init__` 导出。
- Registry 公共查询不返回 handler。
- 架构测试禁止在 ToolRuntime 外访问 handler。

### 7.3 ToolCatalogSnapshot

```python
@dataclass(frozen=True)
class ToolCatalogSnapshot:
    catalog_id: str
    entries: tuple[ToolCatalogEntry, ...]
    created_at_ms: int
```

Core 发起模型请求时保存 `tool_name -> registration_id`。模型返回 ToolCall 后，ExecutionRequest 必须携带当时的 registration ID。

如果工具在模型请求之后被替换或卸载：

```text
request.registration_id != active.registration_id
  -> tool.registration.stale
```

旧调用不能误执行新 handler。

### 7.4 名称与覆盖规则

- 内置短名称为保留名称。
- 外部工具使用命名空间，例如 `mcp__github__create_issue`。
- 名称过长时使用截断前缀和稳定短 hash。
- 默认 `replace=False`。
- 外部工具不能覆盖 builtin。
- 同名非内置工具默认注册失败。
- 覆盖必须显式声明 owner 和 override 配置。
- 覆盖、卸载和恢复均产生新的 registration ID。

## 8. Schema 与 Codec

### 8.1 ToolCodec

```python
T = TypeVar("T")

class ToolCodec(Protocol, Generic[T]):
    @property
    def json_schema(self) -> Mapping[str, object] | None:
        ...

    def decode(self, value: object) -> T:
        ...

    def encode(self, value: T) -> object:
        ...
```

执行链：

```text
raw arguments
  -> input_codec.decode
  -> typed input
  -> access_resolver.resolve
  -> resolved typed input
  -> handler
  -> domain output
  -> output_codec.encode
  -> validated JSON-safe data
  -> renderer
```

`ToolResult.data` 必须是 output codec 编码并验证后的结构化数据。Renderer 接收这份已验证数据，不直接接收未经编码的领域对象。

### 8.2 第一版 Codec

第一版只实现：

- `JsonObjectCodec`
- `DataclassCodec`
- `UnverifiedJsonCodec`，仅供缺少可信 output Schema 的 MCP/外部工具使用。

`JsonObjectCodec` 用于 MCP、动态扩展和迁移期工具；`DataclassCodec` 用于内置工具。`UnverifiedJsonCodec` 仍必须执行 JSON-safe、类型白名单、深度、大小和敏感内容防护，只是不声称完成 Schema 级语义验证。

第一版不绑定 Pydantic。以后可以增加 Pydantic、TypedDict 或 MCP codec adapter，而不改变 ToolRuntime。

### 8.3 Schema 规则

注册阶段：

1. Schema 使用 JSON Schema Draft 2020-12，并且必须合法。
2. 顶层输入必须为 object。
3. Schema 必须可以 JSON 序列化。
4. 不支持的关键字不得静默忽略。
5. properties、required 和参数 description 必须一致。

执行阶段：

```text
input decode 失败
  -> handler 不启动
  -> tool.input.invalid

output encode 或验证失败
  -> 不能返回 success
  -> tool.output.invalid
```

output Schema 只校验 handler 成功产生的领域输出。denied、approval、interaction、timeout、cancelled、interrupted 等控制结果不套用工具 output Schema，但始终经过 ToolResult 协议校验和 final guard。

内置工具默认 `additionalProperties=false`。MCP 和动态扩展遵循其声明的 Schema。

## 9. Handler、Context 与领域输出

### 9.1 ToolExecutionContext

```python
@dataclass(frozen=True)
class ToolExecutionContext:
    cancellation: CancellationToken
    deadline_at_ms: int | None
    progress: ProgressReporter
    effects: EffectReporter
    cleanup: CleanupStack
```

Context 只提供运行控制能力：

- 检查取消。
- 获取 deadline。
- 上报进度。
- 上报实际副作用。
- 注册清理动作。

它不提供 UI、PlanState、Memory、完整 transcript 或直接审批能力。

### 9.2 ToolHandler

```python
class ToolHandler(Protocol, Generic[TInput, TOutput]):
    async def __call__(
        self,
        input: TInput,
        context: ToolExecutionContext,
    ) -> TOutput:
        ...
```

Handler 返回领域输出，不能构造最终 `ToolResult`，不能设置：

- 最终 status。
- approved/approval_id。
- 权限决策。
- 最终耗时。
- `is_error`。
- 最终 workspace_changed。

### 9.3 ToolOutputRenderer

```python
class ToolOutputRenderer(Protocol):
    def render(self, data: Mapping[str, object]) -> tuple[ToolContent, ...]:
        ...
```

Handler 返回完整的领域对象；output codec 将其编码为可验证的 `ToolResult.data`；Renderer 再从已验证数据生成模型可见内容。UI、审计和持久化不依赖脆弱的文本输出。

支持的模型内容：

- `TextContent`
- `ImageContent`
- `ArtifactContent`

大文件使用 opaque ArtifactRef，不能直接暴露本机路径或长期保存 data URL。

## 10. 唯一标准结果

`ToolResult` 是 ToolRuntime 向 Core 返回的唯一 settlement envelope，既可表达终态，也可表达等待审批或输入的暂停态。暂停态不是模型对话中的最终工具结果。

### 10.1 ToolStatus

```python
ToolStatus = Literal[
    "success",
    "error",
    "denied",
    "approval_required",
    "user_input_required",
    "cancelled",
    "timed_out",
    "interrupted",
]
```

### 10.2 ToolResult

```python
@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    status: ToolStatus
    content: tuple[ToolContent, ...] = ()
    data: Mapping[str, object] = field(default_factory=dict)
    error: ToolError | None = None
    effects: tuple[ToolEffect, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    approval: ApprovalChallenge | None = None
    interaction: InteractionRequest | None = None
    timing: ToolTiming = field(default_factory=ToolTiming)
    registration_id: str = ""
    output_validation: Literal["schema_validated", "structurally_validated"] = "schema_validated"
    content_trust: Literal["trusted", "untrusted"] = "trusted"
```

删除重复真值字段：

- `is_error` 由 status 推导。
- `approved` 由审批记录推导。
- `workspace_changed` 由 write/delete effect 推导。
- `affected_paths` 由 effects 投影。
- `error_code` 进入 `ToolError.code`。
- `approval_id` 进入 ApprovalChallenge。

`ToolResultMessage` 可以继续作为对话协议，但只能由 Core 从 ToolResult 单向投影。

### 10.3 结果不变量

```text
success
  -> error=None
  -> approval=None
  -> interaction=None

error/denied/cancelled/timed_out/interrupted
  -> error 必须存在

approval_required
  -> approval 必须存在
  -> handler 尚未启动

user_input_required
  -> interaction 必须存在
```

所有结果必须携带非空 call ID、tool name 和 registration ID，并经过输出验证、大小限制和结果防护。

终态集合为 success、error、denied、cancelled、timed_out 和 interrupted。approval_required、user_input_required 只是同一 attempt 的暂停快照，不写入 final result 槽位，也不投影为模型 ToolResultMessage。

## 11. ToolPolicy 与副作用模型

### 11.1 Effect 类型

```python
ToolEffectKind = Literal[
    "filesystem_read",
    "filesystem_write",
    "filesystem_delete",
    "process_spawn",
    "network_access",
    "credential_access",
    "external_state_read",
    "external_state_write",
    "session_state_read",
    "session_state_write",
]
```

`read_only` 不再是核心真值，只作为派生展示属性。

### 11.2 ToolPolicy

```python
@dataclass(frozen=True)
class ToolPolicy:
    allowed_modes: frozenset[ToolMode]
    declared_effects: frozenset[ToolEffectKind]
    required_permissions: frozenset[str]
    base_risk: RiskLevel
    approval: ApprovalPolicy
    timeout: TimeoutPolicy
    concurrency: ConcurrencyPolicy
    output_limits: OutputLimits
    output_trust: OutputTrustPolicy
```

基础类型第一版固定为：

```text
ToolMode = plan | execute | unrestricted
RiskLevel = low < medium < high < critical
ApprovalPolicy = never | on_risk | always
RuleSource = hard_constraint | configuration | approval_grant | runtime_default
```

`OutputLimits` 至少限制结构化数据字节数、模型内容字节数、artifact 数量和单 artifact 大小。`OutputTrustPolicy` 声明默认 content trust 以及是否允许 structurally validated 输出。`required_permissions` 是工具启用所需的静态 capability 集合，不能替代每次调用的 action/resource/effect 决策。

### 11.3 ToolAccessResolver

静态 Policy 不能描述具体调用。输入校验后，Runtime 调用无副作用的 resolver。Resolver 接收 input codec 生成的 typed input，并同时返回 handler 必须使用的 resolved input 与访问请求：

```python
@dataclass(frozen=True)
class ToolAccessResolution(Generic[TInput]):
    input: TInput
    access: ToolAccessRequest

class ToolAccessResolver(Protocol, Generic[TInput]):
    def resolve(
        self,
        input: TInput,
        context: ToolAccessContext,
    ) -> ToolAccessResolution[TInput]:
        ...
```

Resolver 可以：

- 规范化路径。
- 解析命令。
- 提取 action/resource。
- 推导本次 effect。
- 提升风险等级。
- 生成安全预览。

Resolver 不能执行副作用、修改计划、启动进程、访问网络或请求 UI 审批。

Handler 只能接收 `ToolAccessResolution.input`，不能重新从 raw arguments 解析路径或命令。这样权限判断和实际执行使用同一份规范化输入；文件句柄、符号链接与 TOCTOU 防护仍由 sandbox/资源层负责。

### 11.4 ToolAccessRequest

```python
@dataclass(frozen=True)
class ToolAccessRequest:
    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    effects: frozenset[ToolEffectKind]
    risk: RiskLevel
    reason: str
    safe_preview: Mapping[str, object]
```

资源统一为规范化 URI，例如：

```text
workspace:///src/codepilot/tools/runtime.py
process://shell/powershell
network://api.github.com
mcp://github/create_issue
session://session_123/task_plan
agent://session_123/exploration
credential://github/token
```

## 12. 权限与审批

### 12.1 PermissionRule

```python
PermissionEffect = Literal["allow", "deny", "ask"]

@dataclass(frozen=True)
class PermissionRule:
    action_pattern: str
    resource_pattern: str
    effect: PermissionEffect
    modes: frozenset[ToolMode] = frozenset()
    source: RuleSource = "configuration"
    priority: int = 0
```

规则顺序：

1. 硬安全约束。
2. ToolPolicy 的 mode/effect 限制。
3. 用户或项目配置规则。
4. 动态风险判断。
5. 默认策略。

硬约束不能被审批绕过，包括路径逃逸、禁用能力、Schema 非法、自授权参数和不可恢复危险操作。

空 `modes` 表示适用于全部 mode。同一规则层内依次按 priority、更具体的 action/resource pattern 和 `deny > ask > allow` 决定；不得依赖注册顺序得到结果。

Catalog 可见性不能代替执行授权。

### 12.2 ApprovalChallenge

```python
@dataclass(frozen=True)
class ApprovalChallenge:
    approval_id: str
    request_fingerprint: str
    run_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    registration_id: str
    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    effects: frozenset[ToolEffectKind]
    risk: RiskLevel
    reason: str
    safe_preview: Mapping[str, object]
    allowed_scopes: frozenset[ApprovalScope]
    expires_at_ms: int | None
```

第一版 ApprovalScope：

```text
once
session
project
```

暂不支持 global。

### 12.3 ApprovalGrant

批准后生成不可伪造的 Grant，而不是简单设置 `source=approval_resume`：

```python
@dataclass(frozen=True)
class ApprovalGrant:
    grant_id: str
    approval_id: str
    request_fingerprint: str
    scope: ApprovalScope
    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    issued_at_ms: int
    expires_at_ms: int | None
```

恢复时重新检查：

- approval ID。
- request fingerprint。
- registration identity。
- Schema。
- 资源解析结果。
- 硬安全策略。
- Grant 过期和消费状态。

`approve_once` 必须单次消费。

### 12.4 三种暂停语义

必须区分：

1. ToolRuntime 安全审批：是否允许危险动作执行。
2. Core 计划审批：是否接受计划作为执行合同。
3. Interaction 输入暂停：缺少用户选择或信息。

Core-owned Plan adapter 必须先通过窄 PlanService 原子提交计划操作，才能向 ToolRuntime 返回成功领域输出；随后由 Core 单独发出计划业务审批。不能使用 ToolRuntime 的安全 ApprovalChallenge 表达计划批准。

Interaction 工具只能返回受控 InteractionRequest。ToolRuntime 只持久化暂停状态并返回 `user_input_required`；外层 application runtime 负责把请求路由给 Interface，避免 tools 反向依赖 interfaces。

InteractionResponse 必须携带 interaction ID、request fingerprint、session/tool call ID 和结构化答案。恢复时 ToolRuntime 使用 compare-and-set 消费 response，核对 fingerprint 与 attempt 状态，然后直接结算同一 attempt 的 success ToolResult；Interaction handler 不重复执行。

## 13. timeout、取消、清理与基础并发

### 13.1 TimeoutPolicy

```python
@dataclass(frozen=True)
class TimeoutPolicy:
    default_execution_ms: int
    max_execution_ms: int
    idle_timeout_ms: int | None = None
    cleanup_grace_ms: int = 5_000
```

区分 queue timeout、execution timeout 和流式 idle timeout。

有效 deadline 取 request、tool policy 和 runtime 剩余预算的最小值。

### 13.2 取消

ToolRuntime 为每个 attempt 创建 CancellationToken。

取消顺序：

1. 设置 token。
2. 等待 handler 在 grace period 内协作清理。
3. 超时后取消 asyncio task。
4. 运行 CleanupStack。
5. 返回 cancelled Result，并保留已发生 effect。

`CancelledError` 不能被包装成普通 handler error。

### 13.3 CleanupStack

Shell、MCP、临时文件和子代理工具必须注册资源清理动作。清理采用 LIFO；单个 cleanup 失败不能阻止其他 cleanup，并进入审计信息。

### 13.4 第一版并发

第一版保持简单：

```python
@dataclass(frozen=True)
class ConcurrencyPolicy:
    mode: Literal["parallel", "serial"]
    group: str | None = None

@dataclass(frozen=True)
class ToolRuntimeLimits:
    max_parallel_per_session: int = 4
    max_pending_per_session: int = 32
```

- parallel 工具受 Session semaphore 限制。
- serial 工具按 group 串行。
- workspace 写入使用 `workspace_mutation` group。
- Plan 使用 `task_plan` group。
- Shell 第一版使用 `system_command` group。
- 审批和 Interaction 是执行屏障。
- Core 不再自行 `asyncio.gather()`。

第一版不实现资源级读写锁、优先级队列或复杂公平调度，只保留可扩展接口。

### 13.5 execute_batch

ToolRuntime 先按模型 ToolCall 顺序执行无副作用 preflight：materialize、decode、access resolve、permission/approval 和 queue admission。遇到第一个 ask、input、deny 或 admission failure 时停止接纳后续调用，再从已经接纳的调用构建连续兼容批次：

- 结果顺序与输入顺序一致。
- 只并发连续、兼容的 parallel 工具。
- serial 工具始终形成执行 fence，并按 group 排队；serial 注册的 group 不得为空。
- approval/input 暂停后，后续调用不会通过 preflight，也不会启动。
- 已启动批次必须收集全部结果。
- deny 默认阻止后续批次。
- pending 超过上限返回 `tool.queue.full`。
- 排队时间计入 request deadline；超时返回 queue timeout，不启动 handler。

## 14. Progress 与副作用

### 14.1 ToolProgressEvent

```python
@dataclass(frozen=True)
class ToolProgressEvent:
    tool_call_id: str
    attempt_id: str
    sequence: int
    kind: ToolProgressKind
    message: str = ""
    data: Mapping[str, object] = field(default_factory=dict)
    created_at_ms: int = 0
```

Progress kind：queued、started、message、output_delta、effect_started、effect_completed、heartbeat、cleanup_started。

- sequence 单调递增。
- Progress 不能修改最终状态。
- 高频输出需要限速和合并。
- 事件先脱敏再交给外层。

### 14.2 副作用三阶段

1. ToolPolicy 声明允许的 effect。
2. AccessResolver 根据参数生成计划 effect。
3. Handler 和 Runtime auditor 记录实际 effect。

```python
@dataclass(frozen=True)
class ToolEffect:
    kind: ToolEffectKind
    resource: ToolResource
    operation: str
    status: Literal["started", "completed", "partial", "unknown"]
    certainty: Literal["observed", "reported", "inferred"]
```

最终必须满足：

```text
实际 effect <= 已声明且已授权的 effect
```

超出范围时返回 policy violation，但不能丢弃已经发生的副作用证据。

timeout/cancelled 不代表没有副作用。

## 15. 状态机、持久化与恢复

### 15.1 ToolAttemptState

```text
received
validating
resolving_access
awaiting_approval
awaiting_input
queued
running
cleaning_up
succeeded
failed
denied
timed_out
cancelled
interrupted
```

状态转换必须先持久化，再进入可能产生副作用的下一阶段。

### 15.2 暂停不是终态

`approval_required` 和 `user_input_required` 是 ToolResult 的暂停态，但不作为最终 ToolResultMessage 写入模型对话。

- 它们进入事件、checkpoint 和 RuntimeFrame。
- resume 继续同一 attempt。
- 最终只向模型追加一次终态结果。
- deny 后终态为 denied。

### 15.3 ToolStateStore

tools 定义窄 Store Port，sessions 实现持久化，runtime 注入。

Store 保存：

- 完整 ExecutionRequest。
- Attempt 状态。
- Approval/Interaction challenge。
- ApprovalGrant 消费状态。
- 最终 Result。

状态更新使用 expected-state compare-and-set，防止重复 resume 和重复执行。

### 15.4 崩溃恢复

- received/validating/resolving_access：可重做无副作用阶段。
- awaiting_approval/awaiting_input：恢复 challenge。
- queued：重新调度。
- running/cleaning_up：标记 interrupted。
- mutation 工具不自动重试。
- 只有显式 retry-safe 工具可以自动重试。

无法自动恢复的 interrupted attempt 必须结算为 status=interrupted、error.code=tool.execution.interrupted，并保留已观察到的 effects；不能只停留在内部状态机中。

第一版 RecoveryMode：not_resumable、retry_safe、checkpointed。

## 16. 错误模型

### 16.1 ToolErrorKind

```text
registration
validation
unavailable
permission
approval
interaction
queue_timeout
execution_timeout
cancelled
interrupted
execution
output_validation
policy_violation
resource_cleanup
stale_registration
internal
```

### 16.2 ToolError

```python
@dataclass(frozen=True)
class ToolError:
    code: str
    kind: ToolErrorKind
    message: str
    retryable: bool = False
    recovery_hint: str = ""
    details: Mapping[str, object] = field(default_factory=dict)
```

错误码使用稳定命名空间，例如：

```text
tool.registration.not_found
tool.registration.stale
tool.input.invalid
tool.permission.denied
tool.approval.expired
tool.execution.timeout
tool.execution.cancelled
tool.execution.interrupted
tool.execution.handler_error
tool.output.invalid
tool.effect.policy_violation
tool.cleanup.failed
tool.runtime.internal_error
```

Handler 只能抛出受控 `ToolHandlerError`。未预期异常统一转换为安全错误，原始 traceback 只进入受保护诊断日志。

## 17. Description 规范

Description 只回答：

1. 工具做什么。
2. 什么时候使用。
3. 关键限制。
4. 返回什么信息。

不应包含当前权限模式、用户是否批准、动态 timeout、内部实现或无法证明的安全承诺。

推荐模板：

```text
<一句话说明核心能力。>

Use when:
- <典型场景>

Constraints:
- <关键边界或语义>

Returns:
- <结构化结果重点>
```

质量规则：

- 非空且不超过约 1,200 字符。
- 第一段能独立说明核心行为。
- 非显然参数必须有 Schema description。
- Description 中引用参数时必须使用真实 Schema 参数名。
- 不重复完整 Schema。
- 不堆积大量示例。
- 不承诺并发顺序、完成时机或其他 Runtime 无法保证的动态行为。
- 不描述尚未实现的安全保证。
- Description 与 Schema、Policy 一起进入 registration version hash。

## 18. Extension、Skill 与 Hook

Extension API 只能注册 canonical ToolRegistration，不能直接执行工具或访问 MaterializedTool、ApprovalStore 和 Scheduler 内部对象。

Hook 收敛为：

- `ToolObserver`：只读观察生命周期，不能修改结果和权限。
- `ToolOutputTransformer`：若修改领域输出，必须在 output codec 之前运行；固定顺序为 `domain output -> transformer -> output codec -> ToolResult.data -> renderer -> final guard`。Transformer 不能修改权限、状态或副作用。
- 权限扩展使用独立 PermissionRuleProvider，不能通过普通 before hook 临时放行。

所有来源最终进入相同 Registry 和 ToolRuntime。

Extension/Skill/MCP 按 owner 进行原子批量注册：整批校验成功后才发布新 Catalog snapshot；任一注册失败则整批回滚。卸载或重连按 owner 撤销对应 revision，不影响其他 owner，也不能留下部分可见工具。

## 19. MCP 适配

MCP 必须转换为 canonical ToolRegistration，不能保留特殊执行旁路。

```text
MCP Definition
  -> MCP Adapter
  -> ToolSpec / Codec / Policy / Handler / Renderer
  -> ToolRegistration
  -> ToolRegistry
  -> ToolRuntime
```

约束：

- 名称为 `mcp__<server>__<tool>`，保留原始 server/tool 名用于审计。
- inputSchema 转 input codec。
- outputSchema 转 output codec。
- 无 outputSchema 时使用受限 `UnverifiedJsonCodec` 并标记为 unverified，而不是假装完成强校验。
- 明确只读的 MCP 工具至少声明 network_access 和 external_state_read。
- 无法确认是否写远端状态时，保守声明 external_state_write，默认 medium risk + ask；如果适配器无法安全界定资源或副作用，则拒绝启用。
- server 配置 timeout、并发上限、凭据绑定和 allowlist。
- MCP 文本默认 untrusted。
- 图片和 blob 经过类型、大小和 artifact 控制。
- 不支持内容返回明确 omission。
- 不暴露本机路径。

缺少 outputSchema 的结果记录 `output_validation=structurally_validated`；MCP 文本和外部资源默认记录 `content_trust=untrusted`。这两个字段分别表达“结构验证强度”和“内容信任级别”，不能混用。

## 20. Plan、Interaction 与 Subagent

### 20.1 Plan

- category=plan，source=builtin。
- handler 由 Core 提供并注册。
- ToolRuntime 只做通用校验、权限、串行调度和结果返回。
- Plan adapter 通过 Core 提供的窄 PlanService 完成业务校验和 PlanState 原子提交后，才能返回成功领域输出。
- ToolResult success 表示本次计划操作已提交，不表示计划已获用户批准；计划业务审批仍由 Core 负责。
- Plan 工具使用 `task_plan` 串行组。

### 20.2 Interaction

- handler 由 Core 提供。
- 只能返回受控 InteractionRequest。
- ToolRuntime 转成 user_input_required 暂停结果。
- 外层 application runtime 路由到 Interface，Interface 不参与工具策略。
- InteractionResponse 由 ToolRuntime 校验并单次消费，恢复同一 attempt；不重新运行 handler。

### 20.3 Subagent

- handler 由 Runtime adapter 提供。
- 通过 Session semaphore 限制并发。
- 支持 cancellation、progress 和 checkpointed recovery。
- 子代理结果仍通过统一 ToolResult 返回。

Plan、Interaction、Subagent、Extension 和 MCP registrations 都由顶层 composition root 注入 ToolRuntime。tools 不导入 Core、Runtime adapter 或 Interface 的具体实现。

## 21. 可观测性

ToolEventEnvelope 包含：

- event ID 和 sequence。
- run/session/tool call/attempt/registration ID。
- timestamp。
- 脱敏 payload。

事件类型覆盖 received、validated、access resolved、denied、approval、queued、started、progress、effect、cleanup、completed、failed、timed out、cancelled 和 interrupted。

原则：

1. 同 attempt sequence 单调递增。
2. 先持久化状态，再发送事件。
3. 普通事件只包含 safe preview。
4. 大输出使用 artifact ID。
5. Event sink 失败不能导致重复副作用。
6. 最终 Result 是事实源，事件是证据。

ToolStateStore 保存恢复所需完整数据；Observability 只保存脱敏事件，两者不能混用。

## 22. 模块目录

```text
src/codepilot/
├── protocols/
│   └── tools.py
├── tools/
│   ├── __init__.py
│   ├── contracts.py
│   ├── codecs.py
│   ├── registry.py
│   ├── runtime.py
│   ├── policy.py
│   ├── permissions.py
│   ├── approvals.py
│   ├── scheduling.py
│   ├── cancellation.py
│   ├── progress.py
│   ├── effects.py
│   ├── results.py
│   ├── errors.py
│   ├── rendering.py
│   ├── guards.py
│   ├── resources.py
│   ├── artifacts.py
│   ├── state.py
│   ├── sandbox.py
│   └── builtins/
│       ├── filesystem.py
│       ├── search.py
│       ├── command.py
│       └── workspace.py
├── core/
│   └── tool_adapters/
│       ├── plan.py
│       └── interaction.py
├── runtime/
│   ├── composition.py
│   └── tool_adapters/
│       └── subagents.py
├── sessions/
│   └── tool_state_store.py
└── extensions/
    ├── tool_api.py
    └── mcp/
        ├── adapter.py
        ├── codec.py
        └── renderer.py
```

主要类型归属：contracts 放 Spec、Registration、Request 和 Context；codecs 放 codec；results/errors 放结算协议；resources/effects 放访问与副作用类型；rendering/guards/artifacts 放输出链；state 只定义 Store Port，sessions 实现持久化，runtime/composition.py 负责依赖装配。

## 23. 测试规范

### 23.1 Registration compliance

每个工具必须通过统一契约测试：

- 名称和 registration ID 合法。
- input/output Schema 合法。
- Codec 可往返。
- category/source/policy 完整。
- Description 合规。
- declared effect 与 category 不冲突。
- handler 不能返回最终 ToolResult。
- renderer 输出合法 ToolContent。
- access resolver 无副作用且确定性。

### 23.2 ToolRuntime 管线

测试必须证明：

- input invalid 时 handler 调用次数为 0。
- denied 和 awaiting approval 时 handler 调用次数为 0。
- output invalid 时不能 success。
- handler 异常不穿透 Core。
- final guard 在 transformer/renderer 后执行。
- 同 attempt handler 最多启动一次。

### 23.3 安全与审批

覆盖路径/符号链接逃逸、敏感文件、Shell 工作区外访问、自授权参数、effect 超范围、审批指纹、过期、单次消费和作用域隔离。

### 23.4 timeout、取消、恢复

覆盖 queue/execution timeout、协作取消、强制 task cancel、cleanup 顺序、进程树终止、partial effect、waiting approval 恢复和 mutation interrupted。

### 23.5 第一版并发

只覆盖 Session 上限、parallel 工具并发、同 serial group 串行、审批屏障、结果顺序和 pending 上限。

### 23.6 架构测试

- ToolRuntime 外不能访问 handler。
- Core 不导入 MaterializedTool。
- Interface 不导入 PermissionEngine。
- tools 不导入 PlanState、Memory、ContextGovernor 或 CLI。
- adapters 不直接构造最终 ToolResult。
- 生产代码不直接调用内置 handler。
- 不再引入第二套跨层结果协议。

## 24. 迁移方案

### 阶段 0：冻结旧协议

- 不继续扩展旧 ToolResult/ToolObservation。
- 记录当前行为并建立兼容测试。
- 暂不同时修改 Plan 业务语义。

### 阶段 1：建立新协议对象

新增 Spec、Registration、Codec、ExecutionRequest、Result、Error、Policy、AccessRequest，并提供新 Result 到旧消息协议的适配器。

### 阶段 2：Registry 与 Codec

- 建立 opaque Registry。
- 默认禁止覆盖。
- 增加 Catalog snapshot 和 registration identity。
- 使用标准 JSON Schema validator。
- 为旧定义提供临时 LegacyRegistrationAdapter。

### 阶段 3：单工具执行闭环

先实现 ToolRuntime.execute 的新链路，不同时迁移复杂并发和恢复。选择 workspace_status 或 read 完成第一个端到端切片。

### 阶段 4：逐个迁移 Builtins

推荐顺序：workspace_status、read、ls、grep/find、write、edit、apply_patch、bash/PowerShell。

每迁移一个工具，同时完成 typed input/output、Schema、access resolver、effect、policy、description 和 compliance tests。

### 阶段 5：权限、安全与审批

接入 action/resource/effect 权限、ApprovalGrant、ToolStateStore、敏感文件和 Shell 策略，移除内存 `_pending` 真值。

### 阶段 6：timeout、取消和基础并发

实现通用 timeout、CancellationToken、CleanupStack、Session semaphore、parallel/serial group 和 execute_batch；移除 Core 自行 gather。

### 阶段 7：特殊工具

迁移 Plan、Interaction 和 Subagent adapter，保持安全审批、计划审批和用户输入暂停分离。

### 阶段 8：Extension、Skill、MCP

所有来源改为 canonical registration，删除 MCP 特殊执行旁路，接入 output trust、artifact 和 server 限制。

### 阶段 9：删除旧协议

删除旧公开 execute、ToolCallRequest、PreparedToolCall、ToolObservation、重复 Approval 类型、关键 metadata 语义和不安全 legacy after hook。

保留 Session 已持久化数据的向后读取兼容。

## 25. 验收标准

1. 所有工具只能由 ToolRuntime 启动。
2. 所有来源转换为同一个 ToolRegistration。
3. 输入和输出均经过 Codec 校验。
4. Core 不再负责工具并发调度。
5. ToolRuntime 不理解 PlanState、Memory、Context 或 Interface。
6. 安全审批、计划审批和人机输入完全分离。
7. ToolResult 是唯一跨层执行结果。
8. 关键控制语义不隐藏在 metadata。
9. timeout/cancelled 保留副作用证据。
10. pending approval 可跨进程恢复且不可重复消费。
11. Description、Schema、Policy 和实现同步版本化。
12. 旧模型调用不能误执行热重载后的新 handler。

## 26. OpenCode 调研结论

本设计参考了 OpenCode `dev` 分支提交 `9976269ab1accfc9f9dc98a4a688c516934de422` 中正在迁移的 V2 工具架构。

值得借鉴：

- canonical Tool。
- 输入和输出 codec。
- 领域输出与模型输出分离。
- 小型 ExecutionContext。
- registration identity 和 stale call 防护。
- action/resource/effect 权限模型。
- 统一 settlement boundary。

不直接照搬：

- TypeScript Effect/WeakMap 的实现形式。
- V1/V2 双轨迁移结构。
- leaf tool 自行决定权限的边界。
- 任意 metadata 承载关键语义。
- 本机 output path 和 data URL 持久化。
- 自定义工具静默覆盖 builtin。
- 无上限并发和宿主用户完整 Shell 权限。

OpenCode V2 当前仍有 MCP、plugin、attachment 和 cancellation settlement 未完成项，因此只作为设计方向参考，不作为逐文件移植蓝本。
