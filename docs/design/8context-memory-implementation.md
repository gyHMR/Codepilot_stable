# Context 与 Memory 重构实施设计

## 文档状态

本文是 Codepilot 第一版 Context Governance 与 Memory Management 的联合实施规格。

目标设计分别见：

- [2context-design.md](./2context-design.md)
- [3memory-design.md](./3memory-design.md)

本文不重新讨论领域语义，而是回答：

```text
目标代码应该如何拆分？
Port 和 DTO 应该放在哪里？
现有实现按照什么顺序迁移？
旧 Memory 和 Context Checkpoint 如何升级？
每个阶段需要哪些测试？
满足什么条件才算完成？
```

重构严格限定为四个阶段：

1. 建立契约与兼容基线。
2. 重构 Memory 领域。
3. 重构 Context 领域。
4. 完成 Runtime 切换并删除旧路径。

不得增加阶段 0、阶段 5 或长期并行的新旧实现。阶段内部可以包含多个可独立提交的任务，但只能服务于当前阶段的验收目标。

## 1. 实施目标

本次实施需要最终得到：

- 类型化的 Context Preparation Port，不再使用 `Any -> Any`。
- 五层 Context 物化、预算治理、工具证据投影和辅助 LLM 压缩。
- 八字段 MemoryRecord、两种作用域和五态生命周期。
- 明确分离的 Memory Recall、Proposal 和 Management Port。
- Core 不直接访问 Sessions 或 Memory。
- Context 只通过 MemoryRecallPort 读取 Active Memory。
- Runtime 是 Context、Memory、Session、Core 和 Tools 的唯一组合者。
- Session Boundary 原子保存 CoreState、Messages 和不透明组件 Checkpoint。
- 自动 MemoryProposal 在 Terminal Commit 成功后写入 Candidate。
- 现有 Memory 文件和活动 Context Checkpoint 可以按确定规则迁移。
- 旧 DTO、旧 Memory 字段、旧召回算法和重复写入路径全部删除。

本次实施不包括：

- 向量数据库和 Embedding 检索。
- Prompt cache key 和 Provider 专属缓存编辑。
- 路径条件规则。
- Team Memory 和远程 Memory 同步。
- 独立 Memory 提取模型。
- Memory 自动审批或自动过期。
- 完整 ContextReport / Memory 审计平台。
- Benchmark 数据集扩建；现有 `benchmarks/` 不进入本次提交范围。

## 2. 前置依赖与实施边界

Context 与 Memory 重构依赖以下目标契约已经存在或在第一阶段同时建立：

| 依赖 | 必需能力 |
|---|---|
| Core | `CoreRunInput`、`CoreState`、`CoreBoundary`、`CoreOutcome` 和单一 Driver |
| Sessions | `begin_run`、`inspect_recovery`、`resume_run`、`commit_run_boundary` |
| Tools | Canonical `ToolResult`、ArtifactRef、无副作用 prepare 与 checkpoint |
| Runtime | `RunEnvironment`、`BoundaryPort` Adapter 和唯一 `RunExecutor` |
| LLM | 标准 ModelPort，以及供 Context Summarizer 使用的窄摘要能力 |

依赖方向保持：

```text
protocols -> llm/tools -> core -> sessions/observability
                                     -> extensions -> runtime -> interfaces
```

具体约束：

- Core 不 import `sessions.context` 或 `sessions.memory`。
- Context 可以实现 Core 定义的消费端 Port，但 Core 不依赖其实现。
- Memory 不读取 Session、Run 或 Tool 的文件布局。
- Tools 不 import CoreState、Context、Memory 或 SessionStateService。
- SessionStateService 只保存 Context/Tools 提供的不透明 Checkpoint。
- Runtime 可以 import 并组合以上所有领域。

## 3. 目标调用关系

```text
Runtime
├── SessionRunPort
├── run_core(CoreRunInput, CorePorts)
├── MemoryProposalPort / MemoryManagementPort
├── ContextCheckpointPort
└── ToolControlPort / ToolCheckpointPort

Core
├── ModelPort
├── ContextPreparationPort
├── ToolExecutionPort
└── BoundaryPort

Context
└── MemoryRecallPort
```

禁止建立：

```text
Core -> Session / Memory
Context -> Session / ToolRuntime
Memory -> Session / Core / Tools
Tools -> Context / Memory / Session
Session -> ContextService / MemoryService / ToolRuntime
```

Context 和 Memory 当前仍位于 `sessions/` 源码命名空间中，但它们是独立领域。目录归属不能成为绕过 Port 直接调用 Service 或 Repository 的理由。

## 4. 目标 Port 与 DTO

### 4.1 CoreRunInput

`session_id` 和 `run_id` 必须成为显式执行身份：

```python
@dataclass(frozen=True)
class CoreRunInput:
    session_id: str
    run_id: str
    entry: ModelEntry | ToolResultEntry
    messages: tuple[Message, ...]
    state: CoreState
    mode: RunMode
    model: ModelDescriptor
    limits: CoreLimits
    context_seed: Mapping[str, object]
```

`context_seed` 只允许保存 Core 不解释、原样传给 Context 的启动信息，例如：

- L0 基础指令引用。
- 已恢复的 Context component 引用。
- Runtime 配置产生的 Context feature flags。

`context_seed` 不允许保存：

- `session_id` 或 `run_id`。
- CoreState 的第二份副本。
- Tool approval、deadline 或 registration state。
- Memory Service 或 Repository 对象。

现有 `core/tool_step.py` 从 `context_seed` 读取 `session_id` 的逻辑必须删除。

### 4.2 ContextPreparationPort

Port 由 Core 定义，Context Service 实现：

```python
class ContextPreparationPort(Protocol):
    async def prepare(
        self,
        request: ContextPrepareRequest,
    ) -> PreparedModelContext: ...
```

请求：

```python
@dataclass(frozen=True)
class ContextPrepareRequest:
    session_id: str
    run_id: str
    purpose: Literal["reasoning", "verification", "finalization"]
    directive: str | None
    messages: tuple[Message, ...]
    core_view: CoreContextView
    model: ModelDescriptor
    tool_catalog: ToolCatalogSnapshot
    seed: Mapping[str, object]
```

`CoreContextView` 是 CoreState 的只读投影：

```python
@dataclass(frozen=True)
class CoreContextView:
    goal: str
    mode: RunMode
    plan: TaskPlanView | None
    current_step: TaskStepView | None
    verification: VerificationView
    blocked_reason: str | None
```

Context 不能持有或修改完整 CoreState。

返回：

```python
@dataclass(frozen=True)
class PreparedModelContext:
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[Tool, ...]
    projection_ref: str
```

Core 只消费模型调用所需内容。完整 ContextReport、选择原因、Token 明细和内部 Snapshot 留在 Context/Runtime，不进入 Core 领域协议。

### 4.3 ContextCheckpointPort

Runtime Boundary Adapter 单独持有：

```python
class ContextCheckpointPort(Protocol):
    def checkpoint_state(self) -> Mapping[str, object] | None: ...
    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None: ...
```

CorePorts 不能暴露此 Port。

第一版 Context component 只保存：

```text
compact_snapshot_ref
compacted_until_message_id
可选 schema marker
```

它不能保存完整 Session Messages、CoreState 或 MemoryRecord。

### 4.4 MemoryRecallPort

Context 只能持有召回能力：

```python
class MemoryRecallPort(Protocol):
    def recall(self, query: MemoryQuery) -> MemoryRecallResult: ...
```

```python
@dataclass(frozen=True)
class MemoryQuery:
    user_request: str
    task_goal: str
    current_step: str | None
    active_paths: tuple[str, ...]
    limit: int = 5
```

返回不可变召回投影，而不是可修改 MemoryRecord：

```python
@dataclass(frozen=True)
class RecalledMemory:
    memory_id: str
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str
    source: MemorySource
    rank_reasons: tuple[str, ...]
```

### 4.5 MemoryProposalPort

Runtime 最终化路径持有：

```python
class MemoryProposalPort(Protocol):
    def submit_proposals(
        self,
        batch: MemoryProposalBatch,
    ) -> MemoryProposalReceipt: ...
```

Batch 携带瞬时运行事实：

```python
@dataclass(frozen=True)
class MemoryProposalBatch:
    session_id: str
    run_id: str
    origin: Literal["user_explicit", "agent_finalization"]
    verification_passed: bool
    proposals: tuple[MemoryProposal, ...]
```

`session_id`、`run_id` 和验证状态只用于准入和幂等，不进入八字段 MemoryRecord。

LLM 只能提供：

```text
scope + type + key + content
```

`source` 和 `status` 必须由 Memory Service 根据 Batch origin、用户原始动作和验证状态设置。

### 4.6 MemoryManagementPort

用户管理入口通过 Runtime 使用：

```python
class MemoryManagementPort(Protocol):
    def execute(
        self,
        command: MemoryCommand,
        actor: MemoryActor,
    ) -> MemoryCommandResult: ...
```

`MemoryCommand` 是明确的判别联合：

```text
AddMemory
ListMemory
ShowMemory
ApproveMemory
RejectMemory
EditMemory
DisableMemory
EnableMemory
DeleteMemory
HistoryMemory
PurgeMemory
```

Agent 不能获得此 Port。普通模型响应只能通过 MemoryProposalPort 提出 Candidate。

### 4.7 SessionRunPort 与 BoundaryPort

Runtime 使用 Session 执行 Port：

```python
class SessionRunPort(Protocol):
    def open_session(...) -> SessionSnapshot: ...
    def begin_run(...) -> BeginRunResult: ...
    def inspect_recovery(...) -> RecoveryResult: ...
    def resume_run(...) -> RunState: ...
    def commit_run_boundary(...) -> CommitRunBoundaryReceipt: ...
```

目标 `ResumeRunRequest` 需要允许 Runtime 携带 `prepare_resume` 后生成的 opaque component checkpoint。Sessions 必须在同一次状态更新中：

```text
校验 waiting request 与 checkpoint revision
-> 保存新的 components.tools
-> 清除 WaitingState
-> 将 Run 转为 running
```

Sessions 不解释 `components.tools`，但不能先清除 WaitingState、随后再依赖另一次提交保存 prepared resume 状态。

Core 仍然只依赖：

```python
class BoundaryPort(Protocol):
    async def commit(self, boundary: CoreBoundary) -> None: ...
```

RuntimeSessionBoundaryAdapter 在 commit 前收集：

```text
CoreBoundary.core_state
CoreBoundary.new_messages
CoreBoundary.domain_events
ContextCheckpointPort.checkpoint_state()
ToolCheckpointPort.checkpoint_state()
WorkspaceCheckpoint
```

然后构造唯一 `CommitRunBoundaryRequest`。只有收到 CommitReceipt 后，BoundaryPort 才能返回 Core。

### 4.8 Tools Port 拆分

同一个 ToolRuntime 可以实现三个窄 Protocol：

```python
class ToolExecutionPort(Protocol):
    def catalog_snapshot(...) -> ToolCatalogSnapshot: ...
    def prepare_batch(...) -> ToolBatchPreparation: ...
    async def execute_prepared(...) -> tuple[ToolResult, ...]: ...

class ToolControlPort(Protocol):
    def pending_challenges(...) -> tuple[ToolChallenge, ...]: ...
    def approval_challenge(...) -> ToolChallenge | None: ...
    async def cancel(...) -> bool: ...
    def prepare_resume(...) -> ToolResumePreparation: ...
    async def execute_prepared_resume(...) -> ToolResult: ...

class ToolCheckpointPort(Protocol):
    def checkpoint_state(...) -> Mapping[str, object] | None: ...
    def restore_checkpoint_state(state: Mapping[str, object]) -> None: ...
```

CorePorts 只接受 ToolExecutionPort。ToolControlPort 和 ToolCheckpointPort 只由 Runtime 使用。

当前直接 `resume(response)` 的路径需要拆成无副作用 prepare 和实际 execution，保证审批恢复同样满足“先持久化 Checkpoint，再产生副作用”。

## 5. Context 目标文件布局

```text
src/codepilot/sessions/context/
├── __init__.py
├── contracts.py
├── service.py
├── state.py
├── projection.py
├── budget.py
└── compaction.py
```

### `__init__.py`

受控公共 facade，只导出：

- ContextService 构造入口。
- ContextCheckpointPort 相关稳定类型。
- ContextSummarizerPort。
- 必要的 Context 配置类型。

不能导出内部 selector、mutable state 或压缩 helper。

### `contracts.py`

保存 Context 领域内部稳定对象：

- `ContextLayer`、`RetentionClass`。
- `ContextItem`、`ContextSourceRef`。
- `ContextPressure`、`ContextBudget`。
- `ProjectionPlan`、`ProjectedMessage`、`ProjectedEvidence`。
- `CompactSummary`、`CompactSnapshotRef`。
- `ContextCheckpointState`。
- `ContextSummarizerPort` 和摘要请求/返回 DTO。

Core-facing `ContextPrepareRequest` 和 `PreparedModelContext` 不放在这里，它们属于 Core 消费端契约。

### `service.py`

实现 ContextPreparationPort 和 ContextCheckpointPort，负责：

```text
接收请求
-> 刷新 ContextState
-> 调用 MemoryRecallPort
-> 生成五层候选
-> 生成统一 ProjectionPlan
-> 调用 BudgetSelector
-> 必要时触发 ContextCompactor
-> 最终硬预算校验
-> 返回 PreparedModelContext
```

Service 只做编排，不实现分词、Token 选择或摘要 Prompt 细节。

### `state.py`

保存一次 Session/Run 的 Context 派生状态：

- Active working files。
- Evidence freshness。
- Repository snapshot fingerprint。
- 当前 Compact Snapshot ref 和 cursor。
- Checkpoint 编解码。

状态不是 Session 权威事实，丢失后允许从 Session Messages、ToolResult 和 Workspace 重新计算。

### `projection.py`

负责：

- L0-L4 候选生成。
- L2 Tool Evidence 与 L4 Tool Message 的统一 ProjectionPlan。
- ToolResult、ArtifactRef 和 Evidence 的安全渲染。
- Tool call/result 原子组处理。
- Memory 的 L3 渲染。
- Compact Summary 和未压缩 L4 的拼接。

L2 和 L4 不能分别解析原始 ToolResult。二者必须从同一个 ProjectionPlan 和 source reference 生成。

### `budget.py`

负责：

- Token 估算。
- 有效输入预算计算。
- `normal / tight / critical / overflow` 压力判断。
- Required、Protected、Budgeted、DiscardFirst 选择。
- L4 确定性瘦身。
- 单项上限和 Layer 预算。
- Provider 调用前的最终硬校验。

Budget 模块不能调用 LLM，也不能修改 Session Messages。

### `compaction.py`

负责：

- 辅助 LLM Context Summarizer 调用。
- 结构化 Compact Summary 校验。
- 滚动压缩和 cursor 推进。
- Compact Snapshot Artifact 创建。
- 摘要失败时保留旧 Snapshot 或返回明确失败。

该模块不能写 Memory，也不能删除 Session Message。

## 6. Memory 目标文件布局

```text
src/codepilot/sessions/memory/
├── __init__.py
├── contracts.py
├── repository.py
├── service.py
├── admission.py
└── recall.py
```

持久化布局保持为：

```text
~/.codepilot/memory/
└── memories.jsonl

<workspace>/.codepilot/memory/
└── memories.jsonl
```

User Store 和 Project Store 使用同一 Record Schema，但 Repository 实例、根目录和访问范围必须分离。

### `__init__.py`

只导出：

- Memory Service 构造入口。
- MemoryRecallPort。
- MemoryProposalPort。
- MemoryManagementPort。
- 管理命令和公开结果类型。

不能导出 JSONL 文件操作、内部评分 helper 或可变 Store。

### `contracts.py`

保存：

- 八字段 `MemoryRecord`。
- `MemoryType`、`MemoryScope`、`MemorySource`、`MemoryStatus`。
- `MemoryProposal` 和 `MemoryProposalBatch`。
- `MemoryQuery`、`RecalledMemory`、`MemoryRecallResult`。
- `MemoryCommand` 判别联合及结果。
- Recall、Proposal、Management 三个 Port。

MemoryRecord 必须严格拒绝未知字段，避免旧 Schema 字段继续被静默保留。

### `repository.py`

负责：

- User 和 Project 两个 JSONL Store。
- 按 ID 读取最新快照。
- 原子追加或原子文件重写。
- Logical delete 状态保存。
- Sensitive purge 的物理清除。
- 损坏尾行检测和恢复。
- 旧 Schema dry-run、备份和升级。

Repository 不实现准入、冲突、召回和用户权限。

### `service.py`

实现三个 Memory Port，负责：

- 显式写入和自动 Proposal 编排。
- Candidate 审批。
- Active/Disabled/Superseded/Deleted 生命周期。
- 编辑、启用、禁用、删除、历史和 Purge。
- 同一 `scope + key` 当前版本不变量。
- 调用 AdmissionPolicy 和 MemoryRetriever。

### `admission.py`

负责确定性规则：

- Key 和 Content 规范化。
- 敏感信息拒绝。
- 跨任务价值的基础结构校验。
- 自动 Proposal 上限。
- 精确重复检测。
- 同 Key 冲突识别。
- Source 和初始 Status 的服务端赋值。

该模块不调用 LLM。LLM 只在 Runtime 捕获的最终化 sidecar 中提供 Proposal。

### `recall.py`

负责：

- Active-only 和 Scope 过滤。
- Profile/Feedback 基础召回。
- Project/Experience/Reference 相关性匹配。
- 英文、标识符、路径和中文双字词规范化。
- 可解释排序元组。
- Project 对同 Key User Memory 的覆盖。
- 返回最多 5 条 RecalledMemory。

Context Token 预算仍由 Context 负责，Recall 不构造最终 Prompt。

## 7. 集成文件变更

除 Context 和 Memory 目录外，迁移会有针对性地修改：

| 文件 | 目标变更 |
|---|---|
| `core/contracts.py` | 增加显式 `session_id`、类型化 Context DTO 和 ContextPreparationPort |
| `core/model_step.py` | 按 Decision 传递 Context purpose/directive，消费 PreparedModelContext |
| `core/tool_step.py` | 删除从 context_seed 读取 session_id 的逻辑 |
| `tools/contracts.py` | 将宽 ToolPort 拆成 Execution/Control/Checkpoint Protocol |
| `tools/runtime.py` | 同一 ToolRuntime 实现窄 Port，并提供 prepare_resume/execute_prepared_resume |
| `sessions/contracts.py` | 保留 Session DTO，迁出旧 Context DTO，并让 ResumeRunRequest 接收 prepared component checkpoint |
| `sessions/service.py` | 原子保存 resume component、清除 waiting；继续不解释 Context/Tools/Memory payload |
| `runtime/environment.py` | 组合 CorePorts，持有 Memory 和组件 Checkpoint Port |
| `runtime/session_state_adapter.py` | 在 Boundary 收集 Context/Tools checkpoint 并提交 Sessions |
| `runtime/model.py` | 为 finalization 请求捕获 message + MemoryProposal sidecar |
| `runtime/session_coordinator.py` | 删除旧 MemoryWriter/ContextGovernor 直连和重复 finalization 路径 |

不创建第二个 Runtime Coordinator、第二个 Session Store 或第二套 ToolResult 协议。

## 8. 四阶段迁移顺序

### 阶段一：建立契约与兼容基线

#### 目标

先固定跨域契约和现有行为，不立即替换 Context/Memory 主逻辑。阶段结束时，新 DTO 和窄 Port 已经存在，旧实现通过临时 Adapter 运行。

#### 实施内容

1. 为现有 Context、Memory、Runtime 集成行为补充特征测试。
2. 在 Core contracts 中引入显式 `session_id`、`CoreContextView`、`ContextPrepareRequest`、`PreparedModelContext` 和 ContextPreparationPort。
3. 在 Memory contracts 中建立八字段 Record、Proposal、Query、Command 和三个 Port。
4. 在 Tools contracts 中拆出 Execution、Control 和 Checkpoint Protocol；ToolRuntime 暂时同时实现旧宽接口和新窄接口。
5. 建立 Context/Memory 新目录文件，但只放契约和 Adapter，不迁移领域逻辑。
6. Runtime 使用 Adapter 把当前 PreparedAgentContext 转成 PreparedModelContext。
7. `CoreRunInput.session_id` 替代 `context_seed["session_id"]`。
8. 增加架构 import 测试，禁止新增反向依赖。

#### 临时兼容范围

允许：

- 旧 Context Service 经 Adapter 实现新 ContextPreparationPort。
- 旧 Memory Retriever 经 Adapter 实现新 MemoryRecallPort。
- ToolRuntime 同时满足旧 ToolPort 和新窄 Protocol。

不允许：

- 新代码继续增加 `Any` DTO。
- 新 Context 逻辑读取旧 Memory 多字段模型。
- 新旧数据同时双写。
- 旧 Adapter 跨过第四阶段继续存在。

#### 阶段测试

```text
test/test_core_contracts_v2.py
test/test_runtime_context.py
test/test_runtime_environment.py
test/test_context_memory_architecture.py        # 新增
test/test_context_memory_contracts.py           # 新增
```

必须证明：

- Core 不 import Sessions/Memory。
- ContextPreparationPort 不再使用 Any。
- `session_id` 不再从 context_seed 读取。
- CorePorts 不暴露 MemoryManagementPort、ToolControlPort 或 Checkpoint Port。
- 临时 Adapter 与现有主链路行为一致。

#### 阶段门槛

只有在新契约测试通过、现有 Runtime/Core 测试无回归后，才能开始 Memory 重构。此时不能删除旧 Memory 或 Context 实现。

### 阶段二：重构 Memory 领域

#### 目标

先完成独立可用的新 Memory，因为 Context L3 需要稳定的 MemoryRecallPort。阶段结束时，所有 Memory 写入、管理和召回使用新八字段模型，Context 仍可通过 Adapter 消费。

#### 实施内容

1. 实现严格八字段 MemoryRecord 和枚举校验。
2. 实现 User/Project 双 Repository 和 JSONL 原子操作。
3. 实现 AdmissionPolicy：规范化、敏感信息检查、重复与冲突判断。
4. 实现生命周期和用户管理命令。
5. 实现 Active-only、确定性、中文友好的 Recall。
6. 实现 MemoryProposalPort，但暂不接入最终模型 sidecar。
7. 编写旧 Memory dry-run 和升级器。
8. 将当前 Context Memory Retriever Adapter 切换到新 MemoryRecallPort。
9. 删除旧 MemoryWriter、旧评分字段和旧冲突链字段；保留一次性迁移代码。

#### 旧 Memory 字段映射

类型：

| 旧类型 | 新类型 |
|---|---|
| `preference` | `profile` |
| `correction` | `feedback` |
| `constraint` | `project` |
| `decision` | `project` |
| `workflow` | `experience` |
| `experience` | `experience` |

作用域：

| 旧 Scope | 新 Scope |
|---|---|
| `global` / `user` | `user` |
| `workspace` / `project` | `project` |
| `session` | 不自动迁移，进入迁移报告 |

来源：

| 旧 Source | 新 Source |
|---|---|
| `user_explicit` / `user_approved` / `manual_edit` | `user_explicit` |
| `user_correction` | `user_feedback` |
| 已验证运行来源 | `verified_run` |
| 其他模型或自动来源 | `agent_extracted` |

状态只接受：

```text
candidate
active
disabled
superseded
deleted
```

Key 迁移规则：

1. 优先使用旧 `subject/predicate/type` 生成稳定点分 Key。
2. Project constraint/decision 分别映射到 `project.constraint.*` 和 `project.decision.*`。
3. 无法生成稳定语义 Key 时使用 `legacy.<short_hash>`，并把 Active 降级为 Candidate。
4. 同一 `scope + key` 出现多个冲突 Active 时，迁移必须失败并生成报告，不能按时间自动选一个。

#### 数据迁移过程

```text
读取旧 JSONL
-> 严格校验和 dry-run 报告
-> 构造新记录到临时文件
-> 校验不变量与可重新加载性
-> 备份旧文件为 .legacy.bak
-> 原子替换 memories.jsonl
-> 再次读取验证
```

迁移必须幂等。已经是新八字段 Schema 的文件不能再次转换。失败时保留旧文件，不允许部分替换。

旧 Schema 必须由隔离的 Legacy Reader 解析，不能放宽新 MemoryRecord 的未知字段校验来兼容旧数据。Legacy Reader 在第四阶段切换完成后只保留于一次性迁移入口，不能进入正常 Repository 读取路径。

#### 阶段测试

```text
test/test_memory_contracts.py               # 新增
test/test_memory_repository.py              # 新增
test/test_memory_admission.py               # 新增
test/test_memory_management.py              # 新增
test/test_memory_recall.py                  # 新增
test/test_memory_migration.py               # 新增
test/test_memory.py                         # 迁移现有用例后保留或删除重复
```

必须覆盖：

- Record 只接受八个字段。
- User/Project Scope 文件隔离。
- Candidate 审批和五态转换。
- Active/Disabled 编辑生成 Superseded 旧版本。
- Deleted 不召回，Purge 物理清除。
- 精确重复不创建记录、不刷新时间。
- 同 Key 冲突需要显式替换或审批。
- Project 同 Key 覆盖 User。
- 中文 Query 能召回相关 Content。
- 敏感信息拒绝落盘。
- 旧文件迁移幂等、失败不破坏原文件。

#### 阶段门槛

只有在新 Memory 独立测试全部通过、Context Adapter 可以使用新 Recall、旧 Memory 数据完成 dry-run 后，才能重构 Context。

### 阶段三：重构 Context 领域

#### 目标

用新的五层物化、预算、投影和压缩模块替换当前集中在单个 Service 中的旧逻辑。阶段结束时，所有模型调用通过新 ContextPreparationPort，旧 Adapter 只保留 Runtime 切换用途。

#### 实施内容

1. 将 ContextState、Projection、Budget 和 Compaction 从旧 Service 中拆出。
2. 实现 L0-L4 候选生成和保留等级。
3. 让 L2 Evidence 与 L4 Tool Message 使用同一 ProjectionPlan。
4. 只消费 Canonical ToolResultMessage 和 ArtifactRef，不读取 ToolRuntime 内部状态。
5. 接入新 MemoryRecallPort，生成独立 L3 区块。
6. 实现压力等级、确定性瘦身、L4 原子消息组和最终硬预算检查。
7. 实现辅助 LLM ContextSummarizerPort、结构化 Compact Summary 和滚动压缩。
8. 实现 Compact Snapshot ref/cursor Checkpoint 与恢复。
9. 支持旧 Context Checkpoint 单次升级。
10. 将 ContextReport 限制在内部诊断，不扩展新的审计平台。

#### 旧 Context Checkpoint 迁移

当前活动 Run 可能保存：

```text
compact_summary
compacted_until_message_id
```

兼容加载器按以下方式处理：

- 已知旧字段转换为 Legacy Compact Snapshot 内存视图。
- 旧 Summary 只能服务原 Run，不能写入 Memory。
- 下一次成功压缩或 Boundary 只写新 Snapshot ref/cursor。
- 未知字段或损坏 cursor 阻止 Resume，不能静默忽略。
- Stage Four 删除旧 Checkpoint 写入逻辑，只保留必要的单向读取升级器。

#### 阶段测试

```text
test/test_context_contracts.py               # 新增
test/test_context_layers.py                  # 新增
test/test_context_projection.py              # 新增
test/test_context_budget.py                  # 新增
test/test_context_compaction.py              # 新增
test/test_context_checkpoint.py              # 新增
test/test_context_governance.py              # 迁移现有用例
test/test_context_governor_refactor.py        # 合并后删除重复文件
```

必须覆盖：

- L0 永不被预算裁剪。
- L1 使用 CoreContextView，不读取 CoreState 内部对象。
- L2 和 L4 对同一 ToolResult 使用相同 source reference。
- ToolCall/ToolResult 原子组不会被拆开。
- Artifact 化后模型只看到安全投影和引用。
- `normal/tight/critical/overflow` 阈值正确。
- Tight 可以确定性瘦身 L4，但不能语义删除普通消息。
- Critical 触发辅助 LLM 压缩。
- Summarizer 失败不覆盖旧 Snapshot。
- 最终请求超预算时不调用 Provider。
- Memory Recall 失败降级为空 L3 并保留诊断。
- Candidate/Disabled/Superseded/Deleted 不进入 L3。
- Compact Summary 永远不写 Memory。
- Checkpoint 恢复后可以重新投影相同有效上下文。

#### 阶段门槛

只有在新 Context 对每种压力和恢复场景都通过测试、Provider 硬预算得到保证后，才能切换 Runtime 主链路。旧 Context 仍不能在此阶段提前删除。

### 阶段四：完成 Runtime 切换并删除旧路径

#### 目标

将新 Context、Memory、Session 和 Tools Port 接入唯一 Runtime 主链路，完成最终化和恢复闭环，然后删除所有临时 Adapter、旧 DTO 和旧实现。

#### 实施内容

1. RunEnvironment 只通过窄 Port 组装 CorePorts。
2. RuntimeSessionBoundaryAdapter 收集 Context/Tools checkpoint，并通过唯一 Session commit 提交。
3. Runtime Model Adapter 在 `purpose=finalization` 时捕获用户可见 message 与 MemoryProposal sidecar。
4. Core 只接收 AssistantMessage，不 import 或持久化 MemoryProposal。
5. Terminal Commit 成功后，Runtime 同步调用 MemoryProposalPort 写 Candidate。
6. MemoryProposal 写入失败只生成诊断，不回滚已提交 Run。
7. Tool approval resume 切换为 prepare_resume、持久化 checkpoint、execute_prepared_resume。
8. Resume 按 Sessions -> Context/Tools restore -> Runtime -> Core 的顺序执行。
9. 删除 SessionCoordinator 中旧 Memory finalize 和 Context 直连逻辑。
10. 删除临时 Adapter、旧 DTO、旧字段解析、旧召回和旧压缩路径。
11. 更新设计文档和 README 中的测试命令、文件布局与接口名称。

#### 最终化时序

```text
Context prepare(purpose=finalization)
-> Model Adapter 解析 message + memory_proposals
-> Core 只接收并提交 message
-> CoreOutcome completed
-> Runtime terminal commit
-> CommitReceipt
-> MemoryProposalPort.submit_proposals
-> 最终 RuntimeFrame
```

规则：

- Memory 自动写入不能发生在 Terminal Commit 之前。
- sidecar 缺失或非法时，最终回答仍可完成，MemoryProposal 被丢弃并记录诊断。
- 自动 Proposal 是非权威派生信息，允许失败；Session 终态不允许因此回滚。
- 用户显式 Memory 管理命令失败时必须明确返回失败，不能按自动 Proposal 方式静默降级。

#### Tool Resume 时序

```text
inspect_recovery
-> restore Context/Tools checkpoint
-> ToolControlPort.prepare_resume（无副作用）
-> SessionRunPort.resume_run 原子保存新的 Tools checkpoint 并清除 waiting
-> ToolControlPort.execute_prepared_resume
-> CoreRunInput(ToolResultEntry)
-> run_core
-> after_tools Boundary Commit
```

在保存 prepared resume checkpoint 前不得启动 Tool Handler。

#### 阶段测试

```text
test/test_runtime_context_memory_integration.py    # 新增
test/test_runtime_boundary_contracts.py
test/test_runtime_sessions_state_v2.py
test/test_runtime_core_entry.py
test/test_runtime_environment.py
test/test_sessions_persistence_v2.py
test/test_tool_execution_v2.py
test/test_tool_security_state_v2.py
test/test_web_runtime_integration.py
```

必须覆盖：

- 每次模型调用都经过 ContextPreparationPort。
- before_tools Commit 失败时 Tool Handler 调用次数为 0。
- after_tools 已产生副作用但 Commit 失败时不自动重放 Mutation Tool。
- final message 先完成 Terminal Commit，再写 Candidate。
- MemoryProposal 写入失败不改变 completed Outcome。
- Session commit 失败时不发送成功 RuntimeFrame。
- Resume 不创建新 Run，不重复 ToolResultMessage。
- approval resume 在持久化 prepared state 后才执行。
- Context/Tools component checkpoint 对 Sessions 不透明。
- 旧 Context/Memory Adapter 已不存在。

#### 阶段门槛

完成全量回归、架构测试和删除检查后才能宣告重构完成。不得以“旧路径暂时保留”为理由结束第四阶段。

## 9. 测试策略

### 9.1 测试层次

| 层次 | 重点 |
|---|---|
| Contract | DTO 严格校验、Port 权限、未知字段拒绝 |
| Unit | Admission、Lifecycle、Recall、Budget、Projection、Compaction |
| Repository | JSONL 原子性、损坏恢复、迁移和 Purge |
| Integration | Context + Memory、Core + Context、Runtime + Sessions + Tools |
| Recovery | Checkpoint 恢复、Waiting/Resume、提交失败和不重复副作用 |
| Architecture | import 方向、无 Any Port、无第二写入路径 |
| Regression | CLI/Web/Runtime 和现有 Core/Tools/Sessions 用例 |

### 9.2 Memory 测试矩阵

| 场景 | 期望 |
|---|---|
| 显式用户写入 | Active |
| Agent 自动 Proposal | Candidate |
| 未验证 Experience | 拒绝或 Candidate，不得 Active |
| Candidate approve | Active，旧当前版本 Superseded |
| Active disable/enable | 不召回/重新召回 |
| Active edit | 新 Active + 旧 Superseded |
| Delete | Deleted 且不召回 |
| Purge | 文件中不再存在敏感记录 |
| 同 Key 同 Content | No-op，不刷新时间 |
| 同 Key 不同 Content | 冲突，不自动覆盖 |
| Project 与 User 同 Key | 当前项目只召回 Project |
| 中文相关请求 | 可匹配中文 Content |
| 非 Active 状态 | 永不召回 |
| Recall Store 故障 | Context 降级为空 L3 |

### 9.3 Context 测试矩阵

| 场景 | 期望 |
|---|---|
| Normal 压力 | 五层按正常预算物化 |
| Tight 压力 | 先裁剪低价值项和 Tool 重复投影 |
| Critical 压力 | 触发辅助 LLM Compact Summary |
| Overflow | Provider 调用前失败 |
| L0 超过预算 | 明确失败，不裁剪 L0 |
| 长 ToolResult | Artifact + 安全 Projection |
| 长 L4 对话 | 确定性瘦身后语义压缩 |
| ToolCall/Result 边界 | 原子保留或原子移除 |
| L2/L4 同一工具事实 | source reference 一致 |
| 当前证据与 Memory 冲突 | 当前证据生效 |
| Summarizer 失败 | 保留旧 Snapshot，不制造空 Summary |
| Resume | 使用 checkpoint ref/cursor 重新投影 |

### 9.4 Port 与架构测试

至少增加 AST/import 测试证明：

- `codepilot.core` 不 import `codepilot.sessions` 或 Memory。
- `codepilot.tools` 不 import CoreState、Context 或 Memory。
- Context 不 import SessionStateService 或 ToolRuntime。
- Memory 不 import Core、Context、Tools 或 Session Repository。
- Runtime 是唯一同时 import Core、Sessions、Context、Memory 和 Tools 的组合层。
- CorePorts 不出现 MemoryManagementPort、ContextCheckpointPort 或 ToolCheckpointPort。
- Production ContextPreparationPort 不包含 `Any` 请求或返回。

### 9.5 失败注入测试

必须可注入：

- Memory Repository 写失败。
- Memory Recall 失败。
- Context Summarizer 超时或返回非法结构。
- Context 最终硬预算失败。
- Session Progress/Waiting/Terminal Commit 失败。
- Tool checkpoint 读取失败。
- Tool prepare 后、execution 前进程失败。
- Tool execution 后、after_tools commit 前失败。
- Finalization sidecar 非法。

测试不能只断言异常类型，还要断言：

- Model 或 Tool 是否被调用。
- Session revision 是否变化。
- 是否产生重复 Message。
- 是否产生重复副作用。
- Memory 是否被错误激活。
- Checkpoint 是否仍可恢复。

## 10. 验证命令

阶段内先运行最小相关测试，再运行全量回归：

```powershell
python -m pytest test/test_memory_contracts.py test/test_memory_repository.py test/test_memory_admission.py test/test_memory_management.py test/test_memory_recall.py test/test_memory_migration.py -q

python -m pytest test/test_context_contracts.py test/test_context_layers.py test/test_context_projection.py test/test_context_budget.py test/test_context_compaction.py test/test_context_checkpoint.py -q

python -m pytest test/test_runtime_context_memory_integration.py test/test_runtime_boundary_contracts.py test/test_runtime_sessions_state_v2.py test/test_tool_execution_v2.py -q

python -m pytest test -q
python -m compileall -q src/codepilot
git diff --check
```

项目当前没有配置 Ruff、Mypy 或 Pyright，重构期间不额外引入格式化器或静态检查依赖。类型和架构边界通过 Python 类型声明、严格 DTO 校验和测试保证。

## 11. 验收标准

### 11.1 Context 验收

1. 所有 Model 调用都通过类型化 ContextPreparationPort。
2. L0 永不被预算裁剪。
3. L1 只消费 CoreContextView。
4. L2 和 L4 对 ToolResult 使用同一 ProjectionPlan 和 source reference。
5. L3 只包含 Active Memory。
6. L4 确定性瘦身不删除普通用户/助手语义。
7. Critical 压力使用辅助 LLM 结构化压缩。
8. Compact Summary 和 Snapshot 永不进入长期 Memory。
9. Provider 调用前必须通过最终输入硬预算。
10. Context Checkpoint 丢失只导致重新计算，不修改 Session 权威事实。

### 11.2 Memory 验收

1. MemoryRecord 持久化字段严格等于八个。
2. 只存在 User 和 Project 两种 Scope。
3. 自动 Proposal 只能写 Candidate。
4. 只有用户操作可以批准、编辑、禁用、启用、删除和 Purge。
5. Experience 自动提取必须带有验证通过信号。
6. 同一 `scope + key` 最多一个 Active 或 Disabled 当前版本。
7. 同一 `scope + key` 最多一个 Candidate。
8. Candidate、Disabled、Superseded、Deleted 永不召回。
9. 精确重复不创建新记录，也不刷新更新时间。
10. 当前指令和当前 Workspace/Tool Evidence 始终覆盖 Memory。
11. 删除默认保留 Deleted，敏感信息支持物理 Purge。
12. 旧数据迁移可 dry-run、幂等且失败不破坏原文件。

### 11.3 Port 与集成验收

1. Core 不直接依赖 Sessions 或 Memory。
2. Context 只通过 MemoryRecallPort 读取 Memory。
3. Runtime 是唯一组合五个领域的模块。
4. CorePorts 只包含 Model、Context Preparation、Tool Execution、Boundary、Live Event 和 Cancellation。
5. Session Boundary Adapter 是 CoreBoundary 到 Sessions Commit 的唯一转换入口。
6. Sessions 不解释 Context/Tools component checkpoint。
7. Tools resume 满足 prepare -> persist -> execute。
8. Terminal Commit 成功后才允许提交自动 MemoryProposal。
9. Memory 自动写失败不改变 Session/Core 成功结果。
10. Session Commit 失败时不得继续后续副作用或发送成功结果。

### 11.4 删除验收

生产代码中不得继续出现：

- `ContextPort.prepare(request: Any) -> Any`。
- 从 `context_seed` 获取 `session_id`。
- 旧 Memory `subject/predicate/value/keywords/paths/priority/occurrences` 访问。
- 旧 Memory `confidence/validity/supersedes/superseded_by` 写入。
- SessionCoordinator 直接实例化 MemoryWriter 或调用旧 finalize_run。
- Context 直接读取 Memory Repository。
- Core 直接调用 Tool resume/checkpoint。
- 两套 Context compaction 或两套 Memory recall 主路径。
- 为兼容迁移而长期保留的新旧双写。

## 12. 完成定义

只有同时满足以下条件，Context 与 Memory 重构才算完成：

```text
四个阶段全部通过各自门槛
目标文件布局已经形成
旧数据和活动 Checkpoint 可以单向升级
所有跨域调用经过目标 Port
所有模型调用通过 Context 硬预算
所有自动 Memory 只形成 Candidate
全量 pytest 通过
compileall 通过
git diff --check 通过
旧 Adapter、旧 DTO 和旧主路径已经删除
设计文档与实际实现一致
```

不能以以下状态宣告完成：

- 新旧实现通过 feature flag 长期并存。
- 新 Context 仍依赖旧 MemoryRecord。
- Runtime 仍有第二条 Memory finalization 路径。
- 活动 Run 无法恢复，只能要求用户删除 Session。
- Context 超预算时仍尝试调用 Provider。
- 测试只覆盖成功路径，没有提交失败和恢复场景。
