# Sessions 状态与恢复设计

## 文档状态

本文描述 Codepilot Sessions v2 的目标边界与第一版实现方案，作为后续重构的设计依据。

本文不要求兼容历史 Sessions API、旧 `session.json`、旧 `run.json` 或历史 Checkpoint。重构完成后，旧会话文件不再读取和迁移。

第一版优先保证：

- 状态权威关系明确。
- Run 可以从稳定边界恢复。
- 恢复不会因为消息或事件重复而直接重复模型、工具调用。
- Context、Memory、Planning、Tools 和 Runtime 不再把业务逻辑放入 Sessions。
- 文件持久化足够可靠，但不引入数据库、事件溯源或分布式事务。

## 设计目标

Sessions 是 Agent 运行状态的事实存储与恢复基础设施。

它负责记录：

- 一个 Session 是什么。
- 一个 Run 当前处于什么状态。
- 哪些消息已经提交到会话历史。
- 当前 Run 最近一个稳定恢复位置是什么。
- Runtime 恢复执行前需要读取哪些状态。
- 会话和 Run 中已经发生过哪些审计事件。

它不负责决定：

- Agent 下一步调用模型还是工具。
- 哪些消息、记忆和证据进入本轮 prompt。
- 哪条长期记忆应该写入、召回、替代或删除。
- Plan 是否批准、拒绝、完成或重规划。
- 工具是否允许执行或是否可以安全重试。
- Git 变更应该如何回滚。

```text
Core        -> 决定 Agent loop 如何推进
Context     -> 决定模型本轮看到什么
Memory      -> 决定长期知识如何治理
Tools       -> 决定工具如何准备、审批和执行
Runtime     -> 协调一次 Run 的开始、执行、恢复和结束
Sessions    -> 保存并恢复上述流程产生的运行事实
```

## 核心原则

### 只有两层权威运行状态

Sessions 只有两层权威运行状态：

```text
Session State
  -> Run State
       -> Checkpoint
```

Checkpoint 是 Run State 的组成部分，不是独立事实来源。

### Messages 是内容权威

Messages 保存规范化会话内容：

- `UserMessage`
- `AssistantMessage`
- `ToolResultMessage`

模型上下文和用户可见历史以 Messages 为内容来源，但 Messages 不表示 Run 是否可以恢复，也不表示工具是否应该重新执行。

### Events 不是恢复权威

Events 用于：

- 流式进度。
- 审计。
- Trace。
- Evaluation。
- 问题诊断。

恢复逻辑不通过重放 Events 重建 Run State。Event 丢失不能改变权威运行状态。

### Checkpoint 只描述稳定恢复边界

Checkpoint 不保存 Python 调用栈、流式半成品或 Runtime 对象。它只描述：

- 从哪个稳定业务边界继续。
- 哪条消息是已经提交的消息游标。
- 当前是否等待外部输入。
- Tools、Context 等组件需要恢复的最小状态。
- 恢复前需要比较的工作区状态。

### Sessions 不解释组件状态

Core、Tools 和 Context 可以提交版本化恢复载荷，Sessions 只校验其结构和可序列化性，不解释其中业务字段。

### 第一版采用单写者模型

同一个 Session 同一时刻只能由一个 Runtime 实例写入。Runtime 负责进程内并发控制，Sessions 通过 revision 拒绝过期写入。

第一版不实现分布式事务和多进程租约协议。

## 领域模型

### Session State

Session State 保存会话身份、导航关系、默认运行配置和 Run 引用。

```python
@dataclass(frozen=True)
class SessionState:
    schema_version: int
    session_id: str
    workspace_root: str

    current_run_id: str | None
    last_run_id: str | None

    leaf_message_id: str | None
    parent_session_id: str | None
    parent_run_id: str | None
    session_kind: Literal["primary", "subagent"]

    current_mode: str
    model: ModelRef
    system_prompt_hash: str

    created_at: str
    updated_at: str
    revision: int
```

Session State 不保存：

- Checkpoint。
- Plan State。
- Context Report。
- Context compact summary。
- Memory Record。
- Tool pending state。

### Run State

Run State 是“当前任务进行到哪里”的唯一权威来源。

```python
@dataclass(frozen=True)
class RunState:
    schema_version: int
    run_id: str
    session_id: str

    status: RunStatus
    phase: RunPhase
    stop_reason: str | None

    input_message_id: str
    latest_message_id: str | None

    core_state: dict[str, object]
    checkpoint: RunCheckpoint | None

    result_ref: str | None
    workspace_effects: WorkspaceEffectsSnapshot

    resume_count: int
    created_at: str
    updated_at: str
    started_at: str | None
    ended_at: str | None
    revision: int
```

`core_state` 由 Core 定义，Sessions 透明保存。Planning 状态属于 Core State，不再使用 Sessions 独立 Plan 文件。

### Run Status

```python
RunStatus = Literal[
    "created",
    "running",
    "waiting",
    "completed",
    "failed",
    "cancelled",
]
```

| 状态 | 含义 |
|---|---|
| `created` | Run 已持久化，Core 尚未开始 |
| `running` | Run 正在执行，或进程退出前处于普通执行边界 |
| `waiting` | Run 主动等待审批、用户输入或计划确认 |
| `completed` | Run 成功结束 |
| `failed` | Run 不可继续地失败 |
| `cancelled` | 用户或 Runtime 明确终止 Run |

进程退出或崩溃不能将 Run 写成 `cancelled`。恢复时读取磁盘中最近一次稳定状态。

### Run Phase

```python
RunPhase = Literal[
    "received",
    "model",
    "tools",
    "finalizing",
    "finished",
]
```

`status` 表示生命周期，`phase` 表示当前稳定执行阶段。

### Message Record

```python
@dataclass(frozen=True)
class MessageRecord:
    schema_version: int
    message_id: str
    session_id: str
    run_id: str | None
    parent_id: str | None
    created_at: str
    message: Message
```

Message 使用调用前生成的稳定 ID。相同 ID、相同内容重复写入视为幂等；相同 ID、不同内容视为冲突。

## Checkpoint

### 定位

Checkpoint 是 Run State 内部的可恢复状态：

> 它描述一个非终态 Run 从哪个已提交位置继续，以及继续前必须处理哪些外部条件。

所有非终态 Run 必须具有 Checkpoint，终态 Run 不允许保留活动 Checkpoint。

### 数据结构

```python
@dataclass(frozen=True)
class RunCheckpoint:
    schema_version: int
    checkpoint_id: str

    resume_point: ResumePoint
    message_cursor: MessageCursor

    waiting: WaitingState | None
    components: tuple[ComponentCheckpoint, ...]
    workspace: WorkspaceCheckpoint | None

    created_at: str
```

Checkpoint 嵌入 Run State，因此不重复保存 `run_id`、`status`、`phase`、`core_state` 和 revision。

### Resume Point

```python
ResumePoint = Literal[
    "before_model",
    "after_model",
    "before_tools",
    "after_tools",
    "before_finalization",
]
```

| Phase | 合法 Resume Point |
|---|---|
| `received` | `before_model` |
| `model` | `before_model`, `after_model` |
| `tools` | `before_tools`, `after_tools` |
| `finalizing` | `before_finalization` |
| `finished` | 无 |

### Message Cursor

```python
@dataclass(frozen=True)
class MessageCursor:
    leaf_message_id: str
```

第一版只保存单一 leaf ID，不重复保存 message count、最后一条用户消息、transcript hash 等派生信息。

`message_cursor + resume_point + core_state` 共同决定恢复后是否需要重新调用模型或工具。

### Waiting State

```python
WaitingKind = Literal[
    "tool_approval",
    "user_input",
    "plan_confirmation",
]
```

```python
@dataclass(frozen=True)
class WaitingState:
    kind: WaitingKind
    request_id: str
    payload: dict[str, object]
```

规则：

- `Run.status == "waiting"` 时 `waiting` 必须存在。
- 其他 Run 状态下 `waiting` 必须为 `None`。
- `request_id` 用于拒绝旧审批和重复输入。
- Sessions 不解释 payload 的业务含义。

### Component Checkpoint

```python
CheckpointOwner = Literal["tools", "context"]
```

```python
@dataclass(frozen=True)
class ComponentCheckpoint:
    owner: CheckpointOwner
    schema_version: int
    state: dict[str, object]
```

第一版只允许 Tools 和 Context 提交组件恢复状态：

- Tools 保存待执行调用和审批恢复所需信息。
- Context 保存压缩游标和有长度限制的 compact summary。

Memory 不参与当前 Run 恢复。Plan 已存在于 `core_state`。Observability 不参与恢复。

Context 第一版不建立 `.codepilot/context/` 持久化目录。

```json
{
  "owner": "context",
  "schema_version": 1,
  "state": {
    "compacted_until_message_id": "msg_123",
    "compact_summary": "此前已经完成仓库分析和目标文件定位。"
  }
}
```

compact summary 必须限制最大长度，并只保留最新一份。

### Workspace Checkpoint

```python
@dataclass(frozen=True)
class WorkspaceCheckpoint:
    root: str
    git_head: str | None
    dirty_paths: tuple[str, ...]
    tracked_path_hashes: dict[str, str]
```

第一版不计算全仓库 Hash，只记录当前 Run 已读取、修改或待恢复工具涉及的关键路径。

工作区变化只是恢复条件。Sessions 检测变化，但不决定是否继续。

## 状态不变量

### Session 不变量

- 一个 Session 同时只允许一个非终态 Run。
- 存在非终态 Run 时，`current_run_id` 必须指向它。
- Run 进入终态后，`current_run_id` 必须清空，`last_run_id` 指向该 Run。
- 有活动 Run 时不能切换消息 leaf。
- Fork 不复制活动 Run 和 Checkpoint。

### Run 不变量

- `run_id` 和 `session_id` 创建后不可修改。
- 所有非终态 Run 必须存在 Checkpoint。
- `waiting` Run 必须存在 `WaitingState`。
- 终态 Run 必须满足 `phase == "finished"`、`checkpoint is None`、`ended_at` 非空。
- 终态 Run 不允许恢复到非终态。
- `latest_message_id` 和 Checkpoint cursor 必须属于同一 Session。
- `phase` 与 `resume_point` 必须兼容。
- 所有更新必须匹配 `expected_revision`。

### 合法状态转换

| 当前状态 | 允许转换 |
|---|---|
| `created` | `running`, `failed`, `cancelled` |
| `running` | `running`, `waiting`, `completed`, `failed`, `cancelled` |
| `waiting` | `running`, `failed`, `cancelled` |
| `completed` | 无 |
| `failed` | 无 |
| `cancelled` | 无 |

`running -> running` 表示提交了新的稳定 Checkpoint。

## 稳定提交边界

Core 只能在以下稳定业务边界提交权威状态：

```text
before_model
after_model
before_tools
after_tools
waiting_tool_approval
waiting_user_input
waiting_plan_confirmation
before_finalization
```

### 模型调用

模型调用前提交：

```text
phase = model
resume_point = before_model
```

完整 AssistantMessage 保存后提交：

```text
phase = model
resume_point = after_model
message_cursor = assistant message ID
```

流式半成品不进入 Messages，也不进入 Checkpoint。模型调用中崩溃时，从 `before_model` 重新请求模型。

### 工具调用

工具执行前提交：

```text
phase = tools
resume_point = before_tools
components.tools = Tools 提供的恢复状态
```

完整 ToolResultMessage 保存后提交：

```text
phase = tools
resume_point = after_tools
message_cursor = 最后一条 ToolResultMessage
```

Sessions 不判断工具副作用是否已经发生。恢复时 Tools 解释自己的组件状态，并决定可以重试、需要对账还是需要用户确认。

### 终态提交

终态前提交：

```text
phase = finalizing
resume_point = before_finalization
```

最终结果保存后：

```text
status = completed / failed / cancelled
phase = finished
checkpoint = None
```

## 恢复协议

### 职责分工

```text
Sessions -> 加载和验证权威状态，返回 RecoveryResult
Runtime  -> 协调恢复流程并决定调用哪个执行入口
Core     -> 校验并恢复 core_state
Tools    -> 判断工具恢复状态是否可以继续
Context  -> 重新投影上下文并处理工作区变化
```

Sessions 不直接调用 `run_agent_loop()` 或 `resume_agent_loop()`。

### Recovery Request

```python
@dataclass(frozen=True)
class RecoveryRequest:
    session_id: str
    run_id: str | None = None
    expected_checkpoint_id: str | None = None
    expected_waiting_kind: WaitingKind | None = None
```

未指定 `run_id` 时读取 `SessionState.current_run_id`。

### Recovery Result

```python
RecoveryStatus = Literal[
    "ready",
    "needs_validation",
    "blocked",
    "not_found",
]
```

```python
@dataclass(frozen=True)
class RecoveryResult:
    status: RecoveryStatus
    bundle: RecoveryBundle | None
    issues: tuple[RecoveryIssue, ...]
```

### Recovery Bundle

```python
@dataclass(frozen=True)
class RecoveryBundle:
    session: SessionState
    run: RunState
    messages: tuple[MessageRecord, ...]
    workspace_status: WorkspaceRecoveryState
```

Recovery Bundle 只包含可序列化状态，不包含 Port、函数、Task 或 Runtime 对象。

### Sessions 内部校验

恢复时按固定顺序校验：

1. Session 是否存在。
2. workspace root 是否匹配。
3. 目标 Run 是否存在并属于该 Session。
4. Run 是否为非终态。
5. Checkpoint 是否存在。
6. expected checkpoint ID 和 waiting kind 是否匹配。
7. message cursor 是否存在且 parent chain 完整无环。
8. Run status、phase、resume point 和 waiting 是否构成合法组合。
9. Component schema 是否受支持。
10. 工作区关键路径是否变化。

工作区变化只返回状态，不直接把 Run 标记为失败。

### 恢复决策

Runtime 获得 Recovery Bundle 后：

```text
Tools.inspect_recovery()
Context 重新投影并刷新证据
Core.validate_resume()
```

组件判断可以归一为：

```text
ready
refresh_required
user_confirmation_required
cannot_resume
```

恢复失败不能静默重新执行原始请求。重新开始必须创建新 Run。

### Waiting 恢复

主动等待的 Run 由 Runtime 在组件校验后调用 `resume_run()`：

```text
waiting -> running
校验 checkpoint_id 和 request_id
清除 waiting
增加 resume_count
保留原 Checkpoint，直到新的稳定边界提交
```

进程崩溃后磁盘状态仍为 `running` 时，不需要先修改为 waiting。Runtime 校验后直接从最近 Checkpoint 继续。

## 持久化布局

```text
.codepilot/
├── sessions/<session_id>/
│   ├── session.json
│   ├── messages.jsonl
│   └── events.jsonl
├── runs/<run_id>/
│   ├── run.json
│   ├── events.jsonl
│   └── artifacts/
│       ├── result.json
│       ├── tool_outputs/
│       └── rollback/
└── memory/
    └── memories.jsonl
```

第一版不创建 `.codepilot/context/`。

### 文件写入者

| 文件 | 唯一写入者 |
|---|---|
| `session.json` | Sessions |
| `run.json` | Sessions |
| `messages.jsonl` | Sessions |
| Session/Run events | Sessions Event Sink |
| `memories.jsonl` | Memory Repository |
| Run artifacts | 对应领域服务通过 Artifact Store |

Context 不直接写 Sessions 权威文件。

### 原子写入

`session.json` 和 `run.json` 使用：

```text
同目录临时文件
-> flush
-> fsync
-> os.replace
```

每次更新验证 expected revision，新 revision 等于旧 revision 加一。

### Messages JSONL

- 每条消息使用单行 JSON。
- 写入后 flush 和 fsync。
- 允许忽略或截断唯一一条损坏的末尾记录。
- 中间记录损坏必须报错。
- 未被 Run Checkpoint cursor 引用的尾部消息视为未提交消息。

### 跨文件提交顺序

统一使用：

```text
消息或 Artifact 内容
-> Run State
-> Session State 导航引用
-> Event
```

Run State 是主要提交点。

崩溃处理：

- 消息写入、Run 未更新：忽略未提交消息。
- Run 更新、Session leaf 未更新：根据唯一活动 Run cursor 修复 Session leaf。
- Session 更新、Event 未写入：正常恢复，允许审计缺口。
- Session 指向不存在的 Run：阻塞恢复。
- 多个非终态 Run：阻塞恢复，不自动选择。

## Sessions Service

第一版提供一个聚合门面，封装多文件写入顺序。

```python
class SessionStateService:
    def create_session(...): ...
    def begin_run(...): ...
    def commit_boundary(...): ...
    def resume_run(...): ...
    def finish_run(...): ...
    def inspect_recovery(...): ...
```

### create_session

负责创建 Session State 和基础目录，不负责构建模型、Prompt、Tools 或 Memory。

### begin_run

固定执行：

```text
校验 Session 没有非终态 Run
-> 保存 UserMessage
-> 创建 created/received Run 和 before_model Checkpoint
-> 更新 Session current_run_id、last_run_id 和 leaf
-> 追加 run_created Event
```

### commit_boundary

Runtime 通过该操作提交模型、工具和等待边界：

```python
@dataclass(frozen=True)
class CommitBoundaryRequest:
    session_id: str
    run_id: str
    status: Literal["running", "waiting"]
    phase: RunPhase
    resume_point: ResumePoint
    new_messages: tuple[Message, ...]
    core_state: dict[str, object]
    waiting: WaitingState | None = None
    components: tuple[ComponentCheckpoint, ...] = ()
    workspace: WorkspaceCheckpoint | None = None
```

Sessions 负责分配消息 ID、更新消息链、构造 Checkpoint、提交 Run State、更新 Session leaf 和追加 Event。

### resume_run

只处理 `waiting -> running` 状态转换，不调用 Core 或 Tools。

第一版只保存 `resume_count` 和审计 Event，不保存完整 resume history。

### finish_run

固定执行：

```text
保存最终消息
-> 保存 AgentRunResult Artifact
-> 提交终态 Run
-> 清理 Session current_run_id
-> 追加终态 Event
```

### inspect_recovery

只加载和验证恢复状态，不执行恢复。

## Core 与 Runtime 接入

### RunStatePort

Core 不直接依赖 Sessions。Core contracts 定义：

```python
class RunStatePort(Protocol):
    async def commit(self, boundary: CoreRunBoundary) -> None:
        ...
```

`AgentLoopPorts` 同时包含：

```text
state  -> 权威边界提交
events -> 流式进度和非权威审计
```

Core 不能通过普通 Event 隐式更新 Checkpoint。

Checkpoint 提交失败时，Core 不能跨越边界继续调用 Model 或 Tools。

### RuntimeSessionStateAdapter

Runtime 提供 Adapter：

```text
CoreRunBoundary
-> CommitBoundaryRequest
-> SessionStateService.commit_boundary()
```

Adapter 持有最新 Session/Run revision，避免 revision 在 Gateway 中传播。

### RuntimeSessionCoordinator

当前 `sessions.SessionRuntime` 的协调职责迁入 Runtime：

```text
runtime/session_coordinator.py
```

Coordinator 只保留三段流程：

```text
开始 Run
-> 组装并调用 Core，运行期间由 Ports 工作
-> 结束 Run 和执行最佳努力后处理
```

`SessionController` 也迁入 Runtime，继续作为 `RuntimeGateway` 的应用门面。

## Context 边界

Context 迁为独立源码模块，但第一版不建立独立持久化目录。

```text
src/codepilot/sessions/context/
├── contracts.py
├── service.py
└── runtime_port.py
```

Context 通过只读 `SessionContextReader` 获取：

- 消息链。
- 当前 Run 状态。
- `core_state`。
- Checkpoint 中的 Context component。
- workspace root。

Context 负责：

- Repository snapshot。
- active files。
- evidence freshness。
- Memory recall 集成。
- token 压力。
- 消息选择与压缩。
- ContextReport。

ContextReport 第一版仅作为 `context_projected` Event 保存，不建立独立报告文件。

active files、freshness、repository snapshot 在进程恢复后重新构建，不作为权威状态持久化。

## Memory 边界

Memory 迁为独立模块并拥有 workspace 级长期持久化：

```text
src/codepilot/sessions/memory/
├── contracts.py
├── service.py
└── repository.py
```

第一版可以先保留较少文件，不立即拆分 admission、retrieval 和 conflict。

Memory 负责：

- 记忆准入。
- 召回。
- 冲突检测。
- 状态演化。
- Run 后经验提炼。
- `memories.jsonl`。

Memory 通过 `SessionEvidenceReader` 验证来源引用，不读取 Sessions 文件布局，也不直接写 Session Event。

Context 通过 `MemoryRecallPort` 使用 Memory。

## Planning、Rollback、Commands 与 Subagents

### Planning

- Plan 权威状态属于 Core State。
- 删除 Sessions 独立 `plan_state.json` 和 `active_plan_id`。
- Plan 确认通过 waiting Checkpoint 表达。
- Planning Event 仅用于 UI 和审计。

### Rollback

- Git baseline、rollback plan 和 result 属于 Rollback 服务。
- Sessions 只在 Run State 中保存 baseline 引用和 affected paths 摘要。
- 回滚结果保存为 Run Artifact 和 Event。
- 回滚后不修改已经终止的 Run State。
- Workspace Checkpoint 与 rollback baseline 是两个不同概念。

### Commands

- 命令解析和文本输出属于 Runtime/Interfaces。
- Sessions 只提供结构化 fork、switch leaf、inspect recovery 等能力。
- `/plan`、`/memory`、`/context`、`/rollback` 分别调用对应领域服务。

### Fork

Fork：

- 复制指定 leaf 的消息链。
- 创建新 Session 并设置 parent 引用。
- 不复制活动 Run、Checkpoint 和 Context 内存状态。
- Memory 为 workspace 级共享状态，不复制。

### Subagents

- Subagent orchestration 属于 Runtime。
- 第一版临时 Subagent 使用 Runtime 内存 Registry。
- 第一版暂不支持 Subagent 跨进程恢复。
- 后续需要持久化时复用普通 Child Session/Run，不建立第三套 Subagent 状态模型。

## 当前实现迁移

| 当前文件 | 目标处理 |
|---|---|
| `sessions/contracts.py` | 重建为 Session/Run/Checkpoint/Recovery DTO |
| `sessions/store.py` | 重构为文件系统 Repository 和 State Service |
| `sessions/serde.py` | 保留消息序列化职责 |
| `sessions/conversation.py` | 持久消息归 Sessions，实时流式状态迁 Runtime |
| `sessions/runtime.py` | 协调逻辑迁 RuntimeSessionCoordinator |
| `sessions/controller.py` | 迁入 Runtime |
| `sessions/context.py` | 迁入 `sessions/context/`，仅共享源码命名空间，保持 Context 独立职责 |
| `sessions/memory.py` | 迁入 `sessions/memory/`，仅共享源码命名空间，保持 Memory 独立职责 |
| `sessions/commands.py` | 命令路由迁 Runtime，结构化 Session 操作保留 |
| `sessions/plan_state.py` | Plan 语义迁 Core Planning |
| `sessions/rollback.py` | Git 策略迁 Rollback 模块 |
| `sessions/subagents.py` | 迁 Runtime，第一版使用内存 Registry |
| `sessions/workspace_state.py` | 中立能力收敛到 `sessions/workspace.py`，恢复判断留在 Sessions Recovery |
| Sessions、Context、Rollback 中重复的 Git helper | 低级查询收敛到 `sessions/workspace.py`，领域决策留在各自模块 |
| 各模块重复的路径规范化、Hash 和变更描述 helper | 收敛到 `sessions/workspace.py` |
| `tools/sandbox.py` | 保持在 Tools，仅复用 `sessions/workspace.py` 的路径边界能力 |

## 重构后的源码布局

本节描述重构完成后的目标源码布局，用于指导文件迁移和代码审查。

目录树表达的是职责边界，不要求第一阶段立即创建所有文件。第一版可以合并实现较短、变化一致的文件，但不能跨越本节定义的模块边界。

```text
src/codepilot/
├── protocols/
├── core/
│   ├── contracts.py
│   ├── runner.py
│   ├── state.py
│   └── planning/
│       ├── contracts.py
│       ├── state.py
│       └── service.py
│   └── ...
├── sessions/
│   ├── __init__.py
│   ├── contracts.py
│   ├── service.py
│   ├── repository.py
│   ├── filesystem.py
│   ├── serde.py
│   ├── recovery.py
│   ├── workspace.py
│   ├── context/
│   │   ├── __init__.py
│   │   ├── contracts.py
│   │   ├── service.py
│   │   └── runtime_port.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── contracts.py
│   │   ├── service.py
│   │   └── repository.py
│   └── rollback/
│       ├── __init__.py
│       ├── contracts.py
│       └── service.py
├── tools/
│   ├── sandbox.py
│   ├── security.py
│   └── ...
├── runtime/
│   ├── gateway.py
│   ├── builder.py
│   ├── sessions.py
│   ├── session_controller.py
│   ├── session_coordinator.py
│   ├── session_state_adapter.py
│   ├── live_conversation.py
│   ├── commands.py
│   └── subagents.py
│   └── ...
└── observability/
```

目标依赖方向补充 Workspace 后为：

```text
protocols -> sessions.workspace -> llm/tools -> core -> sessions state/observability
                              -> sessions.context/memory/rollback -> extensions -> runtime -> interfaces
```

这里将 Context、Memory、Rollback 和 Workspace 放入 `sessions/`，仅用于源码目录聚合和可读性，不表示这些领域由 Session State 管理。它们必须保持独立职责，且不得通过 `sessions/__init__.py` 混合导出为同一领域 API。

`sessions/workspace.py` 是中立基础能力模块。第一版不建立 `workspace/` 包，避免路径、状态、变更和 Git 查询被拆成多个需要协同演进的文件：

- 可以被 Tools、Context、Sessions Recovery 和 Rollback 使用。
- 不依赖 Core、Sessions、Context、Rollback、Runtime 或 Interfaces。
- 不拥有工具权限、安全审批、Context freshness 或 Rollback 策略。
- 只提供无状态或显式输入输出的工作区查询与数据转换，不成为新的事实存储。

`tools/sandbox.py` 继续存在，负责工具执行时的路径安全和权限策略；`sessions/rollback/` 继续保持独立领域职责，负责回滚判断与流程。`sessions/workspace.py` 只提供二者可复用的中立机制，因此不会模糊 Tools 与 Rollback 的领域边界。后续只有当该文件的规模和变化轴已经明确分化时，才考虑拆为包。

### `sessions/__init__.py`

职责：

- 导出 Sessions 的稳定公共协议。
- 导出 Session State、Run State、Checkpoint 和 Recovery DTO。
- 导出 `SessionStateService` 的构造入口。

不应包含：

- 业务实现。
- Runtime 装配逻辑。
- Context、Memory、Planning、Tools 或 Rollback 的重新导出。
- 为兼容旧 API 保留的大量别名。

### `sessions/contracts.py`

职责：

- 定义 `SessionState`、`RunState` 和 `MessageRecord`。
- 定义 `RunCheckpoint`、`WaitingState` 和 Component Checkpoint。
- 定义 Recovery Request、Result、Bundle 和 Issue。
- 定义 Session/Run 状态枚举及通用错误类型。
- 定义 Sessions 对外的 Reader、Writer 或 Service Protocol。

不应包含：

- `SessionRunIntent`、`PreparedAgentRun` 等 Runtime 协调对象。
- Context Report、Memory Record 或 Plan State 的领域模型。
- 工具参数校验和审批协议。
- JSON、文件路径或具体持久化实现。
- 根据自然语言或 Event 推导状态的逻辑。

### `sessions/service.py`

职责：

- 实现 `create_session()`。
- 实现 `begin_run()`、`commit_boundary()`、`resume_run()` 和 `finish_run()`。
- 实现 fork、switch leaf 等结构化 Session 操作。
- 统一编排消息、Run、Session 和 Event 的写入顺序。
- 校验状态转换、revision 和 Session/Run 关联不变量。

不应包含：

- 调用 ModelPort 或 ToolPort。
- 调用 `run_agent_loop()`。
- 构建 prompt 或选择 Context。
- 调用 Memory admission、recall 或 finalize。
- 解释 Plan 状态。
- 执行 Git 回滚。
- 解析斜杠命令或生成用户展示文本。

### `sessions/repository.py`

职责：

- 定义并实现 Session State 和 Run State 的数据访问边界。
- 提供类型化 create、load 和 compare-and-swap update。
- 提供 Message Record 的幂等追加和消息链读取。
- 提供 Event 的追加和只读查询。
- 隐藏实际文件布局。

不应包含：

- Agent 生命周期协调。
- 状态转换业务流程。
- 任意字典式 `update_meta()`。
- Event 到 Run State 的反向推导。
- Repository snapshot、Git 状态分析或 Context freshness。
- Memory 文件读写。

第一版中 `repository.py` 可以同时实现 Session、Run、Message 和 Event 的文件访问，不要求拆成多个 Store 类。

### `sessions/filesystem.py`

职责：

- 定义 Sessions 的磁盘路径布局。
- 提供原子 JSON 写入。
- 提供安全 JSONL 追加和读取。
- 提供 flush、fsync、`os.replace()` 和临时文件清理。
- 检测损坏的末尾 JSONL 记录和中间记录。

不应包含：

- Session 或 Run 状态转换。
- Message、Checkpoint 的业务校验。
- Git 命令。
- Context、Memory 或 Rollback artifact 的领域语义。
- Runtime 锁和 asyncio Task 管理。

### `sessions/serde.py`

职责：

- 在 Protocol Message 与持久化字典之间转换。
- 序列化和反序列化 Sessions DTO。
- 严格校验 schema version 和未知字段。

不应包含：

- 文件读写。
- 状态迁移和历史格式兼容。
- 模型 Provider 消息转换。
- prompt 格式化。
- 状态恢复决策。

### `sessions/recovery.py`

职责：

- 实现 `inspect_recovery()`。
- 校验 Session、Run、Checkpoint 和消息链的一致性。
- 比较 Workspace Checkpoint 与当前工作区状态。
- 构造 `RecoveryResult` 和 `RecoveryBundle`。
- 执行有限、确定性的 Session leaf 修复。

不应包含：

- 调用 Core、Tools 或 Context 执行恢复。
- 自动重新开始失败任务。
- 判断工具副作用是否已经发生。
- 重新生成 Context。
- 通过 Event replay 重建 Run State。
- 自动选择多个非终态 Run 中的一个。

### `context/contracts.py`

职责：

- 定义 `ContextPreparationRequest` 和 `PreparedAgentContext`。
- 定义 `ContextSessionView` 和 `SessionContextReader`。
- 定义 Context Checkpoint payload。
- 定义 `MemoryRecallPort` 等 Context 所需窄接口。

不应包含：

- Sessions State 写接口。
- Memory Repository 实现。
- Runtime Gateway DTO。
- 文件路径和持久化布局。

### `context/service.py`

职责：

- 实现上下文投影、选择和压缩。
- 维护进程内 active files、evidence freshness 和 repository snapshot。
- 调用 Memory Recall Port。
- 生成 Context Checkpoint payload。
- 生成精简 `context_projected` Event payload。

不应包含：

- 直接写 `session.json`、`run.json` 或 Messages。
- 独立 Context 状态文件。
- Memory 准入和状态演化。
- Plan 状态流转。
- 工具执行。
- Run status 或 Checkpoint 的直接更新。

### `context/runtime_port.py`

职责：

- 实现 Core 所需的 Context Port。
- 将 Core 的上下文请求转换为 `ContextService.prepare()` 调用。
- 从 Sessions Reader 加载 Context View。
- 将 Prepared Context 返回 Core。
- 将 Context checkpoint payload 暴露给 Runtime State Adapter。

不应包含：

- Context 选择算法本身。
- Session 文件读写。
- Run 生命周期协调。
- Memory 文件访问。

### `memory/contracts.py`

职责：

- 定义 Memory Record、Source Ref、Admission、Recall 和 Finalize DTO。
- 定义 `MemoryRecallPort` 和 `SessionEvidenceReader`。
- 定义 Memory schema 和状态枚举。

不应包含：

- Session、Run 或 Checkpoint DTO。
- Context 投影结构。
- 文件读写实现。
- Runtime 命令解析。

### `memory/service.py`

职责：

- 实现记忆准入。
- 实现记忆召回和冲突处理。
- 实现 Memory Record 状态演化。
- 实现 Run 结束后的经验提炼。
- 通过 Evidence Reader 验证来源引用。

不应包含：

- 直接写 Session Event。
- 修改 Session State 或 Run State。
- 选择最终 prompt 内容。
- 执行工具。
- 读取 Sessions 磁盘布局。

### `memory/repository.py`

职责：

- 独立管理 `.codepilot/memory/memories.jsonl`。
- 提供 Memory Record 的追加、查询和状态更新。
- 保证 Memory schema 和写入一致性。

不应包含：

- Session 文件读写。
- Memory 准入、召回或冲突策略。
- Context Report。
- Runtime 生命周期逻辑。

### `core/contracts.py`

重构新增职责：

- 定义 `RunStatePort`。
- 定义 `CoreRunBoundary` 和 Core Waiting Request。
- 保持 Core 对 Sessions 的零依赖。

不应包含：

- `SessionStateService`。
- `RunCheckpoint` 的文件持久化结构。
- Runtime Controller 或 Gateway DTO。
- Context、Memory 的具体实现。

### `core/runner.py`

重构后职责：

- 在稳定边界调用 `RunStatePort.commit()`。
- 保证边界提交成功后才调用 Model 或 Tools。
- 从 `core_state` 恢复循环状态。
- 产生 `AgentLoopOutcome`。

不应包含：

- 直接读写 Session 文件。
- 根据 Event 保存 Checkpoint。
- Memory 写入或召回。
- Runtime Session 注册和并发控制。

### `core/planning/`

职责：

- 定义 Plan State 和状态流转。
- 校验 propose、approve、reject、progress、complete 和 archive。
- 将 Plan 保存在 Core State 中。

不应包含：

- 独立 `plan_state.json`。
- Session metadata 更新。
- 用户命令文本解析。
- Checkpoint 持久化。

第一版可以继续使用现有 `core/plan.py`，待 Sessions 主链路稳定后再拆为目录。

### `runtime/session_coordinator.py`

职责：

- 协调 Run 的开始、Core 执行、恢复和结束。
- 调用 Sessions、Memory、Planning、Rollback 和 Hooks。
- 组装 ModelPort、ToolPort、ContextPort 和 RunStatePort。
- 区分强一致终态提交和最佳努力后处理。

不应包含：

- JSON 文件读写。
- Context 投影算法。
- Memory 准入、召回和冲突算法。
- Plan 状态机实现。
- Tool 权限和执行策略。
- RuntimeGateway 的 Session Registry。

### `runtime/session_state_adapter.py`

职责：

- 实现 Core `RunStatePort`。
- 将 `CoreRunBoundary` 映射为 `CommitBoundaryRequest`。
- 持有最新 Session/Run revision。
- 将 Tools、Context 的组件恢复状态加入稳定边界提交。

不应包含：

- Core loop 推进逻辑。
- Checkpoint 业务校验的重复实现。
- 文件写入。
- Context 或 Tools 状态内容的解释。

### `runtime/session_controller.py`

职责：

- 作为 RuntimeGateway 的 Session 应用门面。
- 接收 Run、Resume、Continuation 和 Command 意图。
- 调用 RuntimeSessionCoordinator。
- 返回 Runtime 可消费的结构化记录和 View。

不应包含：

- Sessions 文件布局。
- Core loop 内部逻辑。
- Context 和 Memory 领域实现。
- CLI 或钉钉专属渲染。

### `runtime/live_conversation.py`

职责：

- 保存进程内流式 AssistantMessage。
- 保存进程内 pending tool call 展示状态。
- 管理 Runtime Event listeners 和 steering message queue。
- 在 Session 关闭时释放监听器。

不应包含：

- 权威消息历史。
- Session leaf。
- Run Checkpoint。
- 跨进程恢复状态。
- JSON 持久化。

### `runtime/commands.py`

职责：

- 解析斜杠命令。
- 将命令分派到 Sessions、Planning、Memory、Context、Rollback 和 Tools。
- 构造结构化 Command Result。

不应包含：

- Session 或 Run 文件读写。
- Memory、Planning、Rollback 的领域实现。
- CLI 或钉钉专属输出格式。

第一版可以保留单文件；命令数量和依赖明显增长后再拆为 `runtime/commands/` 目录。

### `runtime/subagents.py`

职责：

- 管理当前进程内 Subagent Registry。
- 创建、调度、取消和汇总 Subagent。
- 后续需要持久化时创建普通 Child Session/Run。

不应包含：

- 第三套 Subagent 持久化状态模型。
- 直接写 Session 文件。
- 跨进程恢复逻辑，第一版不实现。

### `sessions/rollback/`

职责：

- 捕获 Git baseline。
- 生成 rollback preview。
- 校验文件是否在 Run 结束后被再次修改。
- 执行允许的回滚动作。
- 保存 rollback artifact。
- 通过 `sessions/workspace.py` 获取中立工作区能力。

不应包含：

- 修改终态 Run State。
- Session 或 Checkpoint 生命周期管理。
- Runtime 命令解析和文本渲染。
- 将 Workspace Checkpoint 与 rollback baseline 混为同一模型。
- 自己实现重复的 Git 命令适配和路径规范化。

### `sessions/workspace.py`

职责：

- 提供路径规范化。
- 将相对路径解析到指定工作区。
- 判断解析后的真实路径是否位于工作区内。
- 提供 symlink 解析后的路径边界检查。
- 提供跨平台路径比较和相对路径表达。
- 提供文件存在性、类型、大小和 Hash 查询。
- 提供关键路径状态快照。
- 比较保存的路径状态与当前文件状态。
- 为 Context、Recovery、Tools 和 Rollback 提供中立文件状态能力。
- 定义中立的文件变更、Patch 和 Diff 数据模型。
- 标准化 affected paths。
- 比较工作区变更前后的文件状态。
- 提供 Git diff 或文件 diff 的结构化结果。
- 为 ToolResult、Rollback 和 Evaluation 提供可复用变更描述。
- 封装 `git status`、`git diff`、`git rev-parse` 等低级命令。
- 将 Git 输出解析为结构化结果。
- 提供 HEAD、branch、dirty paths 和 tracked 状态查询。
- 统一 subprocess 编码、错误和路径处理。

不应包含：

- 判断 Run 是否允许恢复。
- 判断某次回滚是否安全。
- Context freshness 策略。
- Tool 权限和审批策略。
- 文件读写工具的权限判断。
- 用户审批、permission mode、Shell 风险分类或特定 Tool 的 allow/block 规则。
- 执行具体 Tool handler、根据变更判断任务是否完成或自动执行 Rollback。
- Runtime 命令文本渲染。
- Session Event 或 Run State 写入。

### `tools/sandbox.py`

职责：

- 使用 `sessions/workspace.py` 对工具输入路径执行边界检查。
- 根据 Tool Policy 判断路径是否只读、可写或禁止访问。
- 保护 `.codepilot/` 等内部状态路径。
- 将路径边界能力接入工具执行安全流水线。

不应包含：

- 重复实现路径规范化、realpath 和相对路径算法。
- Git 状态、Diff 和 Rollback 策略。
- Session、Run 或 Checkpoint 持久化。
- Context 状态选择。

Workspace 与 Tools 的边界如下：

| 能力 | Workspace | Tools |
|---|---:|---:|
| 路径解析和规范化 | 是 | 使用 |
| 判断真实路径是否位于工作区 | 是 | 使用 |
| 文件 Hash 和状态快照 | 是 | 使用 |
| Git status、diff 和 HEAD | 是 | 使用 |
| 工具是否允许读取或修改 | 否 | 是 |
| permission mode | 否 | 是 |
| 用户审批 | 否 | 是 |
| Shell 风险分类 | 否 | 是 |
| ToolResult 防护 | 否 | 是 |

### 文件合并规则

第一版允许以下合并，以控制文件数量：

- `sessions/repository.py` 可以暂时包含文件系统 Repository 实现。
- `sessions/recovery.py` 可以暂时由 `SessionStateService` 内部实现。
- `context/contracts.py` 和 `context/service.py` 可以先合并。
- `memory/contracts.py` 和 `memory/service.py` 可以先合并。
- `rollback/contracts.py` 和 `rollback/service.py` 可以先合并。
- `sessions/workspace.py` 第一版只实现已经存在复用需求的路径边界、文件 Hash、affected paths/diff DTO 和低级 Git 查询；不为可能出现的需求预建抽象。
- `runtime/session_controller.py` 和 `runtime/session_coordinator.py` 可以在迁移早期共存于一个文件。

不允许以下合并：

- Context 或 Memory 重新放入 `sessions/`。
- Runtime 协调逻辑重新放入 `sessions/service.py`。
- Sessions 文件读写进入 Core。
- Plan 状态机进入 Sessions。
- Tool 执行或审批逻辑进入 Checkpoint 实现。
- Tool Sandbox 和权限策略迁入 Workspace。
- Rollback 策略迁入 Workspace。
- Event 处理逻辑反向修改 Run State。

## 第一版实施顺序

### 阶段 1：建立新状态模型

- 定义 Session State、Run State、Run Checkpoint 和 Recovery DTO。
- 实现状态不变量和状态转换校验。
- 删除旧格式读取和兼容逻辑。

验收：领域模型单元测试覆盖所有合法和非法状态组合。

### 阶段 2：重写文件持久化

- 实现原子 JSON 写入。
- 实现 revision 校验。
- 实现 Message Record 幂等追加。
- 实现 SessionStateService 六个核心操作。
- 停止通过 Event 推导 Run State。

验收：崩溃窗口和跨文件修复规则有确定性测试。

### 阶段 3：Core 显式提交边界

- 在 Core contracts 增加 RunStatePort 和 CoreRunBoundary。
- 在 `before/after model`、`before/after tools`、waiting 和 finalizing 提交边界。
- 提交失败时停止继续调用 Model/Tools。

验收：模型和工具不会越过失败的状态提交边界。

### 阶段 4：迁移 Runtime 协调

- 创建 RuntimeSessionCoordinator 和 RuntimeSessionStateAdapter。
- 迁移 SessionController。
- 删除 Sessions 中的任务推进和生命周期协调。
- RuntimeGateway 保持公共行为一致。

验收：Prompt、审批恢复、计划确认、取消和终态提交走新主链路。

### 阶段 5：迁出 Context 和 Memory

- Context 位于 `sessions/context/`，保持独立领域职责且不建立独立状态目录。
- Context compact state 进入 Checkpoint component。
- ContextReport 仅写 Event。
- Memory 独立管理 `memories.jsonl`。
- Sessions 只提供 Context Reader 和 Evidence Reader。

验收：Sessions 不再 import ContextGovernor、MemoryWriter 或 MemoryRetriever。

### 阶段 6：迁出其余职责

- Planning 状态进入 Core State。
- 提取 `sessions/workspace.py` 中立能力，Tools Sandbox 保持原有安全职责。
- Rollback 策略迁出。
- Commands 迁 Runtime。
- Subagents 迁 Runtime 内存 Registry。
- 删除旧 Sessions API 和无效测试。

验收：Sessions 包只包含状态、消息、事件、恢复和文件持久化能力。

## 第一版不实现

- 历史 Session 文件兼容和迁移。
- Event Sourcing。
- 数据库事务。
- 多进程分布式锁。
- Checkpoint 历史版本。
- 通用 Component Registry。
- Context 独立持久化目录。
- 自动 orphan Run 修复。
- 完整 resume history。
- 全仓库文件 Hash。
- 任意 Shell 操作 exactly-once 保证。
- Subagent 跨进程恢复。

## 测试重点

### 状态模型

- waiting Run 没有 Checkpoint 被拒绝。
- waiting Run 没有 WaitingState 被拒绝。
- 终态 Run 保留 Checkpoint 被拒绝。
- phase 与 resume point 不兼容被拒绝。
- 终态 Run 不能回到 running。
- revision 冲突被拒绝。

### 持久化

- JSON 原子替换失败时保留旧状态。
- 重复 Message ID、相同内容保持幂等。
- 重复 Message ID、不同内容返回冲突。
- 损坏的末尾 JSONL 记录可以安全处理。
- 中间 JSONL 损坏阻塞读取。

### 崩溃窗口

- 消息写入、Run 未提交时忽略尾部消息。
- Run 提交、Session leaf 未更新时确定性修复。
- Event 缺失不影响恢复。
- Session 指向不存在 Run 时阻塞恢复。
- 多个非终态 Run 时阻塞恢复。

### 恢复

- `before_model` 恢复会重新调用模型。
- `after_model` 恢复不会重新调用模型。
- `before_tools` 恢复先交给 Tools 判断。
- `after_tools` 恢复不会重新执行已提交工具结果。
- 旧 checkpoint ID 或 request ID 不能恢复新状态。
- 恢复失败不能静默创建新 Run。

### 架构边界

- Core 不 import Sessions。
- Sessions 不 import Context Service、Memory Service、Planning Service 或 Tool Runtime。
- Context 只能通过 SessionContextReader 读取会话状态。
- Memory 只能通过 SessionEvidenceReader 验证来源。
- Event 不能修改权威 Run State。

## 最终边界

Sessions v2 最终应能够用一句话描述：

> Sessions 保存 Session、Run、Messages 和 Run Checkpoint 的权威状态，提供一致的提交与恢复接口，并记录非权威审计事件；它不决定 Agent 如何推进，也不负责 Context、Memory、Planning、Tools 或 Rollback 的领域策略。
