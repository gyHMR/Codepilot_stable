# Sessions 状态与恢复设计

## 文档状态

本文描述 Sessions v2 当前有效的状态、提交和恢复边界。Sessions 是 Session、Run、Message 和 Checkpoint 的持久化权威，但不决定 Agent 下一步行为。

当前 Sessions v2 的旧 Core payload 可以在恢复时升级为 `CoreState` schema v2；新边界只写新 schema。不提供更早历史 Session API 和文件格式的长期兼容层。

## 职责边界

Sessions 负责：

- 保存 `SessionState`、`RunState`、`MessageRecord` 和 `RunCheckpoint`。
- 提供 begin、progress/waiting/terminal commit、resume 和 recovery inspect。
- 校验 revision、状态转换、消息链和 Checkpoint 不变量。
- 原子写入状态文件，幂等追加消息和审计事件。
- 保存各组件提供的不透明 checkpoint。

Sessions 不负责：

- 调用 `run_core()`、ModelPort 或 ToolPort。
- 解释 `CoreState`、Plan、Tool checkpoint、Context checkpoint 或 Memory 内容。
- 选择 Context、召回 Memory、执行 Tool 或决定审批。
- 管理 Runtime task、deadline、取消和资源释放。
- 通过 Event replay 重建权威状态。

## 权威关系

系统只有两层权威运行状态：

```text
CoreState          任务语义、Plan 和可观察运行事实
Sessions RunState  生命周期、持久化 revision 和恢复边界
```

Runtime Execution State 只存在于进程内。Events 只用于审计和界面，不是恢复权威。Messages 是对话内容权威，Run checkpoint 只记录已成功提交的稳定边界。

## SessionState

`SessionState` 至少保存：

```text
session_id
workspace_root
model
session_kind
parent_session_id / parent_run_id
current_mode
current_run_id
last_run_id
leaf_message_id
revision
created_at / updated_at
```

不变量：

- `current_run_id` 只能指向本 Session 的非终态 Run。
- `leaf_message_id` 必须属于本 Session 的有效消息链。
- 更新必须使用 expected revision。
- 一个 Session 同时只有一个活动 Run。

## RunState

Sessions 的 `RunState` 与 CoreState 是不同概念：

```text
RunState
├── run_id / session_id
├── status
├── phase
├── input_message_id / latest_message_id
├── core_state: dict
├── checkpoint?
├── request_id / request_digest
├── stop_reason / result_ref
├── last_commit_*
├── workspace_effects
├── resume_count
└── revision / timestamps
```

状态：

```text
created -> running -> waiting -> running
created/running/waiting -> completed | failed | cancelled
```

阶段：

```text
received -> model -> tools -> finalizing -> finished
```

终态 Run 必须满足：

- `phase == finished`。
- `checkpoint is None`。
- Session 不再以 `current_run_id` 指向该 Run。
- 最终结果只提交一次。

## MessageRecord

Message 使用追加式记录：

```text
message_id
session_id
run_id
parent_id
created_at
message
```

规则：

- 相同 message_id 与相同内容重复写入是幂等操作。
- 相同 message_id 与不同内容是冲突。
- Session leaf 只在对应 Run 边界成功后推进。
- 实时 token delta 不写 MessageRecord。

## RunCheckpoint

Checkpoint 描述最近一个可恢复的成功边界：

```text
RunCheckpoint
├── checkpoint_id
├── resume_point
├── message_cursor
├── waiting?
├── components[]
├── workspace?
└── created_at
```

有效 resume point：

```text
before_model
after_model
before_tools
after_tools
before_finalization
```

### WaitingState

WaitingState 只保存跨进程恢复所需的公共信息：

```text
kind
request_id
payload
```

Tool approval challenge 的完整权威记录属于 Tools。Sessions 保存 waiting 类型和 request_id；Runtime 通过 ToolPort 查询挑战并生成界面 Frame。

### ComponentCheckpoint

```text
ComponentCheckpoint
├── owner: tools | context | rollback | ...
├── schema_version
└── state
```

Sessions 只验证 owner、schema 和 JSON 可持久化性，不解释 `state`。当前主要组件：

- `tools`：pending attempt、审批或交互恢复状态。
- `context`：Context 压缩和选择所需的最小恢复状态。
- `rollback`：稳定 rollback baseline 引用。

### WorkspaceCheckpoint

Workspace checkpoint 用于判断恢复时工作区是否仍与提交边界一致。它不是 rollback baseline，也不授予 Tool 权限。

## 提交协议

唯一写入口是：

```text
SessionStateService.commit_run_boundary(CommitRunBoundaryRequest)
```

提交类型：

```text
progress
waiting
terminal
```

### Progress Commit

保存：

- `core_state`。
- 新消息和 message cursor。
- Core domain events 与 Runtime 排队的 durable events。
- Tool/Context/Rollback component checkpoint。
- workspace checkpoint。
- phase 和 resume point。

### Waiting Commit

在 Progress 内容之外保存 `WaitingState`，并将 Run 状态改为 waiting。只有提交成功后 Runtime 才能向界面报告暂停。

### Terminal Commit

原子完成：

- 写最终消息和结果引用。
- 设置 completed/failed/cancelled 与 stop reason。
- 清除 Checkpoint。
- 清除 `SessionState.current_run_id`。
- 更新 Session leaf 和 last_run_id。

Terminal commit 失败时不能发送成功 Frame。

### 幂等与并发

- `commit_id` 标识一次逻辑提交。
- 相同 commit_id 与相同 digest 返回已有 receipt。
- 相同 commit_id 与不同内容返回冲突。
- expected Run/Session revision 防止并发覆盖。
- 文件写入中断后可用同一 commit_id 重试，不重复消息、事件或结果。

## Core 与 Runtime 接入

Core 不依赖 Sessions。Core 只定义：

```python
class BoundaryPort(Protocol):
    def commit(self, boundary: CoreBoundary) -> None | Awaitable[None]: ...
```

Runtime 的 `RuntimeSessionStateAdapter` 实现该 Port：

```text
CoreBoundary
  -> 读取 Runtime 持有的 Tool/Context/Rollback checkpoint
  -> CommitRunBoundaryRequest
  -> SessionStateService.commit_run_boundary()
```

映射关系：

| CoreBoundary | Sessions commit | phase | resume point |
|---|---|---|---|
| before_model | progress | model | before_model |
| after_model | progress | model | after_model |
| before_tools | progress | tools | before_tools |
| after_tools | progress | tools | after_tools |
| waiting | waiting | model/tools | after_model/before_tools |
| before_terminal | progress | finalizing | before_finalization |

`CoreBoundary` 不携带组件私有恢复结构。Adapter 从对应 Port 读取组件 checkpoint，并持有最新 Session/Run revision。

`PreparedAgentRun` 是 Runtime 准备与执行之间的下层 handoff DTO：

```text
PreparedAgentRun
├── run_id / session_id
├── loop_input: CoreRunInput
├── context_port
├── state_port: BoundaryPort
├── input_messages
├── rollback_baseline
└── context/memory/plan refs
```

`SessionRunRecord.outcome` 直接保存 `CoreOutcome`，不建立第二个 Core 结果词汇。对外 `AgentRunResult` 和 stop reason 由 Runtime 投影后随 terminal commit 保存。

## 恢复协议

职责分工：

```text
Sessions -> 加载并校验 Session/Run/Checkpoint/消息链
Runtime  -> 校验 Workspace，恢复组件，选择 ModelEntry 或 ToolResultEntry
Tools    -> 恢复 pending attempt，并完成 approval/interaction
Context  -> 恢复上下文组件并重新投影
Core     -> load_core_state 后从统一入口继续
```

恢复顺序：

```text
inspect_recovery
  -> validate checkpoint and workspace
  -> restore Tools/Context components
  -> resume_run(waiting -> running)（仅 waiting 场景）
  -> Runtime prepare CoreRunInput
  -> RunExecutor -> run_core
```

规则：

- Prompt 重试复用 request_id，不能重复创建用户消息。
- Resume 使用原 run_id，不创建第二个 Run。
- Workspace 校验失败时不能启动 Core。
- `before_tools` 后执行结果不确定时禁止自动重放副作用。
- `after_tools` 恢复不能再次执行已提交 ToolResult。
- 终态 Run 不允许恢复执行。

## CoreState Schema 恢复

Sessions 持久化的是 JSON `core_state`，不解释字段。恢复时 Runtime 构造 `CoreRunInput`，由 `load_core_state()`：

- 直接读取 schema v2。
- 将当前 Sessions v2 的无 schema payload 升级为 v2。
- 拒绝未知 schema version 或无法确定 original request 的 payload。

下一次 `CoreBoundary` 只写 `CoreState.to_dict()` 的新 schema，不再双写旧字段或独立 Plan 状态。

## Context 边界

Context 位于 `sessions/context/`，共享源码命名空间但保持独立领域职责。Context preparation DTO 当前由 `sessions/contracts.py` 定义：

```text
AgentContext
ContextPreparationRequest
PreparedAgentContext
PrepareContextFn
```

Context 负责：

- Repository snapshot、active files 和 evidence freshness。
- token 压力、消息选择和压缩。
- Memory recall 集成。
- 生成 ContextReport 和 Context component checkpoint。
- 生成 `context_projected` 等 durable event payload。

Context 不直接写 Session/Run 文件。Runtime Context Port 调用 Context service，并将事件排队到下一次权威边界。

## Memory 边界

Memory 位于 `sessions/memory/`，拥有 workspace 级长期记录和独立 repository。Memory 负责准入、召回、冲突、状态演化和 Run 后经验提炼。

Sessions State Service 不解释 Memory Record。Context 通过窄 Recall 接口使用 Memory；Memory 通过 Evidence Reader 验证来源，不读取 Sessions 文件布局。

## 持久化布局

```text
.codepilot/
├── sessions/<session_id>/
│   ├── session.json
│   ├── messages.jsonl
│   └── events.jsonl
├── runs/<run_id>/
│   ├── run.json
│   └── events.jsonl
└── memory/
    └── memories.jsonl
```

写入规则：

- JSON 状态使用临时文件、flush/fsync 和 `os.replace()`。
- JSONL 追加保持单条记录完整；末尾损坏可以截断，中间损坏阻塞恢复。
- Repository 隐藏磁盘布局；Service 不自行拼接路径。
- Event 缺失不影响状态恢复。

## 源码布局

```text
src/codepilot/sessions/
├── __init__.py
├── contracts.py
├── service.py
├── repository.py
├── filesystem.py
├── serde.py
├── workspace.py
├── context/
├── memory/
└── rollback/
```

文件职责：

| 文件 | 职责 |
|---|---|
| contracts.py | Session/Run/Message/Checkpoint/Recovery DTO，以及下层 handoff/context DTO |
| service.py | 状态转换、边界提交、恢复检查和幂等编排 |
| repository.py | Session/Run/Message/Event 数据访问和 CAS 更新 |
| filesystem.py | 原子 JSON、JSONL、路径布局和损坏检测 |
| serde.py | DTO/Protocol Message 与持久化字典转换 |
| workspace.py | 中立路径、Hash、变更和低级 Git 查询 |
| context/ | 上下文投影、选择、压缩和 checkpoint |
| memory/ | 长期记忆准入、召回、冲突和持久化 |
| rollback/ | 回滚判断、baseline 和执行流程 |

## 架构不变量

1. Sessions 是 Run 生命周期和 Checkpoint 的唯一持久化写入者。
2. CoreBoundary 必须经 Runtime Adapter 转换，Core 不 import Sessions。
3. Events 不修改 RunState，也不能作为恢复权威。
4. Component checkpoint 对 Sessions 不透明。
5. Plan 只持久化在 `core_state.task.plan`，不独立双写。
6. waiting Run 必须有 WaitingState 和 Checkpoint。
7. terminal Run 不得保留 Checkpoint。
8. revision 冲突必须显式失败。
9. Runtime task 状态不得写入 Sessions 充当第二份执行事实。
10. Context、Memory、Tools 和 Rollback 的领域策略不进入 SessionStateService。

## 测试重点

- Session/Run 状态转换和 revision 冲突。
- commit_id 幂等、消息去重和跨文件失败重试。
- Progress、Waiting、Terminal 三类提交。
- 每个 CoreBoundary 到 resume point 的映射。
- 当前 v2 Core payload 读取与新 schema 单写。
- Waiting approval、after_model、before_tools 和 after_tools 恢复。
- Workspace 变化阻塞恢复。
- Event 缺失、JSONL 尾部损坏和中间损坏。
- Core 无 Sessions 反向依赖，Sessions 不解释组件状态。

## 最终边界

> Sessions 保存 Session、Run、Messages 和 Checkpoint 的权威状态，提供一致的提交与恢复接口，并记录非权威审计事件；它不决定 Agent 如何推进，也不负责 Context、Memory、Plan、Tools 或 Runtime 资源的领域策略。
