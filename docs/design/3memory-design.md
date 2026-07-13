# Memory Management 设计

## 文档状态

本文描述 Codepilot 第一版 Memory Management 的目标设计。

设计参考 Claude Code 对长期规则、用户反馈、项目知识和经验沉淀的治理思想，但不追求完整复刻其实现。第一版优先保证边界清楚、行为可解释、用户可控制，并避免为了形式完整引入过多字段、模型调用和状态机。

本文已经确定：

- Memory 只保存跨任务仍有价值的信息。
- Working Summary / Compact Summary 属于 Context，不属于长期 Memory。
- Memory 分为用户级和项目级两个作用域。
- MemoryRecord 只保留八个基础字段。
- 用户明确要求写入的记忆可以直接激活。
- Agent 自动提取的记忆只能成为候选，不能自行批准。
- 自动提取由主模型在最终化调用中完成，不引入独立提取模型。
- 只有 Active Memory 可以进入 Context L3。
- 当前用户指令和当前工作区证据始终覆盖 Memory。
- 删除默认采用逻辑删除，敏感信息提供物理清除例外。

本文暂不设计：

- Session、Workspace 或 Team Memory 作用域。
- 向量数据库和复杂语义检索。
- 每轮对话后的额外 Memory LLM 调用。
- 自动审批或自动提升候选记忆。
- 持久化置信度、有效性、过期时间和版本关系字段。
- 基于路径的条件记忆规则。
- 完整 Memory 审计协议、评测指标和 Benchmark。

## 1. 设计目标

Memory Management 解决的问题是：Agent 在不同任务和不同 Run 之间，应该保留哪些长期有价值的信息，以及这些信息如何被写入、确认、演化和召回。

它需要回答：

```text
哪些信息值得跨任务保存？
用户明确要求和 Agent 自动推断有什么权限差异？
候选记忆如何审批、编辑、禁用和删除？
当前任务需要召回哪些记忆？
重复、冲突和过期信息如何处理？
Memory 与当前指令或工具事实冲突时相信谁？
```

目标数据流：

```text
User Explicit Intent             Verified Run Outcome
         |                                |
         v                                v
   Direct Write                     MemoryProposal
         |                                |
         +----------> MemoryManager <-----+
                           |
                           v
                     MemoryRepository
                           |
                           v
                     MemoryRetriever
                           |
                           v
                    Context L3 Projection
```

Memory Management 不负责：

- 保存完整聊天历史。
- 保存当前任务计划、进度、阻塞和验证状态。
- 保存原始工具输出或大段代码。
- 代替 Workspace、Tool Artifact 或 Session Message 成为事实源。
- 推进 Agent loop 或直接执行工具。
- 决定最终模型请求的整体 Token 分配。
- 将 Context 压缩摘要自动提升为长期知识。

## 2. 领域边界

| 领域 | 权威职责 |
|---|---|
| Session | 已提交的 Session、Run、Message、Checkpoint 和事件 |
| Core | 当前目标、计划、步骤、阻塞和验证状态 |
| Tools / Artifacts | 原始工具结果、文件变化和验证证据 |
| Context | 单次模型调用的分层视图、预算、压缩和 L3 注入 |
| Memory | 跨任务知识的准入、生命周期、冲突治理和召回 |
| Runtime | 协调调用顺序，不实现 Memory 内部策略 |

核心边界：

```text
Session 保存发生过什么。
Core 保存当前任务如何推进。
Tools 保存当前观察和执行证据。
Memory 保存未来任务仍可能有用的知识。
Context 决定本轮模型看到哪些 Memory。
```

Working Summary、Compact Summary、当前计划和当前验证结果即使很重要，也不能因为需要跨模型调用复用而被写入长期 Memory。它们属于 Context 或 Core 的运行状态。

## 3. 核心原则

### 3.1 跨任务价值

Memory 不是聊天归档。只有在当前任务结束后仍可能改变未来回答、规划、工具使用或验证方式的信息，才具备写入资格。

### 3.2 当前事实优先

Memory 是辅助知识，不是当前事实的权威来源。

```text
当前有效指令
> 当前代码、配置和工具证据
> 当前项目 Memory
> 用户级一般 Memory
> 历史对话中的旧说法
```

当 Memory 与当前工作区冲突时，本次任务必须相信当前观察。

### 3.3 用户拥有最终控制权

Agent 可以提出 Candidate Memory，但不能批准、编辑、禁用或删除长期记忆。永久改变 Memory 行为的操作必须来自用户明确指令或审批。

### 3.4 自动提取不等于自动生效

即使一次 Run 已经验证成功，LLM 从该过程概括出的长期经验仍可能过度泛化。因此，自动提取结果默认是 `candidate`，而不是 `active`。

### 3.5 经验必须经过验证

失败本身不能形成 Experience Memory。经验提取至少需要形成：

```text
问题或失败
  -> 修改或解决方法
  -> 验证通过
  -> 提出候选经验
```

验证只能证明本次操作成功，不能代替用户批准长期规则。

### 3.6 精简记录，动态判断

第一版不在记录中持久化主观 `confidence`、复杂 `validity` 或版本关系。信任和适用性由来源、作用域、当前事实、生命周期状态和更新时间在使用时共同判断。

### 3.7 敏感信息不落盘

密钥、Token、密码、隐私数据和完整环境变量值不能进入 Memory。逻辑删除不能替代写入前的敏感信息检查。

## 4. Memory 类型

第一版保留五种类型：

```python
MemoryType = Literal[
    "profile",
    "feedback",
    "project",
    "experience",
    "reference",
]
```

### 4.1 Profile

保存用户长期偏好、沟通习惯和个人约束。

适合保存：

- 默认使用的交流语言。
- 对回答风格的稳定偏好。
- 跨项目都成立的工作习惯。

不适合保存：

- “这次请用英文”。
- 当前任务中的临时格式要求。
- 从一次普通对话中弱推断出的偏好。

Key 示例：

```text
profile.response_language
profile.explanation_detail
```

### 4.2 Feedback

保存用户对 Agent 行为的持续性纠正。

适合保存：

- 不要进行未经请求的重构。
- 修改代码后必须报告未运行的验证。
- 不要把计划当作已经完成的事实。

不适合保存：

- 对某一次回答措辞的局部修改。
- 只针对当前任务的纠正。

Key 示例：

```text
feedback.unrequested_refactor
feedback.unverified_claims
```

### 4.3 Project

保存项目长期约束、约定和关键决策。

适合保存：

- 难以从仓库低成本发现的设计决策。
- 会持续影响实现选择的兼容性要求。
- 用户明确指定的项目约束。

不适合保存：

- 可以直接从配置文件读取的普通事实。
- 当前分支状态或当前任务进度。
- 大段架构文档副本。

Project 不增加单独的子类型字段，通过 Key 表达语义：

```text
project.fact.<name>
project.constraint.<name>
project.decision.<name>
```

例如：

```text
project.constraint.api_backward_compatibility
project.decision.package_manager
```

### 4.4 Experience

保存经过验证、可能复用于未来任务的解决经验和操作注意事项。

适合保存：

- 修改某类语法后需要同步运行的验证。
- 已经通过成功 Run 证明有效的排错步骤。
- 特定项目中容易遗漏的稳定工作流程。

不适合保存：

- 单次失败。
- 尚未验证的根因猜测。
- 偶然成功但无法说明适用条件的操作。

Key 示例：

```text
experience.parser.snapshot_update
experience.migration.validation_order
```

### 4.5 Reference

保存用户指定的重要文档、权威资料或稳定信息入口。

适合保存：

- 项目架构决策的权威文档位置。
- 用户明确指定的规范入口。
- 未来任务需要优先查阅的参考资料。

不适合保存：

- 普通搜索结果。
- 临时网页。
- 文档全文或大段摘录。

Key 示例：

```text
reference.architecture_design
reference.release_policy
```

## 5. Memory 作用域

第一版只保留两个作用域：

```python
MemoryScope = Literal["user", "project"]
```

### 5.1 User Scope

用户级 Memory 对该用户的不同项目生效，适合保存 Profile 和跨项目 Feedback。

建议存储位置：

```text
~/.codepilot/memory/memories.jsonl
```

### 5.2 Project Scope

项目级 Memory 只在当前工作区生效，适合保存项目约束、决策、经验和参考入口。

建议存储位置：

```text
<workspace>/.codepilot/memory/memories.jsonl
```

### 5.3 不引入的作用域

第一版不增加：

- `session`：临时信息属于 Context 或 Core。
- `workspace`：当前版本不再区分 Workspace 和 Project。
- `team`：涉及共享、权限和同步，不属于第一版范围。

同一 Key 可以同时存在 User 和 Project 版本。Project 表示当前项目中的具体规则，User 表示一般偏好。

## 6. MemoryRecord

第一版持久化记录只保留八个字段：

```python
@dataclass
class MemoryRecord:
    id: str
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str
    source: MemorySource
    status: MemoryStatus
    updated_at: datetime
```

字段职责：

| 字段 | 职责 |
|---|---|
| `id` | 管理命令定位一条具体记录 |
| `scope` | 决定用户级或项目级存储与适用范围 |
| `type` | 分类、召回过滤和展示 |
| `key` | 记忆主题，是去重、冲突和版本演化的核心 |
| `content` | 完整表达需要记住的原子信息 |
| `source` | 表示记忆来源，用于动态信任判断 |
| `status` | 控制审批、召回和生命周期 |
| `updated_at` | 管理展示、排序兜底和历史顺序 |

第一版不在每条记录中保存：

```text
schema_version
revision
project_kind
subject / predicate / value
rationale / application
tags / paths
priority / occurrences
confidence / validity / expires_at
supersedes / superseded_by
```

Schema Version 如有需要，由 Repository 或存储格式统一管理，不在每条记录中重复保存。

### 6.1 Key 规范

Key 使用小写点分形式：

```text
<type>.<topic>
<type>.<category>.<topic>
```

例如：

```text
profile.response_language
feedback.unrequested_refactor
project.constraint.api_backward_compatibility
experience.parser.snapshot_update
reference.architecture_design
```

Memory 的逻辑身份是：

```text
scope + key
```

`type` 不属于逻辑身份。修改类型仍然是对同一主题的替换。

## 7. 来源与信任模型

第一版来源类型：

```python
MemorySource = Literal[
    "user_explicit",
    "user_feedback",
    "agent_extracted",
    "verified_run",
]
```

含义：

| Source | 含义 |
|---|---|
| `user_explicit` | 用户明确要求记住或通过管理命令写入 |
| `user_feedback` | 来自用户对 Agent 行为的持续性纠正 |
| `agent_extracted` | Agent 从对话或项目工作中自动提出 |
| `verified_run` | Agent 从完成验证闭环的 Run 中提出 |

不保存主观 Confidence。使用时根据以下因素动态判断：

```text
当前指令和当前事实
生命周期状态
作用域适用性
来源
更新时间
是否存在冲突
```

Memory Source 的一般信任顺序为：

```text
user_explicit
> user_feedback
> verified_run
> agent_extracted
```

来源顺序只用于同等适用条件下的选择，不能覆盖当前用户指令和当前工具证据。

## 8. 写入准入

### 8.1 准入条件

写入 Memory 的信息必须同时满足：

1. **跨任务有价值**：不仅服务当前步骤。
2. **相对稳定**：不会随当前 Run 结束立即失效。
3. **能够影响未来行为**：会改变回答、规划、工具使用或验证方式。
4. **表达明确且原子化**：一条记录只表达一个主题。
5. **作用域明确**：能够判断属于 User 还是 Project。
6. **不存在敏感信息**：不能包含密钥、隐私数据和原始工具输出。

### 8.2 禁止写入

以下内容不得写入长期 Memory：

- 当前任务进度、待办和工作摘要。
- 当前 Run 的临时判断或阻塞原因。
- 原始工具输出、错误日志和大段代码。
- 未验证的根因猜测和失败尝试。
- “这次不要运行测试”一类一次性要求。
- 可以从当前仓库低成本读取的普通事实。
- 密钥、Token、密码、隐私数据和完整环境变量值。
- Context Compact Summary 或其直接改写。

### 8.3 写入流程

```text
对话或 Run 产生信号
        |
        v
形成 MemoryProposal 或明确用户写入
        |
        v
MemoryManager 执行确定性准入检查
        |
        v
规范化 scope / type / key / content
        |
        v
敏感信息、重复和冲突检查
        |
        v
explicit -> active
automatic -> candidate
```

LLM 只能提出 Memory，不能直接修改 JSONL 文件。`source` 和初始 `status` 由 MemoryManager 根据真实触发来源设置，不能由 LLM 自报。

## 9. 显式写入与自动提取

### 9.1 用户明确写入

以下信号可以直接写成 `active`：

- 用户明确说“记住……”。
- 用户明确表达长期范围，例如“以后都……”。
- 用户明确表达项目范围，例如“这个项目始终……”。
- 用户确认一条 Candidate Memory。
- 用户通过 Memory 管理命令手动添加。

普通任务指令即使表达明确，也不能默认成为长期规则。带有“这次”“当前任务”等临时范围的要求只进入当前 Context。

显式写入不需要等到 Run 最终化，应在用户意图确认后立即由 MemoryManager 执行。

### 9.2 自动提取

以下内容可以由 Agent 自动提出，但只能进入 `candidate`：

- 从普通对话推断出的稳定用户偏好。
- 从用户纠正中识别出的长期 Feedback。
- 从项目工作中识别出的潜在约束或决策。
- 从成功验证的 Run 中概括出的 Experience。
- 自动识别的重要 Reference 入口。

以下弱信号不产生候选：

- 用户只表达过一次的轻微措辞偏好。
- 不会改变未来行为的信息。
- 与现有 Active 或 Candidate 完全重复的信息。
- 无法确定作用域或无法原子化表达的信息。
- 仅对当前任务有效的信息。

### 9.3 不引入独立提取模型

第一版由主模型在最终化调用中同时产生用户回答和 MemoryProposal，不增加独立 Memory 提取模型，也不在最终回答已经发送后重新扫描回答文本。

逻辑响应：

```python
@dataclass
class FinalResult:
    message: str
    memory_proposals: list[MemoryProposal]
```

临时 Proposal 可以表示：

```python
@dataclass
class MemoryProposal:
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str
```

`MemoryProposal` 不是持久化记录。实现可以在内部暂时携带提取理由或证据引用，但 MemoryRecord 不因此扩展字段。

最终化流程：

```text
主模型获得任务结果和验证状态
        |
        v
同一次最终调用生成 message + memory_proposals
        |
        +------> Runtime 返回 message
        |
        +------> MemoryManager 校验并保存 candidate
```

约束：

- 每个 Run 最多提出少量候选，第一版上限为 3 条。
- 默认允许返回空列表，大多数任务不应产生 Memory。
- 不在每轮模型调用、每次工具结果或 Context 压缩后提取。
- Run 中断或失败时，默认不自动提取 Experience。
- 自动提取不得直接写成 Active。

## 10. 生命周期

第一版保留五种状态：

```python
MemoryStatus = Literal[
    "candidate",
    "active",
    "disabled",
    "superseded",
    "deleted",
]
```

| Status | 含义 | 是否召回 |
|---|---|---|
| `candidate` | Agent 自动提出，等待用户审批 | 否 |
| `active` | 当前有效记忆 | 是 |
| `disabled` | 暂时停用，可以重新启用 | 否 |
| `superseded` | 已被新版本替代，仅保留历史 | 否 |
| `deleted` | 已逻辑删除 | 否 |

状态流转：

```text
candidate --approve--> active
candidate --reject----> deleted

active ----disable----> disabled
disabled --enable-----> active

active/disabled --edit or replace--> superseded
                                      |
                                      v
                              new active/disabled

candidate/active/disabled/superseded --delete--> deleted
```

只有 `active` 可以进入正常召回。

### 10.1 生命周期不变量

```text
同一 scope + key 最多有一个 active 或 disabled 当前版本。
同一 scope + key 最多有一个 candidate。
superseded 和 deleted 不能进入召回。
Agent 不能自行把 candidate 转成 active。
```

## 11. 审批与管理

### 11.1 审批

只有自动提取产生的 Candidate Memory 需要审批。审批至少展示：

```text
scope
type
key
content
source
```

支持：

- `approve`：Candidate 变成 Active。
- `reject`：Candidate 变成 Deleted。
- `edit-and-approve`：用户修改后激活。

如果 Candidate 与同一 `scope + key` 的当前记录冲突，用户必须选择替换、修改或拒绝。批量审批只处理无冲突项，冲突项不能静默覆盖。

### 11.2 编辑

编辑尚未生效的 Candidate 可以直接修改，因为它还没有影响模型行为。

编辑 Active 或 Disabled Memory 时不覆盖原记录：

```text
旧记录 -> superseded
新记录 -> active 或 disabled
```

新记录使用新 ID，但保持相同 `scope + key`。不需要 `revision`、`supersedes` 和 `superseded_by` 字段，可以通过相同 Key 与 `updated_at` 查看演变顺序。

规则：

- Active 编辑后产生新的 Active。
- Disabled 编辑后产生新的 Disabled。
- Superseded 和 Deleted 不能直接编辑。
- 恢复旧版本时，复制其内容创建新的 Active。
- 修改 Scope 或 Key 视为结束旧主题并创建新主题。

### 11.3 禁用与启用

禁用用于“当前不希望使用，但暂时不删除”的情况：

```text
active -> disabled
```

Disabled Memory：

- 不进入 L3。
- 不参与模型行为。
- 仍可查看和重新启用。
- 仍参与写入去重，防止重复创建。

重新启用前必须检查同一 `scope + key` 是否已经有新的 Active。两个版本不能同时激活。

### 11.4 删除与物理清除

普通删除采用逻辑删除：

```text
status = deleted
```

Deleted Memory：

- 不召回。
- 不参与正常去重和冲突判断。
- 默认只在历史查询中显示。
- 不直接恢复；需要复制为一条新记录。

如果误写入密钥、Token 或隐私信息，必须提供 `purge` 进行物理清除。敏感数据不能仅依靠逻辑删除继续留在文件中。

### 11.5 操作权限

| 操作 | Agent | 用户 |
|---|---:|---:|
| 提出 Candidate | 允许 | 允许 |
| 直接添加 Active | 仅执行明确用户指令 | 允许 |
| 批准候选 | 不允许 | 允许 |
| 编辑 Memory | 不允许自主执行 | 允许 |
| 禁用或启用 | 不允许自主执行 | 允许 |
| 删除或 Purge | 不允许自主执行 | 允许 |
| 确定性 Supersede | MemoryManager 根据用户操作执行 | 间接触发 |

第一版对外提供的管理能力：

```text
list
show
approve
reject
edit
disable
enable
delete
history
purge
```

## 12. 召回

### 12.1 召回时机

召回发生在每次模型调用前，由 Context Governance 构建 L3 时触发：

```text
L1 当前任务状态
        |
        v
生成 MemoryQuery
        |
        v
MemoryRetriever 过滤和排序
        |
        v
ContextGovernor 按 L3 预算选择
        |
        v
注入最终模型请求
```

### 12.2 MemoryQuery

第一版查询只需要：

```python
@dataclass
class MemoryQuery:
    user_request: str
    task_goal: str
    current_step: str | None
    active_paths: list[str]
    limit: int
```

查询不得包含：

- 之前召回的 Memory。
- Agent 自己的历史回答。
- Compact Summary 全文。
- 原始工具输出。

这样可以避免“因为之前召回过，所以后续持续召回”的自我强化。

### 12.3 确定性过滤

召回前必须过滤：

```text
status 必须是 active
scope 必须是当前 user 或当前 project
content 不能为空
不得存在同 scope + key 的更新当前版本
```

其他项目的 Project Memory 不能被召回。

### 12.4 基础召回与相关性召回

Profile 和 Feedback 表示用户偏好与行为纠正，即使与当前请求没有关键词重合也可能适用，因此进入有限的基础候选集。

Project、Experience 和 Reference 必须与当前任务匹配。匹配文本只使用：

```text
record.key + record.content
```

查询文本使用：

```text
user_request + task_goal + current_step + active_paths
```

第一版不引入向量数据库。英文、标识符和路径进行普通规范化分词；中文使用子串或连续双字词匹配，避免把整段中文当作一个 Token。

### 12.5 排序

排序使用可解释的元组，不叠加大量难以维护的魔法分数：

```python
rank = (
    match_level,
    match_count,
    scope_specificity,
    source_trust,
    updated_at,
)
```

匹配等级：

```text
3：Key 或完整短语直接匹配
2：Content 与任务存在多个相关词项
1：Profile / Feedback 基础召回
0：不相关，不召回
```

同等条件下：

- 当前 Project Scope 优先于 User Scope。
- 用户明确来源优先于自动提取来源。
- 更新时间只作为最后的排序兜底，不代表内容一定正确。

召回分数和原因保存在临时 RetrievedMemory 中，用于排查和测试，不写入 MemoryRecord，也不注入模型。

### 12.6 预算职责

职责划分：

- MemoryRetriever 负责过滤和排序，返回少量候选。
- ContextGovernor 负责根据 L3 Token 预算选择最终条目。
- 第一版最多注入 5 条 Memory。
- 过长 Memory 不在召回时临时总结，应在写入准入阶段拒绝或原子化。

Memory 不能抢占 L0-L2 的必要预算，也不能突破最终模型请求的输入硬预算。

## 13. L3 注入

Memory 作为独立 L3 区块注入，不能伪装成 User Message 或历史对话：

```text
[Recalled Memory]
Current instructions and workspace evidence override recalled memory.
Use recalled memory silently unless the user asks or provenance matters.

- [user/profile] profile.response_language: 默认使用中文回答
- [project/project] project.constraint.api_compatibility: 公开 API 必须保持向后兼容
- [project/experience] experience.parser.snapshot: 修改语法后运行快照测试
```

模型可见：

```text
scope + type + key + content
```

模型不需要看到：

```text
id
score
updated_at
生命周期历史
```

Context 内部条目应保留 Memory ID，方便来源定位，但本文暂不扩展完整 ContextReport。

### 13.1 避免重复展示

模型调用是无状态的。只要一条 Active Memory 仍然相关，每次调用重新注入是必要的，不能因为上一轮已经注入就降低权重或跳过。

真正需要避免的是：

- 同一请求出现多条相同 `scope + key` 的 Memory。
- User 和 Project 内容相同时重复注入。
- 已由更高层明确提供的相同内容再次出现在 L3。
- 模型在回答中反复说“我记得你……”或主动展示 Memory 来源。

第一版不增加最近召回次数、曝光计数或重复展示惩罚。这些机制可能让重要约束在长任务中逐渐消失。

## 14. 重复管理

### 14.1 规范化

写入前对 Key 和 Content 做轻量规范化。

Key：

- 转为小写。
- 使用点分层级。
- 去除首尾空白。

Content 比较值：

- 合并连续空白。
- 忽略首尾空白和无意义结尾标点。
- 进行大小写归一化。
- 持久化时保留用户原始表达。

第一版只处理确定性重复，不调用 LLM 判断不同句子是否语义相同。

### 14.2 重复写入

| 已有记录 | 新内容相同 | 处理 |
|---|---|---|
| Active | 是 | 返回已有记录，不创建新记录 |
| Candidate | 是 | 不创建重复候选 |
| Disabled | 是 | 自动提取时忽略；用户明确写入时重新启用 |
| Superseded | 是 | 允许用户恢复为新的 Active |
| Deleted | 是 | 不阻止重新创建 |

同一 Scope 内，不同 Key 但规范化 Content 完全相同，可以确定性去重。不同 Scope 的相同内容可以同时保留，只在召回时消除重复。仅语义近似时不自动合并，交给用户审批。

重复写入不能刷新 `updated_at`，否则重复提取会让旧 Memory 看起来比实际更可信。

## 15. 冲突管理

### 15.1 同作用域冲突

冲突定义为：

```text
scope + key 相同，但规范化后的 content 不同
```

用户明确写入时：

```text
旧 active/disabled -> superseded
新记录 -> active
```

Agent 自动提取时：

```text
旧 active/disabled 保持不变
新记录 -> candidate
```

Candidate 获批替换后，旧记录转为 Superseded，Candidate 转为 Active。已有待审批 Candidate 时，新的自动提取不能持续创建同 Key 候选。

### 15.2 跨作用域冲突

同一个 Key 可以同时有 User 和 Project 版本，因为 Project 可以表达当前项目对一般用户偏好的例外。

```text
user/profile.language = 默认使用中文
project/profile.language = 本项目公开文档使用英文
```

在当前项目中：

```text
project scope > user scope
```

只注入 Project 版本。离开该项目后，User Memory 继续有效。

如果两个作用域内容完全相同，召回时只注入 Project 版本，但不需要删除 User Memory。

### 15.3 当前信息覆盖

需要区分本次覆盖和永久修改。

```text
Memory：默认使用中文回答
当前用户：这次请使用英文
```

本次使用英文，但 Memory 保持不变，因为当前要求是任务级例外。

```text
Memory：项目使用 Poetry
当前 pyproject.toml 和工具结果：项目已经使用 uv
```

本次任务相信当前仓库证据。Agent 可以提出维护建议，但不能自行编辑或删除旧 Memory。

能够确定性识别的相同 Key、完全重复和跨作用域覆盖由 MemoryRetriever 处理。无法确定性识别的语义冲突交给模型按照 L3 优先级说明处理，不增加复杂规则猜测。

## 16. 过期管理

第一版不增加：

```text
stale 状态
expires_at
validity
repository_fingerprint
confidence
```

`updated_at` 只能说明最后更新时间，不能证明内容正确，因此不能根据时间自动删除或禁用 Memory。

过期采用使用时发现策略：

```text
召回 Active Memory
        |
        v
当前指令或工具证据表明它不再成立
        |
        v
本次调用忽略该 Memory
        |
        v
Agent 提出维护建议或替代 Candidate
        |
        v
用户决定替换、禁用或删除
```

Agent 不能因为一次表面冲突就自行禁用 Memory，因为当前要求可能只是临时例外。

对于能够直接从仓库读取的事实发生变化时，通常不应再写一条新的 Project Memory。应以当前文件为准，并建议清理已经失去价值的旧 Memory。

## 17. 第一版完整流程

### 用户明确要求记住

```text
用户明确长期记忆意图
        |
        v
MemoryManager 准入与敏感信息检查
        |
        v
规范化 Key 并检查重复、冲突
        |
        v
写入 Active，必要时 Supersede 旧版本
```

### Agent 自动提出记忆

```text
Run 完成并进入最终化
        |
        v
主模型同一次调用生成 FinalResult
        |
        v
MemoryManager 校验 MemoryProposal
        |
        v
无价值或重复 -> 丢弃
存在冲突或有效新信息 -> Candidate
        |
        v
等待用户审批
```

### 模型调用前召回

```text
ContextGovernor 获取当前任务状态
        |
        v
构造 MemoryQuery
        |
        v
过滤非 Active 和错误作用域
        |
        v
确定性匹配与排序
        |
        v
ContextGovernor 按 L3 预算选择
        |
        v
作为独立 L3 区块注入
```

## 18. 设计不变量

实现和测试至少需要保证：

1. Working Summary 和 Compact Summary 永远不能直接写入长期 Memory。
2. 只有 Active Memory 可以进入 Context L3。
3. Agent 自动提取的 Memory 初始状态只能是 Candidate。
4. Agent 不能批准、编辑、禁用、启用、删除或 Purge Memory。
5. 用户明确的长期写入可以直接成为 Active。
6. Experience 必须来自已经完成验证闭环的 Run，才允许自动提出。
7. 同一 `scope + key` 最多有一个 Active 或 Disabled 当前版本。
8. 同一 `scope + key` 最多有一个 Candidate。
9. 当前用户指令和当前工作区证据始终覆盖 Memory。
10. Project Memory 在当前项目中覆盖同 Key 的 User Memory。
11. Disabled、Superseded 和 Deleted Memory 永远不能被召回。
12. 重复写入不能通过刷新时间伪造新鲜度。
13. 普通删除保留 Deleted 状态，敏感信息必须支持物理 Purge。
14. 自动提取不需要独立模型，也不能从已发送的最终回答文本反向解析。
15. MemoryRetriever 负责过滤和排序，ContextGovernor 负责 L3 预算和最终注入。

## 19. 与其他设计的关系

- Context 五层结构、L3 职责和输入预算见 [2context-design.md](./2context-design.md)。
- Session、Run、Message 和 Checkpoint 的权威边界见 [6sessions-design.md](./6sessions-design.md)。
- Context 与 Memory 的 Port、目标文件布局、四阶段迁移顺序、测试矩阵和验收标准见 [8context-memory-implementation.md](./8context-memory-implementation.md)。
