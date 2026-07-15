# Context Governance 设计

## 文档状态

本文描述 Codepilot 精简版 Context Governance 的目标设计。

设计参考 Claude Code 的分层上下文、工具结果瘦身、语义压缩和完整历史保留思想，但不追求复制其 Provider 专属缓存、服务端 context editing、后台任务或复杂实验能力。

本文已经确定：

- Context 使用五层逻辑结构。
- 每次模型调用前重新物化 Context。
- 工具结果先以有硬上限的完整正文提交；只有请求进入 `tight` 后，旧且可恢复的结果才在本次请求中替换为 Run Artifact 引用。
- 输入预算必须在调用 Provider 前通过硬校验。
- 确定性瘦身处理冗余和工具输出，语义删除由辅助 LLM 完成。
- Working Summary / Compact Summary 属于 Context，不属于长期 Memory。
- Session Messages 始终保留完整、已提交的会话事实。

本文暂不设计：

- ContextReport 的完整审计协议。
- Context 评测指标和 Benchmark。
- Prompt cache key、跨租户缓存或 Provider 专属缓存编辑。
- 路径条件规则。
- Tool Search 和延迟 Tool Schema 加载。
- 向量数据库或复杂语义检索。

## 1. 设计目标

Context Governance 解决的问题是：Session、Core、Tools、Workspace 和 Memory 会持续产生事实，但模型每次调用只能消费有限的输入窗口。

它必须在每次模型调用前回答：

```text
哪些规则必须始终出现？
当前任务和运行状态是什么？
哪些工作集和工具证据仍然有效？
哪些长期记忆与当前问题相关？
哪些历史消息需要保留、瘦身或压缩？
最终请求是否确定没有超过模型窗口？
```

目标数据流：

```text
Session Messages
Core State
Runtime / Tool Capabilities
Workspace / Tool Evidence
Memory Recall
        |
        v
Context Materializer
        |
        v
Token Estimation and Selection
        |
        v
Provider-safe LLM Request
```

Context Governance 不负责：

- 推进 Agent loop。
- 决定任务是否完成。
- 执行工具或判断工具权限。
- 修改 Session、Run 或 Core State。
- 写入、审批或删除长期 Memory。
- 将 Compact Summary 提升为长期 Memory。

## 2. 领域边界

| 领域 | 权威职责 |
|---|---|
| Session | 已提交消息、Run、Checkpoint、事件和来源引用 |
| Core | 当前目标、计划、步骤、阻塞和验证状态 |
| Tools / Artifacts | 原始工具结果、文件变化和验证事实 |
| Context | 单次模型调用的临时视图、预算、选择和压缩 |
| Memory | 跨任务知识的准入、演化和召回 |
| Runtime | 协调各领域，不实现其内部治理策略 |

核心边界：

```text
Session 保存发生过什么。
Core 保存当前任务如何推进。
Tools 保存观察和执行产生了什么。
Memory 保存跨任务仍有价值的知识。
Context 决定模型本轮看到什么。
```

Context 可以保存派生 Artifact 和最小 Checkpoint 引用，但这些内容不能成为 Session 或 Core 的第二套权威状态。

## 3. 核心原则

### 3.1 事实与投影分离

Session Message、Core State、Tool Artifact 和 Memory Record 是来源事实。最终模型请求只是一次临时投影，可以被裁剪、重排、摘要和重新构建。

### 3.2 压缩不删除事实

Context 压缩只改变模型阅读窗口。它不能删除或改写已经提交的 Session Messages，也不能丢失完整 Tool Artifact。

### 3.3 当前事实优先

统一可信度顺序：

```text
当前用户指令
> 当前工作区和工具观察
> 当前有效的验证证据
> 项目稳定规则
> Durable Memory
> Compact Summary
> 历史模型判断
```

Compact Summary 和 Memory 与当前工作区冲突时，必须相信当前观察。

### 3.4 Working Summary 不是 Memory

当前目标、已经完成的工作、活跃文件、验证状态、开放问题和下一步属于 Context 工作摘要。它服务于同一会话的连续性和压缩恢复，不具有跨任务长期权威性。

### 3.5 来源可追溯

动态 Context 条目应尽量携带：

```text
source
source_ref
trust
freshness
estimated_tokens
selection_reason
```

第一版可以在内部对象中维护这些字段，不要求立即形成完整 ContextReport。

### 3.6 分层与保留策略正交

逻辑层表示内容职责，保留级别表示预算压力下能否删除。同一层中可以同时包含不可删除、受保护和可裁剪内容。

## 4. 五层 Context

```text
L0 核心规则 + 作用域指令
L1 运行与能力 + 任务状态
L2 工作集 + 工具证据
L3 记忆召回
L4 对话连续性
```

### 4.1 L0 核心规则与作用域指令

包含：

- Agent 身份和基础行为规则。
- 安全规则。
- 工具使用的稳定规则。
- 用户级无条件指令。
- 项目级无条件指令，例如 AGENTS.md。

L0 中已经判定生效的内容不参与预算裁剪。

第一版不实现路径条件规则。项目指令按 Session 工作区加载，不根据 active file 动态匹配额外规则。

如果 L0 自身过大导致固定成本超过模型窗口，应返回配置错误，而不是截断规则。

### 4.2 L1 运行与能力及任务状态

运行与能力包括：

- 当前 mode。
- permission mode。
- 模型能力。
- 当前可用工具。
- waiting、approval 或 synthetic control 状态。

任务状态来自 Core State，包括：

- current goal。
- plan summary。
- current step / next action。
- completion criteria。
- blocked reason。
- verification status。

L1 必须保留的最小骨架：

```text
current mode
current goal
current step or next action
waiting or blocked state
verification status
```

详细计划、已完成步骤和历史状态可以参与预算裁剪。

当前用户原文只保存在 L4 UserMessage。L1 可以保存结构化目标，但不能复制用户原文并将其伪装成系统规则。

### 4.3 L2 工作集与工具证据

工作集回答“当前正在处理什么”：

- repository snapshot 摘要。
- active files。
- changed paths。
- 相关符号和必要代码片段。
- 当前工作区 delta。

工具证据回答“已经实际观察到什么”：

- 最新工具状态。
- 错误和失败摘要。
- 文件变更及 Hash。
- 验证状态。
- Artifact 引用。
- freshness 和 trust。

L2 不复制 L4 中仍然可见的 ToolResult 正文。它只保存结构化状态和来源引用。

### 4.4 L3 记忆召回

L3 只接收 Memory Service 返回的召回结果：

- Memory 内容。
- Memory ID。
- 来源和置信度。
- 召回原因。
- freshness 或陈旧提示。

Memory 即使来自用户明确写入，也不能伪装成 L0 系统规则。它始终保留 Memory 来源标签。

当前用户指令与 Memory 冲突时，当前用户指令优先。

Memory 召回频率、排序和防重复策略由 Memory 设计确定；Context 只消费召回结果并在预算内选择。

### 4.5 L4 对话连续性

L4 由以下内容构成：

```text
Compact Summary Attachment
+ compact cursor 后的完整消息组
+ 当前用户或工具消息
```

消息必须按合法 API round 分组。Assistant tool call 和对应 ToolResult 不允许分开裁剪。

当前用户消息、未闭合工具链、waiting 请求和近期必要消息属于 L4 的不可删除骨架。

## 5. 保留级别

### Required

- L0 全部内容。
- 当前用户消息。
- 当前 mode、权限和等待状态。
- 当前未闭合的合法 tool call/result 组。
- Provider 调用所需的消息边界。

### Protected

- 当前目标和当前步骤。
- 最新 unresolved error。
- 本轮 affected paths。
- 当前 verification status。
- 当前有效的 Compact Summary。

### Budgeted

- 任务详情和历史计划步骤。
- 工作集详情。
- 相关 Memory。
- 近期已解决证据。
- 普通对话历史。

### DiscardFirst

- stale 或 missing 的重复证据。
- 已经解决且不再相关的工具摘要。
- 重复 Runtime 状态。
- 重复 Memory 投影。
- 冗余元数据和历史模型判断。

## 6. Context 进入机制

### 6.1 每次模型调用前物化

一次 Run 可能包含多次模型调用：

```text
Model -> Tool -> Model -> Tool -> Model
```

每次模型调用前都必须重新物化 Context，因为 Core State、Tool Evidence、Workspace 和 Memory Recall 可能已经变化。

重新物化不等于所有数据都重新持久化。Context 从各权威来源读取当前状态，生成本次请求后即可以丢弃。

### 6.2 分层刷新

第一版采用简单刷新规则，不设计 cache key：

| 层 | 刷新时机 |
|---|---|
| L0 | Session 打开、Prompt 或项目指令变化 |
| L1 | 每次模型调用 |
| L2 | ToolResult、Workspace 或 Core verification 变化后 |
| L3 | Context 请求 Memory Recall 时 |
| L4 | 每次模型调用从已提交消息链和当前 Run 消息构建 |

Resume 后必须从 Session Messages、Core State、Artifacts 和当前 Workspace 重建 L1/L2/L4，不能信任旧进程内缓存。

### 6.3 请求中的物理位置

逻辑层和 Provider 请求位置不是同一概念。建议内部表达为：

```text
system_prompt      = L0
context attachment = L1 + L2 + L3
messages           = L4
tools              = Tool Catalog / schemas
```

L1-L3 使用类型化 Context Attachment，再由 Provider Adapter 渲染为安全消息。动态 Context Attachment 不写入 `messages.jsonl`。

五层内容不能全部拼入 system prompt，同时又在 messages 中重复出现。

### 6.4 物化顺序

```text
1. 读取 Session/Core/Runtime/Tools 的事实快照
2. 组装 L0 和 L1
3. 刷新 Workspace、Working Set 和 Evidence
4. 请求 Memory Recall，形成 L3
5. 从 compact cursor 和消息链形成 L4
6. 估算 Raw Context token
7. 应用选择、瘦身或压缩
8. 生成 Provider 请求
9. 执行消息合法性和最终预算预检
```

## 7. 工具证据与 Artifact

### 7.1 三种对象

#### ToolResult

ToolResult 属于 Session Messages，用于维持合法工具消息链。工具执行层必须对单次结果和并行批次设置硬上限，限额内的正文完整提交到 Session；ContextProjector 不因长度、路径或文件 Hash 改变而替换正文。

请求进入 `tight` 后，ContextThinner 可以把旧且可恢复的 ToolResult 在本次模型请求中替换为确定性首尾摘录和 `artifact_ref`。这个替换不回写 Session。失败、验证、工作区变更、最新未消费工具批次以及仍无替代覆盖的唯一 read 正文不可这样删除。

#### Artifact

Artifact 保存完整原始输出，建议位于：

```text
.codepilot/runs/<run_id>/artifacts/tool_outputs/
```

建议文件名：

```text
<tool_call_id>_<content_hash>.txt
```

写入顺序：

```text
Artifact
-> ToolResult Message
-> Run Checkpoint
-> Session navigation
```

Artifact 写入失败时不能提交一个引用不存在原文的 ToolResult。

#### Evidence

L2 Evidence 是 ToolResult 的派生状态视图。第一版只注入需要脱离消息正文持续可见的三类：

```text
mutation
verification
error
```

概念结构：

```python
ContextEvidence(
    evidence_id,
    kind,
    summary,
    source_message_id,
    source_tool_call_id,
    artifact_ref,
    affected_paths,
    status,
    source_hash,
    workspace_fingerprint,
    freshness,
    created_at,
)
```

### 7.2 Artifact 触发规则

- 每种工具先在执行层限制单次结果；read 还限制行数、字符数和并行批次总字符数。
- 限额内文本完整保留在 Session ToolResult。
- 只有 `tight` 请求级瘦身真正省略旧输出时，才写 Artifact 并在该次请求中使用引用。
- 二进制或 Provider 不可直接消费的内容只保存 Artifact 和描述。
- 文件修改结果优先保留 affected paths、diff summary 和文件 Hash。
- 验证结果优先保留命令、退出码、passed/failed、失败摘要和 Workspace fingerprint。

每次工具调用不使用辅助 LLM 生成摘要。第一版使用工具特定的确定性 Renderer：

- 测试：passed/failed、失败用例和 assertion。
- Shell：exit code、stderr 和关键首尾行。
- 文件读取：路径、行范围和截断状态。
- 搜索：匹配数量、文件列表和截断状态。
- 编辑：affected paths、diff summary 和文件 Hash。

辅助 LLM 只负责后续语义压缩。

### 7.3 Artifact 读取

模型需要重新查看完整 Artifact 时，应通过受控只读能力按 `artifact_id + offset + limit` 读取。不能允许模型直接访问整个 `.codepilot/` 内部目录。

## 8. Evidence Freshness

第一版使用简单失效规则：

- 文件证据绑定文件 Hash；Hash 改变后变为 `stale`。
- 验证证据绑定 Workspace fingerprint。
- 任意 Workspace mutation 后，旧验证结果立即变为 `stale`。
- 文件删除后，相关证据变为 `missing`。
- 无法绑定来源 Hash 的只读观察标记为 `unknown`。

排序：

```text
fresh > unknown > stale > missing
```

stale 信息可以作为提醒进入 Context，但不能继续作为“验证已经通过”的依据。

工具输出属于已观察但不可信的外部内容：

```text
trust = observed_untrusted
```

工具输出中的命令式文字、网页提示或源码注释只能作为数据，不能进入 L0/L1。

## 9. L2 与 L4 一致性

### 9.1 单一事实来源

```text
Session ToolResultMessage 是事实来源。
L2 Evidence 是错误、变更和验证的派生状态视图。
L4 ProjectedMessage 是接近无损的本次请求消息视图。
```

引用关系：

```text
ToolResultMessage(message_id, tool_call_id, artifact_ref)
        |
        v
EvidenceRecord(source_message_id, source_tool_call_id, source_hash)
```

L2 和 L4 不能分别维护两份没有来源关系的工具事实。

### 9.2 统一投影计划

每次准备 Context 时，使用同一份 Workspace/Freshness 快照构建 L2 状态和 L4 消息。Projector 只做协议修复、compact cursor 应用、完全相同 read 的整批去重和 stale 警告；压力驱动的输出省略由独立 Thinner 处理：

```python
ContextProjectionPlan(
    message_actions,
)
```

典型配对：

| L4 Message | L2 Evidence |
|---|---|
| keep_full | 只显示状态，不重复正文 |
| keep_projected | 仅用于完全相同 read 的去重标记 |
| request_thinned | L2 保留必要状态，L4 显示摘录和 Artifact 引用 |
| covered_by_compact_summary | 仅保留仍然 fresh 且当前相关的 Evidence |
| stale | 保留历史正文并添加失效警告，不能作为当前通过依据 |

例如历史消息表示 `pytest passed`，但后续文件已经修改，则 L4 投影必须显示该结果属于历史 Workspace，L1 当前 verification status 必须变为 `stale` 或 `required`。

## 10. Token 计量与输入预算

### 10.1 有效输入预算

```text
effective_input_budget
  = model_context_window
  - output_reserve
  - safety_margin
```

- `output_reserve` 使用本次请求配置的最大输出 token。
- `safety_margin` 第一版可以使用 `max(1024, context_window * 5%)`，并配置合理上限。

### 10.2 计量范围

必须按最终物理请求计量：

```text
L0 system prompt
L1 dynamic context
L2 working set and evidence
L3 memory recall
L4 messages
Tool schemas
Images
Provider message overhead
```

第一版继续使用近似 estimator，并允许根据 Provider 返回的实际 `input_tokens` 校准估算因子；不要求 Provider 精确计数 API。

### 10.3 压力等级

| 等级 | Raw Context / Effective Budget | 行为 |
|---|---:|---|
| normal | `< 60%` | 保留接近无损的上下文视图 |
| tight | `60% - 80%` | 确定性瘦身和低价值裁剪，目标回落到 55% |
| critical | `80% - 100%` | 触发辅助 LLM 压缩并重新物化 |
| overflow | `> 100%` | 禁止普通模型调用，必须压缩或失败 |

阈值属于配置默认值，不应散落在选择算法中。

### 10.4 预算分配顺序

先支付固定成本：

```text
L0 全部
Tool schemas
L1 最小骨架
L4 当前用户消息
当前未闭合合法工具组
```

再支付 Protected 内容：

```text
当前目标和步骤
最新 unresolved error
affected paths
最新 verification
Compact Summary
```

最后对剩余预算使用默认权重：

| 内容 | 剩余预算权重 |
|---|---:|
| L1 任务详情 | 10% |
| L2 工作集与工具证据 | 35% |
| L3 记忆召回 | 10% |
| L4 历史对话 | 45% |

某层未使用的预算可以被其他层借用。层内按以下顺序选择：

```text
retention class
-> relevance
-> freshness
-> recency
```

### 10.5 单项上限

第一版默认建议：

```text
read result:
  <= 30,000 chars and <= 1,000 lines

parallel read batch:
  <= 100,000 chars

recent consumed tool protection under tight pressure:
  min(40,000 tokens, effective_budget * 20%)

single evidence summary:
  <= 512 tokens

single memory:
  <= 600 tokens

L3 total:
  <= effective budget * 10%
```

数值后续通过评测调整。

### 10.6 最终硬校验

```text
Raw estimate
-> selection / thinning / compaction
-> final estimate
-> Provider preflight
-> final estimate <= effective_input_budget
```

最终请求仍然超限时依次执行：

1. 丢弃 DiscardFirst。
2. 缩减 Budgeted。
3. 触发语义压缩。
4. 仍无法满足时返回结构化错误。

不能将已知超限请求交给 Provider 后再依赖 `prompt_too_long`。

## 11. 确定性瘦身

确定性瘦身在 `tight` 压力下执行，不调用 LLM。

### 11.1 L1-L3 处理

- 删除重复运行状态和重复 L2 投影。
- 按保留等级丢弃低价值 L2/L3 条目；stale 本身不等于正文不可见。
- 将旧且可恢复的工具结果降为确定性摘录和 Artifact 引用。
- 减少低相关 Memory 的注入。
- 不生成新的 Compact Summary，不移动 compact cursor。

### 11.2 L4 处理

Artifact 化不能完全解决长聊天问题。用户与助手文本、大量短 ToolResult 和多轮工具组仍会持续增长。

L4 可以安全执行：

- 排除当前 compact cursor 之前、已经由 Compact Summary 覆盖的消息。
- 按 `path + hash + line range` 删除完全相同的旧 read；只按路径或 stale 状态不能删除。
- 将旧的成功 shell/search/list 类输出替换为确定性摘录和 Artifact 引用。
- 仅当较新的 read 区间完整覆盖旧区间时，才允许省略旧 read 正文。
- 为 stale read 保留正文并添加历史状态警告。
- 对 ToolCall 及其全部 ToolResult 按完整并行批次保护和裁剪。

L4 不能确定性执行：

- 直接删除普通 UserMessage。
- 直接删除包含决策的 AssistantMessage。
- 不生成摘要而只保留最近 N 条消息。
- 仅假设 L2 已经保存语义而删除整个工具组。

涉及用户意图、决策和任务过程的语义删除只能由辅助 LLM 压缩完成。

### 11.3 L4 压力信号

审计报告同时记录原始、无损投影和最终请求压力：

```text
conversation_pressure
  = L4 tokens / effective_input_budget
```

当前实现只在无损投影达到 `critical` 或 `overflow` 时执行 LLM 压缩；`tight` 不提前调用摘要模型。

## 12. 辅助 LLM 语义压缩

### 12.1 触发条件

- 总压力进入 `critical`。
- `overflow` 恢复流程要求强制压缩。

### 12.2 可压缩范围

压缩对象只能是 L4 的合法历史前缀：

```text
已有 Compact Summary
+ 上次 cursor 后的早期完整消息组
```

不能进入压缩区：

- 当前用户消息。
- 未闭合 tool call/result。
- 当前 waiting / approval 请求。
- 最新 unresolved error。
- 保留窗口中的近期消息。

当前保留最近约 15% 的有效输入预算，最少 512、最多 8,000 token，并始终按完整 API round 确定边界。

### 12.3 Context Summarizer

```python
class ContextSummarizer(Protocol):
    async def summarize(request: CompactionRequest) -> CompactSnapshot:
        ...
```

第一版约束：

- 单轮模型调用。
- 禁止工具。
- 禁止写 Memory。
- 使用专用摘要 Prompt。
- 限制最大输出 token。
- 格式错误最多重试一次。
- 摘要未达到最低压缩率时视为失败。

### 12.4 Summary 结构

辅助 LLM 输出结构化摘要：

```python
CompactSummary(
    original_goal,
    user_constraints,
    decisions,
    completed_work,
    files_and_symbols,
    important_evidence,
    errors_and_resolutions,
    verification_state,
    open_questions,
    next_actions,
    source_refs,
)
```

要求：

- 用户约束和关键决策保留来源引用。
- 不复制长代码和原始工具输出。
- 验证信息包含 Workspace/source 引用。
- 未验证模型判断明确标记。
- 输出经过 schema 和长度校验。

### 12.5 滚动压缩

```text
旧 Compact Summary
+ cursor 后新增的可压缩消息
-> 新 Compact Summary
-> cursor 前移
```

模型可见上下文只保留最新摘要。结构化字段、source refs 和当前事实优先原则用于降低 summary-of-summary 漂移。

### 12.6 压缩失败

- 存在旧摘要：继续使用旧摘要并缩短近期 Budgeted 尾部。
- 不存在旧摘要：返回 `context_compaction_failed`。
- 不使用消息计数或工具名列表冒充高质量语义摘要。

## 13. Compact Snapshot

### 13.1 Artifact

Compact Summary 保存为非权威 Context Artifact：

```text
.codepilot/runs/<run_id>/artifacts/context/<compact_id>.json
```

概念结构：

```python
CompactSnapshot(
    compact_id,
    session_id,
    run_id,
    compacted_until_message_id,
    source_start_message_id,
    source_digest,
    summary,
    estimated_tokens_before,
    estimated_tokens_after,
    created_at,
)
```

完整 Session Messages 不删除。模型阅读窗口变为：

```text
L0-L3
+ Compact Summary Attachment
+ compacted_until_message_id 之后的完整消息组
```

Compact Summary 是 L4 的逻辑内容，但以动态 Context Attachment 渲染，不伪装成真实 UserMessage。

### 13.2 活动 Run Checkpoint

Context component 只保存最小引用：

```json
{
  "compact_snapshot_ref": "...",
  "compacted_until_message_id": "msg_123",
  "summary_digest": "..."
}
```

恢复活动 Run 时：

1. 加载 Context component。
2. 校验 Artifact 存在。
3. 校验 cursor 属于当前消息链。
4. 校验 source digest。
5. 校验失败时忽略 Snapshot，重新物化和压缩。

### 13.3 跨 Run 复用

Run 终态会清除 Checkpoint。为了避免进程重启或新 Run 后无条件重新压缩，可以同时写非权威 `context_compacted` Event：

```text
compact_snapshot_ref
cursor
source_digest
compression metrics
```

新 Run 可以尝试读取最近一次有效记录。Event 或 Artifact 丢失不能破坏 Session 恢复，只会导致 Context 从完整 Messages 重新构建并再次压缩。

这保持了以下边界：

- Compact Snapshot 不是 Session 权威状态。
- Event 不参与 Run 恢复决策。
- Summary 不进入长期 Memory。
- 所有派生内容都可以从完整事实重新生成。

## 14. Context 生命周期

### Session 打开

```text
加载 L0
-> 加载 Session Messages
-> 尝试读取最近有效 Compact Snapshot
-> 重建 Working Set 和 Evidence
```

### Run 开始

```text
读取 Core State
-> 组装 L1
-> 请求 Memory Recall
-> 形成初始五层 Context
```

### 每次模型调用前

```text
刷新 L1/L2/L3/L4
-> 估算压力
-> 确定性瘦身
-> 必要时 LLM 压缩
-> 统一生成 L2/L4 Projection Plan
-> Provider 预检
-> 硬预算校验
```

### ToolResult 后

```text
Artifact 写入
-> ToolResult 提交
-> Evidence 更新
-> Workspace freshness 失效
-> 下一次模型调用重新物化
```

### Resume

```text
Session 恢复权威 Run/Checkpoint/Messages
-> Context 校验 Compact Snapshot
-> 重建 Workspace 和 Evidence
-> 当前事实覆盖 stale Summary
-> 重新物化
```

### Run 结束

```text
Session 提交终态
-> Compact Artifact 和 Event 保留为可选派生工件
-> 不将 Summary 写入 Memory
```

## 15. 第一版不实现

- Provider `cache_edits`。
- 基于时间的 Prompt cache 微压缩。
- 静态前缀跨组织缓存。
- Prompt section cache key。
- 路径条件规则。
- 后台 Session Memory 进程。
- 分块或多级 Map-Reduce 摘要。
- 自动源码语义检索和向量数据库。
- 自动将 Compact Summary 转为长期 Memory。
- ContextReport 完整审计协议和评测体系。

## 16. 设计不变量

- L0 已生效内容永不被预算裁剪。
- 当前用户消息永不被普通裁剪。
- Tool call/result 不被切断。
- 长输出必须先成功写 Artifact，再提交引用。
- L2 与 L4 使用同一来源引用和同一 freshness 快照。
- stale verification 不能继续表示当前验证通过。
- 语义历史只能通过辅助 LLM 摘要后移出模型窗口。
- 最终请求超过 effective input budget 时禁止调用普通模型。
- Compact Snapshot 丢失不能破坏 Session 恢复。
- Compact Summary 永远不自动进入长期 Memory。

## 17. 实施规格

Context 与 Memory 的 Port、目标文件布局、四阶段迁移顺序、测试矩阵和验收标准见 [8context-memory-implementation.md](./8context-memory-implementation.md)。
