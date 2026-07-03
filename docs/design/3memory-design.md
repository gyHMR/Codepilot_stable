# Codepilot 记忆模块设计

本文说明当前代码中的 Memory v2。它的核心边界很简单：Memory 只保存跨任务可复用的长期知识；当前任务进展、工具日志、文件摘要、新鲜度和临时错误输出都不进入 Memory。

从 Agent 运行流程看，Memory 参与三个时机：

1. run 开始前：检查用户输入是否包含明确长期记忆意图。
2. 每次模型调用前：根据当前任务、活跃文件、错误信号和对话内容召回长期记忆，交给上下文治理链路组装进 prompt。
3. run 结束后：从“失败 -> 修复 -> 验证通过”的闭环中提炼经验型记忆。

## 0. 一句话概述

Codepilot 的记忆系统不是聊天历史缓存，也不是工具日志仓库，而是一套长期知识管理机制：它把用户手写规则、用户纠正、项目约束、设计决策和已验证修复经验保存成结构化 `MemoryRecord`；在每次模型调用前由 `MemoryRetriever.recall()` 按当前任务信号召回；在 run 结束后由 `MemoryWriter.finalize_run()` 从可验证闭环中沉淀经验。

核心代码位置：

| 代码位置 | 职责 |
|---|---|
| `src/codepilot/sessions/memory/records.py` | 定义 `MemoryRecord`、`MemoryQuery`、`MemoryRecall`、`RetrievedMemory` |
| `src/codepilot/sessions/memory/store.py` | 读写 session/project memory 文件 |
| `src/codepilot/sessions/memory/writer.py` | 处理用户输入准入、命令添加、run 结束经验沉淀、手动提升 |
| `src/codepilot/sessions/memory/experience.py` | 从工具结果中识别失败-修复-验证闭环，并合并/提升经验 |
| `src/codepilot/sessions/memory/retriever.py` | 按当前任务信号召回长期记忆 |
| `src/codepilot/sessions/memory/files.py` | 读取 `.codepilot/MEMORY.md`，并对记忆文本做脱敏截断 |
| `src/codepilot/sessions/context/governor.py` | 每次模型调用前调用 `MemoryRetriever.recall()` |
| `src/codepilot/sessions/session.py` | run 生命周期中调用 prompt memory admission 和 finalize memory |
| `src/codepilot/interfaces/cli/commands.py` | `/memory` 命令 |
| `src/codepilot/runtime/service.py` | `get_memory_state()` 只读状态接口 |

## 1. 文件和职责边界

Memory 使用三层文件，不为每个中间过程单独建 ledger。

| 文件 | 谁写 | 什么时候生成 | 内容 |
|---|---|---|---|
| `.codepilot/MEMORY.md` | 用户手动维护 | 用户创建时 | pinned memory。自动流程只读不写 |
| `.codepilot/memory/project.jsonl` | `MemoryStore` | 首次写 project memory | 项目级长期记忆：correction、constraint、decision、promoted experience |
| `.codepilot/sessions/<id>/memory.json` | `MemoryStore` | 首次写 session memory | 本 session 中验证过、但还未提升为 project 的 experience |
| `.codepilot/sessions/<id>/events.jsonl` | `SessionStore` | 首次 session event | `memory_updated`、`memory_retrieved`、`memory_warning` 等事件 |

不会进入 Memory 的内容：

- 当前任务进度：归 `TaskRecoveryStore`。
- 工具原始输出：归 transcript、run result、tool artifact。
- 文件摘要和 active files：归 context state。
- 文件新鲜度和验证是否过期：归 context/run freshness。
- 单次失败日志：归 run evidence，不直接沉淀。

## 2. MemoryRecord 类型

所有自动写入的结构化记忆都使用 `MemoryRecord` v2。

固定字段：

```python
id: str
scope: "session" | "project"
kind: "correction" | "constraint" | "decision" | "experience"
status: "active" | "superseded" | "deleted"
key: str
text: str
triggers: list[str]
related_paths: list[str]
evidence_refs: list[str]
source: "user" | "command" | "run" | "promoted"
supersedes: list[str]
occurrences: int
created_at: str
updated_at: str
```

字段含义：

- `scope` 决定生命周期：`session` 只属于当前会话；`project` 对整个项目可用。
- `kind` 决定管理方式和召回优先级。
- `status` 决定是否可召回：只有 `active` 会进入召回候选。
- `key` 是合并和冲突键，例如 `constraint:project_boundary`、`experience:edit:multiple_matches`。
- `text` 是可以直接渲染给模型的正文。
- `triggers` 是召回触发器，例如 `always`、`topic:context`、`intent:edit_file`、`error:multiple_matches`、`path:src/app.py`。
- `evidence_refs` 只保存短引用，例如 `tool:<id>`、`verification:<id>`、`run:<id>`，不保存大段日志。
- `occurrences` 表示同类经验被验证出现的次数。

旧字段和旧类型已经不属于新契约：`task/file/failure/project` kind、`content` 字段、`trust` 字段、`stale` 状态都不是 Memory v2。

## 3. Agent 运行链路中的 Memory

### 阶段一：创建会话时初始化 memory 组件

`AgentSession.__init__()` 创建：

- `MemoryStore`
- `MemoryWriter`
- `MemoryRetriever`

`MemoryStore` 持有当前 session 的 `memory.json` 路径和项目级 `project.jsonl` 路径。此时不会创建空 memory 文件，只有第一次写入时才会懒创建。

### 阶段二：run 开始前检查用户输入是否应该记住

每次非 continue 的 `AgentSession.run()` 开始时，会进入 `_start_run_lifecycle()`。

如果 `memory_enabled=True`，session 会调用：

```python
self._admit_prompt_memory(text, run_id=run_id)
```

内部再调用：

```python
MemoryWriter.admit_prompt_memory(text, run_id=run_id)
```

这一步只处理用户输入中“明确应该长期记住”的内容。

会写入 Memory 的输入：

| 用户表达 | 写入类型 | scope | 说明 |
|---|---|---|---|
| “请记住...”“以后...”“remember...” | `constraint` | `project` | 显式长期约束或偏好 |
| “纠正一下...”“不是 X 而是 Y”“以后不要...” | `correction` | `project` | 用户纠正，优先级最高 |
| “这是学生学习/求职展示项目，不要做生产级复杂设计” | `constraint` | `project` | 固定项目边界约束 |

不会写入 Memory 的输入：

- 普通任务请求，例如“修复 service.py 并运行测试”。
- 普通代码阅读请求。
- 普通报错描述。
- 单次工具失败。

准入判断在 `decide_prompt_memory_admission()`：

- 先调用 `sanitize_memory_text(text, limit=1200)` 脱敏和截断。
- 如果是 correction，生成 `kind="correction"`。
- 如果是 explicit memory，生成 `kind="constraint"`，并带 `always` trigger。
- 如果是项目边界约束，写固定 `constraint:project_boundary`。
- 其他输入返回 `should_store=False`。

如果成功写入，`AgentSession._admit_prompt_memory()` 会写一条 `memory_updated` session event。

### 阶段三：每次模型调用前召回 Memory

Memory 的召回不是 run 开始时做，而是在每次模型调用前由上下文治理链路触发。

调用链：

```text
LLMStreamRunner.stream_assistant_response()
  -> AgentOptions.prepare_context
  -> ContextGovernor.prepare()
  -> ContextGovernor._recall_memory()
  -> MemoryRetriever.recall(query)
```

`ContextGovernor._recall_memory()` 会构造 `MemoryQuery`：

```python
MemoryQuery(
    text=latest_user_text(context.messages),
    active_paths=sorted(self.state.active_files),
    task_phase=optional_signal(context, "phase"),
    action_intent=optional_signal(context, "action_intent"),
    recent_error=optional_signal(context, "recent_error_code"),
    retrieval_mode=context_mode(context),
)
```

也就是说，召回不是只看用户最后一句话，还会看：

- 当前 active files
- 当前任务阶段
- 当前动作意图
- 最近错误码
- 当前模式：`repair / verify / qa / act`

`MemoryRetriever.recall()` 返回 `MemoryRecall`：

```python
pinned_text: str
always: list[RetrievedMemory]
selected: list[RetrievedMemory]
dropped: dict[str, str]
```

召回结果再由 `ContextGovernor._render_recalled_memory()` 写入上下文的 Memory Recall 区域：

```text
[Pinned memory] ...
[Correction] ... [reasons=...]
[Constraint] ... [reasons=...]
[Decision] ... [reasons=...]
[Experience] ... [reasons=...]
```

如果本轮有结构化 memory record 被召回，`context_prepared` 事件中的 `ContextReport` 会包含：

- `retrieved_memory_ids`
- `memory_retrieval_reasons`

`AgentSession._on_agent_event()` 看到这些 id 后，会追加一条 `memory_retrieved` session event。

### 阶段四：工具执行过程中不沉淀长期记忆

工具结果消息结束时，`AgentSession._on_agent_event()` 会调用：

```python
self._observe_tool_memory(message, run_id=...)
```

但当前 `MemoryWriter.observe_tool_result()` 只是保留扩展点，直接返回空列表。

原因是：工具输出、临时错误、文件摘要、新鲜度和验证证据都属于 context/run 层。Memory 不在工具执行中途立刻写经验，避免单次失败把长期记忆污染成日志堆。

### 阶段五：run 结束后沉淀经验

`AgentSession._complete_run_lifecycle()` 在 run 结果落盘、task recovery 更新后，如果启用 memory，会调用：

```python
self._finalize_memory(result)
```

内部调用：

```python
MemoryWriter.finalize_run(result)
```

`MemoryWriter.finalize_run()` 使用 `ExperienceExtractor.extract(result)` 从 `AgentRunResult.messages` 中查找可复用经验。

当前只有两类确定性经验规则。

第一类：edit 修复经验。

满足条件：

1. 出现 `tool_name == "edit"` 的失败工具结果。
2. `error_code` 是 `multiple_matches` 或 `unexpected_match_count`。
3. 后续出现成功的 edit。
4. 后续出现 verification passed。

生成的 key：

```text
experience:edit:<error_code>
```

生成的 text 大意是：当 edit 因 old_text 不唯一或匹配次数异常失败时，先 read 目标区域，再用更长且唯一的 old_text 或 occurrence_index 重试。

触发器包括：

- `phase:repair`
- `intent:edit_file`
- `error:<error_code>`
- `path:<affected_path>`

第二类：verification 修复经验。

满足条件：

1. 出现 verification failed。
2. 后续出现 verification passed。

生成的 key：

```text
experience:verification:failed_then_passed
```

生成的 text 大意是：验证失败后先读 failure summary 和相关文件，做最小修复，再 rerun 同一个验证命令。

触发器包括：

- `phase:repair`
- `intent:debug_failure`
- `error:verification_failed`
- `path:<affected_path>`

经验沉淀默认写入 session memory，因为单次经验还不一定值得成为项目级知识。

## 4. 召回规则

召回分为四层，顺序固定。

### pinned memory

`.codepilot/MEMORY.md` 是用户手写固定记忆。`MemoryRetriever.pinned_memory()` 每次 recall 都会动态读取它：

- 通过 `load_global_memory(workspace_dir)` 读取。
- 通过 `sanitize_memory_text(..., limit=2000)` 脱敏并截断。
- 不把 pinned memory 伪装成 `MemoryRecord`。

pinned memory 总是进入上下文，只是不出现在 `retrieved_memory_ids` 中。

### correction

active correction 总是进入 `MemoryRecall.always`，分数固定为 1000，原因是 `layer:correction`。

correction 用于用户纠正，优先级高于 constraint、decision、experience。

### always constraint

带 `always` trigger 的 constraint 会进入 `always`。常见来源：

- 用户显式“请记住/以后...”。
- 项目边界约束。
- `/memory add` 添加的普通 constraint。

召回原因里会包含 `layer:always_constraint`，并且可能同时包含 `trigger:always`、`scope:project` 等。

### selected memory

其他 active record 进入候选池后由 `score_memory_record()` 打分。

得分来源：

| 信号 | 规则 |
|---|---|
| `trigger == "always"` | 加分并记录 `trigger:always` |
| `path:<path>` | 如果命中 active_paths，加分并记录 `path:<path>` |
| `topic:<topic>` | 如果 topic 出现在 query text 中，加分并记录 `topic:<topic>` |
| `phase:<phase>` | 如果命中 task_phase，加分 |
| `intent:<intent>` | 如果命中 action_intent，加分 |
| `error:<error>` | 如果命中 recent_error，加分 |
| keyword | query text 和 rendered memory 有词项交集时加分 |
| project scope | project memory 有轻微加分 |
| decision | `qa/plan/design` 模式下加分 |
| constraint | `qa/plan/design` 或 keyword 命中时加分 |
| experience | `repair/verify` 模式、错误码、动作意图、occurrences 会加分 |

选择数量限制：

- constraint 最多 3 条。
- decision 最多 2 条。
- experience 最多 3 条。
- 总数受 `MemoryQuery.limit` 限制，默认 8。

最终 selected 输出顺序是：

```text
constraint -> decision -> experience
```

### dropped

如果 record 不是 active，例如 `deleted` 或 `superseded`，不会进入召回候选，而是进入 `MemoryRecall.dropped`：

```text
memory_id -> status:deleted
memory_id -> status:superseded
```

## 5. 写入、合并、冲突和提升

### 同 key 合并

`MemoryConsolidator.upsert_project_record()` 会先查找同 key、同 kind、active 的 project record。

如果存在：

- 更新 `text`
- 合并 `triggers`
- 合并 `related_paths`
- 合并 `evidence_refs`
- 合并 `supersedes`
- 增加 `occurrences`

这样重复 `/memory add` 或重复明确约束不会不断追加垃圾记录。

### correction 冲突处理

如果新 record 是 `correction`，`_active_conflicts()` 会找出同 key 的其他 active kind：

- constraint
- decision
- experience

这些旧记录会被标记为 `superseded`，并加入新 correction 的 `supersedes`。

同 key correction 之间则走同 key 合并，不重复追加。

### experience 合并

`MemoryConsolidator.upsert_experience()` 只在 session memory 中查找同 key active experience。

如果存在：

- `occurrences += 1`
- 合并 triggers
- 合并 related_paths
- 合并 evidence_refs

如果不存在，则创建新的 session experience。

### 自动提升

当 session experience 的 `occurrences >= 2`，`MemoryConsolidator` 会自动调用 `_promote_experience()`。

提升结果：

- 新建或合并 project experience。
- `source="promoted"`。
- `supersedes=[session_record.id]`。
- 追加 `run:<run_id>` 证据引用。

当前实现不会把 session record 标记为 deleted；它仍保留在 session memory 中，project memory 保存被提升后的跨任务经验。

### 删除

`/memory forget <id>` 调用：

```python
MemoryStore.mark_status(memory_id, "deleted")
```

这是逻辑删除，不物理删除文件内容。deleted record 不再召回，但仍保留审计痕迹。

## 6. CLI 和 API 管理面

CLI 命令在 `src/codepilot/interfaces/cli/commands.py`。

### `/memory`

展示摘要：

- pinned chars
- session active
- project active
- superseded
- deleted

### `/memory list [session|project|correction|experience|deleted]`

读取 session + project memory，按 scope、kind 或 status 过滤。显示格式：

```text
<id> [<scope>/<kind>/<status>] <rendered memory>
```

### `/memory add <text>`

调用 `MemoryWriter.add_project(text)`：

- 默认写 project `constraint`。
- 如果以 `decision:`、`Decision:`、`决策：`、`决策:` 开头，写 project `decision`。
- constraint 默认带 `always` trigger。
- 写入后追加 `memory_updated` 事件。

### `/memory promote <id>`

调用 `MemoryWriter.promote(id)`：

- 只允许 active session experience。
- 输出 project experience。
- 写入后追加 `memory_updated` 事件。

### `/memory forget <id>`

调用 `MemoryStore.mark_status(id, "deleted")`。

`RuntimeService.get_memory_state(session_id)` 返回只读状态：

- pinned 文件路径、字符数、预览。
- session records。
- project records。
- session/project active、deleted、superseded 计数。

## 7. 与 Context、TaskRecovery、RunStore 的边界

### 与 Context

Context 负责本轮 prompt 编排。Memory 只提供 `MemoryRecall`，不决定 token 预算、不裁剪 recent messages、不写 checkpoint。

ContextGovernor 会把 recalled memory 变成 prompt 中的 Memory Recall 区域，并把召回 id/reason 写入 `ContextReport`。

### 与 TaskRecovery

TaskRecovery 保存当前任务目标、步骤、下一步动作。它用于“当前 session 如何继续任务”，不是长期可复用知识。

普通用户任务不会写成 memory；run 开始时 `_begin_task_recovery()` 会处理任务恢复，`_admit_prompt_memory()` 只处理明确长期记忆意图。

### 与 RunStore

RunStore 保存完整 run 结果和工具证据。MemoryWriter 在 run 结束后读取 `AgentRunResult`，只提取少量经过验证的经验。

Memory 不复制 run 里的大段工具输出，只保存 `evidence_refs`。

## 8. 当前没有做什么

代码中未发现以下能力的明确实现：

- 没有 embedding 或向量数据库。
- 没有 LLM 自动总结任意历史并写入 memory。
- 没有置信度学习或复杂评分模型。
- 没有从单次失败中直接沉淀经验。
- 没有把文件摘要、工具日志或 task progress 写入 memory。
- 没有自动写 `.codepilot/MEMORY.md`；它是用户手写 pinned memory。
- `validate_freshness()` 当前返回空列表，Memory 本身不做文件新鲜度判断；新鲜度属于 context/run。

这些取舍是为了让学习型项目的记忆系统保持可解释：什么时候写、为什么写、什么时候召回，都能直接从代码规则看出来。

## 9. 可以写进项目介绍的总结

针对 Coding Agent 在多轮代码任务中容易把临时日志、当前任务进展和长期经验混在一起的问题，Codepilot 设计了分层长期记忆机制：用户手写 `.codepilot/MEMORY.md` 作为最高优先级 pinned memory，用户纠正和项目约束写入 project memory，经过“失败 -> 修复 -> 验证通过”闭环的经验先写入 session memory，并在重复验证后提升为 project experience。每次模型调用前，系统根据当前用户请求、active files、任务意图和错误信号召回 correction、constraint、decision 和 experience，并把召回原因写入 ContextReport，从而让模型获得可复用的长期知识，同时避免工具日志、旧任务状态和单次失败污染记忆库。
