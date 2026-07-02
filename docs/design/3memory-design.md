# Codepilot 记忆模块设计

## 1. 设计结论

Codepilot 的 Memory 只负责长期可复用知识，不保存当前任务进展、工具日志、文件摘要、新鲜度或临时错误输出。它在 sessions 四域中的位置是：

| 子域 | 职责 | Memory 耦合点 |
|---|---|---|
| Session 持久化 | `session.json/messages.jsonl/events.jsonl` 是会话事实源 | 只托管 `sessions/<id>/memory.json` 和 `memory_*` 事件 |
| Run 持久化 | `run.json/events.jsonl` 保存一次运行结果、工具证据、rollback metadata | run 完成后由 `MemoryWriter.finalize_run()` 读取结果提炼经验 |
| ContextGovernor | 每轮把 session/run/context/memory 投影成 prompt | 调用 `MemoryRetriever.recall()`，不改写 memory |
| TaskRecovery/GitRollback | 当前任务恢复、轻量 git 回退 | TaskRecovery 不进入长期 memory，只给经验提取提供信号 |
| Memory | 用户手写记忆、纠正/约束、项目决策、失败成功经验 | 输出 recalled memory 层给 ContextGovernor |

核心代码位置：

- `src/codepilot/sessions/memory/records.py`：`MemoryRecord`、`MemoryQuery`、`MemoryRecall`、`RetrievedMemory`。
- `src/codepilot/sessions/memory/store.py`：`MemoryStore` 读写 session/project memory。
- `src/codepilot/sessions/memory/writer.py`：`MemoryWriter` 写入准入、用户纠正、命令添加、run 经验沉淀。
- `src/codepilot/sessions/memory/experience.py`：`ExperienceExtractor` 和 `MemoryConsolidator`。
- `src/codepilot/sessions/memory/retriever.py`：`MemoryRetriever.recall()` 和 `score_memory_record()`。
- `src/codepilot/sessions/context/governor.py`：ContextGovernor 将 memory recall 注入 `ContextView.recalled_memory`。

## 2. 文件契约

默认不创建空 memory 文件，只有首次写入时懒创建：

| 路径 | Owner | 说明 |
|---|---|---|
| `.codepilot/MEMORY.md` | 用户手动维护 | pinned memory，自动流程只读不写，召回时最高优先级进入 |
| `.codepilot/memory/project.jsonl` | `MemoryStore` | 项目级长期记忆：correction、constraint、decision、promoted experience |
| `.codepilot/sessions/<id>/memory.json` | `MemoryStore` | session 级长期记忆暂存，主要保存未提升的 verified experience |
| `.codepilot/sessions/<id>/events.jsonl` | `SessionStore` | `memory_updated/memory_retrieved/memory_warning` 事件 |

不再使用旧 schema：`task/file/failure/project` kind、`content` 字段、`trust` 字段、`stale` 状态都不是 Memory v2 的公共契约。

## 3. MemoryRecord v2

`MemoryRecord` 固定字段在 `src/codepilot/sessions/memory/records.py`：

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

`text` 是可直接渲染给模型的正文；`key` 是合并/冲突键；`triggers` 是召回触发器，例如 `always`、`topic:context`、`intent:edit_file`、`error:multiple_matches`、`path:src/app.py`。

## 4. 写入规则

### 用户输入

`MemoryWriter.admit_prompt_memory()` 只接收明确长期意图：

- “请记住 / 以后 / remember”等显式记忆请求写 `constraint`。
- “纠正 / 更正 / 不是 X 而是 Y”等写 `correction`。
- 项目边界类强约束写 `constraint:project_boundary`。
- 普通任务、普通读代码、单次错误描述不写 memory。

用户纠正默认写 project scope。`correction` 会 supersede 同 key 的 constraint/decision/experience；同 key 同 kind 的 active project memory 会合并，不追加重复记录。

### 命令写入

`/memory add <text>` 调用 `MemoryWriter.add_project()`，默认写 project `constraint`；如果文本显式表达“决定/decision”，写 `decision`。

`/memory promote <id>` 只允许把 session `experience` 提升成 project `experience`。

`/memory forget <id>` 不物理删除，标记为 `deleted`，后续不召回。

### Run 结束经验

`MemoryWriter.finalize_run()` 只从“失败工具结果 -> 成功修复 -> 验证通过”的闭环提炼 `experience`。当前确定性规则包括：

- edit 因 `multiple_matches` 或 `unexpected_match_count` 失败，后续 edit 成功，并且后续 verification passed。
- verification failed 后，后续 verification passed。

experience 默认写 session memory。同 key 重复经验会合并 `triggers/related_paths/evidence_refs` 并增加 `occurrences`；当 `occurrences >= 2` 且有验证证据时，自动提升为 project experience。

## 5. 召回规则

`MemoryRetriever.recall(query)` 返回 `MemoryRecall`：

```python
pinned_text: str
always: list[RetrievedMemory]
selected: list[RetrievedMemory]
dropped: dict[str, str]
```

召回顺序固定：

1. `.codepilot/MEMORY.md` pinned memory：总是进入，但会脱敏并截断。
2. active correction：总是进入 `always`。
3. always constraint：带 `always` trigger 的约束进入 `always`。
4. decision：在 `qa/plan/design` 或 topic/keyword 匹配时优先进入。
5. experience：仅在 `repair/verify` 或 error/intent/path/keyword 匹配时进入，默认最多 3 条。

`superseded/deleted` 不召回，只在 `MemoryRecall.dropped` 中记录原因。

## 6. 与上下文治理的关系

`ContextGovernor.prepare()` 不再手动拼 pinned memory，也不调用旧 `retrieve()` 主路径，而是调用 `MemoryRetriever.recall()`。随后把：

- pinned memory 渲染为 `[Pinned memory] ...`
- correction/constraint/decision/experience 渲染为对应标签
- 召回原因写入 `ContextReport.memory_retrieval_reasons`
- selected/always 的 record id 写入 `ContextReport.retrieved_memory_ids`

Memory 是 `ContextView.recalled_memory` 的来源之一，不是 prompt 历史，也不是 checkpoint 事实源。critical pressure 下 memory 仍通过 ContextView 投影，不写入 checkpoint 作为长期事实。

## 7. 事件与接口

Session events 使用既有事件类型：

- `memory_retrieved`：一次 context prepare 中有 record 被召回。
- `memory_updated`：用户提示、命令、经验沉淀、提升、删除导致 memory 变化。
- `memory_warning`：memory 流程出现异常但不阻断主任务。

`RuntimeService.get_memory_state()` 返回 pinned 摘要、session/project v2 records，以及 `deleted/superseded` 计数。

`/memory` 展示 pinned/project/session 三层摘要；`/memory list [session|project|correction|experience|deleted]` 展示 v2 records，不再展示 task/stale。

## 8. 设计取舍

当前实现有意不引入 embedding、向量库、置信度、LLM 自由总结或复杂学习评分。对于学习型 Coding Agent，更重要的是：

- 写入准入清晰，普通运行过程不会污染长期记忆。
- 失败成功经验必须有工具和验证证据。
- 同 key 合并和 correction supersede 能阻止 memory 变成日志堆。
- 召回结果可解释，原因可以进入 `ContextReport`。
- 文件数量少，长期 memory 与 session/run/context/task recovery 的边界明确。
