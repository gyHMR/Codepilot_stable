# Codepilot 记忆模块设计

## 1. 设计结论

Codepilot 的记忆模块只负责保存 **可跨任务复用的稳定知识**。当前任务进展、活跃文件、普通工具失败和文件新鲜度不属于长期记忆，它们分别由 task recovery、context state 和 run evidence 管理。

```text
TaskRecoveryStore
  当前任务目标、步骤进展、下一步动作

SessionContextState / ContextGovernor
  活跃文件、近期工具证据、验证证据、文件新鲜度、上下文预算

MemoryStore / MemoryWriter / MemoryRetriever
  项目约束、用户偏好、已验证决策、可复用修复经验、pinned memory
```

这个边界避免把“临时工作状态”误当成“长期记忆”，也让上下文注入更容易解释。

## 2. 什么能进入 Memory

Memory V2 默认只召回 durable memory：

| 类型 | 说明 | 示例 |
|---|---|---|
| `project` | 项目级约束、用户偏好、稳定知识 | “这是学生学习与求职展示项目，避免生产级复杂平台化” |
| `decision` | 已确认的设计决策 | “CLI 是主入口，Web 是未来控制台” |
| `experience` | 有证据链的可复用修复经验 | “edit 多匹配失败后应先 read，再用唯一 old_text” |
| pinned memory | 用户手写固定记忆 | `.codepilot/MEMORY.md` |

以下内容不再作为 durable memory 写入或默认召回：

| 内容 | 新归属 | 原因 |
|---|---|---|
| 当前任务进展 | `TaskRecoveryStore` | 它只服务当前 session 恢复，不是长期知识 |
| 普通 read 文件摘要 | `SessionContextState.active_files/evidence` | 这是短期工作集，文件变更后很快过期 |
| 单次工具失败 | `AgentRunResult` / run evidence | 单次失败噪音大，不能直接沉淀 |
| 文件 hash / stale 状态 | `SessionContextState` / `RepositoryTracker` | 属于上下文新鲜度治理 |

## 3. 写入策略

`MemoryWriter` 是 durable memory 的准入入口。

### 3.1 用户输入

普通用户任务不会创建 `task` memory。`MemoryWriter.admit_prompt_memory()` 只在用户明确表达项目约束时写入 project memory，任务进展交给 `TaskRecoveryStore`。

例如：

```text
这个项目是学生学习和求职展示项目，不要做生产级复杂设计
```

会写入：

```json
{
  "kind": "project",
  "scope": "project",
  "content": {
    "category": "project_constraint",
    "knowledge": "Codepilot 是学生学习与求职展示项目；后续设计应优先保持清晰、可解释、可演示，避免生产级复杂平台化。"
  },
  "trust": "user_given"
}
```

### 3.2 工具结果

普通 `read/edit/bash` 工具结果不直接写入 durable memory。

- read 成功：由 `SessionContextState` 记录 active file 和 evidence。
- 工具失败：保留在 run evidence 中，供本轮修复使用。
- 文件修改：使相关 context evidence / file summary 失效。

### 3.3 Run 结束

只有失败后成功、且有验证通过证据的闭环，才会提炼为 `experience`。

当前第一版使用确定性规则：

- edit 因 `multiple_matches` / `unexpected_match_count` 失败；
- 后续 edit 成功；
- 后续 verification passed；
- 写入 verified experience。

另一个规则是：

- verification failed；
- 后续 verification passed；
- 写入 verification repair experience。

## 4. 任务恢复

任务恢复由 `sessions/history/task_recovery.py` 负责。

它保存：

```json
{
  "goal": "修复失败测试",
  "task_progress": {
    "completed_steps": ["定位失败"],
    "pending_steps": ["重新运行相关验证"],
    "blocked_steps": [],
    "completion_satisfied": false,
    "completion_reason": "incomplete_steps",
    "step_details": {}
  },
  "next_action": "重新运行相关验证"
}
```

`AgentSession.run()` 开始时写入或刷新 task recovery；run 结束后用 `AgentRunResult.task` 更新 projection。下一次 run 会把未完成 projection 传给 `TaskController`，但它不会通过 `MemoryRetriever` 注入为长期记忆。

## 5. 检索与上下文注入

`MemoryRetriever` 只检索：

```python
DURABLE_MEMORY_KINDS = {"project", "decision", "experience"}
```

并跳过 legacy state：

```python
LEGACY_MEMORY_KINDS = {"task", "file", "failure"}
```

`ContextGovernor` 负责把 durable memory 作为 recalled memory 层注入本轮 ContextView：

```text
Stable Rules
Working State
Recalled Memory
Evidence
Recent Messages
```

也就是说，memory 是上下文投影的一个来源，而不是上下文系统本身；失败经验如何总结仍由 MemoryWriter 负责。

## 6. 设计亮点

1. **边界清晰**：task recovery、context evidence、durable memory 分开管理。
2. **保守写入**：普通任务、普通 read、单次失败不会污染长期记忆。
3. **证据驱动**：experience 必须来自真实工具结果和验证闭环。
4. **可解释召回**：`RetrievedMemory.reasons` 说明为什么一条 memory 被选中。
5. **上下文协同**：ContextGovernor 统一决定 working state、evidence、memory、recent history 的投影和压力裁剪。

## 7. 学习路径

1. 读 `sessions/history/task_recovery.py`，理解任务恢复为什么不属于 memory。
2. 读 `sessions/context/state.py`，理解 active files 和 recent evidence。
3. 读 `sessions/memory/writer.py`，理解 durable memory 的写入准入。
4. 读 `sessions/memory/retriever.py`，理解 durable memory 的召回策略。
5. 读 `sessions/context/governor.py` 和 `sessions/context/projector.py`，理解 memory 如何作为上下文来源被注入。
