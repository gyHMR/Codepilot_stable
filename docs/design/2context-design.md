# Codepilot 上下文管理机制设计

本文说明当前代码中的上下文管理机制。更准确的说法是：Codepilot 在每次模型调用前都会跑一条上下文治理链路，把会话历史、仓库状态、工具结果、任务状态、长期记忆和预算压力重新编排成一次可发送给模型的 prompt。

完整历史仍由 session/run 持久化保存；进入 prompt 的只是本轮模型决策需要的规则、当前工作状态、召回记忆、有效证据和少量最近对话。

## 0. 一句话概述

Codepilot 的上下文管理不是简单截断历史消息，而是在每次 LLM 调用前由 `ContextGovernor.prepare()` 统一执行：先整理本轮可用信息，再根据模型窗口计算 token 压力，然后按 `normal / tight / critical` 三档选择、压缩和组装上下文，最后生成 `PreparedAgentContext` 和 `ContextReport`。

核心入口和代码位置：

| 代码位置 | 职责 |
|---|---|
| `src/codepilot/sessions/session.py` 的 `AgentSession._build_context_preparer()` | 创建 `ContextGovernor`，并把 `prepare()` 绑定到 Agent 的 `prepare_context` 回调 |
| `src/codepilot/core/llm_runner.py` 的 `LLMStreamRunner.stream_assistant_response()` | 每次模型调用前执行 `prepare_context`，然后再转换消息并调用 provider |
| `src/codepilot/sessions/context/governor.py` 的 `ContextGovernor.prepare()` | 上下文治理主流程 |
| `src/codepilot/sessions/context/snapshot.py` 的 `SessionSnapshotBuilder.build()` | 整理仓库、工具、artifact、checkpoint、active files 等事实 |
| `src/codepilot/sessions/context/policy.py` 的 `ContextPressurePolicy.evaluate()` | 计算有效预算和上下文压力 |
| `src/codepilot/sessions/context/projector.py` 的 `ContextProjector.project()` | 按层组装 system prompt 和 recent messages |
| `src/codepilot/sessions/context/ledger.py` 的 `ToolArtifactLedger` | 把工具输出写入 artifact，并在高压力下用摘要引用替代原文 |
| `src/codepilot/sessions/context/checkpoint.py` 的 `ContextCheckpointManager` | critical 压力下写入结构化 checkpoint |

## 1. 调用时机

一次用户请求进入系统后，context 不会马上被最终确定。真正的上下文治理发生在“即将调用模型”之前。

运行流程如下：

1. `AgentSession.__init__()` 初始化 `SessionStore`、`MemoryStore`、`MemoryRetriever`、`TaskRecoveryStore`。
2. `AgentSession._build_context_preparer()` 创建 `ContextGovernor`，返回 `self.context_governor.prepare`。
3. `AgentOptions.prepare_context` 保存这个回调。
4. `LLMStreamRunner.stream_assistant_response()` 在每次模型调用前构造 `ContextPreparationRequest`，包含 `session_id`、`model_context_window`、`model_max_output_tokens`。
5. `ContextGovernor.prepare(context, request)` 返回新的 `PreparedAgentContext`。
6. LLM runner 使用准备后的 `system_prompt/messages/tools` 调用模型，并发出 `context_prepared` 事件。

这意味着同一个 run 中如果发生多次“模型 -> 工具 -> 模型”循环，每次模型调用前都会重新治理上下文，而不是只在用户消息进入时治理一次。

## 2. 四阶段治理链路

从原始信息到最终 prompt，当前代码可以按四个阶段理解。

### 阶段一：整理本轮可用信息

入口是 `ContextGovernor.prepare()` 的前半段。它先调用 `SessionSnapshotBuilder.build(context)`，再调用 memory retriever。

这一阶段做四件事。

第一，刷新仓库状态。

`RepositoryTracker.refresh(previous_snapshot)` 会生成新的 `RepositorySnapshot`，并与上一轮快照比较得到 `RepositoryDelta`。快照内容包括：

- workspace root
- project type
- manifest files
- top-level entries
- test directories
- instruction files
- git branch / HEAD
- git status
- instruction file hash
- dirty path hash

如果发现仓库变化，`SessionSnapshotBuilder.build()` 会调用：

- `SessionContextState.invalidate_paths(modified_paths + deleted_paths)`
- `SessionContextState.invalidate_verification()`

这样旧的文件摘要、文件证据和验证结果不会继续被当成 fresh evidence。

第二，观察工具结果。

`SessionSnapshotBuilder.build()` 会遍历当前 `AgentContext.messages` 中的 `ToolResultMessage`：

- `SessionContextState.observe_tool_result()` 记录 active files、tool evidence、verification evidence。
- `ToolArtifactLedger.record_tool_result()` 把工具输出写入 `.codepilot/sessions/<session_id>/artifacts/tool_outputs/*.txt`，并把引用写进 `context_ledger.jsonl`。

注意：当前实现会为观察到的工具结果建立 artifact 引用；是否把完整工具输出继续放进 prompt，由后面的压力阶段决定。

第三，检查新鲜度。

`SessionContextState.validate_sources(repository_fingerprint)` 会重新检查：

- file summaries 是否与磁盘 hash 一致
- active files 是否被修改或删除
- verification evidence 是否对应当前 workspace fingerprint

返回的 `stale_items` 后面会进入 `ContextReport`，并且最多前 8 条会被渲染到 evidence 区域提醒模型。

第四，召回长期记忆。

`ContextGovernor._recall_memory()` 构造 `MemoryQuery`：

- `text`: 最近一条用户消息
- `active_paths`: 当前 active files
- `task_phase`: task signal 中的 phase
- `action_intent`: task signal 中的 action_intent
- `recent_error`: task signal 中的 recent_error_code
- `retrieval_mode`: 由 `context_mode()` 推断出的 `repair / verify / qa / act`

然后调用 `MemoryRetriever.recall(query)`，得到 `MemoryRecall`：

- pinned memory：来自 `.codepilot/MEMORY.md`
- always memory：correction 和 always constraint
- selected memory：按 query 命中的 decision / experience 等
- dropped memory：因为 deleted/superseded 等原因不召回的记录

ContextGovernor 只消费召回结果，不负责总结或写入长期记忆。

### 阶段二：动态预算和压力判断

这一阶段决定本轮 prompt 应该宽松组装，还是需要更激进地裁剪。

代码入口是 `ContextGovernor.prepare()` 中对 `estimate_context_tokens()` 和 `ContextPressurePolicy.evaluate()` 的调用。

Token 估算在 `src/codepilot/llm/overflow.py`：

| 内容 | 估算方式 |
|---|---|
| 文本 | `len(text) // 4`，即默认 4 字符约等于 1 token |
| 图片 | 每张图片按 `IMAGE_TOKEN_ESTIMATE = 1000` token 估算 |
| tool call | 按工具名、参数字符串长度和少量固定开销估算 |
| tool schema | 每个工具按 `TOOL_SCHEMA_TOKEN_ESTIMATE = 200` token 估算 |

ContextGovernor 会估算三组数字：

- `tool_output_tokens`：所有 `ToolResultMessage` 的估算 token。
- `history_tokens`：当前 messages 不含 system prompt 的估算 token。
- `estimated_tokens`：当前 messages + system prompt 的估算 token。

`ContextPressurePolicy.evaluate()` 先计算有效预算：

```text
effective_budget =
  model_context_window
  - model_max_output_tokens
  - safety_margin_tokens
```

默认 `safety_margin_tokens = 1024`，并且有效预算最低为 128。

然后计算：

```text
pressure_ratio = estimated_tokens / effective_budget
```

三档压力规则：

| 压力 | 条件 | 后续影响 |
|---|---|---|
| `normal` | 总体 token 未超过 tight 阈值，工具输出和历史也没有触发压力原因 | 保留更多 recent messages，工具结果原文可以短期保留 |
| `tight` | `pressure_ratio >= 0.72`，或工具输出/历史触发压力原因 | 减少 recent messages，工具结果改为 artifact 摘要 |
| `critical` | `pressure_ratio >= 0.90` | 进一步减少 recent messages，并创建结构化 checkpoint |

额外压力原因：

- 工具输出 token 超过有效预算 25%：`tool_output_pressure`
- 历史消息 token 超过有效预算 50%：`history_pressure`
- 总量超过 tight/critical 阈值：`tight_budget_pressure` / `critical_budget_pressure`

这一步的重点不是精确 tokenization，而是给上下文裁剪提供一个稳定、可解释的压力信号。

### 阶段三：按压力组装 prompt

这一阶段由 `ContextProjector.project()` 完成。虽然类名叫 Projector，但它实际做的是 prompt 组装：把前面整理出的信息按固定层次写入 system prompt，并裁剪 messages。

最终 system prompt 会包含这些层：

```text
原始 system prompt

## Stable Rules
- ...

## Working State
- ...

## Memory Recall
- ...

## Evidence
- ...

## Recent Turns
- ...
```

各层来源如下。

#### stable_rules

`stable_rules(system_prompt)` 从 system prompt 中抽取规则行：

- 包含 `AGENTS`
- 包含 `CLAUDE`
- 包含 `rule`

如果没有明显规则行，就取 system prompt 前 8 行。

注意：原始 system prompt 仍然完整保留在最终 system prompt 的开头；`Stable Rules` 是为了让规则在结构化区域里更醒目，不是替代原始系统提示词。

#### working_state

`working_state_lines()` 写入：

- `context.current_task`
- latest checkpoint 的 goal
- latest checkpoint 的 next actions
- active files，最多前 12 个
- changed files，最多前 12 个

这里不会自动把 active file 的完整源码塞进 prompt。active files 只是告诉模型当前哪些路径与任务相关；如果模型需要内容，仍应通过工具读取。

#### memory_recall

`ContextGovernor._render_recalled_memory()` 将 memory recall 渲染成：

- `[Pinned memory] ...`
- `[Correction] ... [reasons=...]`
- `[Constraint] ... [reasons=...]`
- `[Decision] ... [reasons=...]`
- `[Experience] ... [reasons=...]`

召回到的 record id 和原因会进入 `ContextReport.retrieved_memory_ids` 与 `ContextReport.memory_retrieval_reasons`。

#### evidence

`ContextProjector.render_evidence()` 渲染三类信息：

- freshness 为 `fresh` 的 `ContextEvidence`
- 最近 8 条 artifact 引用
- 前 8 条 stale item

工具输出会先经过 `_compact_evidence_content()`：

- 短输出直接压缩空白后保留。
- 长输出优先抽取包含 `FAILED / ERROR / Traceback / AssertionError / Exception / Exit code` 的重要行。
- 如果没有明显关键行，则写成类似 `N lines, M chars archived` 的摘要。

#### recent_messages

`recent_message_lines()` 写入最近几条消息的文本摘要：

| 压力 | system prompt 中 Recent Turns 行数 |
|---|---|
| `normal` | 最近 6 条 |
| `tight` | 最近 4 条 |
| `critical` | 最近 2 条 |

这部分用于保留对话连续性，但不会把全部 transcript 都塞进 system prompt。

#### messages

真正发送给模型的 messages 也会按压力裁剪：

| 压力 | `project_messages()` 保留消息数 | ToolResultMessage 处理 |
|---|---:|---|
| `normal` | 最近 10 条 | `preserve_full=True`，保留原工具结果 |
| `tight` | 最近 6 条 | 替换为 artifact 摘要 |
| `critical` | 最近 4 条 | 替换为 artifact 摘要 |

`repair_tool_pairs()` 会修复 tool call / tool result 配对问题：如果裁剪后保留了某个 `ToolResultMessage`，但对应的 assistant `ToolCall` 被裁掉了，会补回一个最小 assistant tool call message，避免 provider 报消息结构错误。

#### critical checkpoint

如果压力是 `critical`，ContextGovernor 会调用 `ContextCheckpointManager.create()` 写入结构化 checkpoint，字段包括：

- goal
- active_files
- changed_files
- key_evidence
- verification_state
- open_questions
- next_actions
- source_refs

checkpoint 写在 `context_ledger.jsonl` 中，类型是 `checkpoint`。下一轮 `SessionSnapshotBuilder.build()` 会通过 `ContextCheckpointManager.load_latest()` 读取最新 checkpoint，并把它作为 working state 的优先信息。

### 阶段四：记录报告并交给模型

组装完成后，ContextGovernor 会生成 `ContextReport`，并返回 `PreparedAgentContext`。

`PreparedAgentContext` 包含：

- `system_prompt`: 原始 system prompt + 五个结构化上下文区块
- `messages`: 按压力裁剪和修复后的消息列表
- `tools`: 当前工具列表
- `report`: 本轮上下文治理报告

`ContextReport` 记录：

- `context_id`
- repository fingerprint
- effective token budget
- estimated tokens before / after
- sections
- selected items
- stale items
- repository delta
- memory ids 和 recall reasons
- context mode
- pressure
- checkpoint used / created
- artifact refs
- tokens by layer
- prefix hash
- dynamic hash

`prefix_hash` 来自 `stable_rules`，`dynamic_hash` 来自 working state、memory、evidence、recent messages。当前代码只记录 hash，代码中没有发现把它们映射到具体 provider prefix cache metadata 的实现。

最后，`ContextGovernor._append_context_view()` 会向：

```text
.codepilot/sessions/<session_id>/context_ledger.jsonl
```

追加一条 `type="context_view"` 的记录，包含 pressure、tokens_by_layer、checkpoint、artifact_refs、prefix_hash、dynamic_hash 等摘要信息。

## 3. 上下文来源总览

| 来源 | 事实源 | 如何进入 prompt |
|---|---|---|
| 系统规则 | `AgentOptions.system_prompt`，通常包含 AGENTS.md 等规则 | 原样作为 system prompt 开头，并提取规则行进入 `Stable Rules` |
| 会话消息 | `messages.jsonl` 还原出的 canonical transcript | 只保留 recent messages；数量由压力决定 |
| 仓库状态 | `RepositoryTracker.snapshot()` | 进入 repository fingerprint、changed files、stale 判断 |
| 活跃文件 | `SessionContextState.active_files` | 路径进入 `Working State`，不自动注入源码 |
| 工具结果 | `ToolResultMessage` | 进入 evidence；必要时通过 artifact 摘要进入 messages |
| 验证结果 | `ToolResultMessage.verification` | 进入 verification evidence；工作区变化后会 stale |
| 长期记忆 | `MemoryRetriever.recall()` | 进入 `Memory Recall` |
| checkpoint | `context_ledger.jsonl` 中的 `checkpoint` | critical 后的下一轮进入 `Working State` |
| 工具列表 | `AgentContext.tools` | 原样返回给 LLM runner，名称也记录在 `ContextView.tools` |

## 4. 压力级别对比

| 行为 | normal | tight | critical |
|---|---:|---:|---:|
| system prompt Recent Turns | 6 条 | 4 条 | 2 条 |
| 发送给模型的 messages | 10 条 | 6 条 | 4 条 |
| tool result 原文 | 保留 | 摘要 + artifact ref | 摘要 + artifact ref |
| checkpoint | 不创建 | 不创建 | 创建 |
| evidence | fresh evidence + 最近 artifact + stale 提醒 | 同 normal，但消息更短 | 同 tight，并保留 checkpoint 所需关键证据 |

这三个级别没有做得很复杂，适合当前学习型项目：能解释、能演示、能覆盖大日志和长历史压力，但没有引入 Claude Code 风格的 snip/collapse agent。

## 5. 新鲜度机制

上下文的新鲜度由三层控制。

第一层是仓库 delta。

`RepositoryTracker` 每次模型调用前都会重新计算仓库 fingerprint。只要发现新增、修改、删除、分支变化、HEAD 变化或 instruction 文件变化，`RepositoryDelta.changed` 就会为 true。

第二层是 session context state。

`SessionContextState` 保存：

- active files
- file summaries
- tool evidence
- verification evidence
- last repository snapshot
- observed tool call ids

当工具修改工作区时：

- 相关路径的 file summary 会被标记 stale。
- 相关 path evidence 会被标记 stale。
- verification evidence 会被标记 stale。

第三层是 validate_sources。

`validate_sources(repository_fingerprint)` 会检查文件 hash 和 verification fingerprint。如果发现不一致，会产生 `stale_items`，并进入 `ContextReport.stale_items` 和 evidence 提醒。

## 6. 与 Memory、TaskRecovery、Tool Ledger 的边界

### Memory

Memory 只保存长期可复用知识。ContextGovernor 只调用 `MemoryRetriever.recall()`，不总结失败经验，也不写 memory。

失败-成功-验证通过的经验沉淀由 `MemoryWriter.finalize_run()` 负责，发生在 run 收尾阶段，而不是 context prepare 阶段。

### TaskRecovery

TaskRecovery 保存当前任务目标、步骤进度和下一步动作，属于 session 级恢复信息，不属于长期 memory。

Context 侧能看到的是 `AgentContext.current_task` 和 `task_signal`。`ContextProjector` 当前只把 `current_task` 写入 working state，并用 `task_signal` 帮助判断 context mode 和 memory query。

### Tool Ledger

Tool Ledger 负责解决“大段工具输出是否每轮都塞进 prompt”的问题：

- 原始工具输出写入 artifact 文件。
- `context_ledger.jsonl` 记录 artifact ref。
- normal 压力下近期工具结果可以保留原文。
- tight/critical 压力下工具结果替换成摘要和 artifact 路径。

完整工具结果仍在 transcript 和 artifact 中可恢复，但 prompt 不必每轮携带完整日志。

## 7. 当前没有做什么

代码中未发现以下能力的明确实现：

- 没有 embedding/vector database 级别的文件片段召回。
- 没有独立 file selector 自动把源码片段按相关性注入 prompt。
- 没有 LLM summary builder 把旧历史总结成一条 summary message。
- 没有后台 collapse agent。
- 没有 provider-specific tokenizer，token 估算是字符级近似。
- 没有把 `prefix_hash/dynamic_hash` 映射到 provider cache metadata。
- `context_ledger.jsonl` 当前是 append-only，没有自动 retention/pruning。

这些不是 bug，而是当前学习型项目的取舍：先把主线做清楚，避免为了“像生产系统”而引入难解释的机制。

## 8. 优点和风险

优点：

- 每次模型调用前都会重新整理上下文，能反映最新工具结果和仓库变化。
- 工具输出有 artifact 机制，长日志不会在 tight/critical 下线性污染 prompt。
- 有 `normal/tight/critical` 三档压力，行为可解释。
- 有 stale 检查，验证结果和文件证据不会无限期保持 fresh。
- Memory、TaskRecovery、Context、RunStore 的边界比旧方案更清楚。
- `ContextReport` 能展示压力、token 分层、记忆召回、artifact、checkpoint 等调试信息。

风险：

- active files 只放路径，不放源码内容；复杂修改仍依赖模型主动 read 文件。
- token 估算较粗，可能与真实 provider token 数存在偏差。
- stable rules 只是从 system prompt 中抽取规则行，不能替代更严格的指令文件解析和优先级系统。
- tight/critical 的消息裁剪是按最近 N 条，不是语义相关性排序。
- context ledger 长期 append 可能增长，需要后续 retention 策略。

## 9. 可以写进项目介绍的总结

针对 Coding Agent 在多轮代码任务中容易出现历史过长、工具日志污染、旧验证结果误导和长期记忆混杂的问题，Codepilot 设计了每次模型调用前执行的上下文治理链路：先整理仓库状态、工具结果、active files、checkpoint 和 memory recall，再根据模型窗口计算 `normal/tight/critical` 压力，随后按固定层次组装 system prompt、裁剪 recent messages、将工具输出替换为 artifact 摘要，并记录 `ContextReport`。这样模型每轮拿到的是当前决策最需要的规则、工作状态、长期记忆、有效证据和少量对话连续性，而完整 transcript、run 结果和工具输出仍保存在 session/run/artifact 文件中，可恢复、可调试。
