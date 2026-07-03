# Codepilot 上下文管理机制设计

本文只描述当前代码中的上下文投影治理实现。旧的手动压缩入口、旧上下文编译路径和旧 `context.jsonl/runs.jsonl` 布局已经不再是运行时能力；代码中没有兼容读取旧 session layout 的主路径。

## 0. 机制概述

Codepilot 当前的上下文治理不是“把历史消息列表裁短后发给模型”，而是“从完整 Session 事实源、工具输出 ledger、仓库状态、任务恢复投影和记忆召回中，为本轮模型调用投影一个 ContextView”。`AgentSession` 在创建时无条件构造 `ContextGovernor`，并把 `ContextGovernor.prepare()` 绑定到核心 Agent 的 `prepare_context` 回调；每次模型调用前由 governor 重新读取快照、计算压力、生成分层视图、压缩工具输出、必要时创建 checkpoint，最后返回 `PreparedAgentContext`。

代码证据：
- `AgentSession` 文档明确第 4 项职责是“通过 ContextGovernor 为每轮模型调用投影上下文”：`src/codepilot/sessions/session.py:9`。
- `AgentSession._build_context_preparer()` 创建 `ContextGovernor` 并返回 `self.context_governor.prepare`：`src/codepilot/sessions/session.py:652-665`。
- runtime 装配层不再传入旧 prepare_context，而是交给 session 内部绑定：`src/codepilot/runtime/assembly.py:185`。
- `ContextGovernor.prepare()` 是本轮上下文准备主入口：`src/codepilot/sessions/context/governor.py:79-186`。

## 1. 核心模块、类和入口

| 模块 | 核心类/函数 | 职责 | 代码证据 |
|---|---|---|---|
| `src/codepilot/sessions/layout.py` | `SessionLayout` | 统一定义 sessions/runs/memory/context/artifact 路径，避免各模块散落拼路径。 | `SessionLayout`：`src/codepilot/sessions/layout.py:10`；`context_ledger_file`：`src/codepilot/sessions/layout.py:45`；`tool_outputs_dir`：`src/codepilot/sessions/layout.py:49`；`run_file()`：`src/codepilot/sessions/layout.py:63`。 |
| `src/codepilot/sessions/session.py` | `AgentSession` | 会话编排入口：加载 transcript、绑定 ContextGovernor、持久化消息、在 run 结束后写 memory/task/rollback。 | 初始化读取 `load_session_messages()`：`src/codepilot/sessions/session.py:111`；事件中写消息：`src/codepilot/sessions/session.py:625-646`；绑定 governor：`src/codepilot/sessions/session.py:652-665`。 |
| `src/codepilot/sessions/persistence/store.py` | `SessionStore` | session 事实源：`session.json`、`messages.jsonl`、lazy session events、task recovery 当前投影。 | `ensure_initialized()` 只创建 `session.json/messages.jsonl`：`src/codepilot/sessions/persistence/store.py:52-75`；`append_message()` 写 transcript：`src/codepilot/sessions/persistence/store.py:111-125`；`save_task_recovery()` 写入 session state：`src/codepilot/sessions/persistence/store.py:108-110`。 |
| `src/codepilot/sessions/persistence/run_store.py` | `RunStore` | run 事实源：`run.json`、lazy run events、rollback metadata、freshness 评估。 | `append_run_result()` 写单一 `run.json`：`src/codepilot/sessions/persistence/run_store.py:77-100`；`write_rollback_metadata()`：`src/codepilot/sessions/persistence/run_store.py:114-125`；`evaluate_freshness()`：`src/codepilot/sessions/persistence/run_store.py:142-145`。 |
| `src/codepilot/sessions/context/governor.py` | `ContextGovernor` | 唯一上下文治理入口，串联 snapshot、memory、pressure、projector、checkpoint、ledger report。 | 初始化依赖：`src/codepilot/sessions/context/governor.py:42-77`；`prepare()` 主流程：`src/codepilot/sessions/context/governor.py:79-186`；写 context ledger：`src/codepilot/sessions/context/governor.py:227-254`。 |
| `src/codepilot/sessions/context/snapshot.py` | `SessionSnapshotBuilder` | 从 repository tracker、context state、tool ledger、checkpoint 读取本轮事实快照。 | `SessionSnapshotBuilder.build()` 刷新 repo、处理工具结果、返回 active/changed/artifact/checkpoint：`src/codepilot/sessions/context/snapshot.py:54-81`。 |
| `src/codepilot/sessions/context/policy.py` | `ContextPressurePolicy` | 根据模型窗口、输出预算、安全边界、历史和工具输出规模判定 `normal/tight/critical`。 | `evaluate()`：`src/codepilot/sessions/context/policy.py:17-52`。 |
| `src/codepilot/sessions/context/projector.py` | `ContextProjector` | 把快照和召回信息投影为 `stable_rules/working_state/recalled_memory/evidence/recent_messages/tools` 分层视图。 | `project()`：`src/codepilot/sessions/context/projector.py:61-90`；`project_messages()` 根据压力保留 10/6/4 条消息：`src/codepilot/sessions/context/projector.py:92-112`。 |
| `src/codepilot/sessions/context/ledger.py` | `ToolArtifactLedger` | 大工具输出 artifact 化；prompt 中使用轻量摘要和 artifact ref。 | 初始化使用 `context_ledger.jsonl` 和 `artifacts/tool_outputs`：`src/codepilot/sessions/context/ledger.py:68-77`；`record_tool_result()`：`src/codepilot/sessions/context/ledger.py:79-132`；`project_tool_result()`：`src/codepilot/sessions/context/ledger.py:134-161`。 |
| `src/codepilot/sessions/context/checkpoint.py` | `ContextCheckpointManager` | critical 压力下创建结构化 checkpoint，并写入统一 `context_ledger.jsonl`。 | `ContextCheckpointManager`：`src/codepilot/sessions/context/checkpoint.py:14-23`；`append()`：`src/codepilot/sessions/context/checkpoint.py:49-56`。 |
| `src/codepilot/sessions/memory/store.py` | `MemoryStore` | 负责 session/project memory 文件位置与读写；上下文只召回，不改失败记忆总结算法。 | session memory/project memory 路径：`src/codepilot/sessions/memory/store.py:30-45`。 |
| `src/codepilot/sessions/history/task_recovery.py` | `TaskRecoveryStore` | 保存/读取当前任务恢复投影，实际落在 `session.json.task_recovery`。 | `load_projection()`：`src/codepilot/sessions/history/task_recovery.py:46-47`；`save_projection()`：`src/codepilot/sessions/history/task_recovery.py:64-73`。 |
| `src/codepilot/sessions/history/git_rollback.py` | `build_rollback_metadata()` | 维持轻量 git 回退元数据，不做复杂快照事务。 | `build_rollback_metadata()`：`src/codepilot/sessions/history/git_rollback.py:79`；session 写 rollback metadata：`src/codepilot/sessions/session.py:524-532`。 |

公共导出也只暴露新主线：`src/codepilot/sessions/context/__init__.py:32-48` 导出 `ContextGovernor`、`ContextProjector`、`ContextPressurePolicy`、`SessionSnapshotBuilder`、`ToolArtifactLedger` 等；sessions public API 不再导出旧上下文编译入口。

## 2. 用户请求进入后的上下文构建数据流

1. runtime 通过 `assemble_runtime()` 组装模型、工具、系统提示词和 `AgentSessionOptions`，但 `prepare_context=None`，不在 runtime 层注入旧编译器。证据：`src/codepilot/runtime/assembly.py:145-186`。
2. `AgentSession.__init__()` 初始化 `SessionStore`，确保 `session.json/messages.jsonl` 存在，读取 canonical transcript，再构造核心 `AgentOptions`。证据：`src/codepilot/sessions/session.py:79-145`。
3. `AgentSession._build_context_preparer()` 创建 `ContextGovernor`，注入 workspace、session_id、`SessionContextState`、`MemoryRetriever`，并返回 `prepare` 回调。证据：`src/codepilot/sessions/session.py:652-665`。
4. run 开始前，session 做生命周期钩子、prompt memory admission、task recovery begin、freshness 检查；真正 prompt 投影在 Agent 调模型前通过 prepare_context 执行。证据：`src/codepilot/sessions/session.py:537-566`。
5. `ContextGovernor.prepare()` 调用 `SessionSnapshotBuilder.build()` 读取仓库快照、工具结果、artifact refs、checkpoint、active/changed files。证据：`src/codepilot/sessions/context/governor.py:85`、`src/codepilot/sessions/context/snapshot.py:54-81`。
6. governor 召回 memory，读取 pinned memory，估算历史/工具输出/总体 token，调用 `ContextPressurePolicy.evaluate()`。证据：`src/codepilot/sessions/context/governor.py:86-107`。
7. governor 渲染 evidence；critical 时创建 checkpoint；然后调用 `ContextProjector.project()` 生成 `ContextView`、投影后的 messages 和新的 system prompt。证据：`src/codepilot/sessions/context/governor.py:108-136`。
8. governor 生成 `ContextReport`，写一条 `type="context_view"` 到 `context_ledger.jsonl`，并返回 `PreparedAgentContext`。证据：`src/codepilot/sessions/context/governor.py:137-186`、`src/codepilot/sessions/context/governor.py:227-254`。

## 3. 上下文来源

当前上下文来源可以分成六类：

| 来源 | 进入方式 | 代码证据 |
|---|---|---|
| stable rules / system prompt | `ContextProjector.stable_rules()` 从系统提示词中抽取规则行或前 8 行，最终仍保留完整 base system prompt。 | `src/codepilot/sessions/context/projector.py:115-124`、`src/codepilot/sessions/context/projector.py:153-169`。 |
| canonical transcript | `SessionStore.load_session_messages()` 从 `messages.jsonl` 重建消息树；`project_messages()` 只取最近少量消息。 | `src/codepilot/sessions/persistence/store.py:167-187`、`src/codepilot/sessions/context/projector.py:92-112`。 |
| repository snapshot | `RepositoryTracker.refresh()` 由 `SessionSnapshotBuilder.build()` 调用，delta 变化会使路径和 verification 失效。 | `src/codepilot/sessions/context/snapshot.py:54-60`。 |
| active/changed files | `SessionContextState` 维护 active files；snapshot 同时从 repository delta 收集 changed files。 | `src/codepilot/sessions/context/snapshot.py:78-80`。 |
| tool results / evidence | `SessionSnapshotBuilder` 遍历 `ToolResultMessage`，先观察 evidence，再交给 `ToolArtifactLedger` 记录 artifact/ref。 | `src/codepilot/sessions/context/snapshot.py:62-73`。 |
| memory recall | `ContextGovernor._recall_memory()` 构造 `MemoryQuery`，只接受 durable active memory；pinned memory 通过 retriever 动态读取。 | `src/codepilot/sessions/context/governor.py:191-218`。 |
| task recovery | `TaskRecoveryStore` 从 `session.json.task_recovery` 读取当前任务投影，session 在 run 前交给核心 Agent。 | `src/codepilot/sessions/history/task_recovery.py:46-47`、`src/codepilot/sessions/session.py:563`。 |

代码中未发现明确实现：当前 context projector 不会自动读取 active file 的完整源码片段；`working_state_lines()` 只把 active files 和 changed files 的路径写入 working state。证据：`src/codepilot/sessions/context/projector.py:126-151`。

## 4. 上下文选择策略

当前选择策略以分层投影为中心，而不是旧版“多个 section 按比例塞条目”。

- `stable_rules`：来自 system prompt 的规则行或前 8 行；完整 system prompt 仍是最终 system prompt 的 base。证据：`src/codepilot/sessions/context/projector.py:115-124`、`src/codepilot/sessions/context/projector.py:153-169`。
- `working_state`：当前任务、checkpoint goal/next actions、active files、changed files。证据：`src/codepilot/sessions/context/projector.py:126-151`。
- `recalled_memory`：`MemoryRetriever.retrieve()` 返回后，governor 只保留 `DURABLE_MEMORY_KINDS` 且 `status=="active"` 的记忆。证据：`src/codepilot/sessions/context/governor.py:191-211`。
- `evidence`：只渲染 freshness 为 `fresh` 的 evidence；artifact refs 只放最近 8 条；stale items 最多放 8 条。证据：`src/codepilot/sessions/context/projector.py:27-49`。
- `recent_messages`：按压力保留摘要行，normal/tight/critical 分别保留 6/4/2 条。证据：`src/codepilot/sessions/context/projector.py:142-151`。
- `messages`：真正发给模型的消息列表按压力保留 normal/tight/critical 10/6/4 条，并修复 tool_call/tool_result 配对。证据：`src/codepilot/sessions/context/projector.py:92-112`、`src/codepilot/sessions/context/projector.py:202-217`。
- 工具结果：`project_tool_result()` 在 normal 下可短期保留完整结果；tight/critical 或大输出会用摘要 + artifact ref 替代。证据：`src/codepilot/sessions/context/ledger.py:134-161`。

代码中未发现明确实现：没有独立 file selector 对源码片段进行 BM25/embedding/ranking，也没有模型主动调用的 snip/collapse 工具；当前选择更多是规则化、压力感知和 evidence 新鲜度约束。

## 5. Token 预算、裁剪与结构化压缩

Token 估算由 `estimate_context_tokens()` 完成，governor 分别估算：

- 工具输出 token：`tool_output_tokens(context.messages)`。证据：`src/codepilot/sessions/context/governor.py:93`、`src/codepilot/sessions/context/projector.py:174-180`。
- 历史消息 token：`estimate_context_tokens(context.messages, "")`。证据：`src/codepilot/sessions/context/governor.py:94`。
- 投影前总 token：`estimate_context_tokens(context.messages, context.system_prompt)`。证据：`src/codepilot/sessions/context/governor.py:95-98`。
- 投影后 token：`estimate_context_tokens(prepared_messages, system_prompt)`。证据：`src/codepilot/sessions/context/governor.py:138-140`。

压力预算由 `ContextPressurePolicy.evaluate()` 计算：

- `effective_budget = model_context_window - model_max_output_tokens - safety_margin_tokens`，且至少为 128。证据：`src/codepilot/sessions/context/policy.py:19-24`。
- 默认 safety margin 是 1024，tight ratio 是 0.72，critical ratio 是 0.90，工具输出压力阈值是有效预算的 0.25。证据：`src/codepilot/sessions/context/policy.py:12-16`。
- 工具输出过大标记 `tool_output_pressure`，历史过大标记 `history_pressure`，总量超过阈值进入 `tight/critical`。证据：`src/codepilot/sessions/context/policy.py:28-45`。

裁剪/压缩动作：

- normal：保留较多 recent messages，并允许近期 tool result 原文短期进入 prompt。证据：`src/codepilot/sessions/context/projector.py:92-112`、`src/codepilot/sessions/context/ledger.py:134-161`。
- tight：recent messages 降低到 6 条，tool result 默认投影为 artifact 摘要。证据：`src/codepilot/sessions/context/projector.py:92-112`。
- critical：recent messages 降到 4 条，并创建 `ContextCheckpoint`，后续轮次优先读取 latest checkpoint。证据：`src/codepilot/sessions/context/governor.py:114-126`、`src/codepilot/sessions/context/snapshot.py:77`。

代码中未发现明确实现：当前没有 LLM summary builder、后台 collapse agent、旧历史消息替换成 summary message 的 runtime 路径；CLI 只保留 `/context` 查看最近一次上下文投影治理报告，不再提供手动压缩命令。证据：`src/codepilot/interfaces/cli/commands.py`。

## 6. 更新与失效机制

上下文新鲜度主要由 repository delta 和 context state 驱动：

- `SessionSnapshotBuilder.build()` 每轮刷新 repository snapshot；如果 delta 发生变化，会调用 `invalidate_paths()` 和 `invalidate_verification()`。证据：`src/codepilot/sessions/context/snapshot.py:54-60`。
- 工具结果进入 `SessionContextState.observe_tool_result()`，并携带当时的 repository fingerprint。证据：`src/codepilot/sessions/context/snapshot.py:62-69`。
- 最后通过 `validate_sources(snapshot.fingerprint)` 生成 stale items。证据：`src/codepilot/sessions/context/snapshot.py:74`。
- `AgentSession._check_context_freshness()` 会在 run 前更新 freshness notice。证据：`src/codepilot/sessions/session.py:564`。
- run 层还保留 `RunStore.evaluate_freshness()`，用于已跟踪文件的新鲜度判断。证据：`src/codepilot/sessions/persistence/run_store.py:142-145`。

代码中未发现明确实现：`context_ledger.jsonl` 当前是 append-only，代码中没有按时间、大小或条数的自动 pruning 策略。

## 7. 不同任务阶段的上下文差异

当前差异主要来自 `context_mode()` 和压力级别：

- `context_mode()` 会根据 `task_signal.action_intent`、`recent_error_code`、用户问题文本判断 `repair/verify/qa/act`。证据：`src/codepilot/sessions/context/projector.py:299-311`。
- 该 mode 会进入 `MemoryQuery.retrieval_mode`，影响记忆召回语义。证据：`src/codepilot/sessions/context/governor.py:198-206`。
- 压力级别会影响 recent messages 数量、tool result 是否保留原文、是否创建 checkpoint。证据：`src/codepilot/sessions/context/projector.py:92-112`、`src/codepilot/sessions/context/governor.py:114-126`。

代码中未发现明确实现：没有独立 read-only/plan/execute/replan 的分区预算 profile；`TaskMode` 和 planning budget 主要在 task control 层生效，不直接改变 `ContextProjector.section_reports()` 的预算比例。证据：`src/codepilot/sessions/context/projector.py:171-199`。

## 8. 与工具、任务、记忆、持久化和 git 回退的关系

工具系统：
- 工具结果仍作为 canonical message 写入 `messages.jsonl`；进入 prompt 时由 `ToolArtifactLedger` 决定保留原文还是摘要引用。证据：`src/codepilot/sessions/session.py:625-646`、`src/codepilot/sessions/context/ledger.py:79-161`。

任务控制：
- task recovery 当前投影属于 session state，保存到 `session.json.task_recovery`，不是 memory，也不是 context ledger。证据：`src/codepilot/sessions/persistence/store.py:103-110`、`src/codepilot/sessions/history/task_recovery.py:46-73`。

记忆系统：
- `MemoryStore` 管理 session memory 和 project memory 路径；`ContextGovernor` 只召回，不负责失败记忆总结。证据：`src/codepilot/sessions/memory/store.py:30-45`、`src/codepilot/sessions/context/governor.py:191-224`。

持久化：
- session 事实源是 `session.json/messages.jsonl`；run 事实源是 `run.json/events.jsonl`；context 派生视图只写 `context_ledger.jsonl`。证据：`src/codepilot/sessions/persistence/store.py:52-75`、`src/codepilot/sessions/persistence/run_store.py:77-125`、`src/codepilot/sessions/context/governor.py:227-254`。

git 回退：
- rollback metadata 写入 `run.json.rollback`，回退策略仍是轻量 clean-worktree + affected paths，不引入隐藏分支或事务快照。证据：`src/codepilot/sessions/session.py:524-532`、`src/codepilot/sessions/persistence/run_store.py:114-125`。

## 9. 当前优点和缺陷

优点：

1. 主线清晰：上下文只有 `ContextGovernor.prepare()` 一个入口，避免多套上下文处理路径并存。证据：`src/codepilot/sessions/session.py:652-665`、`src/codepilot/sessions/context/__init__.py:32-48`。
2. 文件契约更瘦：新 session 默认只创建 `session.json/messages.jsonl`，其他 ledger/event/memory 按需懒创建。证据：`src/codepilot/sessions/persistence/store.py:52-75`。
3. 工具输出不再线性污染 prompt：工具输出可 artifact 化，prompt 中放摘要和路径引用。证据：`src/codepilot/sessions/context/ledger.py:79-161`。
4. 有压力感知：normal/tight/critical 三档影响 recent history、tool result 投影和 critical checkpoint。证据：`src/codepilot/sessions/context/policy.py:17-52`、`src/codepilot/sessions/context/projector.py:92-112`、`src/codepilot/sessions/context/governor.py:114-126`。
5. 新鲜度有最小闭环：仓库 delta 会使路径和 verification 失效，避免旧证据长期被当成 fresh evidence。证据：`src/codepilot/sessions/context/snapshot.py:54-74`。

缺陷：

1. active files 只记录路径，不自动注入源码片段；模型可能知道“该看哪个文件”，但仍需要工具读取内容。证据：`src/codepilot/sessions/context/projector.py:126-151`。
2. 阶段策略仍偏轻量；`repair/verify/qa/act` 主要影响 memory query，没有独立 plan/replan 预算 profile。证据：`src/codepilot/sessions/context/projector.py:299-311`、`src/codepilot/sessions/context/governor.py:198-206`。
3. `context_ledger.jsonl` append-only，代码中未发现 retention/pruning；长 session 的调试 ledger 可能增长。证据：`src/codepilot/sessions/context/governor.py:227-254`、`src/codepilot/sessions/context/ledger.py:177-181`。
4. token 估算仍是近似值，代码中未发现 provider-specific tokenizer；预算判断可能与真实 provider token 数有偏差。证据：`src/codepilot/sessions/context/governor.py:93-140`。
5. prefix caching 目前只记录 `prefix_hash/dynamic_hash`，代码中未发现把它映射到 provider cache metadata 的实现。证据：`src/codepilot/sessions/context/governor.py:141-170`。

## 10. 简历/项目文档总结

“针对 Coding Agent 在多轮代码任务中容易出现关键信息遗漏、无关上下文污染和过期信息干扰的问题，项目设计了以 `ContextGovernor` 为唯一入口的上下文投影机制，通过 `SessionLayout + SessionStore/RunStore` 保留完整事实源，通过 `SessionSnapshotBuilder` 汇总仓库、工具、任务和记忆事实，通过 `ContextPressurePolicy` 进行 normal/tight/critical 三档压力判断，并由 `ContextProjector` 生成分层 `ContextView`、tool artifact ledger 和 critical checkpoint，从而让模型每轮在有限 token 预算内优先获得当前决策所需的规则、工作态、记忆召回、有效证据和少量最近对话，同时避免大日志、旧摘要和过期验证结果污染 prompt。”
