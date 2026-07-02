# Codepilot 上下文管理机制设计说明（基于当前代码）

本文只描述当前代码中已经实现的上下文管理机制，不把设想、目标或外部系统能力当成事实。凡代码中没有明确实现的能力，均标注为“代码中未发现明确实现”。

证据引用格式为：`文件路径:行号范围` + 类名/函数名。

## 0. 上下文管理机制概述

当前项目的上下文治理可以概括为：`Session` 保存完整事实源，`ContextGovernor` 在每次模型调用前从会话消息、仓库状态、工具结果、任务状态、记忆召回和 checkpoint 中投影出本轮 `ContextView`，再按 token 压力把不同层级的信息组织进 prompt。也就是说，项目已经从“直接把消息历史交给模型”演进为“从长期会话状态生成一次性的模型决策视图”。

代码证据：

- `AgentContext` 定义模型调用前的原始上下文，包括 `system_prompt`、`messages`、`tools`、`current_task`、`task_recovery_projection` 和 `task_signal`；`PreparedAgentContext` 定义治理后的 `system_prompt`、`messages`、`tools` 和 `ContextReport`。证据：`src/codepilot/core/types.py:71-98`、`src/codepilot/core/types.py:127-170`。
- `ContextView` 明确把本轮上下文分为 `stable_rules`、`working_state`、`recalled_memory`、`evidence`、`recent_messages`、`tools`。证据：`src/codepilot/protocols/context.py:176-185`。
- `ContextGovernor.prepare()` 是当前主路径：构建 snapshot、召回 memory、计算 pressure、生成 evidence、必要时创建 checkpoint、投影 messages/system prompt，并写入 `ContextReport`。证据：`src/codepilot/sessions/context/governor.py:83-191`。
- `AgentSession._build_context_preparer()` 在 `context_governance_enabled` 为真时创建 `ContextGovernor` 并返回 `context_governor.prepare`，否则才走兼容的 `_prepare_context_for_session()`。证据：`src/codepilot/sessions/session.py:639-652`。
- 会话级历史压缩 `_compact_context_if_needed()` 只在 `context_governor is None` 时作为旧路径执行；当前治理开启时不会把历史消息替换成普通 summary 消息。证据：`src/codepilot/sessions/session.py:548-549`、`src/codepilot/sessions/session.py:571-574`、`src/codepilot/sessions/session.py:888-978`。

从真实运行流程看，系统先保留完整会话，再在每轮模型调用前根据当前任务状态、仓库新鲜度、工具执行证据、记忆命中和 token 压力生成 prompt：低压力时保留较多最近消息；压力升高时减少最近消息、工具输出 artifact 化；critical 时生成结构化 checkpoint，把旧工作历史从 prompt 主体中移出，只留下可恢复引用和当前决策需要的信息。

代码证据：

- 压力分为 `normal / tight / critical`，由 `ContextPressurePolicy.evaluate()` 根据 `context_window - max_output - safety_margin` 后的有效预算、历史 token 和工具输出 token 判断。证据：`src/codepilot/sessions/context/policy.py:11-58`。
- `ContextProjector.project_messages()` 在 `normal/tight/critical` 下分别保留最近 `10/6/4` 条消息，并对工具结果执行投影或保留。证据：`src/codepilot/sessions/context/projector.py:92-113`。
- `ContextGovernor.prepare()` 在 critical pressure 下调用 `ContextCheckpointManager.create()` 创建结构化 checkpoint。证据：`src/codepilot/sessions/context/governor.py:115-131`、`src/codepilot/sessions/context/checkpoint.py:22-45`。

## 1. 核心模块、类、函数和调用入口

| 组件 | 代码位置 | 职责 | 代码证据 |
|---|---|---|---|
| 公共上下文协议 | `src/codepilot/protocols/context.py` | 定义仓库快照、上下文条目、压力、artifact 引用、checkpoint、ContextView 和 ContextReport 等公共契约。 | `RepositorySnapshot` / `RepositoryDelta`：`src/codepilot/protocols/context.py:43-79`；`ContextPressure`：`src/codepilot/protocols/context.py:125-139`；`ContextArtifactRef`：`src/codepilot/protocols/context.py:142-159`；`ContextCheckpoint`：`src/codepilot/protocols/context.py:162-173`；`ContextView`：`src/codepilot/protocols/context.py:176-185`；`ContextReport`：`src/codepilot/protocols/context.py:188-217`。 |
| 核心运行上下文类型 | `src/codepilot/core/types.py` | 定义模型调用前后的上下文载体与 `prepare_context` 回调协议。 | `AgentContext`：`src/codepilot/core/types.py:71-98`；`ContextPreparationRequest`：`src/codepilot/core/types.py:127-143`；`PreparedAgentContext`：`src/codepilot/core/types.py:146-162`；`PrepareContextFn`：`src/codepilot/core/types.py:165-170`。 |
| LLM 调用入口 | `src/codepilot/core/llm_runner.py` | 在每次 provider 调用前注入当前任务、调用 `prepare_context`，并把治理后的上下文转换给 LLM provider。 | `LLMStreamRunner.stream_assistant_response()`：先 `_with_current_task_context()`，再调用 `self._config.prepare_context()`，并发出 `context_prepared` 事件，证据：`src/codepilot/core/llm_runner.py:128-153`；随后转换为 provider `Context`，证据：`src/codepilot/core/llm_runner.py:162-184`。 |
| Session 级治理入口 | `src/codepilot/sessions/session.py` | 负责创建 governor、绑定到 Agent 配置、在 run 生命周期中触发 memory/task recovery/freshness，并保存 context report。 | `AgentSession.__init__()` 初始化 memory/task/context governance，证据：`src/codepilot/sessions/session.py:74-107`；`_build_context_preparer()` 创建并返回 `ContextGovernor.prepare()`，证据：`src/codepilot/sessions/session.py:639-652`；`_on_agent_event()` 保存 `context_prepared` report 并转发 memory retrieved 事件，证据：`src/codepilot/sessions/session.py:612-637`。 |
| 当前主治理器 | `src/codepilot/sessions/context/governor.py` | 上下文治理唯一主入口：整合 snapshot、memory、pressure、projector、checkpoint、report 和 view 持久化。 | `ContextGovernor.__init__()` 创建 `RepositoryTracker`、`ToolArtifactLedger`、`ContextCheckpointManager`、`SessionSnapshotBuilder`、`ContextProjector` 等依赖，证据：`src/codepilot/sessions/context/governor.py:41-81`；`prepare()` 完整投影流程，证据：`src/codepilot/sessions/context/governor.py:83-191`；`_append_context_view()` 写入 `context_views.jsonl`，证据：`src/codepilot/sessions/context/governor.py:231-254`。 |
| Session 快照构建器 | `src/codepilot/sessions/context/snapshot.py` | 从当前消息、仓库状态、工具结果、ledger、checkpoint 和 session state 生成一次治理快照。 | `SessionSnapshot` 字段，证据：`src/codepilot/sessions/context/snapshot.py:23-33`；`SessionSnapshotBuilder.build()` 刷新仓库、观察工具结果、写 ledger、校验新鲜度、读取最新 checkpoint，证据：`src/codepilot/sessions/context/snapshot.py:54-82`。 |
| 压力策略 | `src/codepilot/sessions/context/policy.py` | 根据模型窗口、最大输出、安全余量、历史 token 和工具输出 token 判定 `normal/tight/critical`。 | `ContextPressurePolicy` 默认参数，证据：`src/codepilot/sessions/context/policy.py:11-18`；`evaluate()` 计算 `effective_budget`、pressure ratio、reasons 和 level，证据：`src/codepilot/sessions/context/policy.py:20-58`。 |
| 上下文投影器 | `src/codepilot/sessions/context/projector.py` | 把 snapshot、pressure、memory、evidence、recent messages 投影为 `ContextView`、system prompt 和消息后缀。 | `ContextProjector.project()`，证据：`src/codepilot/sessions/context/projector.py:61-90`；`project_messages()`，证据：`src/codepilot/sessions/context/projector.py:92-113`；`compose_system_prompt()`，证据：`src/codepilot/sessions/context/projector.py:165-178`。 |
| 工具输出 ledger | `src/codepilot/sessions/context/ledger.py` | 把完整工具输出保存为 artifact，把 prompt 中的旧工具结果替换为轻量摘要和 artifact 引用。 | `ToolArtifactLedger.__init__()` 定义 `.codepilot/sessions/<session>/artifacts/tool_outputs` 和 `tool_ledger.jsonl`，证据：`src/codepilot/sessions/context/ledger.py:67-75`；`record_tool_result()` 写 artifact、生成 `ContextArtifactRef` 和 ledger entry，证据：`src/codepilot/sessions/context/ledger.py:77-130`；`project_tool_result()` 替换 tool result 内容，证据：`src/codepilot/sessions/context/ledger.py:132-161`。 |
| checkpoint 管理器 | `src/codepilot/sessions/context/checkpoint.py` | critical pressure 时生成并持久化结构化 checkpoint，后续 prepare 读取最新 checkpoint 作为 working state。 | `ContextCheckpointManager.create()`，证据：`src/codepilot/sessions/context/checkpoint.py:22-45`；`append()` / `load_latest()`，证据：`src/codepilot/sessions/context/checkpoint.py:47-66`。 |
| 仓库状态追踪 | `src/codepilot/sessions/context/repository_tracker.py` | 采集仓库 bootstrap、git status、dirty path hash、指令文件 hash 和目录 fingerprint，并计算 delta。 | `RepositoryTracker.snapshot()`，证据：`src/codepilot/sessions/context/repository_tracker.py:25-60`；`refresh()`，证据：`src/codepilot/sessions/context/repository_tracker.py:62-67`；`compare_snapshots()`，证据：`src/codepilot/sessions/context/repository_tracker.py:70-102`。 |
| Session 上下文状态 | `src/codepilot/sessions/context/state.py` | 维护活跃文件、证据、失效状态和仓库快照；不保存完整聊天历史。 | `SessionContextState` 注释说明“不存储 chat history”，证据：`src/codepilot/sessions/context/state.py:276-327`；`observe_tool_result()` 从工具结果中更新 active files/evidence/verification，证据：`src/codepilot/sessions/context/state.py:329-434`；`validate_sources()` 检查文件 hash、存在性和验证 fingerprint，证据：`src/codepilot/sessions/context/state.py:549-620`。 |
| 记忆召回器 | `src/codepilot/sessions/memory/retriever.py` | 根据当前请求、活跃路径、任务阶段、动作意图和错误状态召回 durable memory。 | `MemoryRetriever.retrieve()` 加载 session/project memory、评分、排序、按类型限额，证据：`src/codepilot/sessions/memory/retriever.py:26-54`；`score_memory_record()` 的路径、关键词、trust、mode、phase/intent/error 加权，证据：`src/codepilot/sessions/memory/retriever.py:68-145`。 |
| 任务控制器 | `src/codepilot/core/task_controller.py` | 维护任务阶段、当前步骤、下一步意图、最近错误和验证状态，并把这些状态渲染进上下文。 | `TaskController.render_context()` 输出 `## Current Task`，证据：`src/codepilot/core/task_controller.py:363-429`；`control_signal()` 生成 `phase`、`action_intent`、`recent_error_code`、`last_decision` 等信号，证据：`src/codepilot/core/task_controller.py:477-509`。 |
| 工具执行协调器 | `src/codepilot/core/tool_coordinator.py` | 执行工具、绑定 `ToolResultMessage`、发出工具结束事件，工具结果再被 snapshot/state/ledger 消费。 | `ToolCallCoordinator.bind_tool_result()`，证据：`src/codepilot/core/tool_coordinator.py:188-214`；`execute_batch()` 执行 batch 并返回工具结果，证据：`src/codepilot/core/tool_coordinator.py:287-300`；`_emit_tool_end()` 输出 affected paths、workspace changed、truncated 等事件字段，证据：`src/codepilot/core/tool_coordinator.py:615-654`。 |
| 静态 runtime prompt | `src/codepilot/runtime/context.py`、`src/codepilot/runtime/prompt.py` | 启动时生成静态仓库上下文、prompt 指南、runtime facts；动态 active files/evidence/memory/current task 不在这里处理。 | `build_runtime_context()` 注释说明动态上下文由 context compiler 处理，证据：`src/codepilot/runtime/context.py:69-102`；`build_runtime_system_prompt()` 组装 identity/safety/repository/memory/runtime facts，证据：`src/codepilot/runtime/prompt.py:124-207`。 |
| runtime assembly | `src/codepilot/runtime/assembly.py` | 创建 `AgentSessionOptions`，把 `context_governance_enabled` 传入 session；当前 `prepare_context=None`，由 session 在拿到 session_id 后创建 governor。 | `build_runtime_context()` / system prompt 创建，证据：`src/codepilot/runtime/assembly.py:120-132`；`AgentSessionOptions(... prepare_context=None ...)`，证据：`src/codepilot/runtime/assembly.py:152-188`。 |
| token 估算与 overflow 工具 | `src/codepilot/llm/overflow.py` | 用字符数粗估 token；消息、图片、工具 schema 都有估算逻辑。 | `CHARS_PER_TOKEN=4`、`IMAGE_TOKEN_ESTIMATE=1000`、`TOOL_SCHEMA_TOKEN_ESTIMATE=200`，证据：`src/codepilot/llm/overflow.py:23-25`；`estimate_message_tokens()`、`estimate_context_tokens()`，证据：`src/codepilot/llm/overflow.py:28-66`。 |
| 遗留 ContextCompiler | `src/codepilot/sessions/context/compiler.py` | 兼容路径中的上下文编译器：按 profile 分配预算、选择 active/evidence/memory/history、生成 report。当前主路径不是它。 | `ContextCompiler.compile()` 的完整旧流程，证据：`src/codepilot/sessions/context/compiler.py:272-454`；`ContextPolicy.profile_for_mode()` 的 plan/act/repair/verify/final/qa profile，证据：`src/codepilot/sessions/context/compiler.py:104-179`。 |
| 遗留会话压缩 | `src/codepilot/sessions/context/compaction.py` | 在 governor 未启用时，把旧历史压缩为 `[Context Summary]` 用户消息，并修复 tool call/tool result 配对。 | `decide_context_compaction()`，证据：`src/codepilot/sessions/context/compaction.py:67-124`；`build_compacted_context()`，证据：`src/codepilot/sessions/context/compaction.py:127-157`；`build_llm_compaction_summary()`，证据：`src/codepilot/sessions/context/compaction.py:160-199`。 |

关于“repo indexer / file selector / prompt builder”的代码事实：

- 当前主路径中存在 `RepositoryTracker`，但它追踪的是仓库快照、git 状态、dirty path hash、指令文件 hash 和目录 fingerprint；代码中未发现独立的语义级 repo indexer 或代码片段索引器。证据：`src/codepilot/sessions/context/repository_tracker.py:25-60`。
- 当前主路径中没有专门读取并注入文件正文片段的 file selector；`ContextProjector.working_state_lines()` 只把 active files 和 changed files 的路径列入 working state。证据：`src/codepilot/sessions/context/projector.py:134-152`。遗留 `ContextCompiler._active_file_items()` 会把活跃文件元信息转为 `ContextItem`，但也不是读取文件正文。证据：`src/codepilot/sessions/context/compiler.py:463-497`。
- 当前 prompt builder 分两层：静态系统 prompt 由 `build_runtime_system_prompt()` 生成，治理后 prompt 由 `ContextProjector.compose_system_prompt()` 追加分层上下文。证据：`src/codepilot/runtime/prompt.py:124-207`、`src/codepilot/sessions/context/projector.py:165-178`。

## 2. 用户请求进入系统后的上下文构建数据流

1. runtime 初始化静态系统 prompt 和 session options。

   代码证据：`build_runtime_context()` 生成 `RuntimeContext`，证据：`src/codepilot/runtime/context.py:69-102`；`build_runtime_system_prompt()` 生成静态 prompt，证据：`src/codepilot/runtime/prompt.py:124-207`；`runtime/assembly.py` 把 `context_governance_enabled` 传给 `AgentSessionOptions` 且 `prepare_context=None`，证据：`src/codepilot/runtime/assembly.py:152-188`。

2. `AgentSession` 初始化时加载持久化消息，并创建上下文治理入口。

   代码证据：`AgentSession.__init__()` 加载 session/context messages 并合并 options messages，证据：`src/codepilot/sessions/session.py:109-114`；`_build_context_preparer()` 创建 `ContextGovernor` 并返回 `prepare`，证据：`src/codepilot/sessions/session.py:639-652`。

3. run 开始前，session 处理 memory admission、task recovery 和 freshness check。

   代码证据：`_start_run_lifecycle()` 依次执行 before hooks、`_admit_prompt_memory()`、`_begin_task_recovery()`、设置 `task_recovery_projection`、执行 `_check_context_freshness()`，证据：`src/codepilot/sessions/session.py:520-550`；`_begin_task_recovery()` 写入任务恢复投影，证据：`src/codepilot/sessions/session.py:708-718`；`_check_context_freshness()` 调用 run store freshness evaluation 并可能加入 steering notice，证据：`src/codepilot/sessions/session.py:836-858`。

4. agent loop 用历史消息、当前用户输入和工具规格构造 `AgentContext`。

   代码证据：`run_agent_loop()` 把 `history_messages`、`prompt_messages`、`system_prompt`、`tools` 放入 `AgentContext`，证据：`src/codepilot/core/agent_loop.py:168-176`；`run_agent_loop_continue()` 继续运行时保留已有 messages，包括工具结果，证据：`src/codepilot/core/agent_loop.py:261-269`。

5. 每次 LLM 调用前，任务控制器把当前任务投影和控制信号塞进 `AgentContext`。

   代码证据：`_run_loop()` 在调用 LLM 前设置 `current_context.current_task = task_controller.render_context()` 和 `current_context.task_signal = task_controller.control_signal()`，证据：`src/codepilot/core/agent_loop.py:530-541`；`TaskController.render_context()` 和 `control_signal()` 的具体字段，证据：`src/codepilot/core/task_controller.py:363-429`、`src/codepilot/core/task_controller.py:477-509`。

6. `LLMStreamRunner` 先把 current task 追加进 system prompt，再调用 `prepare_context`。

   代码证据：`LLMStreamRunner.stream_assistant_response()` 先调用 `_with_current_task_context()`，再构造 `ContextPreparationRequest` 并调用 `self._config.prepare_context()`，证据：`src/codepilot/core/llm_runner.py:128-153`；`_with_current_task_context()` 在 system prompt 中追加 `current_task`，证据：`src/codepilot/core/llm_runner.py:456-500`。

7. `ContextGovernor.prepare()` 从完整上下文投影本轮 prompt。

   代码证据：`prepare()` 调用 `snapshot_builder.build(context)`，证据：`src/codepilot/sessions/context/governor.py:90`；召回 memory，证据：`src/codepilot/sessions/context/governor.py:91-95`；估算 token 并评估 pressure，证据：`src/codepilot/sessions/context/governor.py:97-108`；渲染 evidence，证据：`src/codepilot/sessions/context/governor.py:110-114`；critical 时生成 checkpoint，证据：`src/codepilot/sessions/context/governor.py:115-131`；调用 `projector.project()` 输出 view/messages/system prompt，证据：`src/codepilot/sessions/context/governor.py:133-144`；生成 `ContextReport` 并返回 `PreparedAgentContext`，证据：`src/codepilot/sessions/context/governor.py:145-191`。

8. provider 调用完成后，工具结果进入消息历史；下一轮 prepare 会把这些工具结果转成 state/evidence/ledger。

   代码证据：`ToolCallCoordinator.execute_batch()` 执行工具并产出 tool result messages，证据：`src/codepilot/core/tool_coordinator.py:287-300`；`SessionSnapshotBuilder.build()` 遍历 `context.messages` 中的 `ToolResultMessage`，调用 `state.observe_tool_result()` 和 `ledger.record_tool_result()`，证据：`src/codepilot/sessions/context/snapshot.py:61-71`；`ToolArtifactLedger.record_tool_result()` 写 artifact 和 ledger，证据：`src/codepilot/sessions/context/ledger.py:77-130`。

## 3. 上下文来源分析

当前主路径的上下文来源包括以下几类：

1. 静态系统规则与 runtime facts。

   代码证据：`build_runtime_system_prompt()` 组装 identity、safety、repository、memory、extension、runtime facts，证据：`src/codepilot/runtime/prompt.py:124-207`；`ContextProjector.stable_rules()` 从 system prompt 中抽取规则相关行或前 8 行作为 `stable_rules`，证据：`src/codepilot/sessions/context/projector.py:124-131`。

2. 完整会话消息历史。

   代码证据：`AgentSession.__init__()` 从 store 加载 persisted messages，证据：`src/codepilot/sessions/session.py:109-114`；`AgentContext.messages` 是 LLM 调用前的消息列表，证据：`src/codepilot/core/types.py:71-98`；`ContextProjector.project_messages()` 从完整消息列表投影 recent messages，证据：`src/codepilot/sessions/context/projector.py:92-113`。

3. 当前任务与任务恢复状态。

   代码证据：`AgentContext` 包含 `current_task`、`task_recovery_projection`、`task_signal`，证据：`src/codepilot/core/types.py:89-96`；`TaskController.render_context()` 渲染目标、阶段、步骤、下一步、错误、回滚和约束，证据：`src/codepilot/core/task_controller.py:363-429`；`ContextProjector.working_state_lines()` 把 current task、checkpoint、active files、changed files 放入 working state，证据：`src/codepilot/sessions/context/projector.py:134-152`。

4. 仓库快照与仓库变更。

   代码证据：`RepositoryTracker.snapshot()` 收集 repository bootstrap、git status、dirty path hash、instruction hash、directory fingerprint，证据：`src/codepilot/sessions/context/repository_tracker.py:25-60`；`compare_snapshots()` 产出 added/modified/deleted、branch/head/instructions changed，证据：`src/codepilot/sessions/context/repository_tracker.py:70-102`；`SessionSnapshot` 保存 `repository_snapshot` 和 `repository_delta`，证据：`src/codepilot/sessions/context/snapshot.py:23-33`。

5. 活跃文件与变更文件。

   代码证据：`SessionContextState.observe_tool_result()` 从工具结果 metadata/details 中提取 file state 和 affected paths，并调用 `touch_file()`，证据：`src/codepilot/sessions/context/state.py:374-397`；`SessionSnapshotBuilder.build()` 把 `state.ranked_active_files()` 和 `state.changed_paths` 写入 snapshot，证据：`src/codepilot/sessions/context/snapshot.py:74-82`；`ContextProjector.working_state_lines()` 把前 12 个 active/changed files 写入 working state，证据：`src/codepilot/sessions/context/projector.py:146-152`。

6. 工具结果、验证结果和 artifact 引用。

   代码证据：`SessionContextState.observe_tool_result()` 把工具结果加入 evidence，并在 verification 类结果中追加验证证据，证据：`src/codepilot/sessions/context/state.py:404-434`；`ToolArtifactLedger.record_tool_result()` 把完整输出写入 artifact 并记录 `ContextArtifactRef`，证据：`src/codepilot/sessions/context/ledger.py:77-130`；`ContextProjector.render_evidence()` 渲染 fresh evidence、artifact refs 和 stale items，证据：`src/codepilot/sessions/context/projector.py:41-59`。

7. 记忆召回结果。

   代码证据：`ContextGovernor._recall_memory()` 构造 `MemoryQuery`，使用最新用户文本、active paths、task phase、action intent、recent error 和 context mode 召回记忆，证据：`src/codepilot/sessions/context/governor.py:196-214`；`MemoryRetriever.retrieve()` 加载 session/project memory、评分、排序并按 kind 限额，证据：`src/codepilot/sessions/memory/retriever.py:26-54`。

8. checkpoint。

   代码证据：`SessionSnapshotBuilder.build()` 读取 `checkpoints.load_latest()` 并放入 snapshot，证据：`src/codepilot/sessions/context/snapshot.py:74-82`；`ContextCheckpointManager.load_latest()` 返回最新 checkpoint，证据：`src/codepilot/sessions/context/checkpoint.py:64-66`；`ContextProjector.working_state_lines()` 把 checkpoint goal 和 next actions 放入 working state，证据：`src/codepilot/sessions/context/projector.py:141-145`。

9. 工具规格。

   代码证据：`AgentContext.tools` 保存工具定义，证据：`src/codepilot/core/types.py:84`；`ContextGovernor.prepare()` 原样返回 `tools=list(context.tools)`，证据：`src/codepilot/sessions/context/governor.py:188-191`；`LLMStreamRunner.stream_assistant_response()` 把治理后的 tools 放入 provider `Context`，证据：`src/codepilot/core/llm_runner.py:180-184`。

代码中未发现明确实现：当前主路径没有把代码库中“相关函数正文/代码片段”作为独立上下文来源自动注入 prompt；它主要注入 active file 路径、changed file 路径、工具证据和 artifact 摘要。证据：`ContextProjector.working_state_lines()` 只渲染路径和状态，证据：`src/codepilot/sessions/context/projector.py:134-152`。

## 4. 上下文选择策略分析

### 4.1 stable rules 的选择

当前主路径不是重写 system prompt，而是在原始 system prompt 后追加治理分层；同时从 system prompt 中抽取一组 `stable_rules` 放入 `ContextView` 报告。抽取规则是：包含 `AGENTS`、`CLAUDE`、`rule` 的行优先，否则取前 8 个非空行。

代码证据：

- `ContextProjector.stable_rules()` 的抽取逻辑，证据：`src/codepilot/sessions/context/projector.py:124-131`。
- `ContextProjector.compose_system_prompt()` 把 `stable_rules`、`working_state`、`memory`、`evidence`、`recent_turns` 追加到原 system prompt 后，证据：`src/codepilot/sessions/context/projector.py:165-178`。

### 4.2 working state 的选择

working state 只选择当前任务、最近 checkpoint、active files 前 12 个、changed files 前 12 个，不直接注入文件正文。

代码证据：

- `ContextProjector.working_state_lines()` 对 current task、checkpoint、active files、changed files 的选择，证据：`src/codepilot/sessions/context/projector.py:134-152`。
- active files 来源于 `SessionContextState.observe_tool_result()` 和 `touch_file()`，证据：`src/codepilot/sessions/context/state.py:329-397`、`src/codepilot/sessions/context/state.py:436-494`。
- active files 会被排序和裁剪；`DEFAULT_MAX_ACTIVE_FILES = 40`，并按 role/freshness/access_count/last_accessed/path 排序。证据：`src/codepilot/sessions/context/state.py:75-88`、`src/codepilot/sessions/context/state.py:496-509`、`src/codepilot/sessions/context/state.py:666-673`。

### 4.3 evidence 的选择

主路径只把 freshness 为 `fresh` 的 evidence 放入 evidence 区，并附带最多 8 条 artifact refs 和最多 8 条 stale items；大段 evidence 内容会被压缩为错误/失败相关行或 “archived” 摘要。

代码证据：

- `ContextProjector.render_evidence()` 跳过非 fresh evidence，追加 artifacts 和 stale items，证据：`src/codepilot/sessions/context/projector.py:41-59`。
- `_compact_evidence_content()` 对超过 400 字符的证据做压缩，优先保留包含 `error/failed/traceback/exception/assert` 的行，否则写入 “N lines, N chars archived”，证据：`src/codepilot/sessions/context/projector.py:333-355`。
- evidence 由 `SessionContextState.observe_tool_result()` 生成，工具结果证据和验证证据分别追加，证据：`src/codepilot/sessions/context/state.py:404-434`。

### 4.4 recent messages 的选择

主路径按压力等级选择最近消息数量：`normal=10`、`tight=6`、`critical=4`。如果工具结果已经较旧或压力不是 normal，会用 ledger projection 替代完整工具输出；同时会修复 tool result 与 assistant tool call 的配对，避免 provider 消息序列非法。

代码证据：

- `ContextProjector.project_messages()` 的最近消息数和工具结果投影逻辑，证据：`src/codepilot/sessions/context/projector.py:92-113`。
- `ContextProjector.repair_tool_pairs()` 保证 tool result 有对应 assistant tool call，证据：`src/codepilot/sessions/context/projector.py:240-253`。
- `ContextProjector.message_text()` 对 tool result 只保留 tool、status、paths、verification、error、artifact 等轻量字段，证据：`src/codepilot/sessions/context/projector.py:267-286`。

### 4.5 工具输出的选择

工具输出完整内容保存到 artifact，prompt 中默认使用摘要、artifact path、status、verification 等投影文本；`normal` 压力下最近 4 条消息内的工具结果可短期保留原文。

代码证据：

- `ToolArtifactLedger.record_tool_result()` 生成 artifact、summary、visible projection 和 `ContextArtifactRef`，证据：`src/codepilot/sessions/context/ledger.py:77-130`。
- `ToolArtifactLedger.project_tool_result()` 在 `preserve_full=False` 时用 projection 替换工具结果内容，证据：`src/codepilot/sessions/context/ledger.py:132-161`。
- `_projection_text()` 的 prompt 可见内容包括 `[Tool output archived]`、tool/status/artifact/summary/verification，证据：`src/codepilot/sessions/context/ledger.py:214-231`。
- `ContextProjector.project_messages()` 的 `preserve_full = pressure.level == "normal" and index >= len(messages) - 4`，证据：`src/codepilot/sessions/context/projector.py:102-108`。

### 4.6 记忆召回的选择

记忆召回以最新用户文本、活跃文件路径、任务阶段、动作意图、最近错误、context mode 为 query；召回器按路径匹配、关键词匹配、项目 gate、trust、mode、经验适用条件评分，并按 memory kind 限额。

代码证据：

- `ContextGovernor._recall_memory()` 构造 `MemoryQuery`，证据：`src/codepilot/sessions/context/governor.py:196-214`。
- `MemoryRetriever.retrieve()` 的加载、评分、排序、kind limit，证据：`src/codepilot/sessions/memory/retriever.py:26-54`。
- `MemoryRetriever.score_memory_record()` 对 active path、keywords、project gate、trust、retrieval mode、phase/intent/error 的加权，证据：`src/codepilot/sessions/memory/retriever.py:68-145`。
- `_apply_kind_limits()` 限制 `experience=2`、`decision=2`、`project=3`，证据：`src/codepilot/sessions/memory/retriever.py:190-209`。

### 4.7 遗留 ContextCompiler 的选择策略

代码中仍保留 `ContextCompiler`，它的选择策略更接近“按分区预算选条目”：先按 context mode 分配 repository/active_files/recent_evidence/memory/history/task 预算，再按 priority 和 freshness 选择 `ContextItem`，历史消息则从末尾按预算保留。

代码证据：

- `ContextPolicy.profile_for_mode()` 定义 `plan/act/repair/verify/final/qa` 的分区预算比例，证据：`src/codepilot/sessions/context/compiler.py:104-179`。
- `ContextCompiler.compile()` 旧流程中编译 active/evidence/memory/history 并生成 report，证据：`src/codepilot/sessions/context/compiler.py:272-454`。
- `_select_items()` 按 priority 降序、跳过 stale/missing、超预算丢弃或截断第一条，证据：`src/codepilot/sessions/context/compiler.py:640-677`。
- `_select_message_suffix()` 从消息尾部保留历史，并确保最新用户消息保留，证据：`src/codepilot/sessions/context/compiler.py:680-721`。

当前默认主路径是 `ContextGovernor`；`ContextCompiler` 只在 `context_governance_enabled=False` 或外部显式传入旧 `prepare_context` 时走兼容路径。证据：`src/codepilot/sessions/session.py:639-652`、`src/codepilot/runtime/assembly.py:152-188`。

## 5. token 预算和上下文压缩机制

### 5.1 token 估算

项目使用粗粒度 token 估算：默认 `4 chars = 1 token`，图片按 1000 token，工具 schema 按 200 token。

代码证据：

- 常量定义：`CHARS_PER_TOKEN=4`、`IMAGE_TOKEN_ESTIMATE=1000`、`TOOL_SCHEMA_TOKEN_ESTIMATE=200`，证据：`src/codepilot/llm/overflow.py:23-25`。
- `estimate_message_tokens()` 统计 user/assistant/tool 文本、图片、tool call 参数和名称，证据：`src/codepilot/llm/overflow.py:28-53`。
- `estimate_context_tokens()` 统计 system prompt、messages、tools schema，证据：`src/codepilot/llm/overflow.py:56-66`。

### 5.2 pressure 计算

当前主路径不直接等到 overflow 才处理，而是在每轮 prepare 中计算有效输入预算和压力等级。有效预算为 `context_window - max_output - safety_margin`，默认 safety margin 是 1024；超过 72% 进入 tight，超过 90% 进入 critical；工具输出超过有效预算 25% 或历史消息超过有效预算 50% 也会触发 tight reasons。

代码证据：

- `ContextPressurePolicy` 默认阈值，证据：`src/codepilot/sessions/context/policy.py:11-18`。
- `evaluate()` 计算 `effective_budget`、`pressure_ratio`、`tool_output_pressure`、`history_pressure` 和 level，证据：`src/codepilot/sessions/context/policy.py:20-58`。
- `ContextGovernor.prepare()` 在评估 pressure 前计算 `tool_output_tokens`、`history_tokens` 和 `estimated_total`，证据：`src/codepilot/sessions/context/governor.py:97-108`。

### 5.3 裁剪策略

当前主路径的裁剪主要体现在三处：recent messages 数量随 pressure 降低；工具输出 artifact 化；evidence 大内容压缩。它没有对所有 system prompt 分层做严格 hard budget 截断。

代码证据：

- recent messages 裁剪：`normal=10`、`tight=6`、`critical=4`，证据：`src/codepilot/sessions/context/projector.py:92-113`。
- 工具输出 artifact 化：`ToolArtifactLedger.project_tool_result()`，证据：`src/codepilot/sessions/context/ledger.py:132-161`。
- 大 evidence 压缩：`_compact_evidence_content()`，证据：`src/codepilot/sessions/context/projector.py:333-355`。
- `ContextProjector.section_reports()` 生成每层预算报告，但其预算是 `pressure.effective_budget // 6` 的报告值，并未在该函数中执行硬截断。证据：`src/codepilot/sessions/context/projector.py:181-201`。

### 5.4 critical 下的结构化压缩

critical pressure 下，当前主路径会创建结构化 checkpoint，内容包括目标、active files、changed files、key evidence、verification state、open questions、next actions、source refs。后续 prepare 会优先读取最新 checkpoint 并放入 working state。

代码证据：

- `ContextGovernor.prepare()` 在 `pressure.level == "critical"` 时创建 checkpoint，证据：`src/codepilot/sessions/context/governor.py:115-131`。
- `ContextCheckpointManager.create()` 的字段，证据：`src/codepilot/sessions/context/checkpoint.py:22-45`。
- `SessionSnapshotBuilder.build()` 加载 latest checkpoint，证据：`src/codepilot/sessions/context/snapshot.py:74-82`。
- `ContextProjector.working_state_lines()` 渲染 checkpoint goal 和 next actions，证据：`src/codepilot/sessions/context/projector.py:141-145`。

### 5.5 遗留会话压缩

旧路径中存在会话级压缩：当消息数或 token 数达到阈值时，调用 LLM 生成 summary 或 fallback summary，再用 `[Context Summary]` 用户消息替换旧历史并保留最近消息。这条路径当前只在没有 `ContextGovernor` 时执行。

代码证据：

- `AgentSession._compact_context_if_needed()` 估算 token、调用 `decide_context_compaction()`、生成 summary、`build_compacted_context()` 并重写 messages，证据：`src/codepilot/sessions/session.py:908-978`。
- `decide_context_compaction()` 的 message/token 阈值和 retain_recent 策略，证据：`src/codepilot/sessions/context/compaction.py:67-124`。
- `build_compacted_context()` 插入 `[Context Summary]` 并修复 tool pair，证据：`src/codepilot/sessions/context/compaction.py:127-157`。
- `build_llm_compaction_summary()` 使用独立 summary prompt 调用 LLM，证据：`src/codepilot/sessions/context/compaction.py:160-199`。
- 当前治理开启时不走这条路径，证据：`src/codepilot/sessions/session.py:548-549`、`src/codepilot/sessions/session.py:571-574`。

### 5.6 prefix caching 相关实现

当前代码记录了 `prefix_hash` 和 `dynamic_hash`，用于观测稳定前缀和动态后缀变化；代码中未发现把这些 hash 映射到 provider cache metadata 的明确实现。

代码证据：

- `ContextGovernor.prepare()` 计算 `prefix_hash = _hash_text(projection.system_prompt)` 和 `dynamic_hash = _hash_text("\n".join(evidence_lines + memory_lines + projection.recent_summary_lines))`，证据：`src/codepilot/sessions/context/governor.py:145-157`。
- `ContextReport` 包含 `prefix_hash` 和 `dynamic_hash` 字段，证据：`src/codepilot/protocols/context.py:207-214`。
- `LLMStreamRunner.stream_assistant_response()` 只把 prepared system_prompt/messages/tools 放入 provider `Context`，没有 cache metadata 字段，证据：`src/codepilot/core/llm_runner.py:162-184`。

## 6. 上下文更新与失效机制

### 6.1 仓库快照更新

每次 prepare 时，snapshot builder 会刷新仓库状态，并把 delta 应用于 session state：有路径变更则使路径相关上下文失效，仓库变化则使 verification 失效。

代码证据：

- `SessionSnapshotBuilder.build()` 调用 `repository.refresh()` 并设置 `state.last_repository_snapshot`，证据：`src/codepilot/sessions/context/snapshot.py:54-56`。
- 发生 delta 时调用 `state.invalidate_paths(delta.modified_paths + delta.deleted_paths)` 和 `state.invalidate_verification()`，证据：`src/codepilot/sessions/context/snapshot.py:57-59`。
- `RepositoryTracker.compare_snapshots()` 计算 changed、added、modified、deleted、branch/head/instruction 变化，证据：`src/codepilot/sessions/context/repository_tracker.py:70-102`。

### 6.2 工具结果驱动的状态更新

工具结果会更新 active files、changed paths、evidence 和 verification。若工具修改 workspace，会让相关路径与验证状态失效。

代码证据：

- `SessionContextState.observe_tool_result()` 从 tool result details/metadata 读取 file state 和 affected paths，证据：`src/codepilot/sessions/context/state.py:374-385`。
- workspace changed 时 `invalidate_paths()` 和 `invalidate_verification()`，证据：`src/codepilot/sessions/context/state.py:399-402`。
- 工具结果 evidence 和 verification evidence 的写入，证据：`src/codepilot/sessions/context/state.py:404-434`。

### 6.3 文件与验证新鲜度校验

`validate_sources()` 会检查 file summaries 和 active files 的文件存在性、source hash；verification evidence 会和当前 repository fingerprint 比较，不一致则标记 stale。

代码证据：

- file summaries 的 exists/hash 校验，证据：`src/codepilot/sessions/context/state.py:571-588`。
- active files 的 exists/hash 校验，证据：`src/codepilot/sessions/context/state.py:590-604`。
- verification evidence 的 repository fingerprint 校验，证据：`src/codepilot/sessions/context/state.py:606-618`。

### 6.4 memory 新鲜度校验

每次 recall 前会调用 `memory_retriever.validate_freshness()`；召回时只保留 durable 且 active 的 memory。

代码证据：

- `ContextGovernor._recall_memory()` 调用 `self.memory_retriever.validate_freshness()`，证据：`src/codepilot/sessions/context/governor.py:196-201`。
- 同函数过滤 `record.kind in DURABLE_MEMORY_KINDS and record.status == "active"`，证据：`src/codepilot/sessions/context/governor.py:205-214`。
- `MemoryRetriever.validate_freshness()` 委托 writer 校验，证据：`src/codepilot/sessions/memory/retriever.py:56-61`。

### 6.5 session/run 层 freshness steering

run 生命周期开始时还会调用 session store 的 freshness evaluation，如果存在 stale runs，会向当前请求追加 steering notice，提示不要信任过期上下文。

代码证据：

- `_start_run_lifecycle()` 调用 `_check_context_freshness()`，证据：`src/codepilot/sessions/session.py:546-547`。
- `_check_context_freshness()` 调用 `self.run_store.evaluate_freshness()`，并在存在 stale runs 时 append event 和 `_add_context_steering_notice()`，证据：`src/codepilot/sessions/session.py:836-858`。

代码中未发现明确实现：当前主路径没有针对每个 `ContextView` 的 TTL 清理策略，也没有对 `context_views.jsonl`、`tool_ledger.jsonl`、`checkpoints.jsonl` 的自动 retention/pruning。相关写入证据分别是 `src/codepilot/sessions/context/governor.py:231-254`、`src/codepilot/sessions/context/ledger.py:178-181`、`src/codepilot/sessions/context/checkpoint.py:47-50`。

## 7. 不同任务阶段的上下文差异

当前主路径存在“阶段信号影响上下文”的机制，但没有为 read-only、plan、execute、replan 分别实现完全独立的上下文构建管线。

### 7.1 当前主路径的阶段差异

主路径通过 `task_signal` 和 `context_mode` 间接改变上下文：如果存在 recent error 或 debug intent，则进入 `repair`；如果 action intent 包含 verify，则进入 `verify`；如果用户文本像问答，则进入 `qa`；否则是 `act`。这个 mode 会影响 memory query 和 memory scoring，也会影响 `ContextReport.context_mode`。

代码证据：

- `TaskController.control_signal()` 输出 `phase`、`action_intent`、`recent_error_code`、`last_decision` 等信号，证据：`src/codepilot/core/task_controller.py:477-509`。
- `ContextProjector.context_mode()` 根据 `recent_error`、`action_intent` 和最新用户文本返回 `repair/verify/qa/act`，证据：`src/codepilot/sessions/context/projector.py:320-330`。
- `ContextGovernor._recall_memory()` 把 `context_mode` 写入 `MemoryQuery.retrieval_mode`，证据：`src/codepilot/sessions/context/governor.py:196-214`。
- `MemoryRetriever.score_memory_record()` 对 `qa/repair/verify` retrieval mode 加权，证据：`src/codepilot/sessions/memory/retriever.py:106-114`。

### 7.2 任务控制器中的阶段

任务控制器内部会根据工具执行结果进入 acting、verifying、repair、replan、propose_revert、finished 等决策路径；这些结果通过 `current_task` 和 `task_signal` 影响上下文。

代码证据：

- `TaskController.after_tool_results()` 根据 cancelled/denied/tool_error/verification_failed/workspace_changed 等结果决定 next action 和 phase，证据：`src/codepilot/core/task_controller.py:130-274`。
- `TaskController.check_completion()` 会检查 modified files 是否缺少 fresh verification，证据：`src/codepilot/core/task_controller.py:276-336`。
- `TaskController.render_context()` 把 phase、next action、action intent、recent error、rollback guidance 等渲染给模型，证据：`src/codepilot/core/task_controller.py:363-429`。

### 7.3 遗留 ContextCompiler 的阶段 profile

遗留 `ContextCompiler` 有更明确的阶段 profile：`plan/act/repair/verify/final/qa` 会分配不同的 repository、active_files、recent_evidence、memory、history、task 预算比例。

代码证据：

- `ContextPolicy.profile_for_mode()` 的六种 profile，证据：`src/codepilot/sessions/context/compiler.py:104-179`。
- `_resolve_context_mode()` 根据 `last_decision`、`action_intent`、`recent_error`、`phase` 和用户文本选择 `repair/verify/final/plan/qa/act`，证据：`src/codepilot/sessions/context/compiler.py:854-872`。

代码中未发现明确实现：

- 当前 `ContextGovernor` 主路径没有单独的 read-only 上下文构建策略；只看到 `ContextProjector.context_mode()` 的 `repair/verify/qa/act`。证据：`src/codepilot/sessions/context/projector.py:320-330`。
- 当前 `ContextGovernor` 主路径没有显式的 plan/replan profile 分区预算；profile 机制存在于遗留 `ContextCompiler`，不是 governor 主路径。证据：`src/codepilot/sessions/context/compiler.py:104-179`、`src/codepilot/sessions/session.py:639-652`。

## 8. 上下文管理与工具系统、任务控制、记忆系统的关系

### 8.1 与工具系统的关系

工具系统负责执行并产生 `ToolResultMessage`；上下文治理在下一轮 prepare 时消费这些 tool result，更新 active files/evidence，并把完整输出 ledger 化。

代码证据：

- `ToolCallCoordinator.bind_tool_result()` 绑定 tool_call_id、tool_name、status，证据：`src/codepilot/core/tool_coordinator.py:188-214`。
- `SessionSnapshotBuilder.build()` 遍历 `ToolResultMessage` 并调用 `state.observe_tool_result()` 与 `ledger.record_tool_result()`，证据：`src/codepilot/sessions/context/snapshot.py:61-71`。
- `ToolArtifactLedger.record_tool_result()` 把完整输出写入 artifact，同时生成 prompt 可见摘要，证据：`src/codepilot/sessions/context/ledger.py:77-130`。

### 8.2 与任务控制的关系

任务控制器不直接选择文件或裁剪消息，但它给上下文治理提供当前目标、阶段、下一步、错误和验证意图；这些信号影响 working state、context mode 和 memory recall。

代码证据：

- `_run_loop()` 每轮调用前设置 `current_task` 和 `task_signal`，证据：`src/codepilot/core/agent_loop.py:530-541`。
- `ContextProjector.working_state_lines()` 把 `context.current_task` 放入 working state，证据：`src/codepilot/sessions/context/projector.py:134-143`。
- `ContextGovernor._recall_memory()` 把 `task_phase`、`action_intent`、`recent_error` 传给 `MemoryQuery`，证据：`src/codepilot/sessions/context/governor.py:196-214`。

### 8.3 与记忆系统的关系

上下文治理只负责召回和注入记忆，不负责失败记忆如何总结。prompt admission 和 run completion 会触发 memory writer；prepare 阶段只通过 `MemoryRetriever` 读取 memory，并把召回结果渲染为 `Memory Recall`。

代码证据：

- `AgentSession._start_run_lifecycle()` 调用 `_admit_prompt_memory()`，证据：`src/codepilot/sessions/session.py:520-550`。
- `AgentSession._complete_run_lifecycle()` 在 run 完成后 finalize memory，证据：`src/codepilot/sessions/session.py:552-580`。
- `ContextGovernor._recall_memory()` 和 `_render_recalled_memory()` 只召回并渲染，证据：`src/codepilot/sessions/context/governor.py:196-229`。
- `MemoryRetriever.retrieve()` 读取并排序 memory，证据：`src/codepilot/sessions/memory/retriever.py:26-54`。

### 8.4 与持久化和观测的关系

上下文治理把每轮投影、tool ledger、checkpoint 都写成 JSONL 或 artifact，`ContextReport` 则记录 token、pressure、selected/dropped/stale、memory hits、artifact refs、hash 等观测信息。

代码证据：

- `ContextGovernor._append_context_view()` 写 `context_views.jsonl`，证据：`src/codepilot/sessions/context/governor.py:231-254`。
- `ToolArtifactLedger.__init__()` / `record_tool_result()` 写 artifacts 和 `tool_ledger.jsonl`，证据：`src/codepilot/sessions/context/ledger.py:67-130`。
- `ContextCheckpointManager.append()` 写 `checkpoints.jsonl`，证据：`src/codepilot/sessions/context/checkpoint.py:47-50`。
- `ContextReport` 字段包含 pressure、context_view、checkpoint、artifact_refs、tokens_by_layer、prefix_hash、dynamic_hash，证据：`src/codepilot/protocols/context.py:188-217`。
- `AgentSession._on_agent_event()` 保存最新 context report，并发出 memory retrieved 事件，证据：`src/codepilot/sessions/session.py:612-637`。

## 9. 当前上下文管理的优点和缺陷

### 9.1 优点

1. 已经把“完整会话事实源”和“本轮 prompt 投影”分离。

   代码证据：`AgentSession` 加载和保存完整 messages，证据：`src/codepilot/sessions/session.py:109-114`、`src/codepilot/sessions/session.py:612-637`；`ContextGovernor.prepare()` 只返回本轮 `PreparedAgentContext`，证据：`src/codepilot/sessions/context/governor.py:83-191`；当前 governor 开启时不走会话级 summary 替换，证据：`src/codepilot/sessions/session.py:548-549`、`src/codepilot/sessions/session.py:571-574`。

2. 已经实现压力感知的 prompt 投影。

   代码证据：`ContextPressurePolicy.evaluate()` 定义 `normal/tight/critical`，证据：`src/codepilot/sessions/context/policy.py:20-58`；`ContextProjector.project_messages()` 根据 pressure 改变 recent messages 数量和工具输出保留策略，证据：`src/codepilot/sessions/context/projector.py:92-113`；critical 时创建 checkpoint，证据：`src/codepilot/sessions/context/governor.py:115-131`。

3. 工具输出 ledger 化可以降低日志噪声，并保留恢复路径。

   代码证据：完整输出写 artifact，证据：`src/codepilot/sessions/context/ledger.py:88-118`；prompt projection 只保留摘要和 artifact path，证据：`src/codepilot/sessions/context/ledger.py:214-231`；`ContextProjector.render_evidence()` 也只列 artifact refs，证据：`src/codepilot/sessions/context/projector.py:41-59`。

4. 有新鲜度和失效机制，能减少旧验证或旧文件状态污染。

   代码证据：仓库 delta 触发 path/verification invalidation，证据：`src/codepilot/sessions/context/snapshot.py:57-59`；`validate_sources()` 检查文件 hash、存在性、verification fingerprint，证据：`src/codepilot/sessions/context/state.py:549-620`；memory recall 前校验 freshness，证据：`src/codepilot/sessions/context/governor.py:196-201`。

5. 记忆召回已经利用任务阶段、错误和活跃路径，不是简单全文塞入。

   代码证据：`ContextGovernor._recall_memory()` 构造含 active paths 和 task signal 的 query，证据：`src/codepilot/sessions/context/governor.py:196-214`；`MemoryRetriever.score_memory_record()` 对路径、关键词、trust、mode、phase/intent/error 评分，证据：`src/codepilot/sessions/memory/retriever.py:68-145`。

6. `ContextReport` 观测信息较完整，有利于调试上下文治理效果。

   代码证据：`ContextGovernor.prepare()` 写入 pressure、sections、selected_items、stale_items、dropped_items、retrieved_memory_ids、context_view、checkpoint、artifact_refs、tokens_by_layer、hash 等字段，证据：`src/codepilot/sessions/context/governor.py:158-184`；协议字段定义，证据：`src/codepilot/protocols/context.py:188-217`。

### 9.2 缺陷与可能问题

1. 当前 governor 主路径没有严格执行总 token hard cap，存在 prompt 仍过长的风险。

   具体问题：`ContextPressurePolicy` 会计算压力，`ContextProjector` 会减少 recent messages、压缩 evidence 和 artifact 化工具输出；但 `compose_system_prompt()` 仍把原始 system prompt 加上所有治理段落拼接，`section_reports()` 只是报告层预算，没有在每层执行硬截断。因此如果静态 prompt、memory、evidence 或 recent summary 本身很长，仍可能超过 provider 窗口。

   代码证据：pressure 计算，证据：`src/codepilot/sessions/context/policy.py:20-58`；system prompt 直接拼接，证据：`src/codepilot/sessions/context/projector.py:165-178`；section report 只计算报告，证据：`src/codepilot/sessions/context/projector.py:181-201`；`ContextGovernor.prepare()` 计算 after tokens 后没有二次 overflow fallback，证据：`src/codepilot/sessions/context/governor.py:145-191`。

2. stable rules 的保护仍偏启发式，不能保证 AGENTS.md 等指令文件逐字、稳定、按 source hash 注入。

   具体问题：`stable_rules()` 只是从 system prompt 文本中挑行，不是从指令文件源读取并以 hash 固定顺序注入；`RepositoryTracker` 虽然记录 instruction hashes，但 `ContextProjector` 没有基于这些 hash 重建 stable rules。因此规则不一定完整，也可能依赖 runtime prompt 已经如何拼接。

   代码证据：`stable_rules()` 行选择，证据：`src/codepilot/sessions/context/projector.py:124-131`；`RepositoryTracker.snapshot()` 记录 `instruction_hashes`，证据：`src/codepilot/sessions/context/repository_tracker.py:25-60`；`compose_system_prompt()` 只追加文本段，证据：`src/codepilot/sessions/context/projector.py:165-178`。

3. 当前 governor 主路径没有自动注入活跃文件的代码正文，可能导致模型知道文件名但缺少决策证据。

   具体问题：working state 只列 active/changed file path；文件内容仍要依赖后续工具读取。如果模型在下一步需要基于代码细节决策，可能出现“知道该看哪个文件，但本轮 prompt 没有代码证据”的情况。

   代码证据：`working_state_lines()` 只输出 active/changed file path，证据：`src/codepilot/sessions/context/projector.py:146-152`；`SessionContextState.touch_file()` 记录 path、role、reason、source_hash 等元信息，证据：`src/codepilot/sessions/context/state.py:436-494`；代码中未发现 governor 主路径读取文件正文并注入 prompt 的实现。

4. current task 可能重复进入 system prompt。

   具体问题：`LLMStreamRunner` 在调用 `prepare_context` 前已经通过 `_with_current_task_context()` 把 current task 追加到 system prompt；随后 `ContextProjector.working_state_lines()` 又把 `context.current_task` 放进 `Working State`，`compose_system_prompt()` 再追加到 system prompt 后部。这样当前任务可能在同一轮 prompt 中出现两次。

   代码证据：调用 prepare 前注入 current task，证据：`src/codepilot/core/llm_runner.py:128-153`、`src/codepilot/core/llm_runner.py:456-500`；working state 再加入 current task，证据：`src/codepilot/sessions/context/projector.py:134-143`；compose system prompt 追加 working state，证据：`src/codepilot/sessions/context/projector.py:165-178`。

5. checkpoint 和 context view 的持久化缺少 retention/pruning，长会话可能导致治理日志继续膨胀。

   具体问题：`context_views.jsonl`、`checkpoints.jsonl`、`tool_ledger.jsonl` 都是 append-only。代码中未发现按数量、大小或时间清理这些文件的策略。虽然这些文件不一定每轮进入 prompt，但会影响磁盘、resume 和调试成本。

   代码证据：`_append_context_view()` append JSONL，证据：`src/codepilot/sessions/context/governor.py:231-254`；`ContextCheckpointManager.append()` append JSONL，证据：`src/codepilot/sessions/context/checkpoint.py:47-50`；`ToolArtifactLedger._append()` append JSONL，证据：`src/codepilot/sessions/context/ledger.py:178-181`；代码中未发现 retention/pruning 实现。

6. tool ledger 读取方式是全量读取，长会话下可能带来额外开销。

   具体问题：`ToolArtifactLedger.load_entries()` 每次读取整个 `tool_ledger.jsonl`；`artifact_refs()` 基于 `load_entries()` 返回全部 entries。长会话工具调用很多时，prepare 中读取 artifact refs 可能变慢，report/artifact refs 也可能变大。

   代码证据：`load_entries()` 全量遍历文件行，证据：`src/codepilot/sessions/context/ledger.py:163-173`；`artifact_refs()` 返回所有 entries 的 artifact_ref，证据：`src/codepilot/sessions/context/ledger.py:175-176`；`SessionSnapshotBuilder.build()` 每轮调用 `ledger.artifact_refs()`，证据：`src/codepilot/sessions/context/snapshot.py:74-82`。

7. prefix caching 目前停留在 hash 观测层，未接入 provider 级缓存控制。

   具体问题：`ContextReport` 记录了 `prefix_hash` / `dynamic_hash`，但 provider 调用时没有传 cache metadata。这样只能观察稳定性，不能保证模型 provider 的缓存命中。

   代码证据：hash 计算，证据：`src/codepilot/sessions/context/governor.py:145-157`；report 字段，证据：`src/codepilot/protocols/context.py:207-214`；provider `Context` 只包含 system_prompt/messages/tools，证据：`src/codepilot/core/llm_runner.py:162-184`。

8. 当前主路径的阶段隔离较弱，read-only、plan、execute、replan 没有独立上下文预算和选择策略。

   具体问题：governor 主路径只有 `repair/verify/qa/act` 的轻量 `context_mode`，主要影响 memory query 和 report；并没有像遗留 `ContextCompiler` 那样为 `plan/repair/verify/final/qa` 设置分区预算 profile。因此 plan 或 replan 阶段可能无法自动提高架构信息、历史决策或错误证据的预算。

   代码证据：governor 主路径的 `context_mode()`，证据：`src/codepilot/sessions/context/projector.py:320-330`；memory recall 使用该 mode，证据：`src/codepilot/sessions/context/governor.py:196-214`；遗留 profile 存在于 `ContextCompiler`，证据：`src/codepilot/sessions/context/compiler.py:104-179`；session 默认使用 governor，证据：`src/codepilot/sessions/session.py:639-652`。

9. 遗留 `ContextCompiler` 与当前 `ContextGovernor` 并存，概念上容易混淆。

   具体问题：代码中同时存在 `ContextCompiler`、`ContextPolicy`、`ContextItemSection` 和新的 `ContextGovernor`、`ContextPressurePolicy`、`ContextProjector`。如果后续开发者只看旧文档或旧测试，容易误以为 `ContextCompiler.compile()` 仍是主链路。

   代码证据：遗留 compiler 文件仍存在并被导出/测试引用，证据：`src/codepilot/sessions/context/compiler.py:240-454`；session 主路径改为 `ContextGovernor`，证据：`src/codepilot/sessions/session.py:639-652`；runtime assembly 不再传旧 `prepare_context`，证据：`src/codepilot/runtime/assembly.py:152-188`。

## 10. 可写入简历或项目文档的总结

针对 Coding Agent 在多轮代码任务中容易出现关键信息遗漏、无关上下文污染和过期信息干扰的问题，项目设计了以 Session 事实源为基础的上下文分层投影与压力感知治理机制，通过 `ContextGovernor` 在每次模型调用前汇总仓库快照、工具输出 ledger、任务状态、记忆召回、checkpoint 和新鲜度证据，并结合 token 预算评估、压力裁剪、结构化 checkpoint 与可审计 `ContextReport` 实现动态 prompt 编排，从而让模型在有限上下文窗口内优先获得当前决策所需、可追溯且尽量新鲜的信息。
