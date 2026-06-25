# Refactor Notes

本文件记录重构过程中发现、但当前不适合在同一小步里直接修复的问题。

## 已明确的设计边界

- 记忆检索分为三层：`MemoryRecord` 判断硬资格，`build_memory_query()` 从当前上下文提取检索意图，`score_memory_record()` 为单条记忆计算相关性和解释原因；`MemoryRetriever.retrieve()` 只负责加载、排序和数量限制。
- 记忆写入分为 admission 和落盘：`decide_prompt_memory_admission()` 解释用户提示是否包含 durable project knowledge；`MemoryWriter.admit_prompt_memory()` 只按该决策写入稳定项目约束。普通任务进度、工具输出和文件摘要不进入 durable memory，分别归 TaskRecoveryStore 与上下文状态处理。
- 记忆记录采用“构造严格、反序列化宽容”的模型边界：直接构造 `MemoryRecord` 时会拒绝未知 kind/scope/trust/status，防止新代码继续传播脏枚举；`MemoryRecord.from_dict()` 仍会把旧存储中的未知值归一到安全默认值，避免历史会话无法加载。
- 记忆信任度和上下文信任度不是同一个枚举：`MemoryTrust` 允许 `verified` 表示“这条长期记忆有验证证据”，但 `ContextTrust` 只表达模型上下文里的来源强度。`_context_trust_from_memory_trust()` 负责把 verified memory 注入上下文时降为 `observed`，避免在 `ContextCompiler` 里散落临时字符串转换。
- 协议层只保留跨层实际交换的数据契约：`ContextItem` 表示候选上下文片段，`ContextSectionReport` 表示一次编译后的段落报告，`DroppedContextItem` 表示丢弃原因。未被运行期使用的旧 `ContextSection` 草稿已删除，避免和 sessions 层实际负责编译的 `ContextItemSection` 混淆。
- 协议层结构化错误载荷保护跨层错误语义：`ErrorInfo` 要求非空 `code/message`、受限 `source` 和 dict `details`；`LLMErrorInfo` 固定 `source=llm`，校验 `LLMErrorKind`，并清洗 provider/model。这样 core、runtime 和接口层处理错误时可以相信错误码、来源和 LLM 分类是可分派的。
- 协议层用量和费用统计保护非负记账语义：`Usage` 只接受非负整数 token 计数并要求 `cost` 为 `Cost`；`Cost` 只接受非负数值费用。缺少 total 时由分项补齐，显式 total 保留，兼容 provider 只返回总费用或总 token 的场景。这样会话累计、观测指标和 CLI `/usage` 不需要再解释负值或自相矛盾的空 total。
- 协议层运行结果只表达一次 run 的结构化事实：`AgentRunCounters` 保护非负计数，`RunVerification` 限定验证状态和工具调用身份，`TaskSummary` 清洗任务投影并复制证据列表，`AgentRunResult` 校验 run 状态/停止原因并复制消息、验证结果和受影响路径。业务上的任务是否完成仍归 `TaskController` / run decisions 判断，协议对象只保证跨 core、sessions、runtime、interfaces 传递的结果是干净、可分派、不会被调用方后续突变污染的事实快照。
- 协议层模型配置保护 LLM runtime 边界：`Model` 要求非空 id/name/api/provider、受限 input 类型、正数 context/max token，并复制 headers；`ModelCapabilities` 只接受 bool 能力字段。`Model.base_url` 只清洗不强制非空，因为内置 provider、测试模型或 provider 默认 endpoint 可以不在协议对象上声明；工作区自定义模型的 endpoint 必填规则仍归 `WorkspaceModelConfig`。
- 协议层上下文条目自己保护跨层枚举：`ContextItem` 构造时校验 `ContextTrust` 和 `ContextFreshness`，`DroppedContextItem` 构造时校验 `DroppedContextReason`。这样 sessions 层生成的上下文候选和裁剪报告必须符合协议，不再依赖调用方事后清理脏字符串。
- 模型可见工具规范归协议层保护：`Tool` 要求非空 name/description、dict 参数 schema，并复制 parameters。`AgentTool.to_spec()` 只负责把运行时工具定义映射成协议对象，LLM provider 适配器不需要再处理空名称或可变 schema。
- 工具结果状态采用同一份协议词表：`ensure_tool_result_status()` 用于新构造的 `ToolResult` / `ToolResultMessage` 严格拒绝未知状态；`coerce_tool_result_status()` 只用于持久化反序列化，把旧会话中的未知状态归一到安全默认值。`ToolResult` 和 `ToolResultMessage` 都会同步 `status/is_error`，保证非 success 结果不会以 `is_error=False` 继续流入任务规划、审批恢复或上下文证据链。
- 工具定义对象保护 provider spec 边界：`AgentTool` 要求非空 name/label/description、dict 参数 schema、callable 执行器，并复制参数 schema；`to_spec()` 只把模型需要的 name/description/parameters 暴露给 LLM provider。这样注册表、扩展 API 和内置工具装配不需要重复清洗工具定义。
- 工具注册表保护工具与元数据的一一对应：`ToolRegistry.register()` 只接受 `AgentTool`，显式 metadata 必须是 `ToolMetadata` 且 `metadata.name == tool.name`。这样 `ToolRuntime.metadata_for(tool.name)` 不会拿到另一个工具的风险等级、资源范围或审批能力。
- 工具元数据是权限和审批的静态事实契约：`ToolMetadata` 要求非空 name/category/resource_scope，校验 `ToolRiskLevel` 和所有布尔权限字段，并复制 extra。这样 `PermissionPolicy`、CLI 审批展示和 runtime 工具装配看到的风险等级、资源作用域和能力信息不会带着脏值继续传播。
- 工具权限请求和决策是策略边界：`PermissionPolicy` 自己校验并清洗 `ToolPermissionMode`，`ToolRequest` 保护权限检查所需的 tool/source/params，`ToolDecision` 只允许 `allow/deny/approval_required` 三类可分派结果并清洗 details。`ask` 属于权限模式，不是工具执行决策，避免 ToolRuntime、审批载荷和任务控制收到混合语义。
- 工具运行时入口保护调用身份：`ToolRuntimeRequest` 要求非空 `tool_call_id/name/source` 并复制 params；`ToolRuntimeResult` 校验统一的 `ToolResultStatus`、同步 `status/is_error`，并清洗可选 `approval_id`。这样工具执行、审批、任务证据和事件上报共享同一组干净调用身份。
- 上下文编译不直接解释记忆记录的长期价值，只把检索结果和 pinned memory 转成 `ContextItem`，再交给统一预算裁剪流程。
- 上下文条目分区拥有统一的选择、清洗和报告边界：`ContextCompiler` 负责收集 active files、recent evidence、memory 的候选项；`ContextItemSection.compile()` 负责按预算选择条目、过滤过时项、清洗不可信文本并生成 `ContextSectionReport`。这样每个分区为什么进入上下文、为什么被丢弃，可以在同一个对象上解释。
- 会话上下文状态的枚举语义由模型自身保护：`ActiveFile` 使用 `ContextFileRole` 限定文件角色，`ContextEvidence` 使用 `ContextEvidenceKind` 限定证据种类，`FileSummary` / `ContextEvidence` 使用协议层 `ContextFreshness`，`ContextEvidence` 使用 `ContextTrust`，构造时会拒绝未知值。这样 `ContextCompiler` 不需要为 recent evidence 的 kind/trust/freshness 做类型绕行。
- 上下文压缩分为触发决策和执行压缩：`decide_context_compaction()` 负责解释是否压缩、触发原因和保留窗口；`build_compacted_context()` 负责摘要消息拼装和工具调用/结果配对修复；`AgentSession` 只负责编排摘要生成、状态替换和事件记录。
- LLM 压缩摘要生成属于 sessions/context 的上下文管理能力：`build_llm_compaction_summary()` 负责把旧消息格式化为摘要 prompt、传递 provider api key 并提取助手文本；`AgentSession._compact_context_if_needed()` 只决定何时压缩、优先使用自定义 summary builder、在 LLM 摘要为空时降级到 `fallback_summary()`，以及把压缩结果写回 Agent 和 SessionStore。
- 上下文新鲜度分为持久化检测、状态语义和 steering 提示构造：`RunStore.evaluate_freshness()` 负责检查上一轮跟踪文件是否仍匹配工作区；`FreshnessStatus` 明确限定为 `valid/stale/mismatch`，`FreshnessResult` 会拒绝未知状态，并通过 `should_record_event()` / `requires_steering()` 解释检查结果是否值得审计、是否需要提醒 Agent；`build_context_freshness_notice()` 负责把需要提示的状态转换成带 metadata 的 `UserMessage`。`AgentSession._check_context_freshness()` 只负责何时检查、记录 `context_freshness_checked` 事件并注入提示。
- 上下文编译器的会话派生语义是 fork 而不是 clone：`ContextCompiler.fork_for_session()` 复用预算策略和仓库位置，但创建新的 `SessionContextState`，避免分支/新会话共享 active files、recent evidence 和仓库快照造成上下文污染。
- 任务状态对象拥有机械性步骤语义：`TaskStep` 负责单步状态迁移、枚举校验和证据去重；`TaskState` 负责当前步骤定位、任务阶段校验、推进到下一个开放步骤，以及 completed/pending/blocked 投影；`ExecutionDecision` 校验决策动作。`TaskController` 只负责根据工具结果和完成检查决定何时调用这些状态原语。
- 完成门禁结果使用受限 reason：`CompletionCheck` 只允许 `blocked_steps`、`modified_without_fresh_verification`、`incomplete_steps`、`all_steps_completed`，因为这些值会继续驱动 run 层停止原因和 steering 提示。持久化摘要中的 `completion_reason` 仍保持字符串，允许保存 `replan_limit_exceeded` 等更宽的运行结论。
- 任务证据链中的状态词表由模型对象自身保护：`AttemptRecord` 校验 `AttemptStatus`，`ChangeSet` 校验 `ChangeSetStatus`。工具尝试、变更集验证和回滚建议共享同一组状态语义，避免 `TaskController` 在记录执行证据、标记验证失败或提出撤销时继续传播未定义字符串。
- 任务步骤类型词表归 `core.task_state`：`TASK_STEP_KINDS` / `TaskStepKind` 是 planner 输出和运行期 `TaskStep` 的共同边界。`TaskPlanner` 负责把模型输出中的未知 kind 降级为 `other`，`TaskStep` 负责拒绝新代码直接构造出的未知 kind，避免计划解析和任务状态各自维护一份词表。
- 任务计划草稿是模型规划和确定性任务控制之间的边界：`PlannedTaskStep` 自己清洗标题、kind、验收标准和验证提示，并拒绝空标题或未知 kind；`TaskPlanDraft` 要求非空 goal、至少一个步骤和受限 source，并把步骤固定为不可变序列。`TaskPlanner` 可以宽容解析模型 JSON，但进入 `TaskController` 前必须已经是干净计划。
- 任务控制器的输入规范化边界同样宽容：`TaskController._normalize_steps()` 通过 `_coerce_task_step_kind()` 处理 raw dict / planner-like step，未知 kind 降级为 `other`。这样外部输入、模型计划和历史恢复都在进入 `TaskStep` 前完成清洗，核心状态模型只表达干净语义。
- 任务恢复分为两个显式映射：`build_task_recovery_projection()` 负责把 `TaskSummary` 写成会话恢复投影，`build_task_state_from_recovery_projection()` 负责把该投影恢复成下一次 run 的 `TaskState`。`TaskRecoveryStore` 只负责持久化投影，`TaskController` 只负责运行期任务推进。
- 任务恢复投影读取历史数据时保持宽容：`build_task_state_from_recovery_projection()` 会把旧投影里的未知步骤类型降级为 `other`，但 `TaskStep` 直接构造仍然拒绝未知 kind。这样持久化边界可以恢复旧会话，核心状态对象仍能阻止新代码传播脏枚举。
- 任务恢复投影在 core/session 之间统一命名为 `task_recovery_projection`：它不是完整 `TaskState`，也不是 durable memory，而是 `TaskRecoveryStore.active_projection()` 产出的可持久化恢复输入。`Agent.set_task_recovery_projection()` 只负责把这份投影带入下一次 run。
- 会话事件也区分任务恢复和长期记忆：task recovery 写入/收尾失败记录为 `task_recovery_warning`，durable memory 写入/收尾失败才记录为 `memory_warning`。这样观察层不会把“恢复当前任务进度”的问题误解为“长期记忆污染”。
- Agent 主循环分为编排与纯规则：`agent_loop.py` 负责模型调用、工具执行、事件发射和状态落盘；`run_decisions.py` 负责把模型错误、工具调用限制、工具结果和任务完成度翻译成 run 层状态。这样测试可以直接覆盖规则，不必构造完整运行循环。
- 事件分发的基础词表归协议层，事件信封归 core 出口：`ensure_runtime_event_type()` 统一校验 `RuntimeEventType`，`AgentEventEmitter` 负责清洗 run/session 身份、递增 turn/event 序号并覆盖调用方传入的信封字段。具体事件 payload 仍保持 TypedDict 的开放形状，避免在本步把所有 UI/observability 载荷都强行收紧。
- core 运行上下文和循环配置分工明确：`AgentContext` 是一次模型循环输入的快照，构造时复制 messages/tools 和任务恢复/控制信号，避免调用方后续 mutation 污染运行中上下文；`AgentLoopConfig` 是核心循环策略合同，校验模型、回调、工具执行模式、重试/工具迭代上限和任务控制开关。这样 `agent_loop.py` 可以专注编排，不再把配置脏值解释成运行语义。
- Agent 实例配置和内存状态也归 core 边界保护：`AgentOptions` 是创建 Agent 的长期配置快照，构造时校验模型、工具/消息列表、思考级别、工具执行模式、回调、重试/工具上限和任务控制开关；`AgentState` 是运行期内存状态，复制 messages/tools、校验 pending 工具调用集合和流式状态字段。这样 Session 创建 Agent 后，后续外部列表 mutation 不会改变 Agent 的配置或初始状态。
- 模型调用前的上下文准备发生在任务上下文注入之后：`LLMStreamRunner` 先通过 `_with_current_task_context()` 把当前任务投影写入系统提示词，再调用 `prepare_context`。这样 sessions/context 编译器和扩展 prepare hook 看到的是 Agent 实际会发给模型的任务感知上下文，而不是缺少任务段落的旧上下文。
- 会话 run 生命周期分为前置准备和收尾：`_start_run_lifecycle()` 负责 before hooks、记忆准入、任务恢复投影、上下文新鲜度和压缩；`_complete_run_lifecycle()` 负责 run 结果落盘、rollback metadata、任务恢复收尾、记忆收尾、后置压缩和 after hooks。
- 会话回滚结果使用受限状态词表：`GitRollbackResult` 校验 `RollbackStatus`，确保 `AgentSession.revert_run()`、事件记录和接口层只能看到 `reverted/not_eligible/conflict/noop` 四类结果。回滚失败的细节继续放在开放的 `reason` 字符串里，避免把具体 Git/工作区原因塞进状态枚举。
- Runtime active-run 生命周期分为非流式与流式入口：`_run_active_session_call()` 负责非流式调用的 active run 创建、当前 task 绑定、pending approval 记录、状态标记和清理；`_stream_active_session_call()` 负责流式事件转发场景下的同类职责。
- Runtime 的 busy 判断和运行状态查询统一经过 active-run 状态原语：`_current_active_run()` 负责丢弃已完成的残留 `ActiveRun`，`_require_session_idle()` 负责在新消息、continue、审批恢复等入口拒绝真正运行中的会话。这样“已完成残留不算 busy / running”的规则只有一个实现位置。
- Runtime active-run 记录保护自身状态词表：`ActiveRun` 校验 `ActiveRunStatus`，只允许 `running/completed/failed/aborted`。这样 busy 判断、取消运行和状态查询不会因为手工构造的脏状态出现静默绕行。
- Runtime 审批流程分为纯规则和事务编排：`runtime.approval_flow` 定义 `PendingApproval`、`build_pending_approvals()`、`denied_tool_result()` 和 `to_tool_result_message()`；`RuntimeService` 只保存当前进程的 pending approval 表、解析用户决策并编排恢复事务。
- Runtime 审批输入归一化属于审批流程规则：`normalize_approval_decision()` 接受接口层常见别名并返回 `approve/deny/None`；`RuntimeService` 只负责把 `None` 映射为公开错误 `InvalidApprovalDecisionError`。这样 CLI/Web 可以复用同一组别名，而服务门面仍保持自己的错误码契约。
- Runtime pending approval 记录保护恢复身份：`PendingApproval` 要求 `approval_id`、`session_id`、`run_id` 和 `tool_call.id` 非空。审批恢复依赖这些字段作为内存表键、会话归属、run 归档和原工具调用定位，不能把缺失身份的记录交给服务层兜底。
- 审批恢复分为 Runtime 事务和 Session 归档：`_resume_after_tool_approval()` 负责执行/拒绝审批工具、替换 pending 工具结果、记录审批事件并触发继续会话；`AgentSession.continue_after_tool_approval()` 负责把审批工具结果并入 resumed `AgentRunResult`、替换 `Agent.last_run_result`、写 run result 和审批前 baseline 对应的 rollback metadata。
- Session run 结果补证归 `sessions.run_reconciliation`：`merge_approved_tool_result()` 把审批期工具结果、受影响路径、workspace changed 标记和验证证据并入 resumed `AgentRunResult`，并保证同一 approval 不重复计数。`replace_pending_tool_result()` 统一 pending approval 工具结果的匹配/替换语义。`AgentSession` 只保留审批恢复事务编排、消息落盘和事件记录。
- pending approval 替换必须只作用于尚未解决的工具结果：`replace_pending_tool_result()` 即使按 `approval_id` 命中，也要求原消息状态仍为 `approval_required`。这样重复恢复、重放事件或已落盘成功结果不会被新的 replacement 意外覆盖。
- 已批准工具调用的执行入口属于 Agent/Session 边界：`Agent.execute_tool_call_once()` 复用核心工具协调器、before/after tool hooks 和事件发射；`AgentSession.execute_approved_tool_call()` 暴露审批恢复语义；Runtime 只选择工具实现并提交审批决策，不再组装 `AgentLoopConfig` 或直接访问 `Agent._dispatch_event`。
- 工具审批分为“产生请求”和“恢复执行”：`ToolRuntime` / `ApprovalProvider` 只决定工具调用是否需要用户决策，并在未批准时产出 `approval_required` 工具结果；Runtime 层负责登记、批准/拒绝、执行原工具和继续会话。`DeferredApprovalProvider` 是默认的安全提供者，它不是永久拒绝语义，而是避免直接执行高风险工具。
- 工具审批载荷保护恢复身份和展示语义：`ApprovalRequest` 要求非空 `approval_id/tool_call_id/tool_name/risk_level`，复制参数预览并清洗去重 capabilities；`ApprovalDecision` 要求 `approved` 为 bool，并清洗 reason/approval_id。这样 CLI/Web 审批展示、Runtime pending approval 表和后续恢复执行都依赖同一组干净字段。
- Runtime 数据契约归 `runtime.types`，服务门面归 `runtime.service`：CLI/Web/evaluation 等调用方应从 `runtime.types` 导入 `UserInput`、`SessionStatus`、`SessionHandle` 等数据模型，不要通过 `RuntimeService` 模块间接拿类型，避免把服务入口误当成类型聚合层。`runtime.service` 内部用下划线别名引用这些类型，只表达实现依赖，不形成公开转导契约。
- Runtime 包级入口只表达“组装和运行”：`codepilot.runtime` 导出 `RuntimeService`、会话工厂、工作区资源加载和 prompt/command 辅助函数；`CreateAgentSessionOptions`、`UserInput`、`SessionHandle` 等数据契约仍必须从 `codepilot.runtime.types` 导入。这样 evaluation、CLI、Web 从 import 路径上就能区分“调用服务”和“构造数据”。
- Runtime 用户输入对象保护入口快照语义：`UserInput` 构造时裁剪并拒绝空文本，复制并冻结图片路径，避免 CLI/Web/evaluation 构造后再突变请求内容。`RuntimeService` 在调用 `AgentSession.run()` 前把图片 tuple 转回 list，保持外部请求契约不可变、内部会话 API 仍使用原有参数形状。
- Runtime 会话状态对象保护接口展示语义：`SessionStatus` 构造时裁剪必填展示字段、限制 permission mode、拒绝负 message count、要求 running 状态为 bool，并把 warning 文本清洗为不可变 tuple。CLI startup adapter 再把它转成自己的 view model，避免 renderer 直接继承 runtime 的可变列表或脏展示字段。
- Runtime 装配诊断是机器可读证据，不是随意日志字符串：`RuntimeDiagnostic` 限定 `info/warning/error` 三类 severity，要求非空 code/message，并清洗可选 source。Tool assembler 负责产生诊断，RuntimeService 只筛选 warning 汇入 `SessionStatus`，接口层不需要再解释未知严重级别或空诊断码。
- Runtime 配置来源分为内部 resolver 来源和公开解释来源：`resolve_runtime_config()` 内部仍使用 `options/restored_session/workspace/default`，`assembly._build_config_sources()` 显式映射为 `cli/session/project/default`；`ConfigValueSource` 只接受公开来源词表并清洗 location。`ResolvedRuntimeProfile` 复制并冻结 sources，`ResolvedConfigValue` 校验 key/source，保证 CLI `config explain` 展示的是稳定证据快照，不会被后续 dict mutation 或未知来源污染。
- Runtime 能力目录是装配后的只读快照，不是可变注册表：`RegisteredTool` 清洗工具名、来源和 origin，并限制来源为 `builtin/caller/extension/mcp`；`CapabilityCatalog` 复制并冻结工具与命令集合，只接受 `RegisteredTool` / `RegisteredCommand`。实际工具注册、权限和过滤仍归 `tool_assembler` / `ToolRuntime`，接口层只读取快照用于展示和诊断。
- Runtime 装配产物是会话创建时的证据快照：`RuntimeAssembly` 校验 `AgentSessionOptions`、`ResolvedRuntimeProfile`、`RepositoryBootstrap` 和 `CapabilityCatalog` 的归属，并冻结 diagnostics。`RuntimeService.get_assembly()` 暴露的是这份快照，接口层不应把它当作可变装配上下文继续修改。
- Runtime 会话句柄保护三方身份一致性：`SessionHandle.session_id` 会被清洗，并且必须同时匹配 `AgentSession.session_id` 与 `RuntimeAssembly.session_options.session_id`。这样 CLI/Web/evaluation 拿到的句柄可以作为后续 runtime 调用的稳定根身份，不会出现服务表、实际会话和装配快照彼此错位。
- Runtime 启动上下文是系统提示词的静态输入快照：`RuntimeContext` 只保存仓库 bootstrap、prompt guidelines、追加段落、工具说明片段和 memory 文本，不包含 active files、recent evidence 或任务状态。集合字段在构造时复制并冻结，避免配置、扩展或 skills 的后续 mutation 改变已装配会话的 prompt 语义。
- Runtime 系统提示词由结构化计划组装：`PromptSection` 校验段落名、内容、来源和优先级，`PromptPlan.add_section()` 统一处理空可选段落跳过、重复段落名拒绝、按优先级渲染和来源追踪。`build_runtime_system_prompt()` 只声明有哪些段落，不再直接操作裸 `sections` 列表。
- Runtime 命令元数据是接口展示的唯一来源：`RuntimeCommand` 清洗无斜杠命令名、要求非空描述、限制 source 为 `builtin/extension/skill/prompt`，并通过 `to_dict()` 输出 CLI/Web/RPC 可展示字段。`RuntimeService.list_commands()` 不再手写 dict，`list_runtime_commands()` 也按清洗后的命令名合并内置/扩展命令，避免 `/help`、补全和 RPC 命令清单各自解释命令元数据或暴露重复命令。
- CLI RPC 分为协议循环与单请求处理：`run_rpc()` 只负责 JSONL 读写循环和 ready/shutdown；`_handle_rpc_request()` 负责命令分发；`emit_rpc_ok()` / `emit_rpc_error()` 负责稳定响应形状。`test_run_rpc_emits_jsonl_contract_for_state_prompt_errors_and_shutdown` 固定了最小 JSONL 合同。
- CLI RPC 协议响应归 `interfaces.cli.rpc_protocol`：`emit_rpc_ok()` / `emit_rpc_error()` 固定 JSONL 响应形状，`rpc_error_from_exception()` 把 RuntimeServiceError 的稳定 `code` 透传到接口协议；`runner.py` 只保留 stdin/stdout 循环和命令分发。
- CLI RPC ready 握手也属于协议层：`RPC_PROTOCOL_VERSION` 和 `emit_rpc_ready()` 固定 JSONL 初始化消息，`run_rpc()` 不再手写版本号或 ready payload。这样后续升级协议版本时有唯一修改点和直接合同测试。
- CLI RPC 错误载荷保护机器协议不变量：`RpcError` 要求 `code/message` 都是非空字符串并负责裁剪空白；`rpc_error_from_exception()` 对无消息异常使用异常类名兜底。这样 JSONL 客户端始终能拿到可分类、可展示的错误对象。
- CLI RPC 协议 helper 保护可分派字段：`emit_rpc_ready()` 清洗并拒绝空 session id，`emit_rpc_ok()` 清洗并拒绝空 command。runner 仍只负责命令分发和 stdin/stdout 循环，JSONL 客户端看到的握手和响应字段由 `interfaces.cli.rpc_protocol` 统一保证。
- CLI 启动状态归 `interfaces.cli.startup`：`CliStartupState` 是启动横幅/工具栏的 view model，`build_startup_state()` 负责把 runtime 的 `SessionStatus` 转成该 view model；`renderer.py` 只消费状态并输出终端文本，不再拥有数据映射职责。
- CLI startup 只依赖 runtime 数据契约：`interfaces.cli.startup` 从 `runtime.types` 导入 `SessionStatus`，不再通过 `runtime.service` 间接拿类型。这样接口 view model 适配不会把服务门面误当成类型聚合模块。
- CLI 启动 view model 保护终端展示快照：`CliStartupState` 要求 version/model/workspace/session id 非空，校验 permission mode，并把 warnings 清洗为不可变 tuple。`build_startup_state()` 只负责从 runtime `SessionStatus` 映射字段，renderer 只消费干净展示状态，不再继承 runtime 对象的可变集合或空白字符串。
- CLI 运行模式和终端 I/O 类型属于接口层：`RunMode`、`OutputFn`、`InputFn` 统一放在 `interfaces.cli.types`，由 runner 和 renderer 共享。`runtime.types` 不再定义这些终端适配别名，避免 runtime 数据契约混入 CLI 专属概念。
- CLI 运行入口配置由 `RunOptions` 保护：构造时清洗并限制 `mode`，要求非空 `session_id`，检查输入/输出回调可调用，并把退出命令归一成无斜杠的不可变 tuple。这样拼错的运行模式不会静默落入 interactive 分支，runner 的模式分发只处理干净配置。
- CLI runner 门面只导出运行模式入口：`RunOptions`、`run()`、`run_print()`、`run_interactive()`、`run_rpc()`。终端渲染器从 `interfaces.cli.renderer` 或包级 `interfaces.cli` 获取，避免把“运行模式调度”和“终端渲染实现”混成同一个公开模块。
- CLI 包级门面是稳定公共入口：`codepilot.interfaces.cli` 聚合运行入口、启动 view model 和终端渲染器；更细的子模块只表达各自职责边界，例如 `runner` 不再转导 renderer。
- CLI 命令行入口实现归 `interfaces.cli.main`：`build_parser()`、`main()` 和配置子命令处理都在 `main.py`，旧的 `cli.py` 转发/兼容入口已移除，避免同一概念同时存在 “cli” 和 “main” 两个文件名。包级 `interfaces.cli` 不再提前导出名为 `main` 的函数，避免遮蔽 `interfaces.cli.main` 子模块；命令行入口点继续使用 `codepilot.interfaces.cli.main:main`。
- CLI 工具审批提供者只适配终端 I/O：`CliApprovalProvider` 构造时要求 input/output 回调可调用，`request_approval()` 负责渲染 `ApprovalRequest`、读取用户确认并返回工具层 `ApprovalDecision`。审批身份、风险等级和参数预览仍由 tools approval 模型保护，CLI 不重新解释工具权限语义。
- CLI 命令补全只消费 runtime 命令元数据：`CommandCompleter` 接收 `RuntimeCommand`，展示 `name/description`，默认命令源来自 `builtin_commands()`。CLI shell 不再维护第二份命令列表和描述字典，避免 `/help`、命令处理和终端补全三处语义漂移。
- Web 事件适配器只负责接口语义映射和 JSON 化：普通 Agent 运行事件保持 `agent_event`，需要用户显式操作的 `tool_approval_required` 暴露为同名 Web 事件类型，payload 仍保留原始事件字段，避免接口层重新解释工具参数或模型输出。
- Web 包门面导出完整的公开传输契约：`WebEventEnvelope`、`WebEventKind`、`ApprovalDecision`、请求/响应 dataclass 和后端骨架都可从 `codepilot.interfaces.web` 获取。这里的 `ApprovalDecision` 是 Web 传输动作字面量，不是 `tools.ApprovalDecision` 执行结果对象。
- Web 传输对象保护公开协议字段：`WebEventEnvelope` 校验 `WebEventKind`，`WebToolApproval` 校验 `ApprovalDecision`，`WebErrorPayload` 要求非空 `code/message` 并裁剪空白。这样接口层不会构造出前端无法分派的事件类型，也不会把不可分类或不可展示的错误对象透传给浏览器。
- Web 会话引用只表达入口身份，不承载完整会话状态：`WebSessionRef` 要求非空工作区目录、清洗可选 session id，并把空白 session id 归一为 `None`。这样 Web backend 可以明确区分“发送到已有会话”和“基于工作区创建临时会话”，不会把空白身份继续交给 runtime 解释。
- Web 创建会话请求保护公开配置快照：`WebCreateSessionRequest` 要求非空工作区目录，清洗 provider/model/system prompt/session id，并校验布尔开关。这样 `WebConsoleBackend.create_session()` 只负责把干净的接口配置映射到 `CreateAgentSessionOptions`，不再替浏览器输入解释空白字段或字符串布尔值。
- Web prompt 请求保护公开输入快照：`WebPromptRequest` 要求 `session` 是 `WebSessionRef`，裁剪并拒绝空文本，复制并冻结图片路径。Web backend 只把这份干净请求映射成 runtime `UserInput`，不再把浏览器侧可变列表或空文本交给 runtime 兜底解释。
- Web 工具审批请求保护恢复身份：`WebToolApproval` 要求非空 session id 和 tool call id，清洗可选 approval id 与展示 reason，并只允许 `approve/deny` 两类 Web 传输动作。这样 backend 可以稳定选择 `approval_id or tool_call_id` 交给 Runtime 审批恢复，不会把空白身份误当成有效审批键。
- Web 事件信封保护传输快照：`WebEventEnvelope` 校验事件类型、清洗可选 session id，并复制 payload dict。这样事件适配器和 backend 构造信封后，调用方后续修改原始 payload 不会污染已经准备发送给前端的事件。
- Web 只读描述对象也保护展示语义：`WebSessionSummary` 要求非空 session id；`WebRouteSpec` 清洗 HTTP/WS 方法、路径和说明，并拒绝当前公开协议外的方法或非 `/` 开头路径。这样 `describe_web_contract()` 输出的路由清单是可直接展示和校验的接口事实，而不是松散字符串拼装。
- Web 路由规格拥有显式公开序列化：`WebRouteSpec.to_dict()` 只输出 `method/path/description` 三个协议字段，`describe_web_contract()` 不再依赖 dataclass `__dict__`。这样内部对象后续新增实现字段时，不会意外泄漏到浏览器侧合同。
- WebSocket 流辅助只做身份清洗和事件适配：`WebSocketSessionStream` 要求非空 session id，并把 runtime `continue_session()` 事件逐个转成 `WebEventEnvelope`。它不保存会话状态、不解释 Agent 事件，也不绕过 Web schema 的事件信封规则。
- core 门面只导出稳定核心编排对象：`Agent`、run loop、`RunState`、任务控制、上下文准备类型等仍属于 core；协议事件/运行结果从 `codepilot.protocols` 导入，工具定义/工具结果从 `codepilot.tools` 导入，扩展 hook 类型通过 `codepilot.extensions` 面向扩展作者暴露。这样调用方可以从导入路径直接看出概念的归属层。
- `ToolCallCoordinator` 是 core 内部工具执行协调器，不再从 `codepilot.core` 顶层导出；专门测试或内部代码若需要它，应从 `codepilot.core.tool_coordinator` 显式导入，避免接口层误把协调器当作应用服务入口。
- before tool hook 主动拦截工具调用属于 core 策略拒绝，结果状态为 `denied`，并使用 `before_tool_call_blocked` 作为可分派原因；这表示工具从未执行。before/after hook 自身抛出的异常仍属于 hook 执行错误，状态保持 `error`。这样任务规划、观测事件和接口展示可以区分“策略未允许执行”和“执行链路失败”。
- core 并行工具调度只信任显式工具元数据：`can_schedule_tool_in_parallel()` 只允许 `concurrency_safe=True` 且 `exclusive=False` 的工具进入并行批次；没有 metadata、未声明并发安全或要求独占的工具都保守串行。`ToolCallCoordinator` 负责编排批次，具体副作用安全仍由 ToolRuntime/工具实现保证。

## 约束冲突

- `benchmarks/` 的 Git 策略存在冲突：仓库指令要求不要提交 benchmarks，且 `.gitignore` 应包含 benchmarks；但 `test/test_evaluation.py::test_benchmark_sources_are_not_git_ignored` 期望 benchmark sources 不被 git ignore。当前不在代码质量重构中顺手修改该策略，后续需要先明确 benchmark 目录中哪些是源文件、哪些是生成产物。

## 后续重构候选

- `AgentSession` 仍然同时承担会话生命周期、上下文压缩、记忆写入、任务恢复和分支导航职责。当前已先收拢 run 收尾流程、审批结果补证和 pending 工具结果替换规则；后续可以继续按职责提取更小的会话服务对象，但需要避免为学习项目制造过多薄包装。
- CLI RPC 目前已有最小 JSONL 合同测试；如果后续扩展 RPC 命令，建议按命令补充更细粒度的合同用例，而不是只通过交互层间接覆盖。
- CLI 交互式补全目前使用内置 runtime 命令元数据；如果后续要补全扩展命令，应让 `run_interactive()` 在当前会话切换后通过 `RuntimeService.list_commands()` 刷新 shell completer，而不是让 shell 直接读取 Session 或扩展注册表。
