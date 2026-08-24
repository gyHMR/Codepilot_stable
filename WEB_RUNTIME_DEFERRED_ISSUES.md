# Web Runtime Deferred Issues

本文记录 Web Coding Agent 链路中已经确认、但不纳入当前稳定性修复范围的问题。
当前优先处理会直接造成工具不可用、Run 卡住、错误恢复或用户输入丢失的问题；本文事项后续按优先级逐项处理。

## P1: 事件与终态一致性

### 1. 终态 SSE 载荷过大

- 位置：`src/codepilot/interfaces/web/events.py`
- 现状：`run_finished` 和 `run_paused` 会把完整 `SessionRunRecord` 转换为 Web 事件，其中可能包含消息、CoreOutcome、状态和快照。
- 影响：单个 SSE 事件过大，增加序列化时间、EventHub 内存、浏览器解析时间，并重复传输已经持久化的消息。
- 后续方向：终态事件只保留 `run_id`、`status`、`stop_reason`、`final_text`、`affected_paths` 和持久化 revision。
- 验收：大型工具输出或长上下文 Run 的终态事件保持有界，前端仍能通过 timeline 获取完整历史。

### 2. `command_finished` 被投影为 `run.completed`

- 位置：`src/codepilot/interfaces/web/events.py`
- 现状：Session 命令完成与 Agent Run 完成使用同一种 Web 事件。
- 影响：Plan 审批、模式切换等带后续 continuation 的命令会让前端短暂进入 idle，并提前清空 live activity。
- 后续方向：增加独立的 `command.completed` 或 `session.updated` 事件，不触发 Run 终态清理。
- 验收：命令后自动继续执行时，输入框不会短暂开放，活动区不会闪烁或清空。

### 3. 终态先清空 live buffer，再异步刷新 timeline

- 位置：`web/src/events/reducer.ts`、`web/src/pages/WorkspacePage.tsx`
- 现状：收到终态后立即清空流式文本和活动，然后通过 React Query 异步刷新历史。
- 影响：timeline 请求缓慢或失败时，最终回复和工具活动会暂时或永久消失。
- 后续方向：终态记录目标 revision；只有 timeline 成功同步到该 revision 后才释放 live buffer。
- 验收：模拟 timeline 500、超时和延迟时，用户仍能看到完整最终回复与活动。

## P1: SSE 重连与 Session 隔离

### 4. EventSource 重连定时器无法可靠取消

- 位置：`web/src/events/client.ts`
- 现状：`onerror` 创建的定时器未保存；`close()` 后旧定时器仍可能调用 `connect()`，而 `connect()` 会重新设置 `closed=false`。
- 影响：切换 Session 后旧连接可能复活，并把旧 Session 事件发送到当前 reducer。
- 后续方向：保存重连 timer；回调执行前再次检查 `closed`；`close()` 同时取消 timer。
- 验收：快速切换 Session、断网和恢复时，旧 Session 不再建立连接或修改当前页面状态。

### 5. Reducer 不校验事件所属 Session

- 位置：`web/src/events/reducer.ts`
- 现状：Reducer 只根据 event ID 去重，不验证 `event.session_id` 是否是当前 Session。
- 影响：旧连接、延迟事件或测试环境中的跨 Session 事件可以污染当前 UI。
- 后续方向：事件状态中保存当前 Session ID，所有非 reset 事件必须匹配后才能应用。
- 验收：向当前 reducer 注入其他 Session 的事件时，状态保持不变。

### 6. 首次订阅不恢复当前 Run 已产生的 live activity

- 位置：`src/codepilot/interfaces/web/routes/events.py`、`src/codepilot/interfaces/web/events.py`
- 现状：没有 last event ID 时只发送持久化快照；EventHub 已缓存的当前 Run 事件不会重放，而快照不含流式文本和活动。
- 影响：运行中打开页面或切回 Session 时，只能看到订阅之后的新活动。
- 后续方向：为 EventHub 维护有界的当前 Run live projection，快照同时返回当前文本和按 ToolCall 合并后的活动。
- 验收：Run 执行一半后打开新页面，能立即看到此前已经产生的思考和工具状态。

### 7. Snapshot sequence 与下一条 live event 可能重复

- 位置：`src/codepilot/interfaces/web/routes/events.py`
- 现状：Snapshot 使用 `hub.latest_sequence + 1`，但不推进 WebService 的 sequence；下一条 live event 可能使用相同序号。
- 影响：严格顺序检测失效，重连和 gap 恢复难以证明正确。
- 后续方向：由单一事件发布器分配 sequence；Snapshot 使用明确的 watermark，不独立伪造下一序号。
- 验收：Snapshot、replay 和 live event 的 sequence 全程严格递增。

## P1: 历史时间线规范化

### 8. ToolCall 与 ToolResult 使用不同 activity ID

- 位置：`src/codepilot/interfaces/web/service.py`
- 现状：历史 ToolCall 使用 ToolCall ID，ToolResult 使用消息 ID，无法合并为同一个活动。
- 影响：同一工具调用会显示两项，状态可能同时出现 running 和 completed。
- 后续方向：ToolResult activity 使用 `tool_call_id`；后端按 ToolCall ID 合并参数、目标、结果和终态。
- 验收：每个 ToolCall 在历史记录中只显示一项，最终状态与 ToolResult 一致。

### 9. 同一 Run 的多个活动组可能生成重复 React key

- 位置：`web/src/features/conversation/ConversationTimeline.tsx`
- 现状：活动组 key 主要由 `run_id` 生成；同一 Run 多轮模型调用会产生重复 key。
- 影响：React 可能复用错误节点，造成活动漏显示、顺序错位或旧内容出现在底部。
- 后续方向：后端返回稳定唯一的 group ID，前端不再根据 run ID 临时组合历史活动。
- 验收：同一 Run 包含多轮 ToolCall 时无重复 key 警告，刷新前后展示顺序一致。

## P2: 状态与交互体验

### 10. Approval/Interaction resolved 依赖首个非失败 Frame

- 位置：`src/codepilot/interfaces/web/service.py`
- 现状：WebService 通过恢复后是否出现非 failed frame 推断 wait 已解决。
- 影响：恢复直接失败时旧 resolution 会滞留，并可能在后续无关动作中错误生效。
- 后续方向：从持久化 checkpoint 的前后变化生成 wait resolved，不根据 Frame 类型猜测。
- 验收：恢复成功、恢复失败、连续两个审批和重复提交都能保持 Dock 与 checkpoint 一致。

### 11. HTTP 202 到首个 SSE 事件之间存在重复提交窗口

- 位置：`web/src/pages/WorkspacePage.tsx`、`web/src/features/conversation/Composer.tsx`
- 现状：Composer 要等 Snapshot 或首个 Runtime 事件后才进入 busy。
- 影响：用户可在短窗口内再次发送，第二个请求通常得到 `runtime.run_active`，但组件没有稳定展示该错误。
- 后续方向：提交请求开始后立即进入本地 submitting，直到权威 Snapshot 确认 running、waiting 或 terminal。
- 验收：连续按 Enter 只产生一个请求，HTTP 错误会显示且输入内容不会丢失。

### 12. 组件异步错误多为未处理 Promise

- 位置：Composer、ApprovalDock、InteractionDock、ContinuationDock、PlanApproval
- 现状：事件处理器使用 `void asyncCall()`，失败后没有统一错误状态。
- 影响：409、404、网络错误或恢复错误在界面上表现为按钮恢复原状但没有解释。
- 后续方向：统一 action mutation 层，展示结构化 API 错误，并根据错误码刷新 Session Snapshot。
- 验收：所有操作失败都显示可理解的错误，并能通过重试或刷新恢复。

## P2: Core 完成策略一致性

### 13. Plan 与非 Plan Run 的 verification gate 不一致

- 位置：`src/codepilot/core/policy.py`、`src/codepilot/core/state.py`
- 现状：active Plan 的步骤全部完成后可以在 verification assessment 之前终止；无 Plan 的工作区修改会进入 verification required。
- 影响：相同修改因是否创建 Plan 而得到不同完成判定。
- 后续方向：统一采用 verification disposition：passed、failed 或 unavailable；它表示验证过程事实，不表示 Core 能证明实现正确。
- 验收：Plan 与非 Plan 修改执行相同验证策略，且 unavailable 不会造成无限循环。

## 后续测试矩阵

- Windows Web 下执行 `command` 和 raw shell。
- SSE 断线、重连、EventHub overflow 和服务重启。
- 运行中切换 Session，再切回原 Session。
- Approval、Interaction、Continuation 的成功、失败、重复提交和 stale request。
- terminal 后 timeline 延迟、失败和重试。
- 同一 Run 多轮 ToolCall，包含并行调用、审批 barrier 和部分失败。
- 异常 checkpoint 后选择恢复旧 Run或提交新任务。
