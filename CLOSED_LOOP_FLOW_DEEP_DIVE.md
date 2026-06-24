# Codepilot 闭环运行流程深度讲解

本文按当前代码实现解释一条完整链路：

> 用户输入 → 模型接收上下文 → 执行工具 → 需要审批 → 用户批准 → 工具真正执行 → 模型继续推理 → 回复最终结果

重点不是画一个理想化架构图，而是说明当前项目里 `context / memory / session / tool / task` 在每一步到底做了什么、保存了什么、什么时候会变化。

## 0. 先说结论：这条链路由谁负责

Codepilot 当前不是一个“大一统状态机”，而是几个边界比较清楚的模块协作：

| 模块 | 主要职责 | 在本闭环里的关键表现 |
|---|---|---|
| `RuntimeService` | 面向 CLI/Web/API 的门面，管理 session、active run、审批恢复 | 接收用户输入；记录 pending approval；用户批准后恢复工具执行并继续 run |
| `AgentSession` | 管一次会话的消息、事件、run artifact、记忆、上下文压缩 | 调用 agent；保存消息树和 context；写 run result；触发 memory 写入 |
| `agent_loop` | 模型-工具-任务控制主循环 | 调模型；识别工具调用；执行工具；遇到审批暂停；完成后返回 `AgentRunResult` |
| `ToolCallCoordinator` | 单批工具调用的准备、执行、事件上报 | 跑 before/after hook；发 `tool_execution_start/end`；生成 `ToolResultMessage` |
| `ToolRuntime` | 工具运行时权限、审批、真正执行 | 根据权限策略返回 `approval_required` 或真正执行工具 |
| `TaskController` | 轻量任务规划与继续/修复/完成判断 | 记录 attempt/change_set；审批时进入 waiting；修改后要求 fresh verification |
| `ContextCompiler` | 每次模型调用前的上下文治理 | 召回活跃文件、证据、记忆；动态预算；脱敏；产生 `ContextReport` |
| `TaskRecoveryStore` | 保存当前任务恢复投影 | 记录 goal、task_progress、next_action，供下一次 run 恢复 |
| `MemoryWriter / ExperienceExtractor` | 写入 durable memory 与失败经验 | 记录项目约束；run 结束后从验证闭环沉淀经验 |

主要代码入口：

- `RuntimeService.run_message()`：`src/codepilot/runtime/service.py`
- `AgentSession.run()` / `continue_run()`：`src/codepilot/sessions/session.py`
- `run_agent_loop()` / `run_agent_loop_continue()`：`src/codepilot/core/agent_loop.py`
- `ToolCallCoordinator.execute_batch()`：`src/codepilot/core/tool_coordinator.py`
- `ToolRuntime.execute()`：`src/codepilot/tools/runtime.py`
- `TaskController.after_tool_results()`：`src/codepilot/core/task_controller.py`
- `ContextCompiler.compile()`：`src/codepilot/sessions/context/compiler.py`
- `TaskRecoveryStore.update_from_result()`：`src/codepilot/sessions/history/task_recovery.py`
- `MemoryWriter.finalize_run()` / `ExperienceExtractor.extract()`：`src/codepilot/sessions/memory/`

## 1. 两个容易混淆的“上下文处理”

项目里有两层上下文处理，它们不是一回事。

### 1.1 Session 级压缩：防止历史消息无限增长

位置：`AgentSession._check_and_compact_before_prompt()` 和 `_compact_context_if_needed()`。

它发生在每次 `run()` 或 `continue_run()` 调模型前。判断条件大致是：

- `max_context_messages` 超过阈值；
- 或估算 token 超过 `max_context_tokens`；
- 或模型上下文窗口不够。

压缩方式是：

1. 保留最近若干条消息；
2. 把更早的消息做摘要；
3. 用一条 summary message 替换旧历史；
4. 重写 `context.jsonl`；
5. 发 `context_compacted` 事件。

如果 LLM 摘要不可用，会走 fallback summary。这个压缩改变的是 session 当前可见的历史消息。

### 1.2 ContextCompiler：每次模型调用前的动态组装

位置：`ContextCompiler.compile()`。

它不一定永久改变 session 历史，而是在“这次发给模型之前”做一次输入治理：

1. 刷新仓库快照；
2. 观察历史 `ToolResultMessage`，更新活跃文件、证据；
3. 校验 stale 文件和 stale memory；
4. 根据 `context_mode` 分配预算；
5. 选择 active files、recent evidence、memory、history；
6. 对敏感内容脱敏；
7. 组装新的 system prompt 和裁剪后的 messages；
8. 产出 `ContextReport`。

可以理解为：

- Session 压缩是“长期历史瘦身”；
- ContextCompiler 是“本次模型输入怎么装盘”。

## 2. 复杂例子 A：修改求职展示文档，触发审批，然后验证失败，修复后沉淀经验

假设用户输入：

> 请把 `docs/portfolio.md` 里的项目介绍改成更适合求职展示的版本：强调 Codepilot 是学生学习型本地 Coding Agent，但不要写成生产级平台。改完后运行测试验证。

假设当前文件里有两段类似的 “Codepilot 是一个 Coding Agent”，模型第一次 edit 的 `old_text` 不够唯一，导致 edit 失败；随后模型 read 文件、重新 edit；edit/write 需要审批；验证第一次失败，修复后通过。

这个例子会覆盖：

- 用户输入；
- context 组装；
- memory task；
- tool 审批；
- session pending approval；
- 用户批准；
- continue run；
- task 修复/验证判断；
- context 压缩与预算裁剪；
- failure experience 总结。

### 2.1 用户输入进入 Runtime

调用入口：

```text
RuntimeService.run_message(session_id, UserInput(text=...))
```

`RuntimeService` 做：

1. `_validate_request()`：检查 session 是否存在、输入是否为空、同 session 是否已有 active run。
2. `_create_active_run()`：生成 run id，写入 `_active_runs`。
3. 调用 `session.run(...)`。

此时 Runtime 内部状态类似：

```python
_active_runs = {
    "session_xxx": ActiveRun(
        run_id="a1b2...",
        session_id="session_xxx",
        status="running",
    )
}
```

`_pending_approvals` 此时还是空的。

### 2.2 Session 接管：任务恢复、上下文新鲜度、压缩检查

进入：

```text
AgentSession.run(text, run_id=...)
```

它按当前实现大致做：

1. `capture_git_baseline()`：为可能的回滚元数据准备基线。
2. 执行 `before_prompt_hooks`。
3. 调用 `TaskRecoveryStore.begin_task(text, run_id=...)`，保存当前任务恢复投影。
4. 如果用户输入表达了稳定项目约束，`MemoryWriter.remember_task(...)` 只写入 durable project memory。
5. 将未完成的 task recovery projection 塞给 agent：`agent.set_recovered_task(...)`。
6. `_check_context_freshness()`：如果上次 run 追踪的文件已经被外部改了，会注入 steering message。
7. `_check_and_compact_before_prompt()`：必要时做 session 级压缩。
8. 调用 `agent.run(...)`。

此时 task recovery 里会出现或更新一条 session 级任务投影：

```json
{
  "kind": "task",
  "scope": "session",
  "status": "active",
  "content": {
    "goal": "请把 docs/portfolio.md ... 改成更适合求职展示的版本...",
    "constraints": [],
    "confirmed_findings": [],
    "open_questions": [],
    "blocked_on": [],
    "next_action": null
  },
  "source": "user_prompt",
  "trust": "user_given"
}
```

如果用户文本里有类似“学生学习与求职展示，不要生产级复杂设计”，`MemoryWriter._maybe_remember_project_constraint()` 还可能写入 project memory：

```json
{
  "kind": "project",
  "scope": "project",
  "content": {
    "category": "project_constraint",
    "knowledge": "Codepilot 是学生学习与求职展示项目；后续设计应优先保持清晰、可解释、可演示，避免生产级复杂平台化。"
  },
  "source": "user_correction",
  "trust": "user_given"
}
```

### 2.3 Agent 构建上下文：模型真正接收到什么

`Agent._start_run()` 会构造 `AgentContext`：

```python
AgentContext(
    system_prompt=...,
    messages=list(self._state.messages),
    tools=list(self._state.tools),
    recovered_task=active_task_projection,
)
```

然后进入 `run_agent_loop()`。在真正调模型前，如果启用了 `prepare_context`，会进入 `ContextCompiler.compile()`。

这次 ContextCompiler 会：

| 来源 | 可能放入模型上下文的内容 |
|---|---|
| repository snapshot | 当前仓库概况、变更摘要 |
| active files | 之前读过/改过的 `docs/portfolio.md` 摘要 |
| recent evidence | 最近工具结果、验证状态、文件 hash |
| memory | project constraint、已验证经验、手写 pinned memory |
| history | 最近几轮用户/助手/工具消息 |
| current request | 当前用户输入，尽量不丢 |

如果之前已有经验，比如 “edit 多匹配时应先 read 再用更长 old_text”，repair 阶段会优先召回：

```text
[memory.experience]
situation=edit 工具因为 old_text 不唯一或匹配数不符合预期而失败
better_action=先 read 目标区域，再使用更长且唯一的 old_text 或 occurrence_index 进行编辑
applies_when=phase:repair, intent:edit_file, error:multiple_matches
```

`ContextReport` 会记录：

- `context_mode`；
- `budget_profile`；
- 选中了哪些 active/evidence/memory；
- 哪些东西因为 stale 或超预算被丢弃；
- `retrieved_memory_ids`；
- `sanitization` 结果。

注意：当前代码里 `AgentContext.task_signal` 这个字段已经存在，ContextCompiler 也会读取它来辅助 `task_phase/action_intent/recent_error` 检索。任务恢复来自 `TaskRecoveryStore`，长期记忆只作为 durable memory 候选来源参与召回。

### 2.4 TaskController 初始化任务计划

进入 `agent_loop._run_loop()` 后，会创建：

```python
task_controller = TaskController()
task = task_controller.initialize(current_context.messages, recovered_task=...)
```

它会根据用户消息和 recovered task 初始化轻量任务状态。概念上可能是：

```json
{
  "goal": "改写 docs/portfolio.md 并验证",
  "phase": "planning",
  "steps": [
    {"title": "理解目标文件和约束", "status": "in_progress"},
    {"title": "修改 docs/portfolio.md", "status": "pending"},
    {"title": "运行验证", "status": "pending"}
  ],
  "action_intent": null,
  "recent_error_code": null,
  "recent_failure_type": null,
  "change_sets": [],
  "attempts": []
}
```

同时发事件：

```json
{
  "type": "task_plan_created",
  "task": {...}
}
```

这些事件会被 `AgentSession._on_agent_event()` 持久化到 session event log 和 run store。

### 2.5 模型第一次行动：read 文件

模型可能先调用：

```python
ToolCall(
    id="call_read_1",
    name="read",
    arguments={"path": "docs/portfolio.md"}
)
```

`ToolCallCoordinator.execute_batch()` 做：

1. `_emit_tool_start()`：发 `tool_execution_start`。
2. `_prepare()`：
   - 当前上下文是否有这个工具；
   - 非 runtime managed 工具是否允许；
   - before_tool_call hook 是否拦截。
3. `_execute_prepared()`：调用 tool execute。
4. `_finalize()`：
   - after_tool_call hook；
   - 绑定 tool_call_id/tool_name；
   - 发 `tool_execution_end`；
   - 发 `message_start/message_end`，生成 `ToolResultMessage`。

read 成功后：

```python
ToolResultMessage(
    tool_call_id="call_read_1",
    tool_name="read",
    status="success",
    workspace_changed=False,
    metadata={"file_state": {"path": "docs/portfolio.md", "sha256": "..."}}
)
```

模块状态变化：

| 模块 | 变化 |
|---|---|
| tool | read 成功，输出文件内容和 file_state |
| context | `SessionContextState.observe_tool_result()` 后续会把该文件标记为 active file/evidence |
| memory | 不写 durable memory；read 结果只作为 context evidence / active file 使用 |
| task | read 成功只算 evidence collected，不等于任务完成 |
| session | context.jsonl/session.jsonl 追加 ToolResultMessage |

### 2.6 模型第一次 edit：old_text 不唯一，工具失败

模型尝试：

```python
ToolCall(
    id="call_edit_1",
    name="edit",
    arguments={
        "path": "docs/portfolio.md",
        "old_text": "Codepilot 是一个 Coding Agent",
        "new_text": "Codepilot 是面向学生学习与求职展示的本地 Coding Agent",
    }
)
```

假设文件里出现了两次 `old_text`，edit 返回：

```python
ToolResultMessage(
    tool_call_id="call_edit_1",
    tool_name="edit",
    status="error",
    is_error=True,
    error_code="multiple_matches",
    details={
        "reason": "multiple_matches",
        "suggested_action": "use more specific old_text or occurrence_index"
    }
)
```

TaskController 在 `after_tool_results()` 里会看到：

- 有非验证错误；
- `task.action_intent = "debug_failure"`；
- `task.recent_error_code = "multiple_matches"` 或工具给出的错误码；
- 当前 step `failure_count += 1`；
- 决策为 `repair`。

此时不会完成任务，而是让模型继续修。

memory 方面：

- 单次失败只保留在 run/context evidence 中，不直接写 durable memory。
- 当前 `ExperienceExtractor` 只在后续“失败后成功，并且验证通过”时提炼 experience。

### 2.7 repair 阶段上下文：为什么失败经验可能被召回

下一次模型调用前，ContextCompiler 又会编译上下文。

它会看到：

- recent evidence 里有 `edit multiple_matches`；
- memory 里可能已有类似失败经验；
- task 当前处于 repair/debug_failure 语境；
- active file 是 `docs/portfolio.md`。

预算策略上，repair 模式会更偏向 evidence/experience，这样模型更容易看到：

```text
上次 edit 失败：multiple_matches
建议：先 read 目标区域，使用更长 old_text 或 occurrence_index
```

如果 session 消息很多，这里可能同时发生两件事：

1. Session 级压缩把很早的聊天压成 summary；
2. ContextCompiler 仍保留最近失败工具结果、当前用户请求、匹配的 memory experience。

这就是为什么“压缩后仍能修复”的关键：不要依赖很早的原始聊天全文，而是把工具证据和经验结构化保存。

### 2.8 模型第二次 edit：这次需要审批

模型 read 目标区域后，生成更精确 edit：

```python
ToolCall(
    id="call_edit_2",
    name="edit",
    arguments={
        "path": "docs/portfolio.md",
        "old_text": "## 项目介绍\nCodepilot 是一个 Coding Agent，用于...",
        "new_text": "## 项目介绍\nCodepilot 是面向学生学习与求职展示的本地 Coding Agent，用于...",
    }
)
```

这次 edit 参数没问题，但 edit 是写文件操作，需要审批。

`ToolRuntime.execute()` 流程：

1. `PermissionPolicy.decide(...)` 得出 `approval_required`；
2. 调用 approval provider；
3. 当前 recoverable 审批路径下不会直接执行；
4. 返回 `ToolResultMessage(status="approval_required", approval_id="approval_xxx")`。

agent_loop 看到有 `approval_required`，直接返回：

```python
AgentRunResult(
    status="waiting_approval",
    stop_reason="approval_required",
    messages=[
        ...,
        AssistantMessage(content=[ToolCall(id="call_edit_2", ...)]),
        ToolResultMessage(
            tool_call_id="call_edit_2",
            tool_name="edit",
            status="approval_required",
            approval_id="approval_xxx",
        )
    ],
    task=TaskSummary(...)
)
```

TaskController 此时在 `after_tool_results()` 里会：

- block 当前 step；
- `task.phase = "waiting"`；
- `recent_error_code = "approval_required"`；
- decision 为 `wait_approval`。

Runtime 收到结果后调用 `_record_pending_approvals()`，内存里保存：

```python
PendingApproval(
    approval_id="approval_xxx",
    session_id="session_xxx",
    run_id="a1b2...",
    assistant_message=<包含 call_edit_2 的 AssistantMessage>,
    tool_call=<ToolCall edit docs/portfolio.md>,
    reason="..."
)
```

### 2.9 用户批准：恢复执行工具，不重新让模型猜

Web 或外部接口提交：

```text
approval_id=approval_xxx
decision=approve
session_id=session_xxx
```

进入：

```text
RuntimeService.approve_tool_call(...)
```

当前实现的关键保护：

1. `_resolve_pending_approval(approval_id, session_id=session_id)`：保证 approval 属于这个 session。
2. `_session_has_pending_approval_result()`：确认 session 消息里还存在待替换的 approval_required 结果。
3. 创建新的 active run。
4. `capture_git_baseline()`：为这次批准后的实际修改记录回滚基线。
5. `_execute_approved_tool()`：执行已批准工具。

这里有个重要设计点：批准后的执行不会再走 ToolRuntime 的审批判断，否则会再次返回 `approval_required` 形成循环；但它仍然复用 `ToolCallCoordinator`，所以 before/after hook、工具事件、`ToolResultMessage` 生成仍然统一。

执行成功后生成：

```python
ToolResultMessage(
    tool_call_id="call_edit_2",
    tool_name="edit",
    status="success",
    approved=True,
    approval_id="approval_xxx",
    affected_paths=["docs/portfolio.md"],
    workspace_changed=True,
    diff_summary="..."
)
```

然后：

1. `session.replace_tool_result_message(...)` 把原来的 `approval_required` 占位替换成 success；
2. 同时重写 `context.jsonl` 和 `session.jsonl`；
3. 从 `_pending_approvals` 删除该 approval；
4. 写 `tool_approval_decision` 事件；
5. 调用 `session.continue_run(run_id=同一个 active_run.run_id)`。

此时模型看到的不是“用户说批准了”，而是“工具真的执行成功了”，这点很关键。

### 2.10 修改后为什么不能直接完成：TaskController 要 fresh verification

批准 edit 成功后，`RunState.collect_tool_results()` 会记录：

```python
workspace_changed = True
affected_paths = {"docs/portfolio.md"}
fresh_verification_passed = False
```

TaskController 在 completion check 里有一条硬逻辑：

```text
如果 workspace_changed=True 且 fresh_verification_passed=False，
则任务未完成，缺 fresh_verification。
```

所以模型不能只说“我改完了”。它会收到 completion steering，大意是：

> 工作区已经修改，但还没有新鲜验证，请运行验证或说明为什么无法验证。

于是模型可能调用：

```python
ToolCall(
    id="call_test_1",
    name="bash",
    arguments={"command": "python -m pytest test/test_docs_contract.py -q"}
)
```

bash 也可能需要审批。如果需要审批，会重复上面的 approval loop。

### 2.11 验证失败：进入 repair，而不是完成

假设测试失败：

```python
ToolResultMessage(
    tool_call_id="call_test_1",
    tool_name="bash",
    status="error",
    is_error=True,
    verification={
        "status": "failed",
        "command": "python -m pytest test/test_docs_contract.py -q",
        "exit_code": 1,
        "summary": "expected phrase '本地 Coding Agent' but got ..."
    }
)
```

`RunState.collect_tool_results()` 会把 verification 记录进去：

```python
verification = [
    RunVerification(
        tool_call_id="call_test_1",
        tool_name="bash",
        status="failed",
        command="python -m pytest test/test_docs_contract.py -q",
        exit_code=1,
        summary="..."
    )
]
fresh_verification_passed = False
```

TaskController 看到 failed verification 后：

- `task.action_intent = "debug_failure"`；
- `task.recent_error_code = "verification_failed"`；
- `task.recent_failure_type = "verification_failed"`；
- 当前 step failure_count 增加；
- 标记最新 change_set 失败；
- decision 为 `repair`。

如果连续验证失败达到阈值，且已经有 change_sets，TaskController 会转成 `propose_revert`：

```text
报告可能需要撤销的变更并等待用户确认
```

注意：第一版不会自动回滚，它只提出受控回滚建议。

### 2.12 修复后验证通过：任务才真正完成

模型根据失败摘要再 edit 一次，可能又走审批。修复后再次运行同一测试：

```python
ToolResultMessage(
    tool_call_id="call_test_2",
    tool_name="bash",
    status="success",
    verification={
        "status": "passed",
        "command": "python -m pytest test/test_docs_contract.py -q",
        "exit_code": 0,
        "summary": "1 passed"
    }
)
```

此时：

- `RunState.fresh_verification_passed = True`；
- TaskController `_has_passed_verification()`；
- `_complete_verified_steps()`；
- `_mark_latest_changes_verified()`；
- completion check 满足；
- run 最终 `status="completed"`，`stop_reason="final_answer"`。

模型最后回复：

```text
已完成：我已更新 docs/portfolio.md，使描述更偏学生学习与求职展示，并保留“非生产级复杂平台”的定位。验证命令 python -m pytest test/test_docs_contract.py -q 已通过。
```

### 2.13 run 结束后 memory 怎么沉淀经验

`AgentSession.run()` 或 `continue_run()` 结束后会：

1. `store.append_run_result(result)`；
2. `_write_rollback_metadata(result, baseline)`；
3. `_finalize_memory(result)`。

`TaskRecoveryStore.update_from_result()` 会更新当前任务恢复投影：

```json
{
  "task_progress": {
    "completed_steps": [...],
    "pending_steps": [],
    "blocked_steps": [],
    "completion_satisfied": true,
    "completion_reason": "all_steps_completed"
  },
  "confirmed_findings": [
    "Verification passed: python -m pytest test/test_docs_contract.py -q",
    "Run completed: final_answer"
  ],
  "next_action": null
}
```

然后 `ExperienceExtractor.extract(result)` 会尝试总结经验。

当前实现主要提炼两类经验：

#### 经验 1：edit 多匹配失败后成功，并且验证通过

触发条件：

1. 有 `edit` 失败；
2. 错误码是 `multiple_matches` 或 `unexpected_match_count`；
3. 后面有 `edit` 成功；
4. 后面还有 passed verification。

生成 experience：

```json
{
  "kind": "experience",
  "content": {
    "lesson_type": "tool_usage",
    "situation": "edit 工具因为 old_text 不唯一或匹配数不符合预期而失败",
    "failed_attempt": "直接使用不够唯一的 old_text 调用 edit",
    "failure_signal": "multiple_matches",
    "better_action": "先 read 目标区域，再使用更长且唯一的 old_text 或 occurrence_index 进行编辑",
    "applies_when": [
      "phase:repair",
      "intent:edit_file",
      "error:multiple_matches"
    ],
    "maturity": "verified",
    "fingerprint": "..."
  },
  "trust": "verified",
  "related_paths": ["docs/portfolio.md"]
}
```

#### 经验 2：验证失败后修复，并且后续验证通过

触发条件：

1. 某个工具结果里 `verification.status == "failed"`；
2. 后面有 `verification.status == "passed"`。

生成 experience：

```json
{
  "kind": "experience",
  "content": {
    "lesson_type": "verification_repair",
    "situation": "修改后验证失败，需要基于失败日志修复",
    "failed_attempt": "第一次修改没有满足验证期望",
    "failure_signal": "verification_failed",
    "better_action": "先读取失败摘要和相关文件，再做最小修复并重新运行同一验证命令",
    "applies_when": [
      "phase:repair",
      "intent:debug_failure",
      "error:verification_failed"
    ],
    "maturity": "verified",
    "fingerprint": "..."
  }
}
```

`MemoryConsolidator.upsert_experience()` 会按 fingerprint 合并重复经验：

- 如果已有同 fingerprint 经验，增加 `occurrence_count`；
- 合并 evidence refs；
- 如果新经验是 verified，会把 maturity 升级为 verified。

这就是“失败经验总结”的落点。

## 3. 复杂例子 B：长会话触发上下文压缩，但仍能继续审批恢复

假设用户和 agent 已经对话了很多轮：

1. 读了多个文件；
2. 修改过几次；
3. 有一些失败工具调用；
4. 有一条 project memory：“不要生产级复杂设计”；
5. 当前又要修改 `src/codepilot/runtime/service.py`。

这时上下文可能超过限制。

### 3.1 Session 级压缩发生

在 `AgentSession.run()` 或 `continue_run()` 前：

```text
_check_and_compact_before_prompt()
```

如果消息数/token 超阈值：

```text
older messages → summary
recent messages → 保留
```

压缩后 session.messages 可能从：

```text
user1, assistant1, tool1, ..., user35, assistant35, tool35
```

变成：

```text
summary_message, user32, assistant32, tool32, user33, assistant33, tool33, user34, assistant34, tool34
```

重要影响：

- 很早的原始消息可能不在上下文里；
- 但 verified experience、project memory、recent evidence 仍可通过 ContextCompiler 重新注入；
- 工具结果和验证证据如果太旧，可能被标记 stale，不能作为当前完成证据。

### 3.2 ContextCompiler 仍会保护工具消息配对

每次模型调用前，ContextCompiler 还会做 history suffix 选择。

如果裁剪后留下了某条 `ToolResultMessage`，但对应的 assistant tool_call 被裁掉了，`_repair_tool_message_pairs()` 会补一个最小 AssistantMessage，确保 provider 能理解 tool result 对应哪个 call。

这对审批恢复也重要：模型继续时需要看到“助手曾经调用工具 → 工具结果成功/失败”的配对关系。

### 3.3 审批恢复时不依赖模型记住审批请求

审批恢复不靠压缩历史里保留 approval 文本，而靠 Runtime 的 pending approval：

```python
_pending_approvals["approval_xxx"] = PendingApproval(...)
```

批准时还会检查 session 当前 messages 里确实有对应 `approval_required` 占位。也就是说：

- Runtime 负责“哪个工具等待用户批准”；
- Session messages 负责“这条审批占位还在不在”；
- ContextCompiler 负责“继续模型时能不能看到工具成功结果”。

如果压缩或外部操作导致占位消失，`approve_tool_call()` 会拒绝执行，避免工具副作用发生后无法替换历史。

## 4. 复杂例子 C：用户纠偏“不是这个方向，别做生产级”

用户说：

> 不是这个方向，别搞生产级平台化，我这是学生求职展示项目，回到简单清晰的 MVP。

这类输入会影响三个地方。

### 4.1 memory：项目定位可能被写入 project memory

`MemoryWriter.remember_task()` 会调用 `_maybe_remember_project_constraint()`。

如果文本表达了“学生/求职/学习 + 不要生产级/复杂设计”，会写：

```json
{
  "kind": "project",
  "scope": "project",
  "content": {
    "category": "project_constraint",
    "knowledge": "Codepilot 是学生学习与求职展示项目；后续设计应优先保持清晰、可解释、可演示，避免生产级复杂平台化。"
  },
  "source": "user_correction",
  "trust": "user_given"
}
```

后续 ContextCompiler 检索 memory 时，这条 project memory 会作为高价值上下文，提醒模型不要偏离项目定位。

### 4.2 task：可能触发 replan 或 propose_revert

如果用户明确“不是这个方向”“改回去”，TaskController 当前设计倾向于：

- 把用户纠偏当作约束；
- 对连续失败或错误方向的 change_set 提出 `propose_revert`；
- 不自动回滚，而是等待用户确认。

第一版的原则是：识别回滚目标和风险，不直接修改文件。

### 4.3 context：QA/repair 模式更重视历史决定和 project memory

如果当前是问答/纠偏类输入，ContextCompiler 的 `context_mode` 可能更偏 `qa` 或 `repair`：

- QA 更重视 history、decision、project memory；
- repair 更重视 evidence、experience；
- final/verify 更重视 verification/task。

所以这类纠偏不会只成为一条普通聊天记录，而会影响后续检索和预算分配。

## 5. 当前审批闭环的两种模式

这个项目里要区分“同步 CLI 审批”和“可恢复审批”。

### 5.1 CLI 同步审批

`CliApprovalProvider` 会在工具执行时直接问：

```text
Approve once? [y/N]
```

如果用户输入 yes，ToolRuntime 可以继续执行工具；如果 no，返回审批拒绝/需要审批。

### 5.2 Runtime/Web 可恢复审批

本文主讲的是这种：

1. 模型请求工具；
2. ToolRuntime 返回 `approval_required`；
3. agent_loop 返回 `waiting_approval`；
4. Runtime 记录 `_pending_approvals`；
5. Web/API 调 `approve_tool_call(approval_id, decision, session_id=...)`；
6. Runtime 执行已批准工具；
7. 替换占位工具结果；
8. `continue_run()`；
9. 模型基于真实工具结果回复。

这个模式适合 Web UI：用户可以在 UI 上看到待审批工具、参数、风险，然后点击批准。

## 6. 一个完整状态快照：审批前、审批后、最终完成

### 6.1 审批前

```json
{
  "runtime": {
    "active_runs": {},
    "pending_approvals": {
      "approval_xxx": {
        "session_id": "session_xxx",
        "tool_call_id": "call_edit_2",
        "tool_name": "edit"
      }
    }
  },
  "run_result": {
    "status": "waiting_approval",
    "stop_reason": "approval_required",
    "workspace_changed": false
  },
  "session_messages_tail": [
    "UserMessage: 请修改 docs/portfolio.md",
    "AssistantMessage: ToolCall edit call_edit_2",
    "ToolResultMessage: approval_required approval_xxx"
  ],
  "task": {
    "phase": "waiting",
    "recent_error_code": "approval_required",
    "recent_failure_type": "approval_required"
  }
}
```

### 6.2 用户批准、工具执行成功后

```json
{
  "runtime": {
    "pending_approvals": {}
  },
  "session_messages_tail": [
    "UserMessage: 请修改 docs/portfolio.md",
    "AssistantMessage: ToolCall edit call_edit_2",
    "ToolResultMessage: success approved=true workspace_changed=true"
  ],
  "run_result": {
    "workspace_changed": true,
    "affected_paths": ["docs/portfolio.md"]
  }
}
```

### 6.3 验证通过、最终回复后

```json
{
  "run_result": {
    "status": "completed",
    "stop_reason": "final_answer",
    "workspace_changed": true,
    "affected_paths": ["docs/portfolio.md"],
    "verification": [
      {
        "tool_name": "bash",
        "status": "passed",
        "command": "python -m pytest test/test_docs_contract.py -q"
      }
    ]
  },
  "task": {
    "completion_satisfied": true,
    "completed_steps": [
      "理解目标文件和约束",
      "修改 docs/portfolio.md",
      "运行验证"
    ]
  },
  "memory": {
    "task_next_action": null,
    "possible_experience": [
      "edit 多匹配失败后先 read 再精确 edit",
      "验证失败后读取失败摘要并最小修复"
    ]
  }
}
```

## 7. 这个设计对“学生求职型项目”的意义

这个闭环不是生产级平台那种复杂 DAG / 分布式工作流 / 数据库任务引擎，而是一个适合展示和讲解的本地 MVP：

- 能讲清楚用户请求如何变成模型上下文；
- 能讲清楚工具执行如何留下证据；
- 能讲清楚为什么写文件要审批；
- 能讲清楚审批后如何恢复，而不是让模型重新猜；
- 能讲清楚修改后为什么要 fresh verification；
- 能讲清楚失败如何变成下次可复用经验；
- 能讲清楚上下文压缩后为什么不一定丢失关键事实。

这对面试展示很有价值：它体现的是“工程闭环意识”，而不是堆生产级名词。

## 8. 当前实现边界和注意点

当前实现仍然是 MVP，有几个边界要讲清楚：

1. pending approvals 是 Runtime 内存态，不是数据库持久化队列。进程重启后不能自动恢复内存里的 pending approval。
2. 第一版不自动回滚。`propose_revert` 是建议受控回滚并等待用户确认。
3. ExperienceExtractor 是规则模板，不是让 LLM 自由总结经验。
4. ContextCompiler 有动态预算和脱敏，但不是企业级 DLP。
5. 没有 Docker/VM 沙箱；工具安全主要靠权限策略、审批、路径校验和工具自身防线。
6. `AgentContext.task_signal` 已有字段，ContextCompiler 支持读取，但普通主流程里不是每一轮都稳定注入完整 task signal。
7. read/edit/write/bash 的工具证据很重要；如果外部扩展工具不提供 metadata，系统会保守处理，不应乐观放行。

## 9. 面试时可以怎么讲

可以用这段话概括：

> Codepilot 的主流程是一个本地闭环：Runtime 接用户输入并管理 session；Session 在 run 前记录 task recovery、检查上下文新鲜度和压缩历史；AgentLoop 调模型并执行工具；ToolRuntime 负责权限和审批；如果工具需要审批，run 会以 waiting_approval 暂停，Runtime 保存 pending approval。用户批准后，Runtime 校验 session 归属和占位结果，执行已批准工具，替换原 approval_required 工具结果，再 continue_run，让模型基于真实工具结果输出最终答案。TaskController 会防止“文件改了但没验证”就完成，TaskRecoveryStore 会保存任务恢复投影，MemoryWriter 会从失败后成功的验证闭环里提炼 durable experience。
