# 钉钉接入演示脚本

这份脚本用于真实钉钉 Stream Smoke Test。目标不是覆盖所有单元测试，而是让演示者能从手机端完整证明：钉钉入口可启动、可远程发起任务、可审批、可拒绝、可排障，并且不会影响 CLI。

---

## 1. 分支提交边界建议

这条 `codex/dingtalk-interface` 分支建议拆成 4 个提交，方便讲解和回滚：

| 提交 | 内容边界 | 主要验证 |
|---|---|---|
| 1. 替换 Web 为钉钉基础入口 | 删除 `interfaces/web`，新增 `interfaces/dingtalk`、脚本入口、基础 contract 测试 | `test_dingtalk_contract.py`、`test_namespace.py` |
| 2. 手机端 Markdown 体验 | 出站消息支持 markdown，任务接收、审批、状态、help 分组展示 | 钉钉契约测试、fake SDK markdown/fallback |
| 3. 交付闭环增强 | entrypoint 测试、审计 JSONL、审批恢复细化、pending approval 状态详情 | `test_dingtalk_entrypoint.py`、runtime/tools 安全测试 |
| 4. 文档与演示材料 | `0Guide`、钉钉设计文档、这份演示脚本 | 文档人工审阅、真实 smoke test |

不要把 CLI 回归修复和钉钉功能提交混在一起。如果真实 smoke test 暴露 SDK 适配问题，建议单独追加第 5 个提交：`fix: adapt dingtalk stream markdown reply`。

---

## 2. 演示前准备

本机准备：

```powershell
cd E:\Project_python\agent\Codepilot
python -m pytest test/test_dingtalk_entrypoint.py test/test_dingtalk_contract.py test/test_namespace.py -q
```

钉钉准备：

1. 在钉钉开放平台创建 Stream 模式机器人。
2. 获取 `client_id` 和 `client_secret`。
3. 确认手机端发送消息时的 sender staff id。

环境变量：

```powershell
$env:DINGTALK_CLIENT_ID = "<client_id>"
$env:DINGTALK_CLIENT_SECRET = "<client_secret>"
$env:CODEPILOT_DINGTALK_ALLOWED_USERS = "<sender_staff_id>"
```

启动桥接进程：

```powershell
codepilot-dingtalk serve --cwd E:\Project_python\agent\Codepilot
```

预期：

- 进程保持运行，不立即退出。
- 如果 SDK 缺失，错误应提示安装 `codepilot[dingtalk]`。
- 如果 allowed user 缺失，错误应提示配置 `--allowed-user` 或 `CODEPILOT_DINGTALK_ALLOWED_USERS`。

---

## 3. 演示流程

### Step 1: 查看状态

手机发送：

```text
status
```

预期钉钉回复包含：

```text
Codepilot DingTalk status
workspace
session_id
active_run
permission_mode: ask
workspace_state
pending_approvals
```

本地验证：

```powershell
Get-Content .codepilot\dingtalk\audit.jsonl -Tail 5
```

预期审计包含 `message_received` 和 `status_requested`。

### Step 2: 发起只读任务

手机发送：

```text
cp 简要说明这个项目的运行主线，不要修改文件
```

预期钉钉回复分两段：

```text
Codepilot run accepted
Codepilot run finished
```

预期本地状态：

- 没有业务文件被修改。
- `.codepilot/dingtalk/audit.jsonl` 增加 `run_accepted` 和 `run_finished`。

### Step 3: 发起需要写文件的任务

手机发送：

```text
cp 在 docs/design/6dingtalk-smoke-output.md 写一段钉钉 smoke test 记录
```

预期钉钉回复：

```text
Tool approval required
approval_id: approval_xxx
approve approval_xxx
deny approval_xxx
```

本地验证：

```powershell
Get-Content .codepilot\dingtalk\audit.jsonl -Tail 10
```

预期审计包含 `approval_requested`。

### Step 4: 批准工具调用

手机发送：

```text
approve approval_xxx
```

预期钉钉回复：

```text
Tool approval received
Codepilot run finished
affected_paths
```

本地验证：

```powershell
Test-Path docs\design\6dingtalk-smoke-output.md
git status --short
```

预期：

- 文件被创建或修改。
- 审计包含 `approval_received` 和 `approval_finished`。
- 工具执行仍然出现在 `.codepilot/runs/<run_id>/events.jsonl` 和 `trace.json`。

### Step 5: 拒绝工具调用

重新触发一个写文件任务，拿到新的 `approval_id` 后发送：

```text
deny approval_xxx
```

预期：

- 钉钉回复显示审批已收到，最终 run 会说明工具被拒绝或任务未完成。
- 目标文件不应产生对应修改。
- 审计包含 `approval_received` 和 `approval_finished`。

### Step 6: Busy 保护

发送一个较长任务后，马上再发：

```text
cp 再启动一个任务
```

预期：

```text
Codepilot is busy
```

审计预期：

```text
message_rejected
reason: busy
```

### Step 7: Dirty workspace 拒绝

在本地制造一个未提交改动：

```powershell
"dirty check" | Out-File -Encoding utf8 .\dingtalk-dirty-check.txt
```

手机发送：

```text
cp 现在尝试运行一个任务
```

预期：

```text
Remote run refused: workspace is dirty
```

清理演示文件：

```powershell
Remove-Item .\dingtalk-dirty-check.txt
```

### Step 8: 未授权用户拒绝

用不在白名单里的钉钉账号发送：

```text
status
```

预期：

```text
DingTalk sender is not authorized
```

审计预期：

```text
message_rejected
reason: unauthorized_sender
```

### Step 9: Cancel

如果有 active run，手机发送：

```text
cancel
```

预期：

```text
Active Codepilot run cancelled.
```

如果没有 active run，预期：

```text
No active Codepilot run.
```

---

## 4. 演示收尾检查

运行：

```powershell
python -m pytest test/test_cli_refactor.py test/test_cli_shell.py test/test_command_router.py -q
python -m pytest test/test_runtime_service_refactor.py test/test_tool_execution_security.py test/test_tools_security.py -q
```

检查：

- CLI 测试仍通过。
- 钉钉审计文件没有明文 `DINGTALK_CLIENT_SECRET`、API key、password、token。
- `git status --short` 中的业务文件改动符合演示预期。
- `.codepilot/` 内部运行记录可以保留作为证据，但不作为业务代码提交。

如果真实 SDK 的 `reply_markdown()` 参数和 fake SDK 不一致，只修改 `src/codepilot/interfaces/dingtalk/transport.py` 适配，不改 bridge、runtime、tools 主链。
