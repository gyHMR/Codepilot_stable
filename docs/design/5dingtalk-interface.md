# 钉钉远程入口设计

这份文档按真实运行顺序说明 Codepilot 的钉钉接入。它的定位不是新的 Agent Runtime，而是一个远程 interface：手机端发消息，电脑端桥接进程接收消息，然后仍然把任务交给 `RuntimeService -> AgentSession -> ToolRuntime` 主链执行。

如果你要做真实手机端验收，可以直接按 `docs/design/6dingtalk-demo-script.md` 操作；本文负责解释设计边界和安全默认。

钉钉入口和 CLI 完全分开：

- CLI 使用 `codepilot`，入口在 `src/codepilot/interfaces/cli/main.py`。
- 钉钉使用 `codepilot-dingtalk`，入口在 `src/codepilot/interfaces/dingtalk/main.py`。
- 钉钉不会新增 `codepilot dingtalk` 子命令，也不会改变 `codepilot -p`、交互式、RPC 或斜杠命令。

---

## 1. 启动前准备

第一版只支持钉钉 Stream 长连接，需要安装可选依赖：

```bash
pip install "codepilot[dingtalk]"
```

本地桥接进程只从环境变量读取钉钉凭证：

```bash
set DINGTALK_CLIENT_ID=your_client_id
set DINGTALK_CLIENT_SECRET=your_client_secret
```

允许控制 Codepilot 的钉钉用户必须显式配置，二选一：

```bash
codepilot-dingtalk serve --allowed-user user_staff_id
```

或者：

```bash
set CODEPILOT_DINGTALK_ALLOWED_USERS=user_staff_id
```

没有白名单时启动会失败。这是远程改代码入口的安全底线。

---

## 2. 启动桥接进程

推荐从要修改的 Git 工作区启动：

```bash
codepilot-dingtalk serve --cwd E:\Project_python\agent\Codepilot --allowed-user user_staff_id
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--cwd` | 绑定本地工作区，一个桥接进程只服务一个 workspace |
| `--session-id` | 绑定或恢复一个已有 Codepilot session |
| `--model provider/model-id` | 覆盖模型配置 |
| `--allowed-user` | 允许控制的钉钉 sender staff id，可重复 |
| `--allow-dirty` | 允许工作区有本地改动时远程运行 |
| `--verbose-events` | 把非最终工具事件也发到钉钉 |

默认安全策略：

- `tool_permission_mode` 强制为 `ask`。
- 写文件、shell、Git 回退等高风险动作必须审批。
- 工作区默认必须是 Git 仓库且没有用户脏改动。
- `.codepilot/` 内部运行状态不会导致 dirty 拒绝。
- 同一桥接进程同一时间只允许一个 active run。

---

## 3. 手机端命令

钉钉聊天窗口支持这些文本命令：

| 命令 | 作用 |
|---|---|
| `cp <prompt>` | 发起一次 Codepilot 任务 |
| `approve <approval_id>` | 批准一个等待中的工具调用 |
| `deny <approval_id>` | 拒绝一个等待中的工具调用 |
| `status` | 查看 session、workspace、active run、pending approvals |
| `cancel` | 尝试取消当前运行 |
| `help` | 查看命令说明 |

一次典型流程是：

```text
手机发送：cp 修改 README 里的项目说明
钉钉回复：run accepted，显示 session 和权限模式
Codepilot 运行：模型思考、必要时请求工具
钉钉回复：tool approval required，显示 approval_id 和 approve/deny 示例
手机发送：approve approval_xxx
钉钉回复：approval received
Codepilot 继续：审批后的工具调用仍走 ToolRuntime
钉钉回复：run finished，显示 run_id、status、affected_paths
```

钉钉回复默认只发摘要，不发送完整 stdout、环境变量、长日志或密钥。出站内容会经过 `safe_reply()`，内部使用 `redact_artifact()` 脱敏并截断。

---

## 4. 审批恢复

工具审批暂停时，Runtime 会保存 pending approval：

```text
ToolRuntime.execute()
  -> approval_required ToolResult
  -> AgentRunResult(status="waiting_approval")
  -> RuntimeService._record_pending_approvals()
```

手机审批后，钉钉桥接层只调用：

```text
RuntimeService.approve_tool_call(approval_id, "approve" 或 "deny")
```

批准后的工具不会绕过安全边界。它仍然经过：

```text
ToolRuntime.execute_approved()
  -> schema 校验
  -> 权限/耗时元数据
  -> 真实工具执行
  -> ToolResultGuard
  -> AgentSession 继续 run
```

如果审批后又触发新的高风险工具调用，钉钉会返回新的 pending approval，并附上下一条命令示例：

```text
approve approval_next
deny approval_next
```

---

## 5. 审计记录

钉钉入口层会把远程控制过程写入：

```text
.codepilot/dingtalk/audit.jsonl
```

它只记录 interface 层事实，不替代 run trace：

- 收到哪条 message id。
- 来自哪个 sender id 的哈希。
- 当前 command、session_id、run_id、approval_id。
- 工作区状态、拒绝原因、完成状态。

不会记录：

- 完整 prompt。
- 完整 stdout。
- 工具完整参数。
- 环境变量。
- API key、token、password、cookie。

排查时可以这样看：

| 问题 | 先看 |
|---|---|
| 手机消息没反应 | `audit.jsonl` 是否有 `message_received` |
| 被拒绝 | `message_rejected.reason` |
| 工具审批没继续 | `approval_received` 和 `approval_finished` |
| Run 是否结束 | `run_finished.run_id` 和 `status` |
| 详细运行证据 | `.codepilot/runs/<run_id>/events.jsonl` 和 `trace.json` |

审计写入失败不会阻断任务，只会在本地 stderr 输出一条脱敏警告。

---

## 6. 常见失败原因

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动失败：缺少 `DINGTALK_CLIENT_ID` | 没配置钉钉 Stream 凭证 | 设置环境变量后重启 |
| 启动失败：缺少 allowed user | 没配置远程控制白名单 | 添加 `--allowed-user` 或环境变量 |
| 启动失败：`codepilot[dingtalk]` | 没安装可选 Stream SDK | 安装可选依赖 |
| 任务被拒绝：not a Git repository | 工作区不是 Git 仓库 | 从 Git 工作区启动或显式 `--allow-dirty` |
| 任务被拒绝：dirty | 工作区有用户改动 | commit/stash 后再运行，或显式 `--allow-dirty` |
| 第二个任务被拒绝：busy | 当前 session 正在运行 | 等待结束或发送 `cancel` |
| 审批失败：not found | approval_id 不存在、session 不匹配或已失效 | 发送 `status` 查看当前 pending approvals |

---

## 7. 人工 Smoke Checklist

真实手机端验收建议按下面顺序跑：

- `codepilot-dingtalk serve --help` 能显示启动参数。
- 缺少 `DINGTALK_CLIENT_ID` 时启动失败且错误清楚。
- 缺少 allowed user 时启动失败且错误清楚。
- 未授权用户发送 `status` 被拒绝。
- 授权用户发送 `status` 能看到 workspace、session、pending approvals。
- 非 Git workspace 下发送 `cp hello` 默认被拒绝。
- dirty workspace 下发送 `cp hello` 默认被拒绝。
- clean Git workspace 下发送只读任务能收到 run accepted 和 run finished。
- 触发写文件任务时能收到 approval required。
- 手机发送 `approve <approval_id>` 后能继续执行并返回 affected paths。
- 手机发送 `deny <approval_id>` 后 Agent 能收到拒绝结果。
- active run 期间再次发送 `cp` 会收到 busy。
- `cancel` 能取消或明确说明没有 active run。
- `.codepilot/dingtalk/audit.jsonl` 有脱敏审计记录。

完成这些检查后，钉钉入口就达到了第一版“可演示、可恢复、可排障”的目标。
