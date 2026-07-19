# Codepilot

> 一个使用 Python 构建、参考 Claude Code 与 OpenCode 设计的本地 Coding Agent 工程原型。

![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue)
![Version](https://img.shields.io/badge/Version-0.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Codepilot 面向希望学习 Agent 工程、准备 AI 应用开发岗位或构建个人项目的入门级开发者。项目不是只演示一次 LLM 调用的 Demo，而是围绕真实代码任务实现了一条完整链路：理解仓库、治理上下文、调用模型、受控执行工具、验证变更、持久化状态，并在中断后恢复执行。

当前定位是“可用的工程原型”：核心运行链路和主要工程机制已经具备，可以用于学习、调试与继续开发，但仍在持续迭代，不代表已经满足生产环境的稳定性、安全性和兼容性要求。

## 核心能力

| 能力 | 说明 |
|---|---|
| 仓库理解 | 读取项目指令、搜索代码、追踪活动文件并构造当前工作集 |
| 任务执行 | 支持 `read`、`plan`、`build` 三种模式，覆盖分析、规划和代码修改 |
| 模型与工具循环 | Core 根据模型消息、工具结果和用户输入持续决定下一步动作 |
| 工具安全 | 统一进行注册校验、参数解码、权限判断、用户审批、超时和结果防护 |
| Context 治理 | 按 token 预算投影仓库事实、任务状态、工具证据、Memory 和对话历史 |
| 长期 Memory | 管理用户偏好、项目规则和可复用经验的候选、审批、召回与生命周期 |
| 扩展能力 | 支持内置工具、Python Extension、Markdown Skill 和 MCP 远程工具 |
| 恢复与回滚 | 持久化 Session、Run 和 checkpoint，支持暂停恢复与工作区回滚 |
| Plan 与 Subagent | 支持结构化计划、计划审批及受限环境中的探索型子 Agent |
| 多界面接入 | 提供 CLI、Web 工作台、JSONL RPC 和 DingTalk 远程入口 |
| 可观测性 | 记录运行事件、工具活动、Context、Plan、Memory 和完整 Run Trace |

## 典型运行流程

```text
用户任务
  → Session 创建或恢复
  → Context 物化当前模型输入
  → Core 决定调用模型或执行工具
  → Tools 校验权限并执行副作用
  → Core 根据结果继续、等待或终止
  → Runtime 提交边界和终态结果
  → CLI / Web / DingTalk 展示结果
```

典型场景包括：

- 分析陌生代码仓库并解释关键调用链。
- 制定结构化计划后修改代码并运行验证。
- 在写文件或执行高风险命令前暂停并请求用户审批。
- 进程中断后从 Session checkpoint 恢复，而不是重新开始整个任务。
- 在新 Session 中召回已经审批的项目规则和长期经验。

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- 建议在虚拟环境中安装
- 使用 Web 工作台或 DingTalk 时需安装对应可选依赖

### 安装

```bash
git clone https://github.com/gyHMR/Codepilot.git
cd Codepilot
python -m pip install -e ".[dev]"
```

### 配置模型

初始化本地配置：

```bash
codepilot config init
```

编辑工作区中的 `.codepilot/model.local.json`：

```json
{
  "api": "openai-compatible",
  "provider": "deepseek",
  "model_id": "deepseek-v4-flash",
  "base_url": "https://api.deepseek.com/v1",
  "api_key_env": "DEEPSEEK_API_KEY",
  "context_window": 64000,
  "max_tokens": 8192,
  "reasoning": false,
  "vision": false
}
```

推荐通过 `api_key_env` 从环境变量读取密钥，不要把真实 API Key 写入仓库。

检查配置和凭据：

```bash
codepilot config check
codepilot config show
codepilot config explain model
```

### 启动 CLI

```bash
# 交互模式
codepilot

# 执行单次任务
codepilot -p "解释这个项目的核心调用链"

# 指定工作区
codepilot --workspace /path/to/project

# 只读分析
codepilot --mode read -p "分析当前仓库结构"

# 先生成和审批计划
codepilot --mode plan -p "规划一次模块重构"

# 允许执行代码修改任务
codepilot --mode build -p "修复失败测试并验证"

# 所有需要授权的操作都询问用户
codepilot --permission-mode ask
```

任务模式：

| 模式 | 用途 |
|---|---|
| `read` | 分析代码和读取仓库，不执行写操作 |
| `plan` | 先收集事实并形成结构化计划，等待用户决定 |
| `build` | 执行代码修改、命令验证和完整开发任务 |

权限模式：

| 模式 | 用途 |
|---|---|
| `read-only` | 只允许无副作用操作 |
| `workspace-write` | 允许受工作区边界保护的常规修改 |
| `ask` | 工具达到审批条件时暂停并询问用户 |

### 启动 Web 工作台

```bash
python -m pip install -e ".[web]"
codepilot web
codepilot web --workspace /path/to/project --port 8000
```

默认地址为 `http://127.0.0.1:8000`。当前版本没有登录认证，不建议将服务直接暴露到局域网或公网。

开发 Web 前端时，先安装前端依赖，再由一个命令同时启动 FastAPI 自动重载和 Vite HMR：

```bash
cd web
npm ci
cd ..
codepilot web --dev
codepilot web --dev --port 9000 --frontend-port 5174
```

浏览器访问终端输出的前端地址，默认是 `http://127.0.0.1:5173`。工作区仍是执行命令时的目录，也可通过 `--workspace` 指定。

### 启动 DingTalk 入口

```bash
python -m pip install -e ".[dingtalk]"

export DINGTALK_CLIENT_ID="your-client-id"
export DINGTALK_CLIENT_SECRET="your-client-secret"

codepilot-dingtalk serve \
  --cwd /path/to/project \
  --allowed-user <sender_staff_id>
```

DingTalk 远程入口默认使用严格审批策略。工作区存在未提交改动时会拒绝远程 Run，只有显式配置后才允许继续。

## 基本使用

交互模式中的常用命令：

| 命令 | 说明 |
|---|---|
| `/help` | 查看命令帮助 |
| `/status` | 查看当前 Session、Run、模型和模式 |
| `/tools` | 查看当前模型可见工具 |
| `/context` | 查看最近一次 Context 投影报告 |
| `/memory` | 查询和管理长期 Memory |
| `/rollback` | 预览或执行最近 Run 的工作区回滚 |
| `/model` | 查看或切换模型 |
| `/usage` | 查看累计模型用量 |
| `/exit` | 退出交互模式 |

### MCP 与 Skill

MCP 服务最终会被适配为统一的 `ToolRegistration`，继续使用相同的参数校验、权限审批和结果防护链路。示例配置位于：

- [`docs/examples/extensions/demo_mcp_config.json`](docs/examples/extensions/demo_mcp_config.json)
- [`docs/examples/extensions/github_mcp_config.json`](docs/examples/extensions/github_mcp_config.json)

Skill 使用 Markdown 描述专用工作流和参考资源，示例见：

- [`docs/examples/extensions/demo-review/SKILL.md`](docs/examples/extensions/demo-review/SKILL.md)

Python Extension 示例见：

- [`docs/examples/extensions/demo_extension.py`](docs/examples/extensions/demo_extension.py)

## 核心架构

Codepilot 使用单向依赖结构，业务事实由对应模块唯一拥有，其他模块只能引用或投影：

```text
protocols
  ↓
llm / tools
  ↓
core
  ↓
sessions / observability
  ↓
extensions
  ↓
runtime
  ↓
interfaces
```

| 模块 | 职责 |
|---|---|
| `protocols` | 跨层共享的消息、模型、工具和运行结果协议 |
| `llm` | Provider 注册、请求转换、流式事件和错误归一化 |
| `tools` | 工具注册、权限、审批、执行、状态和结果 |
| `core` | 任务目标、计划、Reducer、Policy 和 Agent 主循环 |
| `sessions` | Session、Run、消息、等待状态和 checkpoint 权威存储 |
| `sessions/context` | 工作集投影、预算、压缩、artifact 和 Context checkpoint |
| `sessions/memory` | 长期 Memory 准入、召回和生命周期管理 |
| `observability` | 事件记录、脱敏、摘要和 Run Trace |
| `extensions` | Python Extension、Skill 和 MCP 能力接入 |
| `runtime` | 能力装配、Run 资源、取消、恢复和应用用例协调 |
| `interfaces` | CLI、Web、RPC 和 DingTalk 展示与输入适配 |
| `evaluation` | 离线 Benchmark、Evidence、Scorer 和报告 |

Interface 不拥有业务状态；Runtime 不重新定义 Core 决策；Core 不直接执行工具；Sessions 不代替 Context 或 Memory 创建第二套领域模型。

## 运行主链路

### 普通 Run

```text
用户输入
  → Runtime 准备 Run
  → Context 物化模型输入
  → Core Driver 调用 LLM
  → Reducer 处理模型观察值
  → Terminal Commit
  → Interface 返回结果
```

### 工具执行与审批

```text
模型产生 ToolCall
  → Registry 校验注册快照
  → Codec 解码参数
  → AccessResolver 解析资源和副作用
  → PermissionEngine 决定放行、拒绝或审批
  → ToolRuntime 执行
  → ToolResult 返回 Core
```

需要审批时，Runtime 保存等待状态和 checkpoint，用户决定后使用一次性恢复句柄继续执行，避免工具被重复调用。

### Context 与 Memory

```text
历史消息和仓库事实
  → Context 工作集投影
  → Memory 召回
  → Token 预算和压力判断
  → 必要时压缩或写入 artifact
  → PreparedModelContext
```

Context 摘要只服务当前任务连续性，不会自动变成长久 Memory。长期 Memory 必须经过独立准入和生命周期管理。

### Checkpoint、恢复与回滚

```text
Core Boundary
  → Sessions 持久化 Run checkpoint
  → 进程中断或等待用户
  → Session 重载并恢复 Tools / Context 状态
  → 校验恢复点
  → Core 继续执行
```

工作区 rollback 与崩溃恢复是两套不同语义：恢复解决“任务从哪里继续”，rollback 解决“如何撤销已经产生的文件副作用”。

## 项目结构

```text
Codepilot/
├── src/codepilot/
│   ├── protocols/       # 跨层协议
│   ├── llm/             # 模型 Provider
│   ├── tools/           # 工具安全与执行
│   ├── core/            # Agent 决策主循环
│   ├── sessions/        # 会话、Context、Memory 与回滚
│   ├── observability/   # 事件与 Trace
│   ├── extensions/      # Extension、Skill、MCP
│   ├── runtime/         # 装配和应用协调
│   ├── interfaces/      # CLI、Web、RPC、DingTalk
│   └── evaluation/      # 离线评测
├── test/                # 自动化测试
├── docs/design/         # 详细设计文档
├── docs/examples/       # 扩展示例
└── web/                 # Web 前端源码
```

## 设计文档

README 只介绍核心架构和主链路。建议按以下顺序阅读详细设计：

1. [Core 设计](docs/design/1core-design.md)
2. [Context 设计](docs/design/2context-design.md)
3. [Memory 设计](docs/design/3memory-design.md)
4. [Tools 设计](docs/design/4tool-design.md)
5. [Evaluation 设计](docs/design/5eval-design.md)
6. [Sessions 设计](docs/design/6sessions-design.md)
7. [Runtime 设计](docs/design/7runtime-design.md)

这些文档分别说明权威状态、边界契约、状态机、持久化位置、失败映射和恢复语义。阅读代码时建议遵循架构依赖方向，而不是逐个目录孤立查看。

## 开发与验证

```bash
# 编译全部 Python 源码
python -m compileall -q src/codepilot

# 运行自动化测试
python -m pytest -q test

# 检查协议和架构边界
python -m pytest -q test/test_protocol_contracts.py
python -m pytest -q test/test_context_memory_architecture.py

# 运行工具与 Core 主链路测试
python -m pytest -q test/test_tool_execution_v2.py
python -m pytest -q test/test_core_driver.py
```

评测系统位于 `src/codepilot/evaluation/`。Benchmark 及其运行产物用于离线验证，不属于在线 Run 主链路。

Web 前端开发：

```bash
cd web
npm ci
npm test -- --run
npm run typecheck
npm run build
cd ..
codepilot web --dev
```

## 当前状态与路线图

已经具备：

- 普通模型 Run、工具调用、审批、取消和等待恢复。
- Session/Run 持久化、Context checkpoint 和工作区 rollback。
- Context 分层预算、长期 Memory、结构化 Plan 和探索型 Subagent。
- CLI、Web、JSONL RPC、DingTalk、Extension、Skill 和 MCP 接入。
- 事件追踪、Run Trace、自动化测试和离线评测框架。

持续迭代方向：

- 扩充真实代码仓库 Benchmark 和长期稳定性验证。
- 加强跨平台工具执行和操作系统级隔离能力。
- 改进模型兼容范围、Context 策略和 Subagent 协作质量。
- 完善 Web 工作台交互、认证和部署方式。

## 贡献

欢迎通过 Issue 或 Pull Request 提交问题、测试、文档和改进建议。修改核心模块前，建议先阅读相应设计文档并保持既有依赖方向和权威状态边界。

## License

MIT License，详见 [LICENSE](LICENSE)。
