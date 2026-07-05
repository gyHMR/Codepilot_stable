# Codepilot

> **面向真实代码仓库的本地 Coding Agent 引擎** —— 以工具安全、任务控制、上下文治理、结构化记忆和证据化评测为工程基石。

![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-0.3.0-orange)

---

## Overview

Codepilot 是一个基于 Python 构建的本地编程智能体（Coding Agent），围绕一条完整的真实开发任务主线展开：**理解仓库 → 调用工具 → 修改代码 → 运行验证 → 记录证据**，并在多轮交互中持续维护上下文、持久化记忆和恢复状态。

本项目并非追求成为庞大的生产级 Agent 平台，而是聚焦于 Coding Agent 领域最关键的工程问题，逐一拆解并清晰实现：

- 🤖 模型如何安全、可控地使用工具
- 📐 长上下文场景下哪些信息应当进入 prompt
- 🔁 任务失败后如何优雅推进或恢复
- 🧠 哪些经验值得长期沉淀与复用
- 📊 这些能力如何被系统性地评测与验证

---

## 目录

- [核心能力](#核心能力)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [配置模型](#配置模型)
  - [运行](#运行)
  - [钉钉远程入口](#钉钉远程入口)
- [运行主线](#运行主线)
- [架构分层](#架构分层)
- [任务规划控制](#任务规划控制)
- [上下文治理](#上下文治理)
- [结构化记忆](#结构化记忆)
- [工具安全](#工具安全)
- [会话恢复与回滚](#会话恢复与回滚)
- [评测体系](#评测体系)
- [常用斜杠命令](#常用斜杠命令)
- [开发验证](#开发验证)
- [设计文档](#设计文档)
- [License](#license)

---

## 核心能力

| 能力 | 说明 |
|---|---|
| **本地代码任务执行** | 支持仓库读取、文件搜索、代码编辑、Shell 验证，返回结构化 `AgentRunResult` |
| **受控工具运行时** | 工具调用统一经过**权限决策 → 参数校验 → 用户审批 → 执行 → 结果防护**全链路 |
| **任务规划控制** | 提供 `read` / `edit` / `plan` 三种模式；复杂任务可先只读 discovery，再生成带验收标准的执行计划 |
| **上下文投影治理** | 每次模型调用前动态组装仓库状态、任务进度、工具证据、记忆召回与最近对话，按 token 压力生成本轮 prompt |
| **结构化长期记忆** | 仅沉淀用户显式规则、修正反馈、项目决策及经失败-修复-验证闭环确认的可复用经验 |
| **会话恢复与回滚** | 持久化 session/run 记录，支持任务中断恢复、会话分支及基于 Git clean worktree 的 run 级回滚 |
| **证据化评测体系** | 基于真实运行 trace、工具调用记录、上下文报告、记忆召回率和文件 diff 计算多维评测指标 |

---

## 快速开始

### 安装

```bash
git clone https://github.com/your-username/Codepilot.git
cd Codepilot
pip install -e ".[dev]"
```

### 配置模型

通过交互式命令初始化配置：

```bash
codepilot config init
```

编辑 `.codepilot/model.local.json`：

```json
{
  "api": "openai-compatible",
  "provider": "deepseek",
  "model_id": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key_env": "DEEPSEEK_API_KEY",
  "context_window": 64000,
  "max_tokens": 8192,
  "reasoning": false,
  "vision": false
}
```

> 💡 **推荐**：使用 `api_key_env` 从环境变量读取密钥，避免 API Key 泄露到项目文件中。

验证配置：

```bash
codepilot config check
```

### 运行

```bash
# 交互式模式（默认）
codepilot

# 单次任务
codepilot -p "解释这个项目的主调用链"

# 指定工作区
codepilot --workspace /path/to/project

# 只读分析模式
codepilot --task-mode read -p "分析当前仓库结构"

# 复杂任务规划模式
codepilot --task-mode plan -p "修复失败测试并说明验证方式"

# 高风险操作走用户审批
codepilot --permission-mode ask
```

### 钉钉远程入口

钉钉接入采用独立脚本启动，不与 `codepilot` CLI parser 复用：

```bash
pip install -e ".[dingtalk]"
export DINGTALK_CLIENT_ID="your-client-id"
export DINGTALK_CLIENT_SECRET="your-client-secret"
codepilot-dingtalk serve --cwd /path/to/project --allowed-user <sender_staff_id>
```

手机端发送 `cp <任务描述>` 即可接入主链路（`RuntimeGateway.dispatch() → SessionController → core.run_agent_loop() → ToolPort`）。
钉钉会话默认强制 `tool_permission_mode="ask"`，写文件、Shell、回滚等操作仍需审批；
当 Git 工作区存在未提交改动时，默认拒绝远程 run，可通过 `--allow-dirty` 显式放行。

---

## 运行主线

一次请求的完整链路如下：

```text
CLI / DingTalk / Eval
  → UserAction
  → RuntimeGateway.dispatch()
  → SessionController.prepare_run()
  → core.run_agent_loop(AgentLoopInput, AgentLoopPorts)
  → ModelPort.stream() / ToolPort.execute()
  → SessionController.commit_run()
  → RuntimeFrame
```

### 关键模块索引

| 阶段 | 代码位置 | 职责 |
|---|---|---|
| 接口入口 | `src/codepilot/interfaces/` | CLI、钉钉远程入口、RPC 适配 |
| 应用门面 | `src/codepilot/runtime/gateway.py` | Session 注册、UserAction dispatch、审批恢复、RuntimeFrame 输出 |
| 运行时装配 | `src/codepilot/runtime/assembly.py` | 解析模型、配置、工具、扩展、Prompt 和 Session Options |
| 会话控制 | `src/codepilot/sessions/controller.py` | prepare_run、prepare_resume、commit_run、命令语义 |
| 会话事实源 | `src/codepilot/sessions/session.py` | 持久化、记忆、上下文、任务恢复、回滚内部状态 |
| Agent 主循环 | `src/codepilot/core/loop.py` | 模型回合、工具回合、停止语义、任务完成检查 |
| 任务控制 | `src/codepilot/core/task_control/` | Task mode、Discovery、Planner、Controller、Completion Gate |
| 工具安全 | `src/codepilot/tools/` | 工具契约、注册、权限、Schema、审批、执行、结果防护 |
| 上下文治理 | `src/codepilot/sessions/context/` | Snapshot、Pressure Policy、Projection、Checkpoint、Artifact Ledger |
| 结构化记忆 | `src/codepilot/sessions/memory/` | Memory Record、准入、召回、经验提取、合并与提升 |
| 观测与评测 | `src/codepilot/observability/`, `src/codepilot/evaluation/` | Trace、报告、Benchmark、指标计算 |

> 📖 更详尽的主线导读请见 [docs/design/0Guide.md](docs/design/0Guide.md)。

---

## 架构分层

```
src/codepilot/
├── protocols/      # 跨层数据协议与类型契约
├── llm/            # LLM Provider 抽象与流式事件适配
├── tools/          # 工具注册、权限、安全执行与内置工具
├── core/           # Agent 循环、模型调用协调、任务控制
├── sessions/       # 会话事实源、持久化、上下文、记忆、历史恢复
├── observability/  # 运行 Trace、事件归一化、审计报告
├── extensions/     # Python 扩展、Markdown Skill、MCP 桥接
├── runtime/        # 配置解析、模型/工具/Prompt/Session 装配、服务门面
├── interfaces/     # CLI 与钉钉远程入口
└── evaluation/     # Benchmark、Runner、Evidence、Scorer、Report
```

### 依赖方向

```
protocols → llm/tools → core → sessions/observability → extensions → runtime → interfaces
```

> `evaluation/` 是横切模块，通过公开的 `RuntimeGateway` 和 `RuntimeFrame` 驱动 Agent 运行，不直接侵入 core/sessions 内部状态。

---

## 任务规划控制

> 设计文档：[docs/design/1task-design.md](docs/design/1task-design.md)

任务控制的目标并非替模型做出语义决策，而是将模型的自由执行约束在**可观察、可恢复、可验证**的边界之内。

### 任务模式

| 模式 | 适用场景 | 行为 |
|---|---|---|
| `read` | 代码分析、解释说明 | 仅暴露只读工具，禁止修改工作区 |
| `edit` | 默认开发任务 | 不进行复杂预规划，但持续追踪工具证据、变更与验证结果 |
| `plan` | 复杂/多步骤任务 | 先执行只读 discovery，再合成结构化执行计划 |

### plan 模式主流程

```
PlanningDiscovery
  → 只读工具收集 facts / relevant_files / risks / verification_hints
  → TaskPlanner 生成 goal + steps
  → TaskController 初始化 TaskState
  → 每次工具结果更新步骤状态、证据、失败次数与下一步决策
  → CompletionGate 判断任务是否可以结束
```

### 关键组件

| 组件 | 职责 |
|---|---|
| `discovery.py` | 只读 Scratch ReAct Loop，探索信息不污染主上下文 |
| `planner.py` | 解析并规范化 LLM 输出的 JSON 计划，失败时降级为安全单步计划 |
| `controller.py` | 根据工具结果、验证状态、审批/拒绝与连续失败次数做确定性决策 |
| `rules.py` | 集中放置完成门控、验证失败摘要、重规划与回滚提示规则 |
| `tools.py` | 定义 `complete_task_step` 协议名和识别逻辑 |

`complete_task_step` 的可执行 `AgentTool` 定义在
`src/codepilot/tools/builtins/task_control.py`。core 只解释工具结果里的
`task_control` metadata，tools 层负责创建和执行工具对象。

> 🚨 代码变更后，若未与最新工作区状态达成一致的成功验证，任务不会被标记为完成。

---

## 上下文治理

> 设计文档：[docs/design/2context-design.md](docs/design/2context-design.md)

Codepilot **不将全部历史消息和工具输出直接塞入 prompt**。每次模型调用前，`ContextGovernor.prepare()` 从 Session 事实源投影出精炼的上下文视图。

### 治理链路

```
SessionSnapshotBuilder
  → RepositoryTracker 刷新仓库快照与 delta
  → SessionContextState 记录 active files、evidence、verification
  → ToolArtifactLedger 归档工具输出
  → MemoryRetriever 召回长期记忆
  → ContextPressurePolicy 判断 normal / tight / critical
  → ContextProjector 组装 prompt
  → ContextReport 记录选择结果与 token 分布
```

### 上下文层次

| 层 | 内容 |
|---|---|
| **Stable Rules** | 稳定规则与项目约束 |
| **Working State** | 当前任务、Checkpoint、活跃文件、变更文件 |
| **Memory Recall** | 召回的 correction / constraint / decision / experience |
| **Evidence** | 新鲜工具证据、验证结果、Artifact 引用与过期提醒 |
| **Recent Turns** | 少量最近对话摘要 |

> 当上下文压力达到 `critical` 时，系统自动创建结构化 Checkpoint；长工具输出写入 Artifact，以摘要+引用形式进入 prompt。

---

## 结构化记忆

> 设计文档：[docs/design/3memory-design.md](docs/design/3memory-design.md)

Memory v2 的边界定义十分严格：**仅保存跨任务可复用的长期知识**，不保存当前任务进度、文件摘要、工具原始日志或单次失败输出。

### 记忆类型

| 类型 | 来源 | 用途 |
|---|---|---|
| `correction` | 用户纠正 | 最高优先级召回，修正 Agent 的错误认知 |
| `constraint` | 用户显式记忆、项目边界 | 长期规则与偏好 |
| `decision` | `/memory add` 等命令 | 项目设计决策 |
| `experience` | 失败→修复→验证闭环 | 可复用的修复经验 |

### 记忆介入时机

1. **Run 开始前** — `MemoryWriter.admit_prompt_memory()` 仅接收明确的长期记忆意图
2. **每次模型调用前** — `MemoryRetriever.recall()` 根据任务文本、活跃路径、动作意图、近期错误与检索模式进行召回
3. **Run 结束后** — `ExperienceExtractor` 从已验证的失败→修复→验证闭环中提炼经验，`MemoryConsolidator` 合并重复经验并提升高频经验

> 当前任务恢复由 `TaskRecoveryStore` 独立维护，与长期记忆系统分离。

---

## 工具安全

> 设计文档：[docs/design/4tool-design.md](docs/design/4tool-design.md)

工具模块是 Codepilot 的**执行安全边界**。模型可以**请求**工具调用，但无权直接执行工具，也无法通过参数为自身授权。

### 工具来源

```
内置工具 → 调用方工具 → Python 扩展工具 → MCP 代理工具
```

装配阶段通过 `assemble_tools()` 合并工具、校验定义、绑定 metadata、过滤 read-only 工具，并创建统一的 `ToolRuntime`。

### 执行安全流水线

```
ToolRegistry 查找
  → PermissionPolicy 权限决策
  → SchemaValidator 参数校验
  → ApprovalProvider 用户审批
  → 真实工具执行
  → ToolResultGuard 结果防护
  → ToolObservation / ToolResultMessage
```

### 安全策略要点

- 拦截 `allow_dangerous`、`bypass_approval`、`ignore_workspace_boundary`、`trusted` 等自授权参数
- Shell 命令按风险分类：`verification` / `mutation` / `high_risk` / `unknown`，对应走放行、审批或拒绝
- 文件工具通过 `WorkspaceSandbox` 执行路径边界校验，防止目录逃逸
- Shell 执行过滤敏感环境变量，控制超时与输出长度上限
- 工具结果经 `ToolResultGuard` 脱敏、检测 prompt injection 并标注输出可信度

> ⚠️ 此处的"沙箱"指工作区路径边界与受控执行策略，而非容器或操作系统级的强隔离。

---

## 会话恢复与回滚

Codepilot 持久化记录的要素包括：Session 消息、事件、Run 结果、Context Ledger、Tool Artifacts 与任务恢复投影。中断后使用同一 Session ID 即可恢复上下文。

### Git 回滚安全策略

- Run 开始前要求 Git 工作区为 clean 状态
- 仅处理该 Run 记录的 `affected_paths`
- 若 Run 结束后相关文件被外部修改，自动回滚将被阻塞
- `.codepilot/` 内部文件不参与回滚

### 回滚命令

```text
/rollback            预览最近一次 run 的回滚计划
/rollback <run_id>   预览指定 run
/rollback apply      执行最近一次 run 的回滚
```

> 实现位置：[src/codepilot/sessions/history/git_rollback.py](src/codepilot/sessions/history/git_rollback.py)

---

## 评测体系

> 设计文档：[docs/design/5eval-design.md](docs/design/5eval-design.md)

Evaluation v2 的核心数据流：

```text
Benchmark 描述任务、预期与指标
  → Runner 通过 RuntimeGateway 真实运行 Agent
  → Evidence 从 run trace 与 workspace diff 提取结构化证据
  → Scorer 仅根据 EvalEvidence 计算指标
```

### 默认 Benchmark 目录

```
benchmarks/evaluation_v2/
├── context/
├── memory/
├── planning/
└── security/
```

### 常用评测命令

```bash
# 确定性检查
python -m codepilot.evaluation check

# 运行全部 v2 benchmark
python -m codepilot.evaluation run all

# 运行单个模块
python -m codepilot.evaluation run context
python -m codepilot.evaluation run memory
python -m codepilot.evaluation run planning
python -m codepilot.evaluation run security

# 消融实验
python -m codepilot.evaluation experiment memory --repeat 3
python -m codepilot.evaluation experiment planning --repeat 3

# 静态 A/B 对比
python -m codepilot.evaluation ab context
python -m codepilot.evaluation ab security

# 查看报告
python -m codepilot.evaluation report .codepilot/evals/<eval_id>
```

### Scorer 覆盖维度

| 维度 | 指标 |
|---|---|
| **task** | 任务通过率 |
| **planning** | 步骤完成率、误完成率、修复/重规划成功率、恢复率、证据覆盖率 |
| **context** | 关键上下文命中率、Token 效率、过期上下文率、噪声率 |
| **memory** | 记忆召回命中率、冗余读取率、失败方案复发率 |
| **tool/security** | 工具成功率、非法调用率、危险调用拦截率、良性调用放行率、拒绝后副作用 |

> 评测产物写入 `.codepilot/evals/<eval_id>`，包含 summary、report、case evidence 与 workspace diff。

---

## 常用斜杠命令

交互式模式下可用命令一览：

| 命令 | 说明 |
|---|---|
| `/help` | 查看帮助信息 |
| `/status` | 查看当前 Session、模型、权限与任务模式 |
| `/tools` | 列出当前可用工具 |
| `/context` | 查看最近一次上下文治理报告 |
| `/context items` | 查看本轮上下文选中的条目 |
| `/context stale` | 查看过期上下文提示 |
| `/memory` | 查看结构化记忆概览 |
| `/memory add <text>` | 添加项目级记忆 |
| `/memory promote <id>` | 将 Session Experience 提升为 Project Memory |
| `/memory forget <id>` | 标记删除某条记忆 |
| `/rollback` | 预览 Run 级回滚计划 |
| `/rollback apply` | 执行 Run 级回滚 |
| `/model` | 查看或切换模型 |
| `/usage` | 查看 Token 用量 |
| `/exit` | 退出交互式模式 |

---

## 开发验证

```bash
# 编译检查
python -m compileall -q src/codepilot

# 全量测试
python -m pytest test -q

# 重点模块专项测试
python -m pytest test/test_task_planning.py -q
python -m pytest test/test_context_governor_refactor.py -q
python -m pytest test/test_memory_v2_contract.py -q
python -m pytest test/test_tool_execution_security.py -q
python -m pytest test/test_evaluation_v2.py -q
```

---

## 设计文档

| 文档 | 内容 |
|---|---|
| [docs/design/0Guide.md](docs/design/0Guide.md) | 运行主线导读 |
| [docs/design/1task-design.md](docs/design/1task-design.md) | 任务规划控制 |
| [docs/design/2context-design.md](docs/design/2context-design.md) | 上下文治理 |
| [docs/design/3memory-design.md](docs/design/3memory-design.md) | 结构化记忆 |
| [docs/design/4tool-design.md](docs/design/4tool-design.md) | 工具安全 |
| [docs/design/5eval-design.md](docs/design/5eval-design.md) | Evaluation v2 |

---

## License

MIT License，详见 [LICENSE](LICENSE)。
