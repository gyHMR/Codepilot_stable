# Codepilot

> 面向真实代码仓库的本地 Coding Agent，聚焦工具安全、任务控制、上下文治理、结构化记忆和证据化评测。

![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-0.3.0-orange)

Codepilot 是一个基于 Python 构建的本地编程智能体。它围绕一次真实开发任务的主线展开：理解仓库、调用工具、修改代码、运行验证、记录证据，并在多轮任务中维护上下文、记忆和恢复状态。

这个项目不是要做一个复杂的生产级 Agent 平台，而是把 Coding Agent 的核心工程问题拆开实现清楚：模型如何安全使用工具，长上下文里该看什么，失败后如何继续推进，哪些经验值得长期记住，以及这些能力如何被评测验证。
## 项目定位

Codepilot 面向学生学习和 AI 应用开发求职展示，优先保证代码清晰、链路可解释、功能可演示。项目保留真实 Coding Agent 所需的主链路.

## 核心能力

- **本地代码任务执行**：支持读取仓库、搜索文件、编辑代码、运行 shell 验证，并返回结构化 `AgentRunResult`。
- **受控工具运行时**：工具调用统一经过权限决策、参数校验、用户审批、执行和结果防护。
- **任务规划控制**：支持 `read / edit / plan` 三种任务模式；复杂任务可先进行只读 discovery，再生成带验收标准的执行步骤。
- **上下文投影治理**：每次模型调用前重新整理仓库状态、任务状态、工具证据、记忆和最近对话，按 token 压力生成本轮 prompt。
- **结构化长期记忆**：只沉淀用户显式规则、纠正、项目决策和经失败-修复-验证闭环确认的经验。
- **会话恢复与回滚**：持久化 session/run 记录，支持任务恢复、会话分支，以及基于 Git clean worktree 的 run 级回滚。
- **证据化评测**：通过真实运行 trace、工具调用、上下文报告、记忆召回和文件 diff 计算评测指标。

## 快速开始

### 安装

```bash
git clone https://github.com/your-username/Codepilot.git
cd Codepilot
pip install -e ".[dev]"
```

### 配置模型

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

建议使用 `api_key_env` 从环境变量读取密钥，避免把 API Key 写入项目文件。

```bash
codepilot config check
```

### 运行

```bash
# 交互式模式
codepilot

# 单次任务
codepilot -p "解释这个项目的主调用链"

# 指定工作区
codepilot --workspace /path/to/project

# 只读分析
codepilot --task-mode read -p "分析当前仓库结构"

# 复杂任务规划模式
codepilot --task-mode plan -p "修复失败测试并说明验证方式"

# 高风险操作走审批
codepilot --permission-mode ask
```

## 运行主线

一次请求大致经过下面这条链路：

```text
CLI/Web/Eval
  -> RuntimeService
  -> assemble_runtime()
  -> AgentSession
  -> Agent.run()
  -> run_agent_loop()
  -> ContextGovernor.prepare()
  -> LLMStreamRunner
  -> ToolCallCoordinator
  -> ToolRuntime
  -> ToolResultMessage
  -> TaskController / ContextGovernor / RunStore / MemoryWriter
```

对应的关键文件：

| 阶段 | 代码位置 | 说明 |
|---|---|---|
| 接口入口 | `src/codepilot/interfaces/` | CLI、Web、RPC 适配 |
| 应用门面 | `src/codepilot/runtime/service.py` | session 管理、消息发送、审批恢复、运行查询 |
| 运行时装配 | `src/codepilot/runtime/assembly.py` | 解析模型、配置、工具、扩展、prompt 和 session options |
| 会话事实源 | `src/codepilot/sessions/session.py` | run 生命周期、持久化、记忆、上下文、任务恢复 |
| Agent 主循环 | `src/codepilot/core/agent_loop.py` | 模型调用、工具循环、停止语义、任务完成检查 |
| 任务控制 | `src/codepilot/core/task_control/` | task mode、discovery、planner、controller、completion gate |
| 工具安全 | `src/codepilot/tools/` | 工具契约、注册、权限、schema、审批、执行、结果防护 |
| 上下文治理 | `src/codepilot/sessions/context/` | snapshot、pressure policy、projection、checkpoint、artifact ledger |
| 结构化记忆 | `src/codepilot/sessions/memory/` | memory record、准入、召回、经验提取、合并和提升 |
| 观测与评测 | `src/codepilot/observability/`, `src/codepilot/evaluation/` | trace、报告、benchmark、指标计算 |

更完整的主线导读见 [docs/design/0Guide.md](docs/design/0Guide.md)。

## 架构分层

```text
src/codepilot/
├── protocols/      # 跨层数据协议和类型契约
├── llm/            # LLM provider 抽象和流式事件适配
├── tools/          # 工具注册、权限、安全执行和内置工具
├── core/           # Agent 循环、模型调用协调、任务控制
├── sessions/       # 会话事实源、持久化、上下文、记忆、历史恢复
├── observability/  # 运行 trace、事件归一化、审计报告
├── extensions/     # Python 扩展、Markdown skill、MCP 桥接
├── runtime/        # 配置解析、模型/工具/prompt/session 装配、服务门面
├── interfaces/     # CLI 和 Web 接口
└── evaluation/     # Benchmark、runner、evidence、scorer、report
```

依赖方向保持为：

```text
protocols -> llm/tools -> core -> sessions/observability -> extensions -> runtime -> interfaces
```

`evaluation/` 是横切模块，通过公开的 `RuntimeService` 运行 Agent，不直接修改 core/sessions 内部状态。

## 任务规划控制

设计文档：[docs/design/1task-design.md](docs/design/1task-design.md)

任务控制的目标不是替模型做语义决策，而是把模型的自由执行约束在可观察、可恢复、可验证的边界内。

当前用户可见任务模式：

| 模式 | 适用场景 | 行为 |
|---|---|---|
| `read` | 只读分析、解释代码 | 只暴露只读能力，不允许修改工作区 |
| `edit` | 默认开发任务 | 不做复杂预规划，但仍跟踪工具证据、变更和验证 |
| `plan` | 复杂或多步骤任务 | 先进行只读 discovery，再合成结构化执行计划 |

`plan` 模式的主流程：

```text
PlanningDiscovery
  -> 只读工具收集 facts / relevant_files / risks / verification_hints
  -> TaskPlanner 生成 goal + steps
  -> TaskController 初始化 TaskState
  -> 每次工具结果更新步骤、证据、失败次数和下一步决策
  -> CompletionGate 判断是否允许结束
```

关键实现：

- `discovery.py`：只读 scratch ReAct loop，不把探索消息污染主上下文。
- `planner.py`：解析和规范化 LLM JSON 计划，失败时降级为安全单步计划。
- `controller.py`：根据工具结果、验证状态、审批/拒绝和连续失败做确定性决策。
- `rules.py`：集中放置完成门控、验证失败摘要、重规划和回滚提示规则。
- `tools.py`：提供 runtime-managed `complete_task_step` 工具，允许模型显式完成非修改步骤。

代码变更后，如果没有与最新工作区状态一致的成功验证，任务不会被正常标记为完成。

## 上下文治理

设计文档：[docs/design/2context-design.md](docs/design/2context-design.md)

Codepilot 不把全部历史消息和工具输出直接塞进 prompt。每次模型调用前，`ContextGovernor.prepare()` 会从 session 事实源投影出一个新的上下文视图。

治理链路：

```text
SessionSnapshotBuilder
  -> RepositoryTracker 刷新仓库快照和 delta
  -> SessionContextState 记录 active files、evidence、verification
  -> ToolArtifactLedger 归档工具输出
  -> MemoryRetriever 召回长期记忆
  -> ContextPressurePolicy 判断 normal / tight / critical
  -> ContextProjector 组装 prompt
  -> ContextReport 记录选择结果和 token 分布
```

最终进入模型的内容按层组织：

- `Stable Rules`：稳定规则和项目约束。
- `Working State`：当前任务、checkpoint、active files、changed files。
- `Memory Recall`：召回的 correction、constraint、decision、experience。
- `Evidence`：新鲜工具证据、验证结果、artifact 引用和 stale 提醒。
- `Recent Turns`：少量最近对话摘要。

当上下文压力达到 `critical` 时，系统会创建结构化 checkpoint；长工具输出会写入 artifact，再以摘要和引用进入 prompt。

## 结构化记忆

设计文档：[docs/design/3memory-design.md](docs/design/3memory-design.md)

Memory v2 的边界很严格：只保存跨任务可复用的长期知识，不保存当前任务进度、文件摘要、工具原始日志或单次失败输出。

自动写入的 `MemoryRecord` 类型：

| 类型 | 来源 | 用途 |
|---|---|---|
| `correction` | 用户纠正 | 最高优先级召回，修正 Agent 的错误认知 |
| `constraint` | 用户显式记忆、项目边界 | 长期规则和偏好 |
| `decision` | `/memory add` 等命令 | 项目设计决策 |
| `experience` | 失败-修复-验证闭环 | 可复用修复经验 |

记忆参与三个时机：

1. **run 开始前**：`MemoryWriter.admit_prompt_memory()` 只接收明确的长期记忆意图。
2. **每次模型调用前**：`MemoryRetriever.recall()` 根据任务文本、active paths、action intent、recent error 和 retrieval mode 召回。
3. **run 结束后**：`ExperienceExtractor` 只从已验证的失败-修复-验证闭环中提炼经验，并通过 `MemoryConsolidator` 合并重复经验、提升高频经验。

当前任务恢复由 `TaskRecoveryStore` 维护，和长期 Memory 分离。

## 工具安全

设计文档：[docs/design/4tool-design.md](docs/design/4tool-design.md)

工具模块是 Codepilot 的执行安全边界。模型只能请求工具调用，不能直接执行工具，也不能通过参数给自己授权。

工具来源：

```text
内置工具 -> 调用方工具 -> Python 扩展工具 -> MCP 代理工具
```

装配阶段由 `assemble_tools()` 合并工具、校验定义、绑定 metadata、过滤 read-only 工具，并创建统一 `ToolRuntime`。

`ToolRuntime.execute()` 的执行顺序：

```text
ToolRegistry 查找
  -> PermissionPolicy 权限决策
  -> SchemaValidator 参数校验
  -> ApprovalProvider 用户审批
  -> 真实工具执行
  -> ToolResultGuard 结果防护
  -> ToolRuntimeResult / ToolResultMessage
```

主要安全策略：

- 拦截 `allow_dangerous`、`bypass_approval`、`ignore_workspace_boundary`、`trusted` 等自授权参数。
- shell 命令分类为 `verification / mutation / high_risk / unknown`，不同类别走允许、审批或拒绝。
- 文件工具通过 `WorkspaceSandbox` 做工作区边界校验，防止路径逃逸。
- shell 执行过滤敏感环境变量，控制超时和输出长度。
- 工具结果经过 `ToolResultGuard` 脱敏、prompt injection 检测和输出可信度标注。

注意：这里的“沙箱”是工作区路径边界和受控执行策略，不是容器或操作系统级强隔离。

## 会话恢复与回滚

Codepilot 会持久化 session 消息、事件、run 结果、context ledger、tool artifacts 和任务恢复投影。中断后可以用同一个 session id 恢复上下文。

Git 回滚采用一个很小但可解释的安全子集：

- run 开始前必须是 Git clean worktree；
- 只处理该 run 记录的 `affected_paths`；
- 如果 run 结束后相关文件又被修改，自动回滚会被阻塞；
- `.codepilot/` 内部文件不会参与回滚。

常用命令：

```text
/rollback            预览最近一次 run 的回滚计划
/rollback <run_id>   预览指定 run
/rollback apply      执行最近一次 run 的回滚
```

实现位置：[src/codepilot/sessions/history/git_rollback.py](src/codepilot/sessions/history/git_rollback.py)。

## 评测体系

设计文档：[docs/design/5eval-design.md](docs/design/5eval-design.md)

Evaluation v2 的核心边界是：

```text
Benchmark 描述任务、预期和指标
Runner 通过 RuntimeService 真实运行 Agent
Evidence 从 run trace 和 workspace diff 提取结构化证据
Scorer 只根据 EvalEvidence 计算指标
```

默认 benchmark 目录：

```text
benchmarks/evaluation_v2/
├── context/
├── memory/
├── planning/
└── security/
```

常用命令：

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

# 静态 A/B
python -m codepilot.evaluation ab context
python -m codepilot.evaluation ab security

# 查看报告
python -m codepilot.evaluation report .codepilot/evals/<eval_id>
```

当前 scorer 覆盖：

- task：任务通过率。
- planning：步骤完成率、误完成率、修复/重规划成功率、恢复率、证据覆盖率等。
- context：关键上下文命中率、token efficiency、过期上下文率、噪声率。
- memory：记忆召回命中率、冗余读取、失败方案复发等。
- tool/security：工具成功率、非法调用率、危险调用拦截、良性调用放行、拒绝后副作用。

评测产物会写入 `.codepilot/evals/<eval_id>`，包含 summary、report、case evidence 和 workspace diff。

## 常用斜杠命令

交互式模式下可使用：

| 命令 | 说明 |
|---|---|
| `/help` | 查看帮助 |
| `/status` | 查看当前 session、模型、权限和任务模式 |
| `/tools` | 查看当前可用工具 |
| `/context` | 查看最近一次上下文治理报告 |
| `/context items` | 查看本轮上下文选择了哪些条目 |
| `/context stale` | 查看过期上下文提示 |
| `/memory` | 查看结构化记忆概览 |
| `/memory add <text>` | 添加项目级记忆 |
| `/memory promote <id>` | 将 session experience 提升为 project memory |
| `/memory forget <id>` | 标记删除某条记忆 |
| `/rollback` | 预览 run 级回滚 |
| `/rollback apply` | 执行 run 级回滚 |
| `/model` | 查看或切换模型 |
| `/usage` | 查看 token 用量 |
| `/exit` | 退出 |

## 开发验证

```bash
# 编译检查
python -m compileall -q src/codepilot

# 全量测试
python -m pytest test -q

# 重点模块测试
python -m pytest test/test_task_planning.py -q
python -m pytest test/test_context_governor_refactor.py -q
python -m pytest test/test_memory_v2_contract.py -q
python -m pytest test/test_tool_execution_security.py -q
python -m pytest test/test_evaluation_v2.py -q
```

## 设计文档

| 文档 | 内容 |
|---|---|
| [docs/design/0Guide.md](docs/design/0Guide.md) | 运行主线导读 |
| [docs/design/1task-design.md](docs/design/1task-design.md) | 任务规划控制 |
| [docs/design/2context-design.md](docs/design/2context-design.md) | 上下文治理 |
| [docs/design/3memory-design.md](docs/design/3memory-design.md) | 结构化记忆 |
| [docs/design/4tool-design.md](docs/design/4tool-design.md) | 工具安全 |
| [docs/design/5eval-design.md](docs/design/5eval-design.md) | Evaluation v2 |



## License

MIT License，详见 [LICENSE](LICENSE)。
