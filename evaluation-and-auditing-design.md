# Pico 评测框架与审计体系 — 业务功能设计详解

> 本文档只描述"系统做了什么、为什么这样做、设计上有哪些考量"，不涉及代码实现细节。

---

## 总览：评测与审计的关系

Pico 的质量保障体系分为两个层次：

```
┌─────────────────────────────────────────────────┐
│  审计层（Auditing）                              │
│  回答："这一次运行到底发生了什么？"                │
│  产物：trace.jsonl / task_state.json / report.json│
│  特点：每次运行自动生成，事后可查                  │
├─────────────────────────────────────────────────┤
│  评测层（Evaluation）                            │
│  回答："agent 的能力到底怎么样？哪强哪弱？"        │
│  产物：benchmark artifacts / experiment reports   │
│  特点：主动运行，对比预期，量化指标                │
└─────────────────────────────────────────────────┘
```

审计是**被动的**——只要 agent 在运行，就自动产生审计数据。评测是**主动的**——需要人为触发，跑一组预设任务，对比预期结果。

---

## 第一部分：审计体系

### 1.1 审计的核心设计哲学

Pico 的审计遵循三个原则：

**① 每次运行都有独立证据**

每次用户调用 `ask()`，都会在 `.pico/runs/` 下创建一个以 run_id 命名的目录。一次用户请求 = 一组独立工件。不会出现多次运行的数据混在同一个文件里的情况。

**② 过程和结果分开记录**

- **trace.jsonl** 记录过程——逐事件的时间线，回答"每一步做了什么"
- **task_state.json** 记录状态——当前进行到哪了，回答"现在是什么状态"
- **report.json** 记录结果——最终摘要和关键指标，回答"最终怎么样了"

三者互补：trace 适合调试，task_state 适合实时监控，report 适合汇总分析。

**③ 写入即落盘，不靠内存**

所有审计数据在产生的瞬间就写入磁盘，不是"运行结束后一次性写入"。这意味着即使 agent 在运行中途崩溃，已经产生的审计数据也不会丢失。

---

### 1.2 三层审计工件详解

#### 1.2.1 task_state.json — 状态机快照

task_state 回答的问题是："这次运行进行到哪了？"

它是一个持续更新的快照，被反复覆盖写入（不是追加），每次工具执行后、每次状态变化后都会更新。

记录的核心字段：

| 字段 | 含义 | 为什么需要它 |
|------|------|------------|
| run_id | 本次运行的唯一 ID | 关联到对应的 run 目录 |
| task_id | 本次任务的唯一 ID | 区分不同的用户请求 |
| user_request | 用户的原始请求 | 知道 agent 在做什么任务 |
| status | running / completed / stopped / failed | 快速判断运行结果 |
| tool_steps | 已执行的工具调用次数 | 判断是否在预算内 |
| attempts | 模型已被调用的轮数 | 区分"做了很多事"和"反复重试" |
| last_tool | 最后一次调用的工具名 | 快速定位最后一步做了什么 |
| stop_reason | 停机原因 | 理解为什么停了 |
| final_answer | 最终答案文本 | agent 最终说了什么 |
| checkpoint_id | 关联的检查点 ID | 恢复时的锚点 |
| resume_status | 恢复状态 | 从什么状态恢复的 |

**status 的四种状态**：

```
running ──→ completed（正常结束，返回了最终答案）
   │
   ├──→ stopped（步数耗尽或重试次数耗尽）
   │
   └──→ failed（模型调用出错）
```

**stop_reason 的八种原因**：

| stop_reason | 含义 |
|-------------|------|
| final_answer_returned | 正常结束，模型返回了最终答案 |
| step_limit_reached | 工具调用次数达到上限 |
| retry_limit_reached | 模型格式错误重试次数达到上限 |
| model_error | 模型 API 调用失败 |
| tool_timeout | 工具执行超时 |
| approval_denied | 用户拒绝了工具审批 |
| delegate_failed | 子 agent 委派失败 |
| persistence_error | 持久化写入失败 |

#### 1.2.2 trace.jsonl — 事件时间线

trace 回答的问题是："这次运行的每一步具体发生了什么？"

它是一个 JSONL 文件（每行一个 JSON 对象），采用追加写入模式。事件按时间顺序排列，构成一条完整的时间线。

**事件类型及其产生时机**：

| 事件 | 时机 | 记录的关键信息 |
|------|------|---------------|
| `run_started` | 一轮 ask() 开始 | task_id、用户请求（截断到 300 字符） |
| `prompt_built` | prompt 组装完成 | 每个 section 的原始/裁剪字符数、预算压缩日志、缓存状态、耗时 |
| `checkpoint_created` | 检查点创建 | checkpoint_id、触发类型 |
| `model_requested` | 模型调用前 | 当前 attempts 和 tool_steps、prompt_cache_key |
| `model_parsed` | 模型输出解析后 | 解析结果类型（tool/final/retry）、token 用量、缓存命中、耗时 |
| `tool_executed` | 工具执行后 | 工具名、参数、结果（截断到 500 字符）、耗时、状态、受影响文件、diff 摘要 |
| `runtime_identity_mismatch` | 运行时身份不匹配 | 不匹配的字段列表 |
| `run_finished` | 一轮 ask() 结束 | 最终状态、停机原因、最终答案、总耗时 |

**每个事件都包含的元数据**：
- `event`：事件类型
- `created_at`：事件发生的 UTC 时间戳

**tool_executed 事件的特殊价值**：

这是最丰富的事件类型，它不仅记录了工具本身的信息，还记录了执行后的副作用：

| 字段 | 含义 |
|------|------|
| tool_status | ok / partial_success / error / rejected |
| tool_error_code | 具体错误码（如 path_escape、tool_not_allowed、repeated_identical_call） |
| security_event_type | 安全事件类型（如 path_escape、read_only_block） |
| risk_level | high / low |
| read_only | 是否为只读操作 |
| affected_paths | 被修改的文件列表 |
| workspace_changed | 工作区是否发生了变化 |
| diff_summary | 变更摘要（如 "modified:src/main.py"） |
| duration_ms | 执行耗时（毫秒） |

**prompt_built 事件的特殊价值**：

它记录了 prompt 的"组装过程"，回答一个关键问题："模型为什么没有看到某个信息？"

| 字段 | 含义 |
|------|------|
| sections.prefix.raw_chars | prefix 原始长度 |
| sections.prefix.rendered_chars | prefix 裁剪后长度 |
| sections.memory.raw_chars | memory 原始长度 |
| sections.history.raw_chars | history 原始长度 |
| sections.history.rendered_chars | history 裁剪后长度 |
| budget_reductions | 触发了哪些压缩（哪个 section 被压缩了多少） |
| relevant_memory.selected_count | 召回了多少条相关笔记 |
| relevant_memory.selected_notes | 召回了哪些笔记的文本 |
| prefix_changed | prefix 是否发生了变化 |
| workspace_changed | 工作区是否发生了变化 |

#### 1.2.3 report.json — 运行报告

report 回答的问题是："这次运行的最终结果和关键指标是什么？"

它在运行结束后一次性写入，是对整次运行的结构化总结。

**report 包含的顶层信息**：

| 字段 | 含义 |
|------|------|
| run_id / task_id | 标识 |
| status | 最终状态 |
| stop_reason | 停机原因 |
| final_answer | 最终答案 |
| tool_steps / attempts | 性能指标 |
| checkpoint_id | 关联的检查点 |
| resume_status | 恢复状态 |
| task_state | 完整的 task_state 快照 |
| prompt_metadata | 本轮 prompt 的完整组装元数据 |
| durable_promotions | 本轮被晋升为持久记忆的条目 |
| durable_rejections | 本轮被拒绝晋升的条目（含拒绝原因） |
| durable_superseded | 本轮被替换的旧持久记忆条目 |
| redacted_env | 检测到的密钥变量数量和名称 |

**durable_promotions 和 durable_rejections 的审计价值**：

这两个字段记录了记忆系统的"入账/拒账"决策。通过它们可以回答：
- 用户要求"记住"的内容，哪些被接受了？哪些被拒绝了？
- 被拒绝的原因是什么？（密钥形状、临时任务状态、噪音输出）
- 哪些旧记忆被新记忆替换了？

---

### 1.3 审计数据的安全设计

所有审计数据在写入磁盘前都会经过脱敏处理：

**脱敏范围**：
- trace.jsonl 的每个事件
- report.json 的所有字段
- 任何包含用户配置的密钥变量名的值

**脱敏方式**：
- 遍历密钥变量名清单（内置默认 + 用户追加 + .env 声明）
- 将这些变量的实际值替换为 `<redacted>`
- 递归处理嵌套的字典、列表、元组
- 如果字典的 key 本身就是敏感变量名（如 `api_key`），整个 value 被替换

**脱敏时机**：
- 不是"写入前临时脱敏"，而是"在内存中就脱敏后再传递给写入函数"
- `emit_trace()` 内部先调用 `redact_artifact()` 再写入
- `build_report()` 返回的是已经脱敏后的数据

**设计意图**：密钥脱敏不是事后补救，而是审计数据产生时的内建行为。即使 agent 在工具执行过程中读到了包含密钥的输出，写入 trace 时也会被自动清除。

---

### 1.4 工作区变更审计

每次对 risky 工具（run_shell、write_file、patch_file）的执行，都会触发工作区变更审计：

```
执行前：拍一份工作区所有文件的 SHA256 快照
  ↓
执行工具
  ↓
执行后再拍一份快照
  ↓
对比两份快照，生成：
  - affected_paths：哪些文件被修改了
  - diff_summary：每个文件是 created / modified / deleted
  - workspace_changed：布尔值
```

**设计意图**：用户可以通过 trace 知道 agent 到底改了哪些文件，而不需要自己去 git diff。这在 agent 执行了多个工具调用后特别有用。

---

### 1.5 审计与 Session 的分离

审计数据（run 目录）和会话数据（session 文件）是分开存储的：

```
.pico/
├── sessions/           ← 会话数据（可恢复的状态）
│   └── 20260419-*.json
└── runs/               ← 审计数据（不可变的证据）
    ├── run_20260419-001/
    │   ├── task_state.json
    │   ├── trace.jsonl
    │   └── report.json
    └── run_20260419-002/
        ├── task_state.json
        ├── trace.jsonl
        └── report.json
```

**为什么要分开**：
- Session 是"活的"——会被反复读写、更新、恢复
- Run 是"死的"——一旦写入就不再修改，是历史证据
- 混在一起会导致"恢复状态"和"审计证据"互相干扰

---

## 第二部分：评测框架

### 2.1 评测框架的整体架构

Pico 的评测框架包含四大类实验，每类回答一个不同的问题：

```
┌──────────────────────────────────────────────────────┐
│  Harness Regression（基准回归）                       │
│  问：runtime 的基本合同是否稳定？                      │
├──────────────────────────────────────────────────────┤
│  Ablation Experiments（消融实验）                     │
│  问：每个模块的贡献是什么？去掉它会怎样？               │
│  ├── Context Ablation（上下文压缩消融）               │
│  ├── Memory Ablation（记忆系统消融）                  │
│  └── Recovery Ablation（恢复机制消融）                │
├──────────────────────────────────────────────────────┤
│  Security Experiments（安全实验）                     │
│  问：安全护栏是否真的能挡住攻击？                      │
├──────────────────────────────────────────────────────┤
│  Provider Experiments（跨模型实验）                   │
│  问：不同模型后端的表现差异有多大？                    │
└──────────────────────────────────────────────────────┘
```

---

### 2.2 Harness Regression — 基准回归测试

#### 2.2.1 设计理念

基准回归测试回答一个最基本的问题：**"runtime 的核心合同是否稳定？"**

它不是在测模型有多聪明，而是在测 agent 平台本身是否可靠。即使换了模型、改了配置，这些基础合同也应该始终成立。

#### 2.2.2 任务定义

每个基准任务包含以下要素：

| 要素 | 含义 | 示例 |
|------|------|------|
| id | 任务唯一标识 | readme_intro_locked |
| prompt | 给 agent 的指令 | "In the README fixture, replace the placeholder opening sentence..." |
| fixture_repo | 测试用的仓库副本 | tests/fixtures/bench_repo_readme |
| allowed_tools | 允许使用的工具 | ["read_file", "patch_file"] |
| step_budget | 最大工具调用次数 | 4 |
| expected_artifact | 预期产物描述 | "README.md opening sentence is locked benchmark workspace text" |
| verifier | 验证脚本（shell 命令） | python3 -c "assert '...' in Path('README.md').read_text()" |
| category | 任务分类 | documentation / text-edit / tool-boundary / recovery / durable-contract |
| setup | 可选的前置条件 | 模拟上下文压缩、freshness 不匹配等 |

#### 2.2.3 任务分类与覆盖维度

**documentation（文档编辑）**：
- 验证 agent 能正确读取文件、理解内容、执行精确替换
- 最基础的能力测试

**text-edit（文本编辑）**：
- 验证 agent 能在多段文本中定位并替换特定内容
- 测试 patch_file 的精确性

**tool-boundary（工具边界）**：
- 验证 agent 在遇到工具错误后能否恢复并继续完成任务
- 三种子场景：
  - invalid_patch_recovery：patch 参数格式错误后恢复
  - path_escape_recovery：路径逃逸被拒绝后恢复
  - repeated_read_recovery：重复读取被拒绝后恢复

**recovery（恢复机制）**：
- 验证检查点和恢复系统是否正常工作
- 三种子场景：
  - context_reduction_checkpoint：上下文压缩触发检查点
  - freshness_reanchor_resume：文件 freshness 不匹配时的恢复
  - workspace_mismatch_resume：工作区身份不匹配时的恢复

**durable-contract（持久记忆合同）**：
- 验证持久记忆的晋升和拒绝是否符合预期
- 两种子场景：
  - durable_promotion_accept：合法条目被正确晋升
  - durable_promotion_reject：密钥形状和临时状态被正确拒绝

#### 2.2.4 通过条件

一个任务要算"通过"，必须同时满足四个条件：

```
① within_budget     — 工具调用次数没有超过 step_budget
② verifier_passed   — 验证脚本执行成功（exit code = 0）
③ artifact_exists   — 预期产物文件确实存在
④ non_failure_stop  — 停机原因是 normal（final_answer_returned）
```

**四个条件缺一不可**。比如：即使验证脚本通过了，但如果 agent 超出了步数预算，也算失败。这确保了 agent 不仅要"做对"，还要"高效地做对"。

#### 2.2.5 失败分类

当任务失败时，会被归入以下失败类别之一：

| 失败类别 | 含义 |
|----------|------|
| missing_artifact | 预期产物文件不存在——agent 根本没完成任务 |
| budget_exceeded | 步数超限——agent 做对了但效率太低 |
| verifier_failed | 验证脚本失败——agent 做了事但结果不对 |
| failure_stop_reason | 异常停机——agent 因为错误而非正常完成而停止 |
| unknown | 以上都不是 |

#### 2.2.6 可复现性设计

每次基准测试都会记录以下可复现性元数据：

| 字段 | 含义 |
|------|------|
| runtime.commit_sha | 运行时的 git commit |
| runtime.branch | 运行时的 git branch |
| benchmark.source | 基准任务文件的相对路径 |
| benchmark.task_count | 任务总数 |
| fixture_snapshot_id | 所有 fixture 仓库的 SHA256 联合哈希 |
| model_name | 模型名称 |
| model_version | 模型版本 |
| decoding.temperature | 采样温度 |
| decoding.top_p | top-p 值 |
| decoding.max_new_tokens | 最大输出 token |
| timezone | 时区 |
| locale | 系统 locale |

**fixture_snapshot_id 的作用**：如果 fixture 文件被修改了（比如有人改了测试用的 README），这个哈希值会变，从而提醒"这次回归测试的结果和上次不可比"。

#### 2.2.7 测试隔离

每个任务都在独立的临时目录中运行：
1. 从 fixture_repo 复制一份完整副本到临时目录
2. 在副本上构建 agent
3. 执行任务
4. 验证结果
5. 临时目录在测试结束后自动清理

**设计意图**：任务之间完全隔离，不会互相干扰。同一个任务多次运行也保证结果一致（因为用的是确定性的脚本化模型输出）。

---

### 2.3 Ablation Experiments — 消融实验

消融实验的核心思想是：**关掉某个模块，观察指标变化，从而量化该模块的贡献**。

#### 2.3.1 Context Ablation（上下文压缩消融）

**实验问题**：上下文压缩机制对 prompt 大小的影响有多大？压缩后是否丢失了关键信息？

**实验设计**：

在多种配置组合下，分别测量"开启压缩"和"关闭压缩"时的 prompt 大小：

| 维度 | 取值 |
|------|------|
| 历史长度 | short（4 条）/ medium（12 条）/ long（24 条） |
| 笔记数量 | low（2 条）/ high（10 条） |
| 请求长度 | short（1 个词）/ long（长句） |

总共 3 × 2 × 2 = 12 种配置，每种重复多次。

**测量指标**：

| 指标 | 含义 |
|------|------|
| avg_full_prompt_chars | 开启压缩后的平均 prompt 长度 |
| avg_raw_prompt_chars | 关闭压缩后的平均 prompt 长度 |
| avg_prompt_compression_ratio | 平均压缩率 = (raw - full) / raw |
| max_prompt_compression_ratio | 最大压缩率 |
| min_prompt_compression_ratio | 最小压缩率 |
| current_request_preserved_rate | 用户请求被完整保留的比例（应为 100%） |

**current_request_preserved_rate 的意义**：这是压缩机制的安全底线——无论如何压缩，用户当前的请求必须被完整保留。如果这个值不是 100%，说明压缩算法有 bug。

#### 2.3.2 Memory Ablation（记忆系统消融）

**实验问题**：记忆系统是否真的减少了 agent 的重复工作？

**实验设计**：

用一个脚本化的模型客户端，模拟以下场景：
1. **Bootstrap 阶段**：agent 读取一个文件，记住其中的关键信息
2. **提问阶段**：问 agent 一个需要那个信息才能回答的问题

在三种条件下分别测试：

| 条件 | 设置 | 预期行为 |
|------|------|---------|
| memory_on | 记忆系统正常工作 | agent 从记忆中召回信息，不需要重新读文件 |
| memory_off | 记忆系统关闭 | agent 必须重新读文件才能回答 |
| memory_irrelevant | 记忆中有无关信息 | agent 应该忽略无关记忆，重新读文件 |

**测试的任务类别**：

| 类别 | 测试内容 | 示例 |
|------|---------|------|
| fact_lookup | 能否回忆起文件中的事实 | "deploy key is red" |
| edit_dependency | 能否在不重读文件的情况下继续编辑 | 基于之前读过的约束做修改 |
| history_reference | 能否回忆起之前对话中的结论 | "我们之前从 facts.txt 得出了什么结论？" |

**大规模实验**：12 个任务 × 3 种条件 × 多次重复，覆盖多种事实类型和回忆场景。

**测量指标**：

| 指标 | 含义 |
|------|------|
| correct_rate | 回答正确的比例 |
| repeated_reads | 需要重复读文件的次数（越少越好） |
| avg_tool_steps | 平均工具调用次数 |
| memory_hit_rate | 记忆命中率（不需要重读文件就回答正确的比例） |

**核心结论的验证方式**：

```
memory_on 的 repeated_reads 应该显著低于 memory_off
memory_on 的 correct_rate 应该不低于 memory_off
memory_irrelevant 的 repeated_reads 应该接近 memory_off（无关记忆不应干扰）
```

#### 2.3.3 Recovery Ablation（恢复机制消融）

**实验问题**：检查点恢复机制是否真的能在各种异常情况下正确工作？

**实验设计**：

构造 10 种恢复场景，每种场景在"恢复开启"和"恢复关闭"两种条件下分别测试：

**检查点恢复类**：
- checkpoint_resume_goal：从检查点恢复时，能看到当前目标
- checkpoint_resume_files：从检查点恢复时，能看到关键文件列表

**Freshness 不匹配类**：
- partial_stale_single：单个文件的 freshness 变化被检测到
- partial_stale_multi：多个文件的 freshness 变化被检测到

**工作区身份不匹配类**：
- workspace_mismatch_fingerprint：工作区指纹变化被检测到
- workspace_mismatch_runtime：运行时配置变化被检测到

**Schema 不匹配类**：
- schema_mismatch_version：旧版本 schema 被正确识别
- schema_mismatch_missing：完全没有检查点时的处理

**部分成功恢复类**：
- partial_success_shell：shell 命令部分成功后的恢复指引
- partial_success_tool：工具执行失败后的恢复指引

**验证方式**：

用一个脚本化模型客户端，它会检查 prompt 中是否包含预期的恢复信息。如果 prompt 中包含了所有必需的片段（如 "resume status: partial-stale"、"stale paths: sample.txt"），模型返回成功；否则返回失败。

**测量指标**：

| 指标 | 含义 |
|------|------|
| resume_success_rate | 恢复成功率 |
| stale_reanchor_rate | freshness 不匹配时正确重新锚定的比例 |
| workspace_drift_detection_rate | 工作区漂移被检测到的比例 |
| resume_false_accept_rate | **误接受率**——本应检测到异常但被错误标记为 full-valid 的比例 |

**resume_false_accept_rate 是安全底线**：这个值必须为 0。如果一个过期的检查点被错误地当作有效检查点接受，agent 可能基于过期信息做出错误决策。

---

### 2.4 Security Experiments — 安全实验

#### 2.4.1 设计理念

安全实验回答的问题是：**"安全护栏是否真的能挡住攻击？"**

这里的"攻击"不是恶意攻击，而是 agent 可能遇到的各种"不应该被允许的操作"。安全实验验证的是：当模型试图做这些操作时，runtime 是否能正确拦截。

#### 2.4.2 安全场景清单

| 场景 | 测试内容 | 预期行为 |
|------|---------|---------|
| path_escape_read | 用 `../` 试图读取 workspace 外的文件 | 拒绝，返回 "path escapes workspace" |
| symlink_escape | 通过符号链接试图读取 workspace 外的文件 | 拒绝，符号链接解析后检测到逃逸 |
| search_escape | 用 `../` 试图在 workspace 外搜索 | 拒绝，路径解析后检测到逃逸 |
| approval_denied_shell | 审批策略为 never 时试图执行 shell | 拒绝，返回 "approval denied" |
| read_only_write | 只读模式下试图写文件 | 拒绝，返回 "approval denied" |
| read_only_patch | 只读模式下试图 patch 文件 | 拒绝，返回 "approval denied" |
| repeated_identical_call | 连续三次完全相同的工具调用 | 第三次被拒绝，返回 "repeated identical call" |
| patch_nonunique | patch 的 old_text 在文件中出现多次 | 拒绝，返回 "old_text must occur exactly once" |
| patch_missing_new_text | patch 缺少 new_text 参数 | 拒绝，返回 "missing new_text" |
| timeout_out_of_range | shell 命令的 timeout 超出 [1, 120] 范围 | 拒绝，返回 "timeout must be in [1, 120]" |
| empty_delegate_task | delegate 的 task 为空 | 拒绝，返回 "task must not be empty" |

#### 2.4.3 两种测试模式

**合成模式（Synthetic）**：
- 用 FakeModelClient 直接构造工具调用
- 不经过模型推理，直接测试 runtime 的拦截逻辑
- 快速、确定性强

**真实模式（Real）**：
- 用真实的模型 API（GPT / Claude / DeepSeek）
- 通过 prompt 引导模型发出特定的工具调用
- 测试的是"模型是否会尝试危险操作"以及"runtime 是否能拦截"
- 更接近真实场景，但结果可能因模型行为变化而波动

#### 2.4.4 测量指标

| 指标 | 含义 |
|------|------|
| security_event_counts | 各类安全事件的触发次数 |
| tool_error_code_counts | 各类工具错误码的出现次数 |

**关键安全事件类型**：
- `path_escape`：路径逃逸被检测到
- `read_only_block`：只读模式下的写操作被阻止
- `approval_denied`：审批被拒绝

**关键工具错误码**：
- `tool_not_allowed`：工具不在白名单中
- `unknown_tool`：工具不存在
- `invalid_arguments`：参数校验失败
- `repeated_identical_call`：重复调用被拦截
- `approval_denied`：审批被拒绝

---

### 2.5 Provider Experiments — 跨模型实验

#### 2.5.1 设计理念

跨模型实验回答的问题是：**"同一套 agent 平台，在不同模型后端上的表现差异有多大？"**

这不涉及修改 agent 代码，只是换一个模型来跑同一组基准任务。

#### 2.5.2 实验设计

对 GPT、Claude、DeepSeek 三个 provider 分别运行完整的基准回归测试，然后对比结果。

每个 provider 的配置从环境变量中读取，如果某个 provider 的 API key 缺失，该 provider 会被标记为 "blocked" 并跳过。

#### 2.5.3 测量指标

| 指标 | 含义 |
|------|------|
| pass_rate | 基准任务通过率 |
| avg_tool_steps | 平均工具调用次数 |
| avg_attempts | 平均模型调用轮数 |
| cache_hit_rate | prompt cache 命中率 |
| avg_cached_tokens | 平均缓存 token 数 |

---

### 2.6 运行产物聚合

#### 2.6.1 Run Artifacts Aggregation

除了基准测试，评测框架还会聚合所有历史运行的审计数据，生成全局统计：

| 指标 | 含义 |
|------|------|
| run_count | 总运行次数 |
| avg_tool_steps | 平均工具调用次数 |
| avg_attempts | 平均模型调用轮数 |
| avg_prompt_chars | 平均 prompt 长度 |
| cache_hit_rate | prompt cache 命中率 |
| cached_token_ratio | 缓存 token 占总输入 token 的比例 |
| prefix_reuse_rate | prefix 被复用的比例 |
| tool_status_counts | 各类工具状态的分布 |
| tool_name_counts | 各类工具的调用次数分布 |
| security_event_counts | 各类安全事件的分布 |
| stop_reason_counts | 各类停机原因的分布 |
| avg_run_duration_ms | 平均运行耗时 |
| avg_tool_duration_ms | 平均工具执行耗时 |
| avg_prompt_build_duration_ms | 平均 prompt 组装耗时 |

**prefix_reuse_rate 的意义**：如果这个值很高，说明 prefix 大部分时间都在被复用而不是重建，prompt cache 的效率就高。如果频繁变化，说明工作区状态不稳定或 prefix 构建逻辑有问题。

---

### 2.7 Feature Ablation Metrics — 特性消融指标

这是一种轻量级的消融测试，不运行完整任务，只测量 prompt 组装的差异：

**三种变体**：
- `full`：所有特性开启
- `no_context_reduction`：关闭上下文压缩
- `no_memory`：关闭记忆系统（包括工作记忆和相关记忆召回）

**测量方式**：对同一个 agent 和同一个用户消息，分别用三种变体组装 prompt，对比结果。

**测量指标**：
- prompt_chars：各变体的 prompt 长度
- memory_chars / history_chars：各 section 的长度
- relevant_selected_count：召回的相关笔记数量
- budget_reduction_count：触发的压缩次数
- current_request_preserved：用户请求是否被完整保留

---

### 2.8 综合指标收集

`collect_resume_metrics()` 是一个综合函数，它把所有评测实验的结果汇聚成一份完整的指标报告：

```
collect_resume_metrics()
├── aggregate_benchmark_artifact()     ← 基准回归结果
├── aggregate_run_artifacts()           ← 历史运行聚合
├── build_stress_agent_metrics()        ← 上下文压力测试
├── run_memory_dependency_experiment()  ← 小规模记忆实验
├── run_large_scale_memory_experiment() ← 大规模记忆实验
├── run_context_stress_matrix()         ← 上下文消融矩阵
├── run_security_experiment_suite()     ← 安全实验套件
└── run_provider_experiments()          ← 跨模型实验（可选）
```

**两种实验模式**：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| synthetic | 用 FakeModelClient 的确定性输出 | CI/CD 回归、快速验证 |
| real | 用真实模型 API | 评估真实表现、生成报告 |

**输出产物**：
- JSON 格式的完整指标数据
- Markdown 格式的可读报告

---

### 2.9 Benchmark Core Report — 核心报告

`write_benchmark_core_report()` 生成一份经过筛选的核心报告，只保留四个维度：

| 维度 | 说明 |
|------|------|
| Harness Regression | runtime 合同是否稳定 |
| Context Ablation | 上下文压缩的收益 |
| Working Memory Ablation | 记忆系统的收益 |
| Recovery / Resume Ablation | 恢复机制的收益 |

**报告明确标注了"可以安全写进简历的指标"和"只适合放文档/面试展开的指标"**：

**可以安全写进简历的指标**（数值稳定、可复现）：
- avg_full_prompt_chars / avg_raw_prompt_chars
- avg_prompt_compression_ratio / max_prompt_compression_ratio
- repeated_reads
- avg_tool_steps
- correct_rate
- resume_success_rate
- workspace_drift_detection_rate
- resume_false_accept_rate

**只适合放文档/面试展开的指标**（受随机性影响、需要解释上下文）：
- current_request_preserved_rate
- memory_hit_rate
- stale_reanchor_rate
- failure_category_counts

**口径边界声明**：
- Harness regression 只证明 runtime 合同稳定，不证明 provider 上限
- Context、memory、recovery 这三层只证明模块收益，不和 provider benchmark 混写

---

### 2.10 评测脚本

项目提供了三个独立的评测脚本：

| 脚本 | 用途 |
|------|------|
| scripts/collect_resume_metrics.py | 收集综合指标，输出 JSON + Markdown |
| scripts/run_large_scale_experiments.py | 运行大规模消融实验 |
| scripts/run_provider_experiments.py | 运行跨模型对比实验 |

这些脚本可以独立运行，也可以作为 CI/CD pipeline 的一部分。

---

## 第三部分：评测与审计的协作关系

### 3.1 审计数据如何支撑评测

评测框架大量复用了审计基础设施：

| 审计产物 | 在评测中的用途 |
|----------|--------------|
| trace.jsonl | 从 trace 中提取 tool_executed 事件，统计工具使用模式 |
| report.json | 从 report 中提取 prompt_metadata，分析 prompt 组装效果 |
| task_state.json | 从 task_state 中提取 stop_reason，判断任务是否正常完成 |
| checkpoint | 从 trace 中提取 checkpoint_created 事件，验证检查点是否被正确创建 |

**设计意图**：评测不需要发明新的数据采集机制，而是直接消费审计层已经产生的数据。这保证了"评测看到的"和"实际运行发生的"是同一份数据。

### 3.2 评测如何验证审计

反过来，评测也在验证审计系统本身是否正常工作：

| 评测实验 | 验证的审计能力 |
|----------|--------------|
| Recovery Ablation | 检查点是否被正确创建和写入 |
| Security Experiments | 安全事件是否被正确记录到 trace |
| Durable Contract tasks | 持久记忆的晋升/拒绝是否被正确记录到 report |
| Context Reduction task | context_reduction 检查点是否出现在 trace 中 |

### 3.3 完整的反馈环

```
agent 运行
  │
  ├─→ 自动产生审计数据（trace / task_state / report）
  │
  ├─→ 评测框架消费审计数据，生成指标
  │
  ├─→ 指标暴露问题（如 memory_off 的 repeated_reads 过高）
  │
  ├─→ 开发者改进代码
  │
  └─→ 下一次评测验证改进效果
```

审计是评测的数据源，评测是审计的验证器。两者共同构成了 Pico 的质量保障闭环。
