# 评测系统设计思路

## 设计目标

评测系统解决的问题是：Coding Agent 不能只靠“最终回答看起来不错”来判断效果。它需要用可复现任务、真实运行证据和固定指标回答：

- 是否真的修好了问题。
- 是否找到关键上下文。
- 是否召回正确记忆。
- 是否能按任务计划推进。
- 是否能安全处理危险工具调用。
- 失败时能否复盘原因。

Codepilot 的评测系统不是在线平台，也不是 LLM Judge。它是一个本地、证据驱动的评测链路。

```text
benchmark 描述任务
  -> runner 真实运行 Agent
  -> 收集 trace 和 workspace diff
  -> 形成 evidence
  -> scorer 计算指标
  -> artifact 保存结果
```

## 核心原则

### benchmark 只描述任务和期望

benchmark 不应该知道 Agent 内部实现。它只声明：

- 任务是什么。
- 使用哪个 fixture。
- 运行前如何准备。
- 最终用什么检查。
- 需要计算哪些指标。
- 哪些上下文、记忆或安全行为是期望的。

### runner 只负责真实运行和收集证据

runner 像普通用户一样打开会话、发送消息、执行步骤。它不直接操控 core、tools、memory 或 context 内部状态。

### scorer 只基于 evidence

评分层不回头访问 runtime 内部，不重新读一堆原始文件。它只消费整理后的证据对象。这样指标公式稳定、可解释，也容易测试。

## 评测对象

当前评测围绕几个 Agent 关键能力：

| 能力 | 关注点 |
---|---
| task | 最终任务是否通过检查 |
| context | 是否命中关键上下文，是否被噪声误导 |
| memory | 是否召回目标记忆，是否避免重复失败路径 |
| planning | 是否完成步骤，失败后是否能修复或恢复 |
| tool | 工具调用是否成功，是否有大量无效调用 |
| security | 危险工具是否被拦截，拒绝后是否无副作用 |

这些指标不替代人工判断，但能把 Agent 行为拆成可分析的部分。

## Benchmark 设计

一个 benchmark case 应包含：

- 唯一 ID。
- 所属模块。
- fixture 名称。
- 用户 prompt 或多步骤 scenario。
- setup 操作。
- checks。
- metrics。
- expected。
- tags。

fixture 应该是可控的小型真实项目，而不是过度简化的字符串替换任务。理想 fixture 包含：

- 业务代码。
- 测试。
- 当前文档。
- 旧文档或干扰信息。
- 可注入 bug 的 mutation。
- 可选项目记忆。

这样可以同时测试代码理解、上下文选择、工具执行和验证闭环。

## Runner 设计

runner 的职责是执行 case：

```text
复制 fixture 到隔离 workspace
  -> 应用 setup
  -> 打开 runtime session
  -> 发送 prompt 或 scenario steps
  -> 收集运行事件和结果
  -> 执行 checks
  -> 计算 workspace diff
  -> 构建 evidence
  -> 调用 scorer
  -> 写 artifact
```

runner 不应该绕过 runtime 直接调用 core。评测要覆盖真实用户路径，否则无法评估 runtime、sessions、tools、context 和 memory 的协同。

## Evidence 设计

evidence 是评测系统的核心中间层。它把原始运行过程压成 scorer 能理解的事实。

evidence 应包含：

- case 是否通过。
- 最终回答。
- run ids。
- 工具调用列表。
- 上下文报告。
- 记忆召回记录。
- 任务步骤和任务摘要。
- workspace diff。
- check 结果。
- 安全拦截证据。

scorer 只看 evidence，不关心这些证据最初来自哪个具体文件或事件格式。

## 指标设计

指标应遵循几个原则：

1. 可复现：相同 evidence 得到相同分数。
2. 可解释：分子、分母、样本数都能展示。
3. 可缺省：没有样本时返回 N/A，而不是误算成 0。
4. 可组合：一个 case 可以声明多个指标。
5. 不依赖模型再判断：尽量不用 LLM Judge。

典型指标：

| 指标类型 | 示例 |
---|---
| 任务结果 | checks 是否全部通过 |
| 上下文 | 关键上下文命中率、噪声率、stale 率 |
| 记忆 | 目标记忆召回率、失败路径复现率 |
| 规划 | 步骤完成率、虚假完成率、恢复成功率 |
| 工具 | 成功率、无效调用率 |
| 安全 | 危险拦截率、正常放行率、拒绝后副作用率 |

## Artifact 设计

评测结果必须可复盘。一次运行应保存：

- manifest。
- summary。
- human-readable report。
- 每个 case 的输入定义。
- 每个 case 的 result。
- evidence。
- scores。
- steps 或事件摘要。
- workspace diff。
- run 关联信息。

失败排查顺序应该清楚：

```text
summary
  -> report
  -> failed case result
  -> scores
  -> evidence
  -> workspace diff
  -> run artifacts
```

## 实验与 A/B

有些能力适合真实 runtime on/off 消融，例如 memory、planning。它们可以通过开关对比：

```text
off
  -> run suite
on
  -> run suite
delta
```

有些能力适合确定性 A/B，例如上下文选择策略、安全策略。它们可以直接用固定输入比较策略效果，不必每次跑真实模型。

设计上不要强行把所有评测都做成真实模型实验。真实模型实验更接近用户路径，但成本高、波动大；确定性实验更稳定，适合策略单元。

## 与主系统边界

| 模块 | 评测如何交互 |
---|---
| runtime | 像用户一样打开会话和发送动作 |
| observability | 读取运行 trace 和事件证据 |
| sessions | 通过 run artifact 和结果间接观察 |
| context | 通过上下文报告评估命中和噪声 |
| memory | 通过召回 id 和事件评估 |
| tools | 通过工具调用证据评估成功率和安全性 |

评测系统不应该修改 core 内部决策，也不应该为了指标而让主系统暴露额外内部对象。

## 设计取舍

当前评测系统不做：

- 在线排行榜。
- LLM Judge 主观评分。
- 大型复杂业务 fixture。
- 对所有模块强制真实 on/off。
- 过度复杂断言树。

它追求的是本地可运行、可复现、可审计：

```text
任务是什么
Agent 实际做了什么
证据在哪里
指标怎么算
失败如何复盘
```
