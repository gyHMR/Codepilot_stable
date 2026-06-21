# 轻量评估改造方案

## 目标

- 用少量指标解释 Context、Memory、Planning、Tool Security 的效果。
- 同时支持确定性机制测试和真实模型 on/off 消融。
- 每个模块保留 2 个 Benchmark，并允许以后继续添加 JSON 用例。
- 统一评估入口，删除重复脚本和重复 Benchmark。

## 指标

- Context：Key Context Hit Rate、Token Efficiency、Stale Context Rate。
- Memory：Memory Retrieval Hit Rate、Redundant Read Reduction Rate、Failed Attempt Recurrence Rate。
- Planning：Evidence Coverage Rate、False Completion Rate、Repair/Replan Success Rate。
- Security：Dangerous Tool Block Rate、Mutation After Denial Rate、Benign Tool Pass Rate。

## Benchmark

- `benchmarks/evaluation/context/`：关键上下文命中、过期上下文排除。
- `benchmarks/evaluation/memory/`：记忆召回、重复读取。
- `benchmarks/evaluation/planning/`：证据完成、失败后修复。
- `benchmarks/evaluation/security/`：危险工具阻止、正常工具放行。

用例继续使用 JSON，通过 `metrics` 声明指标，通过 `expected` 提供简单答案。

## 运行方式

```bash
python -m codepilot.evaluation check
python -m codepilot.evaluation run context
python -m codepilot.evaluation experiment memory --repeat 3
python -m codepilot.evaluation report <eval_dir>
```

- `check`：使用确定性模型流验证机制和指标。
- `run`：使用真实模型运行一个模块或全部模块。
- `experiment`：对 Context、Memory、Planning 做 on/off 对照。
- `report`：根据已有产物重新生成指标报告。

## 实现原则

- 复用现有 Run Artifact 和 Assertion，不重新建设事件系统。
- 指标集中放在 `evaluation/metrics.py`，不散落在 Executor 和脚本中。
- Experiment 只支持两个变体、简单重复和平均值对比。
- 分母为 0 的指标记为 `N/A`。
- 不计算综合总分，不做显著性检验。



# 附加介绍
这个评估模块采用一种轻量的“指标优先”理念：

```text
Benchmark 定义任务和预期
→ Agent 真实运行
→ Runtime 记录事件与报告
→ Assertion 检查任务是否正确
→ Metrics 计算模块表现
→ 输出 JSON 和 Markdown 报告
```

## 1. 评估理念

评估分成两层。

### 确定性机制测试

用于验证：

- 指标公式是否正确；
- Runtime 是否记录了所需数据；
- 模块开关是否真的生效；
- Benchmark 是否合法；
- 报告和 Experiment 是否能正常生成。

它不调用真实模型，本质上是可重复的自动化测试。

```bash
python -m codepilot.evaluation check
```

### 真实模型评估

让真实模型执行 Benchmark，回答：

- Agent 最终有没有完成任务；
- Context、Memory、Planning、安全模块表现如何；
- 模块开启后是否比关闭时更好。

---

## 2. Benchmark 结构

Benchmark 位于：

[benchmarks/evaluation](/E:/Project_python/agent/Codepilot/benchmarks/evaluation)

```text
context/   2 个
memory/    2 个
planning/  2 个
security/  2 个
```

每个 JSON 主要包含：

```json
{
  "id": "context-key-hit",
  "domain": "context",
  "fixture": "context_heavy",
  "prompt": "任务描述",
  "metrics": [
    "context.key_context_hit_rate"
  ],
  "expected": {
    "key_context": ["src/sample.py"]
  },
  "assertions": []
}
```

其中：

- `prompt/steps`：Agent 要执行的任务；
- `fixture`：隔离运行的小型项目；
- `metrics`：这个任务计算哪些指标；
- `expected`：指标计算需要的简单标准答案；
- `assertions`：检查最终任务结果是否正确。

以后扩展时，只需继续增加 Benchmark JSON 和对应 fixture。

---

## 3. 当前指标

### Context

- `Key Context Hit Rate`：关键上下文命中率；
- `Token Efficiency`：关键上下文 token 占最终上下文 token 的比例；
- `Stale Context Rate`：被选中过期上下文项的比例。

### Memory

- `Memory Retrieval Hit Rate`：预期记忆是否成功召回；
- `Redundant Read Count`：同一文件第二次及以后读取的次数；
- `Failed Attempt Recurrence Rate`：相同失败操作重复出现的比例。

在 Memory 消融实验中还会生成：

- `Redundant Read Reduction Rate`：开启 Memory 后重复读取降低了多少。

### Planning

- `Evidence Coverage Rate`：已完成步骤中带有证据的比例；
- `False Completion Rate`：声称完成但最终验证失败的比例；
- `Repair/Replan Success Rate`：发生修复或重规划后最终成功的比例。

### Tool Security

- `Dangerous Tool Block Rate`：危险工具调用阻止率；
- `Mutation After Denial Rate`：拒绝后仍发生修改的比例；
- `Benign Tool Pass Rate`：正常工具正确放行率。

---

## 4. 常用命令

### 运行确定性检查

```bash
python -m codepilot.evaluation check
```

只检查某个模块：

```bash
python -m codepilot.evaluation check context
python -m codepilot.evaluation check memory
```

### 运行真实模型评估

运行一个模块：

```bash
python -m codepilot.evaluation run context
python -m codepilot.evaluation run memory
python -m codepilot.evaluation run planning
python -m codepilot.evaluation run security
```

运行全部模块：

```bash
python -m codepilot.evaluation run all
```

如果项目没有模型配置：

```bash
python -m codepilot.evaluation run context \
  --provider deepseek \
  --model deepseek-chat
```

默认会尝试读取项目的 `.codepilot/model.local.json` 或 `.codepilot/settings.json`。

### 运行消融实验

目前支持 Context、Memory 和 Planning：

```bash
python -m codepilot.evaluation experiment context --repeat 3
python -m codepilot.evaluation experiment memory --repeat 3
python -m codepilot.evaluation experiment planning --repeat 3
```

实验会运行：

```text
off 变体：关闭模块
on 变体：开启模块
```

然后比较指标均值和任务通过率。

安全模块不提供关闭安全机制的实验。

### 查看已有报告

```bash
python -m codepilot.evaluation report .codepilot/evals/<eval_id>
```

---

## 5. 输出放在哪里

默认输出目录：

```text
.codepilot/evals/<eval_id>/
```

主要结构：

```text
<eval_id>/
├── manifest.json
├── summary.json
├── report.md
├── cases/
│   └── <case_id>/
│       ├── definition.json
│       ├── result.json
│       ├── metrics.json
│       ├── assertion-results.json
│       ├── workspace.diff
│       └── run-refs.json
└── workspaces/
    └── <case_id>/
```

最常看的文件：

- `summary.json`：整个 Suite 的通过率和指标平均值；
- `report.md`：适合阅读和项目展示的报告；
- `cases/<case_id>/metrics.json`：单个任务的详细指标；
- `cases/<case_id>/result.json`：任务完整评估结果。

底层运行证据位于隔离工作区：

```text
workspaces/<case_id>/.codepilot/runs/<run_id>/
├── events.jsonl
├── state.json
├── result.json
└── report.json
```

消融实验额外生成：

```text
.codepilot/evals/<experiment_id>/
├── experiment.json
└── report.md
```

其中 `experiment.json` 保存 on/off 均值和变化量，`report.md` 提供可直接展示的对比表。
