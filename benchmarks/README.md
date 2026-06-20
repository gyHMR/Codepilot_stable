# Codepilot Benchmarks

目录按 Eval 层次组织：

- `coding/`：真实模型代码任务；
- `harness/`：确定模型输出下的 Runtime 合同回归；
- `recovery/`：Session 与工作区恢复场景；
- `fixtures/`：每个案例使用的独立小型工程。

示例：

```python
import asyncio

from codepilot.evaluation import EvaluationService, EvalRunOptions
from codepilot.runtime import CreateAgentSessionOptions

options = EvalRunOptions(
    fixtures_root="benchmarks/fixtures",
    artifact_root=".codepilot/evals",
    benchmark_name="coding-smoke",
    session_options=CreateAgentSessionOptions(
        workspace_dir=".",
        provider="deepseek",
        model_id="deepseek-chat",
    ),
)

result = asyncio.run(
    EvaluationService().run_suite(
        "benchmarks/coding",
        options,
    )
)
print(result.summary)
```

Harness Regression 可通过
`CreateAgentSessionOptions(stream_fn=...)` 注入 Session 级确定性模型流。
脚本行为属于 Python callable，不写入 JSON；可复用的 Harness 脚本与案例组合应放在
`harness/` 的 Python runner 中。基础贯通测试见 `test/test_evaluation.py`。
