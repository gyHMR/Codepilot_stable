# Recovery Scenarios

这里放置 `EvalScenario` JSON，支持以下步骤：

`prompt`、`cancel`、`modify_file`、`restart`、`continue`、`verify`。

恢复案例应使用稳定的小型 fixture，并通过 Run/Trace Verifier 同时检查恢复后的
最终结果、Run 关联和事件生命周期。
