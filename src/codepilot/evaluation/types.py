from __future__ import annotations

"""多维度 Agent 评估的稳定领域类型。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codepilot.observability import AuditBundle
from codepilot.runtime.types import CreateAgentSessionOptions


# 评估领域
EvalDomain = Literal[
    "runtime", "coding", "context", "memory", "security", "planning", "recovery",
]
# 评估维度
EvalDimension = Literal[
    "coding_outcome", "runtime_contract", "context_governance",
    "memory", "tool_security", "task_planning", "recovery", "efficiency",
]
# 评估总体结果
EvalOverall = Literal["passed", "failed", "invalid_case", "execution_error"]
# 维度状态
DimensionStatus = Literal["passed", "failed", "error", "not_applicable"]
# 断言状态
AssertionStatus = Literal["passed", "failed", "error", "skipped"]
# 断言类型：命令/文件/差异/运行/追踪/上下文/记忆/安全/任务
AssertionType = Literal[
    "command", "file", "diff", "run", "trace", "context", "memory", "security", "task",
    "metric",
]
# 场景步骤类型
ScenarioStepType = Literal[
    "prompt", "cancel", "modify_file", "restart", "continue", "verify", "inspect",
]


@dataclass(frozen=True)
class AssertionSpec:
    """断言规格：定义一个评估断言的类型、维度和选项。"""
    type: AssertionType
    dimension: EvalDimension
    options: dict[str, Any] = field(default_factory=dict)
    required: bool = True  # 是否为必需断言（失败则整体不通过）


@dataclass(frozen=True)
class ScenarioStep:
    """场景步骤：一个多步评估场景中的单个操作。"""
    type: ScenarioStepType
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalBudgets:
    """评估预算：限制模型调用、工具调用和超时。"""
    max_model_attempts: int | None = None  # 最大模型调用次数
    max_tool_calls: int | None = None      # 最大工具调用次数
    max_replans: int | None = None         # 最大重规划次数
    timeout_seconds: int = 120             # 超时时间（秒）


@dataclass(frozen=True)
class EvalRuntimeProfile:
    """评估运行时配置：控制上下文治理、记忆、权限等行为。"""
    context_governance_enabled: bool = True   # 是否启用上下文治理
    memory_enabled: bool = True               # 是否启用记忆
    task_control_enabled: bool = True         # 是否启用任务控制
    permission_mode: str = "workspace-write"  # 权限模式
    scripted_stream: str | None = None        # 脚本化流式响应（用于测试）


@dataclass(frozen=True)
class EvalCase:
    """评估用例：单轮 prompt 驱动的评估定义。"""
    id: str                        # 用例 ID
    domain: EvalDomain             # 所属领域
    fixture: str                   # fixture 目录名
    prompt: str                    # 用户提示词
    assertions: list[AssertionSpec]  # 断言列表
    metrics: list[str] = field(default_factory=list)  # 需要计算的指标
    expected: dict[str, Any] = field(default_factory=dict)  # 指标答案
    budgets: EvalBudgets = EvalBudgets()
    runtime: EvalRuntimeProfile = EvalRuntimeProfile()
    tags: list[str] = field(default_factory=list)

    @property
    def timeout_seconds(self) -> int:
        return self.budgets.timeout_seconds


@dataclass(frozen=True)
class EvalScenario:
    """评估场景：多步骤的状态化评估定义。"""
    id: str                          # 场景 ID
    domain: EvalDomain
    fixture: str
    steps: list[ScenarioStep]        # 步骤列表
    assertions: list[AssertionSpec]
    metrics: list[str] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)
    budgets: EvalBudgets = EvalBudgets()
    runtime: EvalRuntimeProfile = EvalRuntimeProfile()
    tags: list[str] = field(default_factory=list)

    @property
    def timeout_seconds(self) -> int:
        return self.budgets.timeout_seconds


@dataclass(frozen=True)
class AssertionResult:
    """断言结果：单个断言的执行结果。"""
    name: str
    dimension: EvalDimension
    status: AssertionStatus
    summary: str
    expected: object | None = None
    actual: object | None = None
    evidence_refs: list[str] = field(default_factory=list)  # 证据引用
    required: bool = True


@dataclass(frozen=True)
class DimensionResult:
    """维度结果：一个评估维度下所有断言的聚合结果。"""
    dimension: EvalDimension
    status: DimensionStatus
    summary: str
    assertion_results: list[AssertionResult]
    metrics: dict[str, float | int | bool] = field(default_factory=dict)


@dataclass
class EvalResult:
    """评估结果：单个用例/场景的完整执行结果。"""
    case_id: str
    overall: EvalOverall
    session_id: str | None
    run_ids: list[str]
    dimensions: list[DimensionResult]
    failure_categories: list[str]         # 失败分类列表
    metrics: dict[str, Any]
    artifact_dir: str                     # 产物目录
    error: str | None = None
    duration_ms: int | None = None        # 总耗时（毫秒）


@dataclass
class EvalSuiteResult:
    """评估套件结果：一次完整评估运行的所有用例结果。"""
    eval_id: str
    results: list[EvalResult]
    artifact_dir: str
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRunOptions:
    """评估运行选项。"""
    fixtures_root: str | Path                       # fixture 根目录
    session_options: CreateAgentSessionOptions      # 会话配置
    artifact_root: str | Path = ".codepilot/evals"  # 产物根目录
    eval_id: str | None = None                      # 评估 ID（自动生成）
    benchmark_name: str = ""                        # 基准名称
    keep_workspace: bool = True                     # 是否保留工作区
    runtime_overrides: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceChange:
    """工作区变更记录。"""
    path: str
    status: Literal["added", "modified", "deleted"]


@dataclass
class EvalEvidence:
    """评估证据：收集运行过程中的审计数据，供断言使用。"""
    workspace: Path                                    # 工作区路径
    baseline: dict[str, str]                           # 基线文件哈希
    session_id: str | None = None                      # 会话 ID
    audit_bundles: list[AuditBundle] = field(default_factory=list)      # 审计包
    freshness_history: list[dict[str, Any]] = field(default_factory=list)  # 新鲜度历史
    changes: list[WorkspaceChange] = field(default_factory=list)        # 工作区变更
    step_results: list[dict[str, Any]] = field(default_factory=list)    # 步骤结果

    @property
    def run_ids(self) -> list[str]:
        return [bundle.run_id for bundle in self.audit_bundles]

    def select_bundle(self, requested: object = "latest") -> AuditBundle | None:
        if not self.audit_bundles:
            return None
        if requested == "first":
            return self.audit_bundles[0]
        if requested == "latest" or requested is None:
            return self.audit_bundles[-1]
        return next(
            (
                bundle
                for bundle in self.audit_bundles
                if bundle.run_id == str(requested)
            ),
            None,
        )


EvalDefinition = EvalCase | EvalScenario
