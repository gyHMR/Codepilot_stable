from __future__ import annotations

"""
工作区资源加载模块。

负责从工作区的 `.codepilot/` 目录加载配置文件：
1) settings.json: 模型和运行时参数配置
2) model.local.json: 自定义模型配置（本地模型或非内置 provider）
3) prompt.md: 自定义系统提示词
4) tools.json: 启用的内置工具列表
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from codepilot.core import PlanningBudgetProfile, TaskMode, ToolExecutionMode
from codepilot.protocols import Model, ModelCapabilities


# 支持的模型 API 协议集合
SUPPORTED_MODEL_APIS = {"openai-compatible", "anthropic-messages"}


@dataclass(frozen=True)
class WorkspaceModelConfig:
    """工作区自定义模型配置。

    从 `.codepilot/model.local.json` 加载，用于配置非内置的自定义模型。

    Attributes:
        api: API 协议标识（"openai-compatible" 或 "anthropic-messages"）。
        provider: provider 名称。
        model_id: 模型 ID。
        base_url: API 端点基础 URL。
        api_key: 明文 API Key（可选，优先使用环境变量）。
        api_key_env: API Key 对应的环境变量名（可选）。
        context_window: 上下文窗口大小（默认 128K）。
        max_tokens: 最大输出 token 数（默认 8192）。
        reasoning: 是否支持推理模式。
        vision: 是否支持图片输入。
    """

    api: str
    provider: str
    model_id: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    context_window: int = 128_000
    max_tokens: int = 8192
    reasoning: bool = False
    vision: bool = False

    def __post_init__(self) -> None:
        """初始化后校验：确保 API 协议、必填字段和数值范围有效。"""
        if self.api not in SUPPORTED_MODEL_APIS:
            raise ValueError(f"Unsupported API protocol: {self.api}")
        if not self.provider or not self.model_id or not self.base_url:
            raise ValueError("provider, model_id and base_url are required")
        if self.context_window <= 0 or self.max_tokens <= 0:
            raise ValueError("context_window and max_tokens must be positive")

    def to_model(self) -> Model:
        """将工作区模型配置转换为通用的 Model 对象。"""
        return Model(
            id=self.model_id,
            name=self.model_id,
            api=self.api,
            provider=self.provider,
            base_url=self.base_url,
            reasoning=self.reasoning,
            input=["text", "image"] if self.vision else ["text"],
            context_window=self.context_window,
            max_tokens=self.max_tokens,
            capabilities=ModelCapabilities(
                tools=True,
                vision=self.vision,
                streaming=True,
                reasoning=self.reasoning,
                system_prompt=True,
                tool_choice=self.api == "openai-compatible",
                parallel_tool_calls=self.api == "openai-compatible",
            ),
        )

    def resolve_api_key(self) -> str | None:
        """解析 API Key：优先从环境变量读取，其次使用明文配置。"""
        if self.api_key_env:
            value = os.getenv(self.api_key_env)
            if value:
                return value
        return self.api_key

    def build_api_key_resolver(self) -> Callable[[str], str | None]:
        """构建 API Key 解析器函数（忽略 provider 参数，始终返回本配置的 Key）。"""
        return lambda _provider: self.resolve_api_key()


@dataclass
class WorkspaceSettings:
    """工作区 settings.json 配置。

    所有字段都是可选的，未设置时使用 RuntimeDefaults 中的默认值。
    通过 WorkspaceResourceLoader._load_settings() 从 JSON 文件加载。

    Attributes:
        provider: provider 名称。
        model_id: 模型 ID。
        system_prompt: 自定义系统提示词。
        thinking_level: 推理级别。
        tool_execution: 工具执行模式。
        retry_enabled: 是否启用重试。
        max_retries: 最大重试次数。
        retry_base_delay_ms: 重试基础延迟（毫秒）。
        read_only_mode: 是否为只读模式。
        block_dangerous_bash: 是否阻止危险 bash 命令。
        bash_allow_patterns: bash 命令白名单。
        bash_block_patterns: bash 命令黑名单。
        edit_require_unique_match: edit 是否要求唯一匹配。
        prompt_guidelines: 额外的提示词准则。
        append_system_prompt: 追加到系统提示词末尾的文本。
        tool_snippets: 工具说明片段。
        extension_paths: 扩展加载路径。
        skill_paths: 技能加载路径。
        prompt_debug_sources: 是否包含调试来源信息。
        mcp_servers: MCP 服务器配置。
    """

    provider: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    thinking_level: Optional[str] = None
    tool_execution: Optional[ToolExecutionMode] = None
    task_mode: Optional[TaskMode] = None
    planning_budget_profile: Optional[PlanningBudgetProfile] = None
    max_tool_calls_per_turn: Optional[int] = None
    retry_enabled: Optional[bool] = None
    max_retries: Optional[int] = None
    retry_base_delay_ms: Optional[int] = None
    read_only_mode: Optional[bool] = None
    tool_permission_mode: Optional[str] = None
    block_dangerous_bash: Optional[bool] = None
    bash_allow_patterns: Optional[list[str]] = None
    bash_block_patterns: Optional[list[str]] = None
    edit_require_unique_match: Optional[bool] = None
    prompt_guidelines: Optional[list[str]] = None
    append_system_prompt: Optional[str] = None
    tool_snippets: Optional[dict[str, str]] = None
    extension_paths: Optional[list[str]] = None
    skill_paths: Optional[list[str]] = None
    prompt_debug_sources: Optional[bool] = None
    mcp_servers: Optional[list[dict[str, Any]]] = None
    shell_timeout_seconds: Optional[int] = None
    shell_max_timeout_seconds: Optional[int] = None
    shell_stdout_limit: Optional[int] = None
    shell_stderr_limit: Optional[int] = None
    shell_allowed_env: Optional[list[str]] = None


@dataclass
class WorkspaceResources:
    """工作区资源汇总。

    Attributes:
        settings: settings.json 中的配置。
        model: model.local.json 中的自定义模型配置（不存在时为 None）。
        prompt: prompt.md 中的自定义系统提示词（不存在时为 None）。
        enabled_tools: tools.json 中启用的工具列表（不存在时为 None）。
    """

    settings: WorkspaceSettings
    model: WorkspaceModelConfig | None
    prompt: Optional[str]
    enabled_tools: Optional[list[str]]


class WorkspaceResourceLoader:
    """工作区资源加载器。

    从工作区的 `.codepilot/` 目录加载所有配置文件，
    并进行类型安全的解析和校验。

    使用示例::

        loader = WorkspaceResourceLoader("/path/to/workspace")
        resources = loader.load()
        if resources.model:
            model = resources.model.to_model()
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.resource_root = self.workspace_dir / ".codepilot"
        self.settings_file = self.resource_root / "settings.json"
        self.model_file = self.resource_root / "model.local.json"
        self.prompt_file = self.resource_root / "prompt.md"
        self.tools_file = self.resource_root / "tools.json"

    def load(self) -> WorkspaceResources:
        """加载所有工作区资源文件。"""
        return WorkspaceResources(
            settings=self._load_settings(),
            model=self._load_model(),
            prompt=self._load_prompt(),
            enabled_tools=self._load_tools(),
        )

    def _load_settings(self) -> WorkspaceSettings:
        """加载并解析 settings.json。"""
        if not self.settings_file.exists():
            return WorkspaceSettings()
        raw = self._safe_load_json(self.settings_file)
        if not isinstance(raw, dict):
            return WorkspaceSettings()

        # 校验 tool_execution 枚举值
        tool_execution = raw.get("tool_execution")
        if tool_execution not in {"parallel", "sequential"}:
            tool_execution = None
        raw_task_mode = raw.get("task_mode")
        task_mode = (
            raw_task_mode
            if isinstance(raw_task_mode, str)
            and raw_task_mode in {"read", "edit", "plan"}
            else None
        )
        raw_planning_budget_profile = raw.get("planning_budget_profile")
        planning_budget_profile = (
            raw_planning_budget_profile
            if isinstance(raw_planning_budget_profile, str)
            and raw_planning_budget_profile in {"conservative", "balanced", "wide"}
            else None
        )
        permission_mode = raw.get("tool_permission_mode")
        if permission_mode not in {"read-only", "workspace-write", "ask"}:
            permission_mode = None

        return WorkspaceSettings(
            provider=raw.get("provider") if isinstance(raw.get("provider"), str) else None,
            model_id=raw.get("model_id") if isinstance(raw.get("model_id"), str) else None,
            system_prompt=raw.get("system_prompt") if isinstance(raw.get("system_prompt"), str) else None,
            thinking_level=raw.get("thinking_level") if isinstance(raw.get("thinking_level"), str) else None,
            tool_execution=tool_execution,
            task_mode=task_mode,
            planning_budget_profile=planning_budget_profile,
            max_tool_calls_per_turn=self._to_positive_int(raw.get("max_tool_calls_per_turn")),
            retry_enabled=raw.get("retry_enabled") if isinstance(raw.get("retry_enabled"), bool) else None,
            max_retries=self._to_positive_int(raw.get("max_retries")),
            retry_base_delay_ms=self._to_positive_int(raw.get("retry_base_delay_ms")),
            read_only_mode=raw.get("read_only_mode") if isinstance(raw.get("read_only_mode"), bool) else None,
            tool_permission_mode=permission_mode,
            block_dangerous_bash=raw.get("block_dangerous_bash")
            if isinstance(raw.get("block_dangerous_bash"), bool)
            else None,
            bash_allow_patterns=self._to_string_list(raw.get("bash_allow_patterns")),
            bash_block_patterns=self._to_string_list(raw.get("bash_block_patterns")),
            edit_require_unique_match=raw.get("edit_require_unique_match")
            if isinstance(raw.get("edit_require_unique_match"), bool)
            else None,
            prompt_guidelines=self._to_string_list(raw.get("prompt_guidelines")),
            append_system_prompt=raw.get("append_system_prompt")
            if isinstance(raw.get("append_system_prompt"), str)
            else None,
            tool_snippets=self._to_string_map(raw.get("tool_snippets")),
            extension_paths=self._to_string_list(raw.get("extension_paths")),
            skill_paths=self._to_string_list(raw.get("skill_paths")),
            prompt_debug_sources=raw.get("prompt_debug_sources")
            if isinstance(raw.get("prompt_debug_sources"), bool)
            else None,
            mcp_servers=self._to_object_list(raw.get("mcp_servers")),
            shell_timeout_seconds=self._to_positive_int(raw.get("shell_timeout_seconds")),
            shell_max_timeout_seconds=self._to_positive_int(raw.get("shell_max_timeout_seconds")),
            shell_stdout_limit=self._to_positive_int(raw.get("shell_stdout_limit")),
            shell_stderr_limit=self._to_positive_int(raw.get("shell_stderr_limit")),
            shell_allowed_env=self._to_string_list(raw.get("shell_allowed_env")),
        )

    def _load_model(self) -> WorkspaceModelConfig | None:
        """加载并解析 model.local.json。"""
        if not self.model_file.exists():
            return None
        raw = self._safe_load_json(self.model_file)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid model config JSON: {self.model_file}")
        try:
            return WorkspaceModelConfig(
                api=str(raw.get("api", "")),
                provider=str(raw.get("provider", "")),
                model_id=str(raw.get("model_id", "")),
                base_url=str(raw.get("base_url", "")),
                api_key=raw.get("api_key") if isinstance(raw.get("api_key"), str) else None,
                api_key_env=raw.get("api_key_env") if isinstance(raw.get("api_key_env"), str) else None,
                context_window=int(raw.get("context_window", 128_000)),
                max_tokens=int(raw.get("max_tokens", 8192)),
                reasoning=bool(raw.get("reasoning", False)),
                vision=bool(raw.get("vision", False)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model config {self.model_file}: {exc}") from exc

    def _load_prompt(self) -> Optional[str]:
        """加载 prompt.md 文件内容。"""
        if not self.prompt_file.exists():
            return None
        text = self.prompt_file.read_text(encoding="utf-8").strip()
        return text or None

    def _load_tools(self) -> Optional[list[str]]:
        """加载并解析 tools.json 中的 enabled 列表。"""
        if not self.tools_file.exists():
            return None
        raw = self._safe_load_json(self.tools_file)
        if not isinstance(raw, dict):
            return None
        enabled = raw.get("enabled")
        if not isinstance(enabled, list):
            return None
        return [item for item in enabled if isinstance(item, str)]

    @staticmethod
    def _safe_load_json(path: Path) -> Any:
        """安全加载 JSON 文件，解析失败时返回 None。"""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _to_positive_int(value: Any) -> Optional[int]:
        """将值转换为正整数（排除 bool 类型），无效时返回 None。"""
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    @staticmethod
    def _to_string_list(value: Any) -> Optional[list[str]]:
        """将值转换为字符串列表，过滤非字符串元素。"""
        if not isinstance(value, list):
            return None
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _to_string_map(value: Any) -> Optional[dict[str, str]]:
        """将值转换为字符串字典，过滤非字符串的键值。"""
        if not isinstance(value, dict):
            return None
        result: dict[str, str] = {}
        for k, v in value.items():
            if isinstance(k, str) and isinstance(v, str):
                result[k] = v
        return result

    @staticmethod
    def _to_object_list(value: Any) -> Optional[list[dict[str, Any]]]:
        """将值转换为字典列表，过滤非字典元素。"""
        if not isinstance(value, list):
            return None
        result: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                result.append(dict(item))
        return result
