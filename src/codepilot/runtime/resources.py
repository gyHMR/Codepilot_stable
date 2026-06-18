from __future__ import annotations

"""Workspace resource loading:
1) settings.json: model and runtime parameters
2) prompt.md: system prompt
3) tools.json: enabled built-in tools
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from codepilot.core import ToolExecutionMode
from codepilot.protocols import Model, ModelCapabilities


SUPPORTED_MODEL_APIS = {"openai-compatible", "anthropic-messages"}


@dataclass(frozen=True)
class WorkspaceModelConfig:
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
        if self.api not in SUPPORTED_MODEL_APIS:
            raise ValueError(f"Unsupported API protocol: {self.api}")
        if not self.provider or not self.model_id or not self.base_url:
            raise ValueError("provider, model_id and base_url are required")
        if self.context_window <= 0 or self.max_tokens <= 0:
            raise ValueError("context_window and max_tokens must be positive")

    def to_model(self) -> Model:
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
        if self.api_key_env:
            value = os.getenv(self.api_key_env)
            if value:
                return value
        return self.api_key

    def build_api_key_resolver(self) -> Callable[[str], str | None]:
        return lambda _provider: self.resolve_api_key()


@dataclass
class WorkspaceSettings:
    provider: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    thinking_level: Optional[str] = None
    tool_execution: Optional[ToolExecutionMode] = None
    max_context_messages: Optional[int] = None
    retain_recent_messages: Optional[int] = None
    max_context_tokens: Optional[int] = None
    retry_enabled: Optional[bool] = None
    max_retries: Optional[int] = None
    retry_base_delay_ms: Optional[int] = None
    read_only_mode: Optional[bool] = None
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


@dataclass
class WorkspaceResources:
    settings: WorkspaceSettings
    model: WorkspaceModelConfig | None
    prompt: Optional[str]
    enabled_tools: Optional[list[str]]


class WorkspaceResourceLoader:
    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.resource_root = self.workspace_dir / ".codepilot"
        self.settings_file = self.resource_root / "settings.json"
        self.model_file = self.resource_root / "model.local.json"
        self.prompt_file = self.resource_root / "prompt.md"
        self.tools_file = self.resource_root / "tools.json"

    def load(self) -> WorkspaceResources:
        return WorkspaceResources(
            settings=self._load_settings(),
            model=self._load_model(),
            prompt=self._load_prompt(),
            enabled_tools=self._load_tools(),
        )

    def _load_settings(self) -> WorkspaceSettings:
        if not self.settings_file.exists():
            return WorkspaceSettings()
        raw = self._safe_load_json(self.settings_file)
        if not isinstance(raw, dict):
            return WorkspaceSettings()

        tool_execution = raw.get("tool_execution")
        if tool_execution not in {"parallel", "sequential"}:
            tool_execution = None

        return WorkspaceSettings(
            provider=raw.get("provider") if isinstance(raw.get("provider"), str) else None,
            model_id=raw.get("model_id") if isinstance(raw.get("model_id"), str) else None,
            system_prompt=raw.get("system_prompt") if isinstance(raw.get("system_prompt"), str) else None,
            thinking_level=raw.get("thinking_level") if isinstance(raw.get("thinking_level"), str) else None,
            tool_execution=tool_execution,
            max_context_messages=self._to_positive_int(raw.get("max_context_messages")),
            retain_recent_messages=self._to_positive_int(raw.get("retain_recent_messages")),
            max_context_tokens=self._to_positive_int(raw.get("max_context_tokens")),
            retry_enabled=raw.get("retry_enabled") if isinstance(raw.get("retry_enabled"), bool) else None,
            max_retries=self._to_positive_int(raw.get("max_retries")),
            retry_base_delay_ms=self._to_positive_int(raw.get("retry_base_delay_ms")),
            read_only_mode=raw.get("read_only_mode") if isinstance(raw.get("read_only_mode"), bool) else None,
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
        )

    def _load_model(self) -> WorkspaceModelConfig | None:
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
        if not self.prompt_file.exists():
            return None
        text = self.prompt_file.read_text(encoding="utf-8").strip()
        return text or None

    def _load_tools(self) -> Optional[list[str]]:
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
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _to_positive_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    @staticmethod
    def _to_string_list(value: Any) -> Optional[list[str]]:
        if not isinstance(value, list):
            return None
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _to_string_map(value: Any) -> Optional[dict[str, str]]:
        if not isinstance(value, dict):
            return None
        result: dict[str, str] = {}
        for k, v in value.items():
            if isinstance(k, str) and isinstance(v, str):
                result[k] = v
        return result

    @staticmethod
    def _to_object_list(value: Any) -> Optional[list[dict[str, Any]]]:
        if not isinstance(value, list):
            return None
        result: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                result.append(dict(item))
        return result

