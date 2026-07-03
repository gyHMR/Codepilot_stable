from __future__ import annotations

# 新手导读：模型解析负责从选项、配置和内置目录中找出最终 Model。
# 关注点：它只决定“用哪个模型”，不负责真正调用模型。

"""
模型解析模块。

负责从多种来源中确定最终使用的模型，按优先级依次尝试：
1) options.model: 调用方直接指定的模型对象（最高优先级）
2) options.provider + options.model_id: 从内置模型目录解析
3) 恢复的会话元数据中的 provider + model_id
4) 工作区 model.local.json 中的自定义模型配置
5) 工作区 settings.json 中的 provider + model_id

以上均未匹配时抛出 ValueError。
"""

from dataclasses import dataclass
from typing import Awaitable, Callable

from codepilot.llm.models import get_model
from codepilot.protocols import Model

from .config import RuntimeInputs
from codepilot.runtime.contracts import ConfigValueSource, CreateAgentSessionOptions


@dataclass(frozen=True)
class ResolvedModel:
    """解析完成的模型配置。

    Attributes:
        model: 解析得到的 Model 对象。
        get_api_key: API Key 获取函数（可选，为 None 时使用环境变量默认逻辑）。
    """

    model: Model
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = None
    source: ConfigValueSource = ConfigValueSource(kind="default")


def resolve_model(
    options: CreateAgentSessionOptions,
    inputs: RuntimeInputs,
) -> ResolvedModel:
    """按优先级从多种来源解析模型配置。

    优先级顺序：
    1) options.model（直接指定）
    2) options.provider + options.model_id（从内置目录查找）
    3) 恢复的会话元数据中的 provider + model_id
    4) 工作区 model.local.json（自定义模型）
    5) 工作区 settings.json 中的 provider + model_id

    Args:
        options: 创建会话的配置选项。
        inputs: 运行时输入数据（含工作区资源和会话元数据）。

    Returns:
        ResolvedModel 对象，包含 Model 和 API Key 获取函数。

    Raises:
        ValueError: 所有来源都无法解析模型时抛出。
    """
    resources = inputs.resources

    # 优先级 1：直接指定的模型对象
    model = options.model
    if model is not None:
        return ResolvedModel(
            model=model,
            get_api_key=options.get_api_key,
            source=ConfigValueSource(kind="cli"),
        )

    # 优先级 2：provider + model_id（从内置目录查找）
    if options.provider and options.model_id:
        return ResolvedModel(
            model=get_model(options.provider, options.model_id),
            get_api_key=options.get_api_key,
            source=ConfigValueSource(kind="cli"),
        )

    # 优先级 3：恢复的会话元数据
    restored_meta = inputs.restored_meta or {}
    provider = restored_meta.get("provider")
    model_id = restored_meta.get("model_id")
    if isinstance(provider, str) and isinstance(model_id, str):
        return ResolvedModel(
            model=get_model(provider, model_id),
            get_api_key=options.get_api_key,
            source=ConfigValueSource(
                kind="session",
                location=options.session_id,
            ),
        )

    # 优先级 4：工作区 model.local.json
    if resources and resources.model is not None:
        return ResolvedModel(
            model=resources.model.to_model(),
            get_api_key=options.get_api_key or resources.model.build_api_key_resolver(),
            source=ConfigValueSource(
                kind="project",
                location=".codepilot/model.local.json",
            ),
        )

    # 优先级 5：工作区 settings.json
    if resources and resources.settings.provider and resources.settings.model_id:
        return ResolvedModel(
            model=get_model(resources.settings.provider, resources.settings.model_id),
            get_api_key=options.get_api_key,
            source=ConfigValueSource(
                kind="project",
                location=".codepilot/settings.json",
            ),
        )

    # 所有来源均未匹配
    raise ValueError(
        "Unable to resolve model: create .codepilot/model.local.json "
        "or provide --model provider/model-id"
    )
