from __future__ import annotations

"""
================================================================================
工具注册表 (Tool Registry) 模块
================================================================================

本模块是 Codepilot 工具系统的核心组件，负责以下职责：

1. **工具注册与查询** — 通过 ToolRegistry 类提供工具定义的增删改查功能。
   每个运行时会话 (runtime session) 持有一个独立的 ToolRegistry 实例，
   用于管理该会话中可用的所有工具。

2. **工具目录暴露** — 将注册的工具以 ToolCatalogView 的形式暴露给外部，
   供 UI 层或 AI 模型了解当前可用的工具列表及其元数据。

3. **工具调用准备** — 在执行工具调用之前，对原始参数进行解析、展开和类型
   强制转换 (coercion)，确保参数符合工具的 JSON Schema 定义。若参数不合法，
   则返回结构化的错误信息，而非直接抛出异常。

4. **内置工具元数据** — 定义了一组内置工具的元数据（如 ls、read、write 等），
   包括风险级别、读写属性、适用作用域等，供工具注册时使用。

模块结构：
- ToolRegistry 类：核心注册表
- 参数处理函数：_parse_arguments、_unwrap_arguments、_coerce_arguments、_coerce_value
- 辅助函数：prepare_error_content、builtin_metadata、get_builtin_tool_metadata 等
- 常量：READ_ONLY_TOOL_NAMES、MUTATING_TOOL_NAMES
- 内置元数据字典：_BUILTIN_METADATA
"""

import json
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Iterable

from codepilot.protocols import (
    CLOSE_PLAN_TOOL,
    CREATE_BUILD_PLAN_TOOL,
    PLAN_TOOL_NAMES,
    PROPOSE_PLAN_TOOL,
    TextContent,
    UPDATE_PLAN_PROGRESS_TOOL,
)

from .contracts import (
    PreparedToolCall,
    PreparedToolCallResult,
    ToolCallRequest,
    ToolCatalogItem,
    ToolCatalogView,
    ToolDefinition,
    ToolMetadata,
)

# ---------------------------------------------------------------------------
# 工具名称分类常量
# ---------------------------------------------------------------------------
# 只读工具名称集合：这些工具不会修改文件系统或执行副作用操作。
READ_ONLY_TOOL_NAMES = {"ls", "read", "grep", "find", "workspace_status", *PLAN_TOOL_NAMES}

# 可变工具名称集合：这些工具会修改文件系统或执行 shell 命令等副作用操作。
MUTATING_TOOL_NAMES = {"write", "edit", "apply_patch", "bash"}


# ==============================================================================
# ToolRegistry 类 — 工具注册表
# ==============================================================================
@dataclass
class ToolRegistry:
    """内存中的工具注册表，每个已打开的运行时会话持有一个实例。

    该注册表以字典形式存储 ToolDefinition 对象，支持按名称注册、查询、
    列出和准备工具调用。
    """

    # 内部存储：工具名称 -> ToolDefinition 的映射字典
    _tools: dict[str, ToolDefinition] = field(default_factory=dict)

    # --------------------------------------------------------------------------
    # 注册方法
    # --------------------------------------------------------------------------

    def register(self, tool: ToolDefinition, *, replace: bool = True) -> None:
        """注册单个工具定义到注册表中。

        Args:
            tool: 要注册的 ToolDefinition 实例。
            replace: 若为 True（默认），同名工具将被覆盖；
                     若为 False，同名工具已存在时抛出 ValueError。

        Raises:
            TypeError: 传入的 tool 不是 ToolDefinition 实例。
            ValueError: replace=False 且同名工具已注册时抛出。
        """
        # 类型检查：确保只接受 ToolDefinition 实例
        if not isinstance(tool, ToolDefinition):
            raise TypeError("ToolRegistry.register expects ToolDefinition")
        # 若不允覆盖且工具已存在，抛出错误
        if not replace and tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        # 存储工具定义
        self._tools[tool.name] = tool

    def extend(self, tools: Iterable[ToolDefinition], *, replace: bool = True) -> None:
        """批量注册多个工具定义。

        Args:
            tools: 可迭代的 ToolDefinition 集合。
            replace: 传递给每个 register 调用的覆盖标志，默认为 True。
        """
        for tool in tools:
            self.register(tool, replace=replace)

    # --------------------------------------------------------------------------
    # 查询方法
    # --------------------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition | None:
        """按名称获取工具定义。

        Args:
            name: 工具名称。

        Returns:
            对应的 ToolDefinition 实例，若不存在则返回 None。
        """
        return self._tools.get(name)

    def metadata_for(self, name: str) -> ToolMetadata | None:
        """按名称获取工具的元数据。

        Args:
            name: 工具名称。

        Returns:
            工具的 ToolMetadata，若工具不存在则返回 None。
        """
        tool = self.get(name)
        return tool.metadata if tool is not None else None

    # --------------------------------------------------------------------------
    # 列表与目录方法
    # --------------------------------------------------------------------------

    def list(self, *, current_mode: str | None = None) -> list[ToolDefinition]:
        """列出注册表中所有（或按模式过滤后的）工具定义。

        工具按 (category, name) 排序，确保输出顺序稳定。

        Args:
            current_mode: 当前运行模式（如 "read"、"plan"、"build"）。
                          若为 None，返回所有工具；
                          若指定，仅返回在该模式下可见的工具。

        Returns:
            排序后的 ToolDefinition 列表。
        """
        # 按类别和名称排序
        tools = sorted(self._tools.values(), key=lambda item: (item.metadata.category, item.name))
        if current_mode is None:
            return tools
        # 按模式的可见性过滤
        return [tool for tool in tools if tool.metadata.visible_in(current_mode)]

    def catalog(self, *, current_mode: str = "build") -> ToolCatalogView:
        """生成当前模式下的工具目录视图。

        该视图将工具的规格说明（用于 AI 模型）和元数据打包在一起，
        供 UI 层或 Agent 层使用。

        Args:
            current_mode: 当前运行模式，默认为 "build"。

        Returns:
            ToolCatalogView 实例，包含 ToolCatalogItem 元组。
        """
        return ToolCatalogView(
            tuple(
                ToolCatalogItem(spec=tool.to_spec(), metadata=tool.metadata)
                for tool in self.list(current_mode=current_mode)
            )
        )

    # --------------------------------------------------------------------------
    # 工具调用准备方法
    # --------------------------------------------------------------------------

    def prepare_call(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any] | str | None,
        current_mode: str,
        source: str = "agent",
        policy_context: Any = None,
        assistant_message: Any = None,
        context: Any = None,
    ) -> PreparedToolCallResult:
        """为工具调用做预处理：校验、解析参数并构造请求对象。

        这是工具调用的入口网关，在工具实际执行之前完成以下检查：
        1. 工具是否存在（不存在则给出相似名称提示）
        2. 工具在当前模式下是否可用
        3. 参数是否为合法 JSON
        4. 参数是否符合工具的 JSON Schema（类型强制转换、必填项检查等）

        Args:
            run_id: 当前运行 ID，用于追踪和关联。
            tool_call_id: 工具调用 ID（通常由模型生成）。
            name: 工具名称。
            arguments: 原始参数，可以是 dict、JSON 字符串或 None。
            current_mode: 当前运行模式。
            source: 调用来源，默认为 "agent"。
            policy_context: 策略上下文，用于权限判断等。若为 None 则使用默认值。
            assistant_message: 关联的 assistant 消息（可选）。
            context: 附加上下文（可选）。

        Returns:
            PreparedToolCallResult 实例：
            - 若成功，其 call 字段包含 PreparedToolCall（含 ToolDefinition 和 ToolCallRequest）。
            - 若失败，其 error_code 和 message 字段包含错误详情。
        """
        # 步骤 1：查找工具定义
        tool = self.get(name)
        if tool is None:
            # 工具不存在时，利用模糊匹配给出建议名称
            hint = _tool_name_hint(name, self._tools)
            return PreparedToolCallResult(
                error_code="tool_not_found",
                message=f"Tool '{name}' not found.{hint}",
                recovery_hint=hint.strip(),
            )

        # 步骤 2：检查工具在当前模式下是否可见
        if not tool.metadata.visible_in(current_mode):
            return PreparedToolCallResult(
                error_code="tool_not_available_in_mode",
                message=f"Tool '{name}' is not available in {current_mode} mode.",
                recovery_hint=(
                    "Use an available tool for the current mode. "
                    "If build actions are required, stop and ask the user to approve "
                    "the plan with /plan approve."
                ),
            )

        # 步骤 3：解析原始参数（处理 dict / JSON 字符串 / None 等不同形式）
        parsed = _parse_arguments(arguments)
        if parsed.error is not None:
            return PreparedToolCallResult(
                error_code="invalid_tool_arguments",
                message=parsed.error,
                recovery_hint="Pass tool arguments as a JSON object matching the schema.",
            )

        # 步骤 4：展开嵌套参数（处理 {"arguments": {...}} 包裹形式）
        unwrapped = _unwrap_arguments(parsed.value, tool.parameters)

        # 步骤 5：按工具 Schema 进行类型强制转换和校验
        coerced = _coerce_arguments(unwrapped, tool.parameters)
        if coerced.errors:
            return PreparedToolCallResult(
                error_code="invalid_tool_arguments",
                message="Tool arguments failed schema validation: " + "; ".join(coerced.errors),
                recovery_hint="Correct the arguments and retry the same exact tool name.",
            )

        # 步骤 6：构造 ToolCallRequest 请求对象
        request = ToolCallRequest(
            run_id=run_id,
            tool_call_id=tool_call_id,
            name=name,
            arguments=coerced.value,
            metadata=tool.metadata,
            current_mode=current_mode,
            source=source,  # type: ignore[arg-type]
            policy_context=policy_context or _default_policy_context(),
            assistant_message=assistant_message,
            context=context,
        )

        # 步骤 7：返回成功结果，包含工具定义和请求对象
        return PreparedToolCallResult(call=PreparedToolCall(definition=tool, request=request))


# ==============================================================================
# 错误内容准备函数
# ==============================================================================

def prepare_error_content(result: PreparedToolCallResult) -> tuple[TextContent, ...]:
    """将 PreparedToolCallResult 中的错误信息转换为 TextContent 元组。

    该函数用于将结构化的工具调用准备错误转换为模型可消费的文本内容格式。

    Args:
        result: 包含错误信息的 PreparedToolCallResult 实例。

    Returns:
        TextContent 元组，包含错误消息和恢复提示。
    """
    # 构建基础错误文本：优先使用 message，其次 error_code，最后使用默认文本
    text = result.message or result.error_code or "Tool call could not be prepared."
    # 如果有恢复提示，追加到错误文本中
    if result.recovery_hint:
        text += f"\nRecovery hint: {result.recovery_hint}"
    # 包装为 TextContent 元组返回
    return (TextContent(text=text),)


# ==============================================================================
# 默认策略上下文工厂函数
# ==============================================================================

def _default_policy_context() -> Any:
    """创建默认的工具策略上下文实例。

    使用延迟导入避免循环依赖问题。

    Returns:
        一个新的 ToolPolicyContext 实例（默认值）。
    """
    from .contracts import ToolPolicyContext

    return ToolPolicyContext()


# ==============================================================================
# 参数处理相关数据类
# ==============================================================================

@dataclass(frozen=True)
class _ParsedArguments:
    """参数解析结果容器。

    不可变 (frozen) 数据类，用于在参数解析阶段传递解析结果或错误。

    Attributes:
        value: 解析后的参数字典。若解析失败，为空字典。
        error: 解析错误消息。若解析成功，为 None。
    """
    value: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class _CoercedArguments:
    """参数强制转换结果容器。

    不可变 (frozen) 数据类，用于在类型强制转换阶段传递转换结果或错误列表。

    Attributes:
        value: 转换后的参数字典。
        errors: 转换过程中的错误消息元组。若全部成功，为空元组。
    """
    value: dict[str, Any]
    errors: tuple[str, ...] = ()


# ==============================================================================
# 参数解析函数
# ==============================================================================

def _parse_arguments(value: dict[str, Any] | str | None) -> _ParsedArguments:
    """解析原始工具调用参数，将其统一转换为 dict 形式。

    支持三种输入形式：
    1. None — 视为空参数，返回空字典。
    2. dict — 直接返回其副本。
    3. str — 作为 JSON 字符串解析，解析结果必须是 JSON 对象。
    4. 其他类型 — 返回错误。

    Args:
        value: 原始参数值，可为 None、dict 或 JSON 字符串。

    Returns:
        _ParsedArguments 实例，包含解析后的字典或错误信息。
    """
    # 情况 1：参数为 None，返回空字典
    if value is None:
        return _ParsedArguments({})

    # 情况 2：参数已是字典，返回其浅拷贝
    if isinstance(value, dict):
        return _ParsedArguments(dict(value))

    # 情况 3：参数为 JSON 字符串
    if isinstance(value, str):
        text = value.strip()
        # 空字符串视为空参数
        if not text:
            return _ParsedArguments({})
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            # JSON 解析失败，返回错误
            return _ParsedArguments({}, f"arguments must be valid JSON: {exc.msg}")
        # 解析结果必须是 JSON 对象（dict），不能是数组或基本类型
        if not isinstance(parsed, dict):
            return _ParsedArguments({}, "arguments must decode to a JSON object")
        return _ParsedArguments(parsed)

    # 情况 4：不支持的参数类型
    return _ParsedArguments({}, "arguments must be an object or JSON object string")


# ==============================================================================
# 参数展开函数
# ==============================================================================

def _unwrap_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """处理被额外包装的参数形式。

    某些模型会将参数以 {"arguments": {...}} 的形式传入（即参数被包裹在
    一个名为 "arguments" 的键中）。本函数检测这种模式并自动展开。

    展开条件：
    1. 参数字典恰好只有一个键 "arguments"。
    2. 该键的值本身也是一个字典。
    3. 工具 schema 的 properties 中不包含名为 "arguments" 的顶级参数。
    4. 嵌套字典的键与 schema 中定义的属性名有交集（说明确实是参数包裹，
       而非工具的某个参数恰巧叫 "arguments"）。

    Args:
        arguments: 原始参数字典（已通过 _parse_arguments 处理）。
        schema: 工具的 JSON Schema 定义。

    Returns:
        展开后的参数字典。若不满足展开条件，则原样返回。
    """
    # 检查是否为 {"arguments": {...}} 的包裹形式
    if set(arguments) != {"arguments"} or not isinstance(arguments.get("arguments"), dict):
        return arguments

    # 获取 schema 中定义的顶层属性
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return arguments

    # 如果 schema 本身就有一个名为 "arguments" 的参数，则不展开
    # （避免把合法参数当作包裹层误拆）
    nested = arguments["arguments"]
    if any(key in properties for key in nested):
        # 嵌套字典的键与 schema 属性名有交集，确认为参数包裹，展开
        return dict(nested)

    # 键与 schema 属性名无交集，保持原样（可能是其他用途的嵌套）
    return arguments


# ==============================================================================
# 参数类型强制转换函数
# ==============================================================================

def _coerce_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> _CoercedArguments:
    """根据工具的 JSON Schema 对参数进行类型强制转换和校验。

    处理流程：
    1. 检查必填 (required) 参数是否全部提供。
    2. 对每个已提供的参数，根据 schema 中的类型定义进行强制转换。
    3. 若 schema 设置了 additionalProperties=false，检查是否有未声明的参数。

    Args:
        arguments: 展开后的参数字典。
        schema: 工具的 JSON Schema 定义。

    Returns:
        _CoercedArguments 实例，包含转换后的参数和错误列表。
    """
    # 若 schema 为空或非 dict，不做任何校验，直接返回
    if not isinstance(schema, dict) or not schema:
        return _CoercedArguments(dict(arguments))

    value = dict(arguments)
    errors: list[str] = []

    # 提取 schema 中的属性定义和必填项列表
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")

    # 检查必填参数是否都已提供
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in value:
                errors.append(f"missing required argument: {name}")

    # 对每个已提供的参数进行类型强制转换
    for name, raw_schema in properties.items():
        if not isinstance(name, str) or name not in value or not isinstance(raw_schema, dict):
            continue
        converted, error = _coerce_value(value[name], raw_schema, name)
        if error is not None:
            errors.append(error)
        else:
            value[name] = converted

    # 若 schema 禁止额外属性，检查是否存在未声明的参数
    additional = schema.get("additionalProperties", True)
    if additional is False:
        unknown = sorted(name for name in value if name not in properties)
        errors.extend(f"unexpected argument: {name}" for name in unknown)

    return _CoercedArguments(value=value, errors=tuple(errors))


# ==============================================================================
# 单个值的类型强制转换函数
# ==============================================================================

def _coerce_value(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, str | None]:
    """对单个参数值进行类型检查和强制转换。

    根据 schema 中声明的类型，尝试将值转换为目标类型。支持的类型包括：
    - integer：整数（bool 类型不会被当作 int，因为 Python 中 bool 是 int 的子类）
    - number：数字（整数或浮点数）
    - boolean：布尔值
    - string：字符串
    - array：数组，支持嵌套元素的递归校验以及 minItems/maxItems 约束
    - object：对象，支持嵌套属性的递归校验

    还支持 enum（枚举值校验）：若 schema 指定了枚举值列表，则值必须是其中之一。

    Args:
        value: 要转换的原始值。
        schema: 该参数的 JSON Schema 片段。
        path: 参数路径，用于生成有意义的错误消息（如 "arg[0].name"）。

    Returns:
        (转换后的值, 错误消息) 元组。若转换成功，错误消息为 None。
    """
    # ---- 枚举值校验 ----
    # 若 schema 定义了 enum 列表，值必须是其中之一
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        return None, f"{path} must be one of {enum_values!r}"

    # 获取期望的类型
    expected = _schema_type(schema)

    # ---- 整数类型转换 ----
    if expected == "integer":
        # Python 中 bool 是 int 的子类，需要显式排除
        if isinstance(value, bool):
            return None, f"{path} must be integer"
        # 已经是 int，直接返回
        if isinstance(value, int):
            return value, None
        # 字符串形式的整数，尝试转换
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value), None
        return None, f"{path} must be integer"

    # ---- 数字类型转换 ----
    if expected == "number":
        # 同样排除 bool 类型
        if isinstance(value, bool):
            return None, f"{path} must be number"
        # int 和 float 都直接接受
        if isinstance(value, (int, float)):
            return value, None
        # 字符串形式的数字，尝试转换为 float
        if isinstance(value, str):
            try:
                return float(value), None
            except ValueError:
                return None, f"{path} must be number"

    # ---- 布尔类型转换 ----
    if expected == "boolean":
        # 已是 bool，直接返回
        if isinstance(value, bool):
            return value, None
        # 字符串形式的布尔值（"true" / "false"，不区分大小写）
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true", None
        return None, f"{path} must be boolean"

    # ---- 字符串类型转换 ----
    if expected == "string":
        # 已是字符串则直接返回，否则调用 str() 转换
        return value if isinstance(value, str) else str(value), None

    # ---- 数组类型转换 ----
    if expected == "array":
        # 必须是 list 类型
        if not isinstance(value, list):
            return None, f"{path} must be array"

        # 获取数组元素的 schema 定义
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            # 对每个数组元素递归调用 _coerce_value 进行校验和转换
            converted_items: list[Any] = []
            errors: list[str] = []
            for index, item in enumerate(value):
                converted, error = _coerce_value(item, item_schema, f"{path}[{index}]")
                if error:
                    errors.append(error)
                else:
                    converted_items.append(converted)
            # 若有元素校验失败，返回所有错误
            if errors:
                return None, "; ".join(errors)
            value = converted_items

        # 检查数组的最小长度约束 (minItems)
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return None, f"{path} must contain at least {min_items} item(s)"

        # 检查数组的最大长度约束 (maxItems)
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            return None, f"{path} must contain at most {max_items} item(s)"

        return value, None

    # ---- 对象类型转换 ----
    if expected == "object":
        # 必须是 dict 类型
        if not isinstance(value, dict):
            return None, f"{path} must be object"
        # 对嵌套对象递归调用 _coerce_arguments 进行校验
        nested = _coerce_arguments(value, schema)
        if nested.errors:
            return None, "; ".join(nested.errors)
        return nested.value, None

    # 无法确定类型或类型不受支持，原样返回
    return value, None


# ==============================================================================
# Schema 类型推断函数
# ==============================================================================

def _schema_type(schema: dict[str, Any]) -> str | None:
    """从 JSON Schema 片段中提取类型声明。

    支持以下 JSON Schema 类型声明方式：
    1. 单一类型字符串：{"type": "string"}
    2. 类型数组（取第一个非 null 类型）：{"type": ["string", "null"]}
    3. 隐式对象类型：若 schema 中含有 properties/required/additionalProperties
       关键字，则推断为 "object" 类型（即使没有显式声明 type）。

    Args:
        schema: JSON Schema 片段字典。

    Returns:
        推断出的类型字符串（如 "string"、"integer"、"object" 等），
        若无法推断则返回 None。
    """
    raw = schema.get("type")

    # 情况 1：type 是单一字符串，直接返回
    if isinstance(raw, str):
        return raw

    # 情况 2：type 是数组，返回第一个非 "null" 的字符串类型
    # （JSON Schema 中 null 通常与另一个类型组合使用，表示可选）
    if isinstance(raw, list):
        return next((item for item in raw if isinstance(item, str) and item != "null"), None)

    # 情况 3：没有显式 type，但有关键字暗示是对象类型
    if any(key in schema for key in ("properties", "required", "additionalProperties")):
        return "object"

    return None


# ==============================================================================
# 工具名称模糊匹配提示函数
# ==============================================================================

def _tool_name_hint(name: str, tools: dict[str, ToolDefinition]) -> str:
    """当工具名称不存在时，生成包含相似名称建议的提示文本。

    使用 difflib.get_close_matches 进行模糊匹配，找出最接近的已注册工具名称。

    Args:
        name: 用户提供的（可能错误/不存在的）工具名称。
        tools: 已注册工具的字典（名称 -> ToolDefinition）。

    Returns:
        提示字符串，如 " Did you mean 'read'? Tool names must match exactly."
        若无相似名称，则返回 " Tool names must match exactly."
    """
    # 在所有已注册工具名称中寻找最接近的匹配
    matches = get_close_matches(name, tools.keys(), n=1)
    if matches:
        return f" Did you mean '{matches[0]}'? Tool names must match exactly."
    return " Tool names must match exactly."


# ==============================================================================
# 内置工具元数据工厂函数
# ==============================================================================

def builtin_metadata(
    name: str,
    *,
    category: str,
    read_only: bool,
    risk_level: str,
    scopes: tuple[str, ...],
    requires_approval: bool = False,
    network_access: bool = False,
    credential_required: bool = False,
    extra: dict[str, Any] | None = None,
) -> ToolMetadata:
    """创建内置工具的 ToolMetadata 实例。

    这是一个工厂函数，为内置工具提供一致的元数据构造方式。
    它封装了 ToolMetadata 的构造细节，并根据 read_only 属性自动推导
    concurrency_safe（并发安全）和 exclusive（排他性）属性。

    Args:
        name: 工具名称。
        category: 工具类别（如 "filesystem"、"search"、"shell" 等）。
        read_only: 是否为只读工具。只读工具被标记为并发安全且非排他。
        risk_level: 风险级别（如 "low"、"medium"、"high"）。
        scopes: 工具适用的作用域元组（如 ("read", "plan", "build")）。
        requires_approval: 是否需要用户审批，默认为 False。
        network_access: 是否需要网络访问，默认为 False。
        credential_required: 是否需要凭证，默认为 False。
        extra: 额外的自定义元数据字典，默认为空字典。

    Returns:
        构造好的 ToolMetadata 实例。
    """
    return ToolMetadata(
        name=name,
        category=category,
        read_only=read_only,
        # 只读工具是并发安全的；可变工具不是
        concurrency_safe=read_only,
        # 只读工具不排他；可变工具是排他的（同一时间只能执行一个）
        exclusive=not read_only,
        requires_approval=requires_approval,
        risk_level=risk_level,  # type: ignore[arg-type]
        scopes=scopes,
        network_access=network_access,
        credential_required=credential_required,
        extra=extra or {},
    )


# ==============================================================================
# 内置工具元数据查询函数
# ==============================================================================

def get_builtin_tool_metadata(name: str) -> ToolMetadata | None:
    """按名称获取内置工具的元数据。

    从预定义的 _BUILTIN_METADATA 字典中查找指定工具的元数据。

    Args:
        name: 工具名称（如 "read"、"write"、"bash" 等）。

    Returns:
        对应的 ToolMetadata 实例，若该名称不是内置工具则返回 None。
    """
    return _BUILTIN_METADATA.get(name)


# ==============================================================================
# 内置工具元数据定义字典
# ==============================================================================
# 该字典为每个 Codepilot 内置工具预定义了元数据，包括：
# - category: 工具类别（filesystem / search / shell / workspace / plan）
# - read_only: 是否只读（决定并发安全和排他性）
# - risk_level: 风险等级（low / medium）
# - scopes: 适用的运行模式作用域
#
# 各内置工具说明：
# - ls:        列出目录内容（只读文件系统操作）
# - read:      读取文件内容（只读文件系统操作）
# - write:     写入文件（可变文件系统操作，有副作用）
# - edit:      编辑文件（可变文件系统操作，有副作用）
# - apply_patch: 应用补丁（可变文件系统操作，有副作用）
# - grep:      内容搜索（只读搜索操作）
# - find:      文件查找（只读搜索操作）
# - bash:      执行 Shell 命令（可变操作，有副作用）
# - workspace_status: 工作区状态查询（只读工作区操作）
# - propose_plan/create_build_plan/update_plan_progress/close_plan: 管理 Task Plan 状态

_BUILTIN_METADATA: dict[str, ToolMetadata] = {
    # ---- 文件系统工具 ----
    "ls": builtin_metadata(
        "ls",
        category="filesystem",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
    "read": builtin_metadata(
        "read",
        category="filesystem",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
    "write": builtin_metadata(
        "write",
        category="filesystem",
        read_only=False,
        risk_level="medium",
        scopes=("build",),
    ),
    "edit": builtin_metadata(
        "edit",
        category="filesystem",
        read_only=False,
        risk_level="medium",
        scopes=("build",),
    ),
    "apply_patch": builtin_metadata(
        "apply_patch",
        category="filesystem",
        read_only=False,
        risk_level="medium",
        scopes=("build",),
    ),

    # ---- 搜索工具 ----
    "grep": builtin_metadata(
        "grep",
        category="search",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),
    "find": builtin_metadata(
        "find",
        category="search",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),

    # ---- Shell 工具 ----
    "bash": builtin_metadata(
        "bash",
        category="shell",
        read_only=False,
        risk_level="medium",
        scopes=("build",),
    ),

    # ---- 工作区状态工具 ----
    "workspace_status": builtin_metadata(
        "workspace_status",
        category="workspace",
        read_only=True,
        risk_level="low",
        scopes=("read", "plan", "build"),
    ),

    # ---- 计划工具 ----
    PROPOSE_PLAN_TOOL: builtin_metadata(
        PROPOSE_PLAN_TOOL,
        category="plan",
        read_only=True,
        risk_level="low",
        scopes=("plan",),
    ),
    CREATE_BUILD_PLAN_TOOL: builtin_metadata(
        CREATE_BUILD_PLAN_TOOL,
        category="plan",
        read_only=True,
        risk_level="low",
        scopes=("build",),
    ),
    UPDATE_PLAN_PROGRESS_TOOL: builtin_metadata(
        UPDATE_PLAN_PROGRESS_TOOL,
        category="plan",
        read_only=True,
        risk_level="low",
        scopes=("build",),
    ),
    CLOSE_PLAN_TOOL: builtin_metadata(
        CLOSE_PLAN_TOOL,
        category="plan",
        read_only=True,
        risk_level="low",
        scopes=("build",),
    ),
}


# ==============================================================================
# 模块公开接口
# ==============================================================================
# __all__ 明确声明了本模块对外暴露的公共 API，包括常量、类和函数。
# 以下划线开头的内部函数（如 _parse_arguments、_coerce_value 等）不在其中。

__all__ = [
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "ToolRegistry",
    "builtin_metadata",
    "get_builtin_tool_metadata",
    "prepare_error_content",
]
