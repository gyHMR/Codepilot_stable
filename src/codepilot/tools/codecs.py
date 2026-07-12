"""基于 JSON Schema Draft 2020-12 的规范输入/输出编解码器。

本文件提供三种编解码器和一个 Schema 验证函数：

编解码器层次：
1. JsonObjectCodec     — 严格的 JSON Schema 验证编解码器（核心基础）
2. DataclassCodec      — 在 JsonObjectCodec 之上做 dataclass 类型转换（最常用）
3. UnverifiedJsonCodec — 不要求 Schema 的宽松编解码器（用于外部/MCP 输出）

典型使用：
    write_codec = DataclassCodec(WriteInput, write_json_schema)
    decoded = write_codec.decode({"path": "foo.py", "content": "..."})
    # decoded 是 WriteInput(path="foo.py", content="...")
"""

import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Generic, Mapping, TypeVar, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class ToolCodecError(ValueError):
    """编解码错误 —— 工具值无法被其声明的编解码器解码或编码。

    在输入验证失败（如缺少必需字段、类型错误、额外的属性）时抛出。
    """


class ToolSchemaError(ValueError):
    """Schema 错误 —— 工具编解码器用无效的 JSON Schema 构造时抛出。

    在 Schema 本身不符合 Draft 2020-12 规范时抛出（不是值验证失败）。
    """


class JsonObjectCodec:
    """严格的 JSON 对象编解码器 —— 使用已验证的 Draft 2020-12 Schema。

    这是最基础的编解码器，对输入/输出进行严格的 JSON Schema 验证：
    - 输入必须是 JSON 对象（dict）
    - 必须符合声明的 JSON Schema
    - 输出也必须符合相同的 Schema

    使用 Draft202012Validator 进行验证，确保：
    - 字段类型正确（string / integer / boolean 等）
    - 必需字段都被提供了
    - 没有额外的未声明属性（additionalProperties: false 时）

    参数:
        schema: JSON Schema（Draft 2020-12），必须声明 type=object
    """

    def __init__(self, schema: Mapping[str, object]) -> None:
        """初始化编解码器，验证并保存 Schema。

        处理流程:
        1. 深拷贝 schema 防止外部篡改
        2. 调用 validate_json_schema 校验 schema 合法性
        3. 创建 Draft202012Validator 实例用于后续值验证
        """
        self._schema = _mapping_copy(schema, "schema")
        validate_json_schema(self._schema, require_object=True)
        self._validator = Draft202012Validator(self._schema)

    @property
    def json_schema(self) -> Mapping[str, object]:
        """返回 Schema 的深拷贝（防止外部篡改内部 Schema）。"""
        return deepcopy(self._schema)

    def decode(self, value: object) -> dict[str, object]:
        """解码（验证）输入值 —— 将外部 JSON 对象解码为 Python dict。

        参数:
            value: 待验证的 JSON 对象（通常来自 LLM 的参数）

        返回:
            验证通过后的 dict（经过 JSON Schema 校验）

        抛出:
            ToolCodecError: 值不符合 Schema 要求时
        """
        return self._validate(value)

    def encode(self, value: object) -> dict[str, object]:
        """编码（验证）输出值 —— 将 Python dict 编码回 JSON 对象。

        注意：encode 和 decode 做同样的验证流程，
        因为输出也需要符合相同的 Schema。

        参数:
            value: 待验证的 Python dict（通常来自工具处理器返回值）

        返回:
            验证通过后的 dict

        抛出:
            ToolCodecError: 值不符合 Schema 要求时
        """
        return self._validate(value)

    def _validate(self, value: object) -> dict[str, object]:
        """核心验证方法 —— 检查值是否符合 Schema。

        验证步骤:
        1. 检查是否为 Mapping 类型（dict 或其子类）
        2. 深拷贝并转为普通 dict
        3. 使用 Draft202012Validator 迭代所有错误
        4. 如果有错误，格式化为人类可读的消息后抛出

        错误格式化示例:
            "path: 'some_field' is a required property"
        """
        if not isinstance(value, Mapping):
            raise ToolCodecError("Tool value must be a JSON object")
        copied = _mapping_copy(value, "tool value")
        errors = sorted(self._validator.iter_errors(copied), key=lambda item: list(item.path))
        if errors:
            first = errors[0]
            path = ".".join(str(item) for item in first.absolute_path)
            prefix = f"{path}: " if path else ""
            raise ToolCodecError(prefix + first.message)
        return copied


TDataclass = TypeVar("TDataclass")


class DataclassCodec(Generic[TDataclass]):
    """Dataclass 编解码器 —— 在 JsonObjectCodec 验证后做类型转换。

    这是最常用的编解码器类型。它在 JSON Schema 验证后，
    将 dict 自动转换为指定的 dataclass 类型：
    - decode: JSON dict → 类型化的 dataclass 实例
    - encode: dataclass 实例 → JSON dict（通过 asdict）

    这使得工具处理器可以直接操作类型化的 Python 对象，
    而不需要手动从 dict 中提取字段。

    类型参数:
        TDataclass: 目标 dataclass 类型（如 WriteInput、ReadInput）

    参数:
        dataclass_type: 目标 dataclass 类型
        schema: JSON Schema（传递给内部的 JsonObjectCodec）
    """

    def __init__(self, dataclass_type: type[TDataclass], schema: Mapping[str, object]) -> None:
        """初始化 dataclass 编解码器。

        处理流程:
        1. 验证 dataclass_type 确实是 dataclass
        2. 创建内部的 JsonObjectCodec 用于 Schema 验证
        """
        if not is_dataclass(dataclass_type):
            raise TypeError("dataclass_type must be a dataclass")
        self._type = dataclass_type
        self._object_codec = JsonObjectCodec(schema)

    @property
    def json_schema(self) -> Mapping[str, object]:
        """返回底层 JSON Schema。"""
        return self._object_codec.json_schema

    def decode(self, value: object) -> TDataclass:
        """解码：JSON dict → 类型化的 dataclass 实例。

        两步处理:
        1. 先用 JsonObjectCodec 验证 JSON Schema
        2. 用验证后的 dict 构造 dataclass 实例

        参数:
            value: 来自 LLM 的 JSON 参数

        返回:
            指定类型的 dataclass 实例（如 WriteInput(path="...", content="...")）

        抛出:
            ToolCodecError: Schema 验证失败或 dataclass 构造失败
        """
        decoded = self._object_codec.decode(value)
        try:
            return self._type(**decoded)
        except TypeError as exc:
            raise ToolCodecError(f"Cannot construct {self._type.__name__}: {exc}") from exc

    def encode(self, value: TDataclass) -> dict[str, object]:
        """编码：dataclass 实例 → JSON dict。

        两步处理:
        1. 用 asdict 将 dataclass 转为 dict
        2. 用 JsonObjectCodec 验证符合 Schema

        参数:
            value: 工具处理器返回的 dataclass 实例

        返回:
            编码后的 JSON dict

        抛出:
            ToolCodecError: 类型不匹配或 Schema 验证失败
        """
        if not isinstance(value, self._type):
            raise ToolCodecError(f"Expected {self._type.__name__} output")
        return self._object_codec.encode(cast(Mapping[str, object], asdict(value)))


class UnverifiedJsonCodec:
    """未验证的 JSON 编解码器 —— 用于不声明 Schema 的外部输出。

    这个编解码器不做 JSON Schema 验证，只做基本的结构安全检查：
    - 最大嵌套深度（防止深度攻击）
    - 最大序列化字节数（防止体积攻击）
    - JSON 可序列化性检查

    主要用于 MCP 工具输出等外部源的响应，
    因为这些输出可能没有可用的 Schema 定义。

    参数:
        max_bytes: 序列化后的最大字节数（默认 1MB）
        max_depth: JSON 最大嵌套深度（默认 32）
    """

    def __init__(self, *, max_bytes: int = 1_000_000, max_depth: int = 32) -> None:
        """初始化未验证编解码器，设置安全限制。"""
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        self._max_bytes = max_bytes
        self._max_depth = max_depth

    @property
    def json_schema(self) -> None:
        """返回 None —— 表示此编解码器不需要 Schema 验证。"""
        return None

    def decode(self, value: object) -> object:
        """解码：检查 JSON 安全性和体积限制后返回。

        参数:
            value: 任意 JSON 值

        返回:
            验证通过的值（经过序列化-反序列化循环，确保 JSON 安全性）
        """
        return self._validate(value)

    def encode(self, value: object) -> object:
        """编码：同 decode，做同样的安全检查。

        参数:
            value: 任意 JSON 值

        返回:
            验证通过的值
        """
        return self._validate(value)

    def _validate(self, value: object) -> object:
        """核心验证方法：嵌套深度 + 序列化体积检查。

        流程:
        1. 检查 JSON 嵌套深度是否超过限制
        2. 尝试序列化为 JSON 字符串（检查 JSON 兼容性）
        3. 检查序列化后的字节数是否超过限制
        4. 重新解析以返回干净的 JSON 对象
        """
        if _json_depth(value) > self._max_depth:
            raise ToolCodecError(f"JSON value exceeds maximum depth {self._max_depth}")
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ToolCodecError("Tool value must be JSON-safe") from exc
        if len(encoded.encode("utf-8")) > self._max_bytes:
            raise ToolCodecError(f"JSON value exceeds maximum size {self._max_bytes} bytes")
        return json.loads(encoded)


# ── 模块级辅助函数 ────────────────────────────────────────────────────────────


def _mapping_copy(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    """深拷贝一个映射并确保键都是字符串。

    将任意 Mapping 转为普通的 dict[str, object]，
    非字符串键会被转换为字符串。
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): _json_copy(item) for key, item in value.items()}


def validate_json_schema(
    schema: Mapping[str, object],
    *,
    require_object: bool = False,
) -> dict[str, object]:
    """验证 JSON Schema 的合法性，返回防御性深拷贝。

    用于确保注册时提供的 JSON Schema 符合 Draft 2020-12 规范。
    在编解码器构造时和注册验证时调用。

    参数:
        schema: 待验证的 JSON Schema
        require_object: 如果为 True，要求 schema 声明 type=object

    返回:
        验证通过并深拷贝后的 schema dict

    抛出:
        ToolSchemaError: Schema 不合法时
    """
    copied = _mapping_copy(schema, "schema")
    if require_object and copied.get("type") != "object":
        raise ToolSchemaError("JSON object codec schema must declare type=object")
    try:
        Draft202012Validator.check_schema(copied)
    except SchemaError as exc:
        raise ToolSchemaError(f"Invalid JSON Schema: {exc.message}") from exc
    return copied


def _json_copy(value: object) -> object:
    """递归深拷贝一个 JSON 兼容的值。

    将 dict → dict（字符串键）、list/tuple → list，
    其他值 → deepcopy。确保结果只包含 JSON 安全的类型。
    """
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    return deepcopy(value)


def json_value(value: object) -> object:
    """返回防御性的纯 JSON 投影，用于 Schema 和元数据。

    对外暴露的 JSON 值提取函数，确保返回值只包含 JSON 安全类型。
    """
    return _json_copy(value)


def _json_depth(value: object) -> int:
    """递归计算 JSON 值的最大嵌套深度。

    用于 UnverifiedJsonCodec 的深度攻击防护。
    dict 和 list 的深度为其子项最大深度 + 1，
    标量值的深度为 0。
    """
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


__all__ = [
    "DataclassCodec",
    "JsonObjectCodec",
    "ToolCodecError",
    "ToolSchemaError",
    "UnverifiedJsonCodec",
    "validate_json_schema",
    "json_value",
]