"""不透明的规范工具注册目录 —— 管理工具注册信息的增删改查。

本文件是工具注册中心，负责：
1. 接收和验证 ToolRegistration
2. 生成不可变的 registration_id（基于注册内容的哈希）
3. 管理工具注册的生命周期（注册、查询、卸载）
4. 生成工具目录快照（catalog snapshot，给 LLM 选择工具用）
5. 检测过期注册（stale registration，防止模型使用旧信息）
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Iterable

from .codecs import validate_json_schema
from .contracts import ToolRegistration, ToolSpec
from .security import ToolMode, ToolPolicy


class ToolRegistryError(ValueError):
    """工具注册错误 —— Registry 相关异常的基类。"""


class ToolRegistrationConflictError(ToolRegistryError):
    """注册冲突 —— 工具名已存在且 replace=False。"""


class ToolRegistrationNotFoundError(ToolRegistryError):
    """注册未找到 —— 按名称查询时没有对应的注册。"""


class StaleToolRegistrationError(ToolRegistryError):
    """过期注册 —— 客户端使用了旧的 registration_id 调用已变更的工具。"""


@dataclass(frozen=True)
class ToolCatalogEntry:
    """工具目录条目 —— 暴露给 LLM 的工具信息。

    与完整的 ToolRegistration 不同，ToolCatalogEntry 只包含
    LLM 需要知道的信息和运行时需要验证的信息：
    - spec: 工具名称、描述、参数 Schema（给 LLM 看）
    - category: 分类（给计划/执行调度用）
    - source: 来源（决定信任级别）
    - policy: 安全策略（给权限引擎用）
    - registration_id: 唯一标识（用于检测过期）
    - version: 版本号
    - owner: 拥有者

    参数:
        spec: 工具规格（名称、描述、I/O Schema）
        category: 工具分类
        source: 工具来源
        policy: 安全策略
        registration_id: 注册 ID（内容哈希）
        version: 版本号
        owner: 拥有者标识
    """

    spec: ToolSpec
    category: str
    source: str
    policy: ToolPolicy
    registration_id: str
    version: str
    owner: str


@dataclass(frozen=True)
class ToolCatalogSnapshot:
    """工具目录快照 —— 某一时刻所有可用工具的完整视图。

    当 LLM 需要了解可用工具时，Runtime 调用 catalog_snapshot() 生成此快照。
    快照中包含 catalog_id（所有工具的哈希）和完整的条目列表。
    LLM 在选择工具后，会在 ToolExecutionRequest 中携带 registration_id，
    Runtime 用这些 ID 检查快照是否仍然有效（未被更新覆盖）。

    参数:
        catalog_id: 快照的唯一 ID（基于所有注册内容的哈希）
        entries: 所有工具条目的元组（按 category + name 排序）
        created_at_ms: 快照创建时间戳
    """

    catalog_id: str
    entries: tuple[ToolCatalogEntry, ...]
    created_at_ms: int


@dataclass(frozen=True)
class _MaterializedTool:
    """内部使用的物化工具 —— 包含完整的注册信息和生成 ID。

    ToolRegistry 内部使用此类型关联 ToolRegistration 和
    计算出的 registration_id，对外暴露 ToolCatalogEntry。
    """

    registration_id: str
    registration: ToolRegistration


@dataclass
class ToolRegistry:
    """工具注册中心 —— 管理所有工具注册的目录。

    核心功能：
    1. register() / register_batch() — 注册一个或多个工具
    2. extend() — 批量注册（自动遍历 Iterable）
    3. entry() — 查询单个工具的目录条目
    4. catalog_snapshot() — 生成当前快照（按 mode 过滤）
    5. materialize() — 物化（按 name + registration_id 获取完整注册）
    6. unregister() / unregister_owner() — 卸载工具

    注册规则：
    - 同名工具不能重复注册（除非 replace=True）
    - 内置工具（builtin）不能被外部来源替换
    - 保留名称（reserved）只能由内置工具注册

    参数:
        _registrations: 工具名 → _MaterializedTool 的映射
        _reserved_names: 保留名称集合（运行时预留）
        _revision: 修订号（每次变更递增）
    """

    _registrations: dict[str, _MaterializedTool] = field(default_factory=dict)
    _reserved_names: set[str] = field(default_factory=set)
    _revision: int = 0

    def reserve(self, names: Iterable[str]) -> None:
        """预留工具名称 —— 预留给内置工具使用。

        被预留的名称只能由 source="builtin" 的工具注册，
        外部工具不能使用这些名称。

        参数:
            names: 需要预留的名称列表
        """
        values = {str(name).strip() for name in names}
        if any(not name for name in values):
            raise ValueError("reserved tool names cannot be empty")
        self._reserved_names.update(values)

    def register(self, registration: ToolRegistration, *, replace: bool = False) -> str:
        """注册一个工具（便捷方法）。

        参数:
            registration: 工具注册信息
            replace: 是否允许替换已有的同名工具

        返回:
            生成的 registration_id
        """
        if not isinstance(registration, ToolRegistration):
            raise TypeError("ToolRegistry.register expects ToolRegistration")
        return self.register_batch(
            (registration,),
            owner=registration.owner,
            replace=replace,
        )[0]

    def extend(
        self,
        registrations: Iterable[ToolRegistration],
        *,
        replace: bool = False,
    ) -> tuple[str, ...]:
        """扩展注册 —— 批量注册（自动遍历 Iterable）。

        参数:
            registrations: 工具注册的 Iterable
            replace: 是否允许替换

        返回:
            registration_id 的元组
        """
        ids: list[str] = []
        for registration in registrations:
            ids.append(self.register(registration, replace=replace))
        return tuple(ids)

    def register_batch(
        self,
        registrations: Iterable[ToolRegistration],
        *,
        owner: str,
        replace: bool = False,
    ) -> tuple[str, ...]:
        """批量注册工具（核心注册方法）。

        注册流程：
        1. 验证所有参数
        2. 检查是否有重名（同一批内）
        3. 逐个验证每个工具注册
        4. 检查名称冲突（保留名、已存在名）
        5. 生成 registration_id（基于注册内容和修订号）
        6. 将物化后的工具存入字典

        参数:
            registrations: 工具注册的 Iterable
            owner: 这批注册的拥有者（必须与每个注册的 owner 一致）
            replace: 是否允许替换已有的同名工具

        返回:
            registration_id 的元组（顺序与输入一致）

        抛出:
            ToolRegistrationConflictError: 名称冲突时
        """
        items = tuple(registrations)
        if not items:
            return ()
        owner = str(owner).strip()
        if not owner:
            raise ValueError("registration batch owner cannot be empty")
        if any(not isinstance(item, ToolRegistration) for item in items):
            raise TypeError("register_batch expects ToolRegistration values")
        if any(item.owner != owner for item in items):
            raise ValueError("registration owner does not match batch owner")
        names = [item.spec.name for item in items]
        if len(names) != len(set(names)):
            raise ToolRegistrationConflictError("registration batch contains duplicate tool names")

        for registration in items:
            self._validate(registration)
            if (
                registration.spec.name in self._reserved_names
                and registration.source != "builtin"
            ):
                raise ToolRegistrationConflictError(
                    f"Tool name is reserved by the runtime: {registration.spec.name}"
                )
            existing = self._registrations.get(registration.spec.name)
            if existing is not None and not replace:
                raise ToolRegistrationConflictError(
                    f"Tool already registered: {registration.spec.name}"
                )
            if (
                existing is not None
                and existing.registration.source == "builtin"
                and registration.source != "builtin"
            ):
                raise ToolRegistrationConflictError(
                    f"External tool cannot replace builtin: {registration.spec.name}"
                )

        revision = self._revision
        materialized: list[_MaterializedTool] = []
        for registration in items:
            revision += 1
            materialized.append(
                _MaterializedTool(
                    registration_id=_registration_id(registration, revision),
                    registration=registration,
                )
            )
        for item in materialized:
            self._registrations[item.registration.spec.name] = item
        self._revision = revision
        return tuple(item.registration_id for item in materialized)

    def entry(self, name: str) -> ToolCatalogEntry | None:
        """查询单个工具的目录条目。

        参数:
            name: 工具名称

        返回:
            ToolCatalogEntry 或 None（未找到时）
        """
        item = self._registrations.get(name)
        return _entry(item) if item is not None else None

    def catalog_snapshot(self, *, mode: ToolMode | None = None) -> ToolCatalogSnapshot:
        """生成工具目录快照。

        可以按 mode 过滤（如只返回 plan 模式下可用的工具）。
        对条目按 (category, name) 排序以确保快照的确定性。
        catalog_id 由所有条目的 registration_id 哈希生成。

        参数:
            mode: 如果指定，只返回此模式下可用的工具

        返回:
            ToolCatalogSnapshot（包含 catalog_id、条目列表、时间戳）
        """
        entries = tuple(
            sorted(
                (
                    _entry(item)
                    for item in self._registrations.values()
                    if mode is None or mode in item.registration.policy.allowed_modes
                ),
                key=lambda item: (item.category, item.spec.name),
            )
        )
        payload = {
            "revision": self._revision,
            "registrations": [item.registration_id for item in entries],
        }
        catalog_id = "catalog_" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        return ToolCatalogSnapshot(catalog_id, entries, int(time.time() * 1000))

    def materialize(self, name: str, registration_id: str) -> _MaterializedTool:
        """物化工具 —— 按名称和 registration_id 获取完整注册。

        此方法同时做两项检查：
        1. 工具是否存在（name 在目录中）
        2. 工具是否过期（registration_id 是否匹配当前版本）

        如果 registration_id 与当前注册不符，说明客户端使用的
        快照已经过期，需要重新获取。

        参数:
            name: 工具名称
            registration_id: 客户端持有的注册 ID

        返回:
            _MaterializedTool 包含完整的注册信息

        抛出:
            ToolRegistrationNotFoundError: 工具未找到
            StaleToolRegistrationError: 工具注册已过期
        """
        item = self._registrations.get(name)
        if item is None:
            raise ToolRegistrationNotFoundError(f"Tool registration not found: {name}")
        if item.registration_id != registration_id:
            raise StaleToolRegistrationError(
                f"Tool registration is stale: {name} ({registration_id})"
            )
        return item

    def unregister(self, name: str, *, owner: str | None = None) -> bool:
        """卸载指定工具。

        参数:
            name: 要卸载的工具名称
            owner: 如果指定，只有拥有者匹配时才卸载

        返回:
            True 表示成功卸载，False 表示未找到
        """
        item = self._registrations.get(name)
        if item is None or (owner is not None and item.registration.owner != owner):
            return False
        del self._registrations[name]
        self._revision += 1
        return True

    def unregister_owner(self, owner: str) -> int:
        """卸载指定拥有者的所有工具。

        参数:
            owner: 拥有者标识

        返回:
            卸载的工具数量
        """
        names = [
            name
            for name, item in self._registrations.items()
            if item.registration.owner == owner
        ]
        for name in names:
            del self._registrations[name]
        if names:
            self._revision += 1
        return len(names)

    @staticmethod
    def _validate(registration: ToolRegistration) -> None:
        """验证注册的一致性 —— 确保 input_schema 与 input_codec 的 Schema 一致。

        检查规则：
        1. input_codec 必须声明 Schema（不能是 None）
        2. input_codec 的 Schema 必须与 spec.input_schema 完全一致
        3. output_codec 如果是 UnverifiedJsonCodec（Schema=None），
           则 spec.output_schema 也必须是 None
        4. output_codec 如果有 Schema，必须与 spec.output_schema 完全一致

        参数:
            registration: 待验证的工具注册

        抛出:
            ToolRegistryError: 一致性验证失败
        """
        input_schema = registration.input_codec.json_schema
        if input_schema is None:
            raise ToolRegistryError("Canonical tool input codec must declare a schema")
        if _json(input_schema) != _json(validate_json_schema(registration.spec.input_schema, require_object=True)):
            raise ToolRegistryError("ToolSpec input schema does not match input codec schema")
        output_schema = registration.output_codec.json_schema
        if output_schema is None:
            if registration.spec.output_schema is not None:
                raise ToolRegistryError("Unverified output codec requires output_schema=None")
        elif registration.spec.output_schema is None or _json(output_schema) != _json(
            validate_json_schema(registration.spec.output_schema)
        ):
            raise ToolRegistryError("ToolSpec output schema does not match output codec schema")


def _entry(item: _MaterializedTool) -> ToolCatalogEntry:
    """将物化工具转为目录条目（对外暴露的安全视图）。"""
    registration = item.registration
    return ToolCatalogEntry(
        spec=registration.spec,
        category=registration.category,
        source=registration.source,
        policy=registration.policy,
        registration_id=item.registration_id,
        version=registration.version,
        owner=registration.owner,
    )


def _registration_id(registration: ToolRegistration, revision: int) -> str:
    """生成工具注册 ID —— 基于注册内容 + 修订号的 SHA256 哈希。

    registration_id 是对工具"身份"的完整摘要：
    名称、描述、Schema、版本、策略等全部被包含在哈希中。
    任何字段的变更都会产生不同的 ID。

    这实现了 "content-addressable registration" 模式：
    客户端通过 registration_id 引用的工具不会被静默更新。
    """
    policy = registration.policy
    payload = {
        "name": registration.spec.name,
        "description": registration.spec.description,
        "input_schema": _json(registration.spec.input_schema),
        "output_schema": _json(registration.spec.output_schema),
        "schema_version": registration.spec.schema_version,
        "version": registration.version,
        "implementation_version": registration.implementation_version,
        "category": registration.category,
        "source": registration.source,
        "owner": registration.owner,
        "policy": {
            "allowed_modes": sorted(policy.allowed_modes),
            "declared_effects": sorted(policy.declared_effects),
            "required_permissions": sorted(policy.required_permissions),
            "base_risk": policy.base_risk,
            "approval": policy.approval,
            "timeout": {
                "default_execution_ms": policy.timeout.default_execution_ms,
                "max_execution_ms": policy.timeout.max_execution_ms,
                "cleanup_grace_ms": policy.timeout.cleanup_grace_ms,
            },
            "concurrency": {
                "mode": policy.concurrency.mode,
                "group": policy.concurrency.group,
                "max_parallel": policy.concurrency.max_parallel,
            },
            "output_trust": policy.output_trust.default_content_trust,
            "allow_structurally_validated": policy.output_trust.allow_structurally_validated,
            "output_limits": {
                "max_data_bytes": policy.output_limits.max_data_bytes,
                "max_content_bytes": policy.output_limits.max_content_bytes,
                "max_artifacts": policy.output_limits.max_artifacts,
                "max_artifact_bytes": policy.output_limits.max_artifact_bytes,
            },
        },
        "revision": revision,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"reg_{digest[:28]}"


def _json(value):
    """将值转为 JSON 兼容的纯 Python 结构。"""
    if value is None:
        return None
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


__all__ = [
    "StaleToolRegistrationError",
    "ToolCatalogEntry",
    "ToolCatalogSnapshot",
    "ToolRegistrationConflictError",
    "ToolRegistrationNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
]