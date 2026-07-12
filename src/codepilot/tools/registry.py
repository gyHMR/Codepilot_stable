from __future__ import annotations

"""Opaque canonical tool registration catalog."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Iterable

from .codecs import validate_json_schema
from .contracts import ToolRegistration, ToolSpec
from .security import ToolMode, ToolPolicy


class ToolRegistryError(ValueError):
    pass


class ToolRegistrationConflictError(ToolRegistryError):
    pass


class ToolRegistrationNotFoundError(ToolRegistryError):
    pass


class StaleToolRegistrationError(ToolRegistryError):
    pass


@dataclass(frozen=True)
class ToolCatalogEntry:
    spec: ToolSpec
    category: str
    source: str
    policy: ToolPolicy
    registration_id: str
    version: str
    owner: str


@dataclass(frozen=True)
class ToolCatalogSnapshot:
    catalog_id: str
    entries: tuple[ToolCatalogEntry, ...]
    created_at_ms: int


@dataclass(frozen=True)
class _MaterializedTool:
    registration_id: str
    registration: ToolRegistration


@dataclass
class ToolRegistry:
    _registrations: dict[str, _MaterializedTool] = field(default_factory=dict)
    _revision: int = 0

    def register(self, registration: ToolRegistration, *, replace: bool = False) -> str:
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
        item = self._registrations.get(name)
        return _entry(item) if item is not None else None

    def catalog_snapshot(self, *, mode: ToolMode | None = None) -> ToolCatalogSnapshot:
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
        item = self._registrations.get(name)
        if item is None:
            raise ToolRegistrationNotFoundError(f"Tool registration not found: {name}")
        if item.registration_id != registration_id:
            raise StaleToolRegistrationError(
                f"Tool registration is stale: {name} ({registration_id})"
            )
        return item

    def unregister(self, name: str, *, owner: str | None = None) -> bool:
        item = self._registrations.get(name)
        if item is None or (owner is not None and item.registration.owner != owner):
            return False
        del self._registrations[name]
        self._revision += 1
        return True

    def unregister_owner(self, owner: str) -> int:
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
                "idle_timeout_ms": policy.timeout.idle_timeout_ms,
                "cleanup_grace_ms": policy.timeout.cleanup_grace_ms,
            },
            "concurrency": {
                "mode": policy.concurrency.mode,
                "group": policy.concurrency.group,
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
