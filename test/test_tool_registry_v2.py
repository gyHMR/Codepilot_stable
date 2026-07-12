from __future__ import annotations

from dataclasses import dataclass

import pytest


def _policy(*, effects=frozenset({"filesystem_read"})):
    from codepilot.tools.security import (
        ConcurrencyPolicy,
        OutputLimits,
        OutputTrustPolicy,
        TimeoutPolicy,
        ToolPolicy,
    )

    return ToolPolicy(
        allowed_modes=frozenset({"plan", "execute"}),
        declared_effects=effects,
        required_permissions=frozenset({"workspace.read"}),
        base_risk="low",
        approval="never",
        timeout=TimeoutPolicy(default_execution_ms=5_000, max_execution_ms=10_000),
        concurrency=ConcurrencyPolicy(mode="parallel"),
        output_limits=OutputLimits(),
        output_trust=OutputTrustPolicy(),
    )


def _registration(*, version: str = "1.0.0"):
    from codepilot.tools.codecs import JsonObjectCodec
    from codepilot.tools.contracts import ToolRegistration, ToolSpec
    from codepilot.tools.security import ToolAccessRequest, ToolAccessResolution, ToolResource

    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    async def handler(input, context):
        _ = context
        return {"text": input["path"]}

    class Renderer:
        def render(self, data):
            return ()

    class Resolver:
        def resolve(self, input, context):
            _ = context
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=("read",),
                    resources=(ToolResource(f"workspace:///{input['path']}"),),
                    effects=frozenset({"filesystem_read"}),
                    risk="low",
                    reason="Read workspace file",
                ),
            )

    return ToolRegistration(
        version=version,
        implementation_version="1",
        spec=ToolSpec(
            name="read_v2",
            description="Read a UTF-8 workspace file.",
            input_schema=input_schema,
            output_schema=output_schema,
        ),
        category="filesystem",
        source="builtin",
        owner="codepilot.builtin",
        policy=_policy(),
        input_codec=JsonObjectCodec(input_schema),
        output_codec=JsonObjectCodec(output_schema),
        handler=handler,
        renderer=Renderer(),
        access_resolver=Resolver(),
    )


def test_canonical_codecs_validate_and_round_trip_values() -> None:
    from codepilot.tools.codecs import (
        DataclassCodec,
        JsonObjectCodec,
        ToolCodecError,
        UnverifiedJsonCodec,
    )

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "count": {"type": "integer", "minimum": 1},
        },
        "required": ["path", "count"],
        "additionalProperties": False,
    }
    codec = JsonObjectCodec(schema)

    assert codec.decode({"path": "src/app.py", "count": 2}) == {
        "path": "src/app.py",
        "count": 2,
    }
    with pytest.raises(ToolCodecError, match="count"):
        codec.decode({"path": "src/app.py", "count": "2"})

    @dataclass(frozen=True)
    class Payload:
        path: str
        count: int

    dataclass_codec = DataclassCodec(Payload, schema)
    value = dataclass_codec.decode({"path": "src/app.py", "count": 3})

    assert value == Payload(path="src/app.py", count=3)
    assert dataclass_codec.encode(value) == {"path": "src/app.py", "count": 3}

    unverified = UnverifiedJsonCodec(max_bytes=128)
    assert unverified.json_schema is None
    assert unverified.encode({"items": [1, True, None]}) == {"items": [1, True, None]}
    with pytest.raises(ToolCodecError, match="JSON-safe"):
        unverified.encode({"bad": {1, 2}})


def test_registry_exposes_opaque_snapshots_and_rejects_stale_identity() -> None:
    from codepilot.tools import ToolCatalogSnapshot as PublicToolCatalogSnapshot
    from codepilot.tools.registry import (
        StaleToolRegistrationError,
        ToolRegistrationConflictError,
        ToolRegistry,
    )

    registry = ToolRegistry()
    first = _registration()
    first_id = registry.register(first)

    entry = registry.entry("read_v2")
    snapshot = registry.catalog_snapshot(mode="execute")

    assert entry is not None
    assert entry.registration_id == first_id
    assert not hasattr(entry, "handler")
    assert snapshot.entries == (entry,)
    assert snapshot.catalog_id
    assert isinstance(snapshot, PublicToolCatalogSnapshot)

    with pytest.raises(ToolRegistrationConflictError, match="already registered"):
        registry.register(_registration())

    second_id = registry.register(_registration(version="1.0.1"), replace=True)
    assert second_id != first_id

    with pytest.raises(StaleToolRegistrationError, match="stale"):
        registry.materialize("read_v2", first_id)

    current = registry.materialize("read_v2", second_id)
    assert current.registration_id == second_id
    assert current.registration.spec.name == "read_v2"


def test_registration_identity_tracks_description_policy_and_implementation() -> None:
    from dataclasses import replace

    from codepilot.tools.registry import ToolRegistry
    from codepilot.tools.security import OutputLimits

    registry = ToolRegistry()
    registration = _registration()
    baseline = registry.register(registration)
    described = registry.register(
        replace(
            registration,
            spec=replace(
                registration.spec,
                description="Read one UTF-8 workspace file with canonical validation.",
            ),
        ),
        replace=True,
    )
    policy_changed = registry.register(
        replace(
            registration,
            policy=replace(
                registration.policy,
                output_limits=OutputLimits(max_data_bytes=128_000),
            ),
        ),
        replace=True,
    )
    implementation_changed = registry.register(
        replace(registration, implementation_version="2"),
        replace=True,
    )

    assert len({baseline, described, policy_changed, implementation_changed}) == 4
