from __future__ import annotations

"""Canonical registrations for Runtime-owned exploration subagents."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path

from codepilot.tools.codecs import JsonObjectCodec
from codepilot.tools.contracts import (
    ToolExecutionRequest,
    ToolBatchPreparation,
    ToolExecutionPort,
    ToolHandlerError,
    ToolRegistration,
    ToolSpec,
)
from codepilot.tools.registry import ToolCatalogSnapshot
from codepilot.tools.results import TextContent, ToolError, ToolResult
from codepilot.tools.security import (
    ConcurrencyPolicy,
    OutputLimits,
    OutputTrustPolicy,
    TimeoutPolicy,
    ToolAccessRequest,
    ToolAccessResolution,
    ToolEffect,
    ToolPolicy,
    ToolResource,
)

from .runner import (
    DEFAULT_READ_ONLY_TOOL_NAMES,
    DISPATCH_EXPLORATION_TOOL,
    LIST_EXPLORATION_AGENTS_TOOL,
    MAX_TASKS_PER_BATCH,
    ExplorationCoordinator,
    SubagentStore,
)


_MUTATING_EFFECTS = frozenset(
    {"filesystem_write", "filesystem_delete", "external_state_write", "session_state_write"}
)


@dataclass
class RestrictedToolPort:
    """Read-only ToolExecutionPort used by exploration subagents."""

    base: ToolExecutionPort | None
    allowed_names: frozenset[str] = DEFAULT_READ_ONLY_TOOL_NAMES
    forced_mode: str = "plan"
    _prepared_batches: dict[str, tuple[ToolExecutionRequest, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _prepared_sequence: int = field(default=0, init=False, repr=False)

    def catalog_snapshot(self, *, mode=None) -> ToolCatalogSnapshot:
        if self.base is None:
            return ToolCatalogSnapshot("catalog_restricted_empty", (), 0)
        snapshot = self.base.catalog_snapshot(mode=self.forced_mode)
        entries = tuple(
            entry
            for entry in snapshot.entries
            if entry.spec.name in self.allowed_names
            and not (entry.policy.declared_effects & _MUTATING_EFFECTS)
        )
        payload = {
            "base_catalog_id": snapshot.catalog_id,
            "registrations": [entry.registration_id for entry in entries],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        return ToolCatalogSnapshot(
            catalog_id=f"catalog_restricted_{digest}",
            entries=entries,
            created_at_ms=snapshot.created_at_ms,
        )

    def pending_challenges(self) -> tuple[object, ...]:
        return ()

    async def execute(self, request: ToolExecutionRequest) -> ToolResult:
        if self.base is None:
            return _restricted_denied(request, "restricted_base_missing")
        if request.tool_name not in self.allowed_names:
            return _restricted_denied(request, "restricted_tool_denied")
        allowed = {
            entry.spec.name: entry.registration_id
            for entry in self.catalog_snapshot().entries
        }
        if allowed.get(request.tool_name) != request.registration_id:
            return _restricted_denied(request, "restricted_registration_denied")
        preparation = self.base.prepare_batch(
            (replace(request, mode=self.forced_mode),)
        )
        if preparation.results:
            result = preparation.results[0]
        else:
            results = await self.base.execute_prepared(preparation.batch_id or "")
            if len(results) != 1:
                raise RuntimeError("Restricted Tool execution returned an invalid batch")
            result = results[0]
        if any(effect.kind in _MUTATING_EFFECTS for effect in result.effects):
            return _restricted_denied(
                request,
                "restricted_mutating_effect",
                effects=result.effects,
            )
        return result

    async def execute_batch(
        self,
        requests: tuple[ToolExecutionRequest, ...] | list[ToolExecutionRequest],
    ) -> list[ToolResult]:
        return [await self.execute(request) for request in requests]

    def prepare_batch(
        self,
        requests: tuple[ToolExecutionRequest, ...] | list[ToolExecutionRequest],
    ) -> ToolBatchPreparation:
        items = tuple(requests)
        if not items:
            raise ValueError("RestrictedToolPort.prepare_batch requires requests")
        if any(not isinstance(item, ToolExecutionRequest) for item in items):
            raise TypeError("RestrictedToolPort expects ToolExecutionRequest values")
        self._prepared_sequence += 1
        batch_id = f"restricted_batch:{self._prepared_sequence}"
        self._prepared_batches[batch_id] = items
        return ToolBatchPreparation(batch_id=batch_id)

    async def execute_prepared(self, batch_id: str) -> tuple[ToolResult, ...]:
        try:
            requests = self._prepared_batches.pop(batch_id)
        except KeyError as exc:
            raise ValueError(f"Restricted prepared batch not found: {batch_id}") from exc
        return tuple(await self.execute_batch(requests))

def _restricted_denied(
    request: ToolExecutionRequest,
    error_code: str,
    *,
    effects=(),
) -> ToolResult:
    message = f"Tool '{request.tool_name}' is not allowed in read-only subagent runs."
    return ToolResult(
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        status="denied",
        content=(TextContent(message),),
        error=ToolError(code=error_code, kind="permission", message=message),
        effects=tuple(effects),
        registration_id=request.registration_id,
    )


def create_subagent_registrations(
    *,
    workspace: Path,
    session_provider: Callable[[], object],
) -> list[ToolRegistration]:
    workspace = Path(workspace).resolve()
    return [
        _list_registration(workspace, session_provider),
        _dispatch_registration(workspace, session_provider),
    ]


def _list_registration(
    workspace: Path,
    session_provider: Callable[[], object],
) -> ToolRegistration:
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "focus_paths": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "agents": {"type": "array", "items": {"type": "object"}},
            "has_reports": {"type": "boolean"},
            "next_action": {"type": "string"},
        },
        "required": ["agents", "has_reports"],
        "additionalProperties": False,
    }

    async def handler(input, context):
        context.cancellation.raise_if_cancelled()
        session = session_provider()
        session_id = str(getattr(session, "session_id"))
        agents = SubagentStore(workspace, session_id).list_agents(
            query=_optional_text(input.get("query")),
            focus_paths=_string_list(input.get("focus_paths")),
        )
        result: dict[str, object] = {"agents": agents, "has_reports": bool(agents)}
        if not agents:
            result["next_action"] = "Call dispatch_exploration to create read-only exploration subagents."
        context.effects.report(
            ToolEffect(
                kind="session_state_read",
                resource=ToolResource(f"session://{session_id}/subagents"),
                operation="list exploration reports",
                status="completed",
                certainty="observed",
            )
        )
        return result

    return _registration(
        name=LIST_EXPLORATION_AGENTS_TOOL,
        description=(
            "Plan mode only. Inspect reports already produced by dispatch_exploration. This tool does "
            "not create or run subagents; use it to filter or compare existing exploration evidence."
        ),
        input_schema=input_schema,
        output_schema=output_schema,
        effects=frozenset({"session_state_read"}),
        handler=handler,
        renderer=_ListRenderer(),
    )


def _dispatch_registration(
    workspace: Path,
    session_provider: Callable[[], object],
) -> ToolRegistration:
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_TASKS_PER_BATCH,
                "items": {
                    "type": "object",
                    "properties": {
                        "subagent_id": {"type": "string"},
                        "purpose": {"type": "string", "minLength": 1},
                        "instruction": {"type": "string", "minLength": 1},
                        "focus_paths": {"type": "array", "items": {"type": "string"}},
                        "expected_output": {
                            "type": "string",
                            "enum": ["architecture", "flow", "open", "risk", "tests"],
                        },
                        "critical": {"type": "boolean"},
                    },
                    "required": ["purpose", "instruction"],
                    "additionalProperties": False,
                },
            },
            "max_parallel": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TASKS_PER_BATCH,
            },
            "reuse": {
                "type": "string",
                "enum": ["auto", "force_refresh", "no_reuse"],
            },
        },
        "required": ["tasks"],
        "additionalProperties": False,
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "batch_status": {"type": "string"},
            "reports": {"type": "array", "items": {"type": "object"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "created_subagent_ids": {"type": "array", "items": {"type": "string"}},
            "reused_subagent_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "batch_status",
            "reports",
            "warnings",
            "created_subagent_ids",
            "reused_subagent_ids",
        ],
        "additionalProperties": False,
    }

    async def handler(input, context):
        session = session_provider()
        session_id = str(getattr(session, "session_id"))
        context.cancellation.raise_if_cancelled()
        await context.progress.report(
            "subagent_dispatch_started",
            data={"task_count": len(input.get("tasks", []))},
        )
        coordinator = ExplorationCoordinator(
            workspace=workspace,
            session_id=session_id,
            model=session.controller.model,
            model_port=session.model_port,
            tool_port=RestrictedToolPort(session.tool_port),
            store=SubagentStore(workspace, session_id),
        )
        try:
            result = await coordinator.dispatch(dict(input))
        except ValueError as exc:
            raise ToolHandlerError("subagent.invalid_request", str(exc)) from exc
        context.cancellation.raise_if_cancelled()
        resource = ToolResource(f"session://{session_id}/subagents")
        context.effects.report(
            ToolEffect(
                kind="session_state_read",
                resource=resource,
                operation="read exploration reports",
                status="completed",
                certainty="observed",
            )
        )
        if any(not report.get("reused") for report in result.get("reports", [])):
            context.effects.report(
                ToolEffect(
                    kind="session_state_write",
                    resource=resource,
                    operation="store exploration reports",
                    status="completed",
                    certainty="observed",
                )
            )
        await context.progress.report(
            "subagent_dispatch_completed",
            data={"batch_status": result.get("batch_status", "unknown")},
        )
        return result

    return _registration(
        name=DISPATCH_EXPLORATION_TOOL,
        description=(
            "Plan mode only. Dispatch the minimum necessary number of read-only exploration subagents "
            "for independent repository questions. Use distinct scopes, prefer reuse=auto, and integrate "
            "the returned evidence before proposing the final Task Plan."
        ),
        input_schema=input_schema,
        output_schema=output_schema,
        effects=frozenset({"session_state_read", "session_state_write"}),
        handler=handler,
        renderer=_DispatchRenderer(),
    )


def _registration(
    *,
    name: str,
    description: str,
    input_schema: dict[str, object],
    output_schema: dict[str, object],
    effects: frozenset[str],
    handler,
    renderer,
) -> ToolRegistration:
    input_codec = JsonObjectCodec(input_schema)
    output_codec = JsonObjectCodec(output_schema)

    class Resolver:
        def resolve(self, input, request):
            resource = ToolResource(f"session://{request.session_id}/subagents")
            return ToolAccessResolution(
                input=input,
                access=ToolAccessRequest(
                    actions=(name,),
                    resources=(resource,),
                    effects=effects,
                    risk="low",
                    reason=f"Run Runtime-owned subagent operation {name}",
                ),
            )

    return ToolRegistration(
        version="1.0.0",
        implementation_version="1",
        spec=ToolSpec(name, description, input_schema, output_schema),
        category="delegation",
        source="builtin",
        owner="runtime.subagents",
        policy=ToolPolicy(
            allowed_modes=frozenset({"plan"}),
            declared_effects=effects,
            required_permissions=frozenset(),
            base_risk="low",
            approval="never",
            timeout=TimeoutPolicy(90_000, 120_000, cleanup_grace_ms=5_000),
            concurrency=ConcurrencyPolicy(mode="parallel"),
            output_limits=OutputLimits(max_data_bytes=512_000, max_content_bytes=32_000),
            output_trust=OutputTrustPolicy(),
        ),
        input_codec=input_codec,
        output_codec=output_codec,
        handler=handler,
        renderer=renderer,
        access_resolver=Resolver(),
    )


class _ListRenderer:
    def render(self, data):
        return (
            TextContent(
                text=(
                    f"Found {len(data['agents'])} stored exploration report(s)."
                    if data["has_reports"]
                    else str(data.get("next_action", "No exploration reports found."))
                )
            ),
        )


class _DispatchRenderer:
    def render(self, data):
        summary = {
            "batch_status": data["batch_status"],
            "report_count": len(data["reports"]),
            "warnings": data["warnings"],
        }
        return (TextContent(text=json.dumps(summary, ensure_ascii=False, sort_keys=True)),)


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _optional_text(item)) is not None]


__all__ = ["RestrictedToolPort", "create_subagent_registrations"]
