from __future__ import annotations

"""Restricted tool port views used by nested read-only runners."""

from dataclasses import dataclass, replace
from typing import Any

from codepilot.protocols import TextContent, Tool

from .contracts import (
    ToolCatalogItem,
    ToolCatalogView,
    ToolInvocation,
    ToolObservation,
    ToolPort,
    ToolResumeDecision,
)


DEFAULT_READ_ONLY_TOOL_NAMES = frozenset(
    {
        "ls",
        "read",
        "grep",
        "find",
        "workspace_status",
    }
)


@dataclass
class RestrictedToolPort(ToolPort):
    """Expose a small allowlist over another ToolPort."""

    base: ToolPort | None
    allowed_names: frozenset[str] = DEFAULT_READ_ONLY_TOOL_NAMES
    forced_mode: str = "read"

    def catalog(self, current_mode: str = "read") -> ToolCatalogView:
        if self.base is None:
            return ToolCatalogView()
        try:
            catalog = self.base.catalog(self.forced_mode or current_mode)
        except TypeError:
            catalog = self.base.catalog()  # type: ignore[call-arg]
        return ToolCatalogView(tuple(self._allowed_catalog_items(catalog)))

    async def execute(self, invocation: ToolInvocation) -> ToolObservation:
        if self.base is None:
            return _denied_observation(invocation, "restricted_base_missing")
        if invocation.name not in self.allowed_names:
            return _denied_observation(invocation, "restricted_tool_denied")
        restricted = replace(invocation, current_mode=self.forced_mode)
        observation = await self.base.execute(restricted)
        if observation.workspace_changed:
            return _denied_observation(invocation, "restricted_workspace_changed")
        return observation

    async def resume(self, decision: ToolResumeDecision) -> ToolObservation:
        return ToolObservation(
            tool_call_id="",
            name="approval_resume",
            status="denied",
            content=(TextContent(text="Restricted tool port does not support approval resume."),),
            metadata={
                "approval_id": decision.approval_id,
                "error_code": "restricted_resume_denied",
            },
        )

    def _allowed_catalog_items(self, catalog: Any) -> list[ToolCatalogItem]:
        items: list[ToolCatalogItem] = []
        if isinstance(catalog, ToolCatalogView):
            for item in catalog.items:
                if item.spec.name in self.allowed_names and item.metadata.read_only:
                    items.append(item)
            return items
        for spec in _tool_specs(catalog):
            if spec.name not in self.allowed_names:
                continue
            items.append(
                ToolCatalogItem(
                    spec=spec,
                    metadata=_synthetic_read_metadata(spec.name),
                )
            )
        return items


def _tool_specs(catalog: Any) -> list[Tool]:
    if isinstance(catalog, dict):
        value = catalog.get("tools")
        if isinstance(value, (list, tuple)):
            return [_as_tool(item) for item in value]
        return [_as_tool(catalog)]
    if isinstance(catalog, (list, tuple)):
        return [_as_tool(item) for item in catalog]
    return [_as_tool(catalog)]


def _as_tool(item: Any) -> Tool:
    if isinstance(item, Tool):
        return item
    if isinstance(item, str):
        return Tool(name=item, description=item, parameters={})
    if hasattr(item, "to_spec"):
        spec = item.to_spec()
        if isinstance(spec, Tool):
            return spec
    if isinstance(item, dict):
        name = str(item.get("name") or item.get("id") or "")
        return Tool(
            name=name,
            description=str(item.get("description") or name),
            parameters=dict(item.get("parameters") or item.get("input_schema") or {}),
        )
    name = str(getattr(item, "name", ""))
    return Tool(
        name=name,
        description=str(getattr(item, "description", "") or name),
        parameters=dict(getattr(item, "parameters", {}) or {}),
    )


def _synthetic_read_metadata(name: str) -> Any:
    from .contracts import ToolMetadata

    return ToolMetadata(
        name=name,
        category="restricted",
        read_only=True,
        concurrency_safe=True,
        exclusive=False,
        requires_approval=False,
        risk_level="low",
        scopes=("read", "plan", "build"),
    )


def _denied_observation(invocation: ToolInvocation, error_code: str) -> ToolObservation:
    return ToolObservation(
        tool_call_id=invocation.tool_call_id,
        name=invocation.name,
        status="denied",
        content=(TextContent(text=f"Tool '{invocation.name}' is not allowed in read-only subagent runs."),),
        metadata={"error_code": error_code},
    )


__all__ = ["DEFAULT_READ_ONLY_TOOL_NAMES", "RestrictedToolPort"]
