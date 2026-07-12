from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from codepilot.sessions.workspace import file_state_for_path


@dataclass(frozen=True)
class SubagentStore:
    """Process-local registry for exploration subagents."""

    workspace_dir: str | Path
    session_id: str
    _registries: ClassVar[dict[tuple[str, str], dict[str, Any]]] = {}

    def _registry(self) -> dict[str, Any]:
        key = (str(Path(self.workspace_dir).resolve()), self.session_id)
        return self._registries.setdefault(key, {"agents": {}, "reports": {}})

    def list_agents(self, *, query: str | None = None, focus_paths: list[str] | None = None) -> list[dict[str, Any]]:
        query_text = (query or "").strip().lower()
        focus = {Path(path).as_posix() for path in focus_paths or []}
        items: list[dict[str, Any]] = []
        for agent in self._registry()["agents"].values():
            if query_text and query_text not in str(agent).lower():
                continue
            agent_paths = set(agent.get("focus_paths", []))
            if focus and not focus.intersection(agent_paths):
                continue
            item = dict(agent)
            latest = self.latest_report(item["subagent_id"])
            item["latest_report"] = latest
            item["stale"] = self.report_is_stale(latest)
            items.append(item)
        return sorted(items, key=lambda item: item.get("updated_at", ""), reverse=True)

    def ensure_agent(self, *, subagent_id: str, purpose: str, scope_key: str, focus_paths: list[str]) -> dict[str, Any]:
        agents = self._registry()["agents"]
        now = _utc_now_iso()
        current = dict(agents.get(subagent_id, {}))
        profile = {
            **current,
            "subagent_id": subagent_id,
            "purpose": purpose,
            "scope_key": scope_key,
            "focus_paths": sorted(set(current.get("focus_paths", [])) | {Path(path).as_posix() for path in focus_paths}),
            "created_at": current.get("created_at", now),
            "updated_at": now,
        }
        agents[subagent_id] = profile
        return dict(profile)

    def find_by_scope(self, scope_key: str) -> dict[str, Any] | None:
        return next((dict(item) for item in self._registry()["agents"].values() if item.get("scope_key") == scope_key), None)

    def latest_report(self, subagent_id: str) -> dict[str, Any] | None:
        reports = self.reports(subagent_id)
        return reports[-1] if reports else None

    def reports(self, subagent_id: str) -> list[dict[str, Any]]:
        return [dict(item) for item in self._registry()["reports"].get(subagent_id, [])]

    def append_report(
        self,
        *,
        subagent_id: str,
        purpose: str,
        scope_key: str,
        focus_paths: list[str],
        report: dict[str, Any],
        evidence_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        self.ensure_agent(subagent_id=subagent_id, purpose=purpose, scope_key=scope_key, focus_paths=focus_paths)
        payload = {
            **report,
            "report_id": report.get("report_id") or f"subreport_{uuid4().hex[:12]}",
            "subagent_id": subagent_id,
            "status": str(report.get("status") or "completed"),
            "focus_paths": list(focus_paths),
            "evidence_states": [
                file_state_for_path(self.workspace_dir, path)
                for path in (evidence_paths or focus_paths)
            ],
            "created_at": _utc_now_iso(),
        }
        self._registry()["reports"].setdefault(subagent_id, []).append(payload)
        return dict(payload)

    def report_is_stale(self, report: dict[str, Any] | None) -> bool:
        if report is None:
            return False
        for saved in report.get("evidence_states", []):
            path = saved.get("path")
            if not isinstance(path, str):
                continue
            current = file_state_for_path(self.workspace_dir, path)
            if current.get("exists") != saved.get("exists") or current.get("sha256") != saved.get("sha256"):
                return True
        return False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["SubagentStore"]
