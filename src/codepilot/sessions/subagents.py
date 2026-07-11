from __future__ import annotations

"""Session-scoped storage for plan-mode exploration subagents."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .store import SessionLayout
from .workspace_state import file_state_for_path


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SubagentStore:
    """Persist compact exploration reports under one session directory."""

    workspace_dir: str | Path
    session_id: str

    @property
    def root(self) -> Path:
        return SessionLayout.for_workspace(self.workspace_dir, self.session_id).session_dir / "subagents"

    @property
    def index_file(self) -> Path:
        return self.root / "index.json"

    def list_agents(
        self,
        *,
        query: str | None = None,
        focus_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        index = self._read_index()
        agents = list(index.get("agents", {}).values())
        query_text = _clean_text(query).lower()
        focus = {_normalize_path(path) for path in focus_paths or [] if _clean_text(path)}
        result: list[dict[str, Any]] = []
        for agent in agents:
            if query_text and query_text not in json.dumps(agent, ensure_ascii=False).lower():
                continue
            agent_focus = {_normalize_path(path) for path in agent.get("focus_paths", [])}
            if focus and not _paths_overlap(focus, agent_focus):
                continue
            item = dict(agent)
            latest = self.latest_report(str(agent.get("subagent_id") or ""))
            item["latest_report"] = latest
            item["stale"] = self.report_is_stale(latest) if latest is not None else False
            result.append(item)
        return sorted(result, key=lambda item: str(item.get("updated_at") or ""), reverse=True)

    def ensure_agent(
        self,
        *,
        subagent_id: str,
        purpose: str,
        scope_key: str,
        focus_paths: list[str],
    ) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        index = self._read_index()
        agents = dict(index.get("agents", {}))
        now = _utc_now_iso()
        existing = agents.get(subagent_id)
        if isinstance(existing, dict):
            profile = {
                **existing,
                "purpose": purpose or existing.get("purpose", ""),
                "scope_key": scope_key or existing.get("scope_key", ""),
                "focus_paths": _unique_paths([*existing.get("focus_paths", []), *focus_paths]),
                "updated_at": now,
            }
        else:
            profile = {
                "subagent_id": subagent_id,
                "purpose": purpose,
                "scope_key": scope_key,
                "focus_paths": _unique_paths(focus_paths),
                "created_at": now,
                "updated_at": now,
                "last_status": "",
                "last_report_id": None,
                "last_summary": "",
            }
        agents[subagent_id] = profile
        self._write_index({**index, "agents": agents, "updated_at": now})
        self._write_profile(subagent_id, profile)
        return dict(profile)

    def find_by_scope(self, scope_key: str) -> dict[str, Any] | None:
        key = _clean_text(scope_key)
        if not key:
            return None
        for agent in self._read_index().get("agents", {}).values():
            if isinstance(agent, dict) and agent.get("scope_key") == key:
                return dict(agent)
        return None

    def latest_report(self, subagent_id: str) -> dict[str, Any] | None:
        rows = self.reports(subagent_id)
        return rows[-1] if rows else None

    def reports(self, subagent_id: str) -> list[dict[str, Any]]:
        path = self._reports_file(subagent_id)
        if not path.exists():
            return []
        return _read_jsonl(path)

    def append_report(
        self,
        *,
        subagent_id: str,
        purpose: str,
        scope_key: str,
        focus_paths: list[str],
        report: dict[str, Any],
        evidence_paths: list[str],
    ) -> dict[str, Any]:
        profile = self.ensure_agent(
            subagent_id=subagent_id,
            purpose=purpose,
            scope_key=scope_key,
            focus_paths=focus_paths,
        )
        report_id = f"report_{uuid4().hex[:12]}"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "report_id": report_id,
            "subagent_id": subagent_id,
            "purpose": purpose,
            "scope_key": scope_key,
            "focus_paths": _unique_paths(focus_paths),
            "created_at": _utc_now_iso(),
            **dict(report),
            "evidence_states": self._evidence_states(evidence_paths),
        }
        _append_jsonl(self._reports_file(subagent_id), payload)
        self._update_agent_summary(
            subagent_id,
            {
                **profile,
                "last_status": str(payload.get("status") or ""),
                "last_report_id": report_id,
                "last_summary": str(payload.get("summary") or ""),
                "updated_at": payload["created_at"],
            },
        )
        return payload

    def report_is_stale(self, report: dict[str, Any] | None) -> bool:
        if not isinstance(report, dict):
            return False
        states = report.get("evidence_states")
        if not isinstance(states, list):
            return False
        for state in states:
            if not isinstance(state, dict):
                continue
            path = state.get("path")
            if not isinstance(path, str) or not path:
                continue
            try:
                current = file_state_for_path(self.workspace_dir, path)
            except ValueError:
                return True
            if _state_changed(state, current):
                return True
        return False

    def _update_agent_summary(self, subagent_id: str, profile: dict[str, Any]) -> None:
        index = self._read_index()
        agents = dict(index.get("agents", {}))
        agents[subagent_id] = profile
        self._write_index({**index, "agents": agents, "updated_at": _utc_now_iso()})
        self._write_profile(subagent_id, profile)

    def _evidence_states(self, paths: list[str]) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for path in _unique_paths(paths):
            try:
                states.append(file_state_for_path(self.workspace_dir, path))
            except ValueError:
                continue
        return states

    def _read_index(self) -> dict[str, Any]:
        payload = _read_json(self.index_file)
        if payload is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "agents": {},
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
        agents = payload.get("agents")
        if not isinstance(agents, dict):
            payload["agents"] = {}
        return payload

    def _write_index(self, payload: dict[str, Any]) -> None:
        payload["schema_version"] = SCHEMA_VERSION
        payload["session_id"] = self.session_id
        _write_json(self.index_file, payload)

    def _write_profile(self, subagent_id: str, profile: dict[str, Any]) -> None:
        _write_json(self._agent_dir(subagent_id) / "profile.json", profile)

    def _agent_dir(self, subagent_id: str) -> Path:
        return self.root / subagent_id

    def _reports_file(self, subagent_id: str) -> Path:
        return self._agent_dir(subagent_id) / "reports.jsonl"


def _state_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    for key in ("exists", "size", "mtime_ns", "sha256"):
        if previous.get(key) != current.get(key):
            return True
    return False


def _paths_overlap(first: set[str], second: set[str]) -> bool:
    for left in first:
        for right in second:
            if left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/"):
                return True
    return False


def _unique_paths(paths: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        text = _normalize_path(path)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _normalize_path(path: Any) -> str:
    return _clean_text(path).replace("\\", "/").strip("/")


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["SubagentStore"]
