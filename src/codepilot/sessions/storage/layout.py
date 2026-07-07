from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionLayout:
    """Resolve session-owned file paths from one small contract."""

    workspace_dir: Path
    session_id: str

    @classmethod
    def for_workspace(cls, workspace_dir: str | Path, session_id: str) -> "SessionLayout":
        return cls(Path(workspace_dir), session_id)

    @property
    def codepilot_dir(self) -> Path:
        return self.workspace_dir / ".codepilot"

    @property
    def session_dir(self) -> Path:
        return self.codepilot_dir / "sessions" / self.session_id

    @property
    def session_file(self) -> Path:
        return self.session_dir / "session.json"

    @property
    def messages_file(self) -> Path:
        return self.session_dir / "messages.jsonl"

    @property
    def session_events_file(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def task_state_file(self) -> Path:
        return self.session_dir / "task_state.json"

    @property
    def context_ledger_file(self) -> Path:
        return self.session_dir / "context_ledger.jsonl"

    @property
    def tool_outputs_dir(self) -> Path:
        return self.session_dir / "artifacts" / "tool_outputs"

    @property
    def project_memory_file(self) -> Path:
        return self.codepilot_dir / "memory" / "memories.jsonl"

    def run_dir(self, run_id: str) -> Path:
        return self.codepilot_dir / "runs" / run_id

    def run_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def run_events_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"


__all__ = ["SessionLayout"]
