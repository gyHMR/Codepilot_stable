from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkspaceSandbox:
    """Resolve paths while enforcing the workspace boundary."""

    workspace_dir: str | Path

    @property
    def root(self) -> Path:
        return Path(self.workspace_dir).resolve()

    def resolve_path(self, path_text: str | Path) -> Path:
        path = Path(path_text)
        target = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        self.ensure_within_workspace(target)
        return target

    def ensure_within_workspace(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path escapes workspace boundary") from exc
        return target


def file_state_for_path(workspace_dir: str | Path, path: str | Path) -> dict[str, Any]:
    sandbox = WorkspaceSandbox(workspace_dir)
    target = sandbox.resolve_path(path)
    relative = target.relative_to(sandbox.root).as_posix()
    if not target.exists() or not target.is_file():
        return {
            "path": relative,
            "exists": False,
            "workspace_path": str(sandbox.root),
        }
    stat = target.stat()
    return {
        "path": relative,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(target),
        "workspace_path": str(sandbox.root),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
