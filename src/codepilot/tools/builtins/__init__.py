from __future__ import annotations

"""Canonical builtin registration composition."""

from pathlib import Path

from ..contracts import ToolRegistration
from ..sandbox import ShellExecutionPolicy, WorkspaceSandbox
from .files import create_file_registrations
from .search import create_search_registrations
from .shell import create_command_registration, create_shell_registration
from .workspace import create_workspace_status_registration


def create_builtin_registrations(
    workspace_dir: str | Path,
    enabled_names: list[str] | None = None,
    *,
    edit_require_unique_match: bool = True,
    shell_policy: ShellExecutionPolicy | None = None,
) -> list[ToolRegistration]:
    sandbox = WorkspaceSandbox(Path(workspace_dir))
    enabled = set(enabled_names) if enabled_names else None

    def allow(name: str) -> bool:
        return enabled is None or name in enabled

    registrations: list[ToolRegistration] = []
    registrations.extend(
        create_file_registrations(
            sandbox,
            allow=allow,
            edit_require_unique_match=edit_require_unique_match,
        )
    )
    registrations.extend(create_search_registrations(sandbox, allow=allow))
    if allow("workspace_status"):
        registrations.append(create_workspace_status_registration(sandbox))
    if allow("command"):
        registrations.append(create_command_registration(sandbox, policy=shell_policy))
    if allow("bash"):
        registrations.append(create_shell_registration(sandbox, policy=shell_policy))
    return registrations


__all__ = [
    "create_builtin_registrations",
    "create_file_registrations",
    "create_search_registrations",
    "create_command_registration",
    "create_shell_registration",
    "create_workspace_status_registration",
]
