from __future__ import annotations

"""Session persistence boundary.

The storage package has one job: keep session facts on disk in predictable
files.  Runtime code should use ``SessionStore`` and ``RunStore`` instead of
building paths or serializing messages by hand.
"""

from .layout import SessionLayout
from .repository import (
    GitInfo,
    RepositoryBootstrap,
    build_repository_bootstrap,
    render_repository_context,
)
from .run_store import FreshnessResult, FreshnessStatus, RunStore
from .serde import message_from_dict, message_to_dict
from .session_store import SessionOpenMetadata, SessionStore, load_session_open_metadata, new_session_id

__all__ = [
    "FreshnessResult",
    "FreshnessStatus",
    "GitInfo",
    "RepositoryBootstrap",
    "RunStore",
    "SessionLayout",
    "SessionOpenMetadata",
    "SessionStore",
    "build_repository_bootstrap",
    "load_session_open_metadata",
    "message_from_dict",
    "message_to_dict",
    "new_session_id",
    "render_repository_context",
]
