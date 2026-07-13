"""Runtime boundary for interface actions and agent execution."""

from .gateway import RuntimeGateway
from .opening import SessionOpenIntent
from .environment import RunEnvironment, RunEnvironmentFactory, RunResourceScope
from .executor import RunExecutor
from .coordinator import RunCoordinator
from .contracts import CommitReceipt, RunCommitIdentity, RuntimeExecutionState, TerminalOutcome
from .lifecycle import RuntimeLifecycle

__all__ = [
    "RuntimeGateway",
    "SessionOpenIntent",
    "RunEnvironment",
    "RunEnvironmentFactory",
    "RunResourceScope",
    "RunExecutor",
    "RunCoordinator",
    "CommitReceipt",
    "RunCommitIdentity",
    "RuntimeExecutionState",
    "RuntimeLifecycle",
    "TerminalOutcome",
]
