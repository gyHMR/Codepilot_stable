from __future__ import annotations

"""In-process Runtime execution lifecycle; never persisted by Sessions."""

from dataclasses import dataclass

from .contracts import RuntimeExecutionState, TerminalOutcome


_TRANSITIONS: dict[RuntimeExecutionState, frozenset[RuntimeExecutionState]] = {
    "new": frozenset({"preparing", "cancelling"}),
    "preparing": frozenset({"executing", "resuming", "cancelling", "finalizing"}),
    "executing": frozenset({"waiting", "cancelling", "finalizing"}),
    "waiting": frozenset({"resuming", "cancelling", "released"}),
    "resuming": frozenset({"executing", "cancelling", "finalizing"}),
    "cancelling": frozenset({"finalizing"}),
    "finalizing": frozenset({"terminal"}),
    "terminal": frozenset({"released"}),
    "released": frozenset(),
}


@dataclass
class RuntimeLifecycle:
    run_id: str
    state: RuntimeExecutionState = "new"
    terminal_outcome: TerminalOutcome | None = None

    def __post_init__(self) -> None:
        self.run_id = _required_text(self.run_id, "run_id")
        if self.state != "terminal" and self.terminal_outcome is not None:
            raise ValueError("Only terminal state can carry terminal_outcome")
        if self.state == "terminal" and self.terminal_outcome is None:
            raise ValueError("Terminal state requires terminal_outcome")

    def transition(
        self,
        target: RuntimeExecutionState,
        *,
        terminal_outcome: TerminalOutcome | None = None,
    ) -> None:
        if self.state == "released" and target == "released":
            return
        if target not in _TRANSITIONS[self.state]:
            raise ValueError(f"Invalid Runtime transition: {self.state} -> {target}")
        if target == "terminal":
            if terminal_outcome not in {"completed", "failed", "cancelled"}:
                raise ValueError("Terminal transition requires terminal_outcome")
            self.terminal_outcome = terminal_outcome
        elif terminal_outcome is not None:
            raise ValueError("terminal_outcome is only valid for terminal transition")
        self.state = target

    def mark_released(self) -> None:
        self.transition("released")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


__all__ = ["RuntimeLifecycle"]
