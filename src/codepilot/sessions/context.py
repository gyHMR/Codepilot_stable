from __future__ import annotations

"""Per-turn context projection for the model."""

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from codepilot.core.contracts import AgentContext, ContextPreparationRequest, PreparedAgentContext
from codepilot.llm.estimation import (
    ContextUsageCalibrator,
    estimate_context,
    estimate_context_tokens,
    estimate_text_tokens,
    estimate_tools_tokens,
)
from codepilot.protocols import (
    ContextArtifactRef,
    ContextFreshness,
    ContextItem,
    ContextPressure,
    ContextReport,
    ContextSectionReport,
    ContextTrust,
    ContextView,
    DroppedContextItem,
    Message,
    RepositoryDelta,
    RepositorySnapshot,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.core.plan import MAX_PLAN_ITEMS
from codepilot.sessions.memory import MemoryQuery, MemoryRecall, render_memory
from codepilot.sessions.plan_state import PlanStateStore
from codepilot.sessions.store import SessionStore, build_repository_bootstrap
from codepilot.sessions.workspace_state import file_state_for_path


ContextFileRole = Literal["target", "test", "dependency", "config", "reference"]
ContextEvidenceKind = Literal["tool_result", "verification", "observation"]

_LAYER_ORDER = ("system", "task_plan", "working_set", "memory", "conversation")
_LAYER_BUDGET_RATIOS = {
    "system": 0.10,
    "task_plan": 0.10,
    "working_set": 0.32,
    "memory": 0.08,
    "conversation": 0.40,
}
_KEEP_COUNTS = {
    "normal": {"task_plan": 32, "working_set": 18, "memory": 5, "conversation": 20},
    "tight": {"task_plan": 32, "working_set": 12, "memory": 4, "conversation": 12},
    "critical": {"task_plan": 32, "working_set": 8, "memory": 2, "conversation": 8},
}
_CONTEXT_FILE_ROLES = {"target", "test", "dependency", "config", "reference"}
_CONTEXT_EVIDENCE_KINDS = {"tool_result", "verification", "observation"}
_CONTEXT_FRESHNESS_VALUES = {"fresh", "stale", "missing", "unknown"}
_CONTEXT_TRUST_VALUES = {"observed", "derived", "user_given", "model_claim"}
_ACTIVE_FILE_ROLE_PRIORITY = {
    "target": 5,
    "test": 4,
    "dependency": 3,
    "config": 2,
    "reference": 1,
}
_ACTIVE_FILE_FRESHNESS_PRIORITY = {"fresh": 3, "unknown": 2, "stale": 1, "missing": 0}
_INTERNAL_DIR_NAMES = {".git", ".codepilot", ".pytest_cache", "__pycache__"}
_LONG_TOOL_OUTPUT_CHARS = 4000


@dataclass
class ActiveFile:
    path: str
    role: ContextFileRole
    reason: str
    source_hash: str | None = None
    access_count: int = 1
    last_accessed_at: float = field(default_factory=time.time)
    freshness: ContextFreshness = "unknown"

    def __post_init__(self) -> None:
        _ensure_file_role(self.role)
        _ensure_context_freshness(self.freshness)


@dataclass
class FileSummary:
    path: str
    summary: str
    source_hash: str
    relevant_symbols: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    freshness: ContextFreshness = "fresh"

    def __post_init__(self) -> None:
        _ensure_context_freshness(self.freshness)


@dataclass
class ContextEvidence:
    kind: ContextEvidenceKind
    content: str
    trust: ContextTrust
    source: str
    source_hash: str | None = None
    workspace_fingerprint: str | None = None
    freshness: ContextFreshness = "unknown"
    path: str | None = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _ensure_context_evidence_kind(self.kind)
        _ensure_context_trust(self.trust)
        _ensure_context_freshness(self.freshness)


@dataclass
class SessionContextState:
    """Small in-memory ledger of working files and evidence."""

    workspace_dir: Path
    active_files: dict[str, ActiveFile] = field(default_factory=dict)
    max_active_files: int = 40
    file_summaries: dict[str, FileSummary] = field(default_factory=dict)
    evidence: list[ContextEvidence] = field(default_factory=list)
    last_repository_snapshot: RepositorySnapshot | None = None
    observed_tool_call_ids: set[str] = field(default_factory=set)

    def observe_tool_result(
        self,
        message: ToolResultMessage,
        *,
        repository_fingerprint: str | None = None,
    ) -> None:
        if message.tool_call_id and message.tool_call_id in self.observed_tool_call_ids:
            return
        if message.tool_call_id:
            self.observed_tool_call_ids.add(message.tool_call_id)

        details = message.details if isinstance(message.details, dict) else {}
        state = details.get("file_state")
        if not isinstance(state, dict):
            state = message.metadata.get("file_state")
        path = state.get("path") if isinstance(state, dict) else None
        source_hash = state.get("sha256") if isinstance(state, dict) else None
        read_paths = _metadata_paths(message.metadata.get("read_paths"))

        paths = [str(item) for item in message.affected_paths]
        if _is_successful_read_result(message):
            paths.extend(path for path in read_paths if path not in paths)
        if isinstance(path, str) and path not in paths:
            paths.append(path)

        role: ContextFileRole = (
            "target"
            if message.workspace_changed or _is_successful_read_result(message)
            else "reference"
        )
        for item in paths:
            self.touch_file(
                item,
                role=role,
                reason=f"{message.tool_name} tool result",
                source_hash=source_hash if item == path else None,
            )

        if message.workspace_changed:
            self.invalidate_paths(paths)
            self.invalidate_verification()

        text = _tool_result_text(message)
        if text:
            self.evidence.append(
                ContextEvidence(
                    kind="tool_result",
                    content=text,
                    trust="observed",
                    source=message.tool_name,
                    source_hash=source_hash if isinstance(source_hash, str) else None,
                    workspace_fingerprint=repository_fingerprint,
                    freshness="fresh",
                    path=path if isinstance(path, str) else None,
                )
            )
            self.evidence = self.evidence[-80:]

        if message.verification:
            self.evidence.append(
                ContextEvidence(
                    kind="verification",
                    content=str(message.verification),
                    trust="observed",
                    source=message.tool_name,
                    workspace_fingerprint=repository_fingerprint,
                    freshness="fresh",
                )
            )
            self.evidence = self.evidence[-80:]

    def touch_file(
        self,
        path: str,
        *,
        role: ContextFileRole,
        reason: str,
        source_hash: str | None = None,
    ) -> None:
        role = _ensure_file_role(role)
        normalized = Path(path).as_posix()
        current = self.active_files.get(normalized)
        if current is None:
            self.active_files[normalized] = ActiveFile(
                path=normalized,
                role=role,
                reason=reason,
                source_hash=source_hash,
                freshness="fresh" if source_hash else "unknown",
            )
            self._prune_active_files()
            return

        current.access_count += 1
        current.last_accessed_at = time.time()
        current.reason = reason
        if role == "target" or current.role == "reference":
            current.role = role
        if source_hash:
            current.source_hash = source_hash
            current.freshness = "fresh"
        self._prune_active_files()

    def invalidate_paths(self, paths: list[str]) -> None:
        for path in paths:
            normalized = Path(path).as_posix()
            summary = self.file_summaries.get(normalized)
            if summary is not None:
                summary.freshness = "stale"
            for evidence in self.evidence:
                if evidence.path == normalized:
                    evidence.freshness = "stale"
            active = self.active_files.get(normalized)
            if active is not None:
                active.freshness = "stale"

    def invalidate_verification(self) -> None:
        for evidence in self.evidence:
            if evidence.kind == "verification":
                evidence.freshness = "stale"

    def validate_sources(self, repository_fingerprint: str) -> list[str]:
        stale: list[str] = []
        for path, summary in list(self.file_summaries.items()):
            state = file_state_for_path(self.workspace_dir, path)
            if not state.get("exists"):
                summary.freshness = "missing"
            elif state.get("sha256") != summary.source_hash:
                summary.freshness = "stale"
            else:
                summary.freshness = "fresh"
            if summary.freshness != "fresh":
                stale.append(f"file_summary:{path}:{summary.freshness}")

        for path, active in list(self.active_files.items()):
            state = file_state_for_path(self.workspace_dir, path)
            if not state.get("exists"):
                active.freshness = "missing"
            elif active.source_hash and state.get("sha256") != active.source_hash:
                active.freshness = "stale"
            elif active.source_hash:
                active.freshness = "fresh"
            else:
                active.freshness = "unknown"
            if active.freshness in {"stale", "missing"}:
                stale.append(f"active_file:{path}:{active.freshness}")

        for evidence in self.evidence:
            if (
                evidence.kind == "verification"
                and evidence.workspace_fingerprint
                and evidence.workspace_fingerprint != repository_fingerprint
            ):
                evidence.freshness = "stale"
            if evidence.freshness in {"stale", "missing"}:
                stale.append(f"evidence:{evidence.source}:{evidence.freshness}")
        return stale

    def _prune_active_files(self) -> None:
        if self.max_active_files <= 0:
            self.active_files.clear()
            return
        if len(self.active_files) <= self.max_active_files:
            return
        ranked = sorted(
            self.active_files.items(),
            key=lambda item: _active_file_rank(item[0], item[1]),
            reverse=True,
        )
        self.active_files = dict(ranked[: self.max_active_files])


@dataclass(frozen=True)
class ContextPressurePolicy:
    safety_margin_tokens: int = 1024
    tight_ratio: float = 0.72
    critical_ratio: float = 0.90
    tool_output_ratio: float = 0.25

    def evaluate(
        self,
        request: ContextPreparationRequest,
        *,
        estimated_tokens: int,
        tool_output_tokens: int,
        history_tokens: int,
    ) -> ContextPressure:
        effective_budget = max(
            128,
            request.model_context_window
            - request.model_max_output_tokens
            - self.safety_margin_tokens,
        )
        pressure_ratio = estimated_tokens / effective_budget if effective_budget > 0 else 1.0
        reasons: list[str] = []
        if tool_output_tokens >= int(effective_budget * self.tool_output_ratio):
            reasons.append("tool_output_pressure")
        history_pressure = history_tokens >= int(effective_budget * 0.50)
        if history_pressure:
            reasons.append("history_pressure")
        if pressure_ratio >= self.critical_ratio:
            level = "critical"
            reasons.append("critical_budget_pressure")
        elif pressure_ratio >= self.tight_ratio or history_pressure:
            level = "tight"
            if pressure_ratio >= self.tight_ratio:
                reasons.append("tight_budget_pressure")
        else:
            level = "normal"
        return ContextPressure(
            level=level,
            effective_budget=effective_budget,
            estimated_tokens=max(0, estimated_tokens),
            reasons=_dedupe(reasons),
        )


class RepositoryTracker:
    """Low-cost repository snapshot before each model call."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()

    def snapshot(self) -> RepositorySnapshot:
        bootstrap = build_repository_bootstrap(self.workspace)
        git_status = _git_lines(self.workspace, ["status", "--porcelain"])
        dirty_path_hashes = _dirty_path_hashes(self.workspace, git_status)
        instruction_hashes = {
            path: _sha256(self.workspace / path)
            for path in bootstrap.instruction_files
            if (self.workspace / path).is_file()
        }
        payload = [
            bootstrap.workspace_root,
            bootstrap.project_type or "",
            *bootstrap.manifest_files,
            *bootstrap.top_level_entries,
            *bootstrap.test_directories,
            *(f"{key}:{value}" for key, value in sorted(instruction_hashes.items())),
            *(f"{key}:{value}" for key, value in sorted(dirty_path_hashes.items())),
            bootstrap.git.branch if bootstrap.git and bootstrap.git.branch else "",
            bootstrap.git.head_sha if bootstrap.git and bootstrap.git.head_sha else "",
            *git_status,
        ]
        fingerprint = hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()
        return RepositorySnapshot(
            workspace_root=bootstrap.workspace_root,
            project_type=bootstrap.project_type,
            manifest_files=list(bootstrap.manifest_files),
            top_level_entries=list(bootstrap.top_level_entries),
            test_directories=list(bootstrap.test_directories),
            instruction_files=list(bootstrap.instruction_files),
            branch=bootstrap.git.branch if bootstrap.git else None,
            head_sha=bootstrap.git.head_sha if bootstrap.git else None,
            git_status=git_status,
            fingerprint=fingerprint,
            instruction_hashes=instruction_hashes,
            dirty_path_hashes=dirty_path_hashes,
        )

    def refresh(
        self,
        previous: RepositorySnapshot | None,
    ) -> tuple[RepositorySnapshot, RepositoryDelta]:
        current = self.snapshot()
        return current, compare_snapshots(previous, current)


class ContextGovernor:
    """Prepare one model-visible context with a readable linear flow."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        session_id: str,
        state: SessionContextState | None = None,
        memory_retriever: Any | None = None,
        pressure_policy: ContextPressurePolicy | None = None,
        plan_state_store: PlanStateStore | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.store = store or SessionStore(self.workspace_dir, self.session_id)
        self.state = state or SessionContextState(workspace_dir=self.workspace_dir)
        self.memory_retriever = memory_retriever
        self.pressure_policy = pressure_policy or ContextPressurePolicy()
        self.plan_state_store = plan_state_store or PlanStateStore(self.store)
        self.repository = RepositoryTracker(self.workspace_dir)
        self.tool_ledger = ToolArtifactLedger(store=self.store)
        self.calibrator = ContextUsageCalibrator(self.workspace_dir)

    async def prepare(
        self,
        context: AgentContext,
        request: ContextPreparationRequest,
    ) -> PreparedAgentContext:
        provider, model = _request_provider_model(request)
        correction_factors = self.calibrator.factors_for(provider, model)

        snapshot, delta = self.repository.refresh(self.state.last_repository_snapshot)
        self.state.last_repository_snapshot = snapshot
        if delta.changed:
            self.state.invalidate_paths([*delta.modified_paths, *delta.deleted_paths])
            self.state.invalidate_verification()

        artifact_refs = self._observe_tools(context, request, snapshot)
        stale_items = self.state.validate_sources(snapshot.fingerprint)
        plan_state = _plan_state_from_context(context, self.plan_state_store)
        run_signals = _run_signals_from_context(context)
        visible_messages = _without_published_plan_summaries(
            context.messages,
            plan_state,
        )

        raw_estimate = estimate_context(
            visible_messages,
            context.system_prompt,
            context.tools,
            correction_factors=correction_factors,
        )
        history_tokens = estimate_context_tokens(
            visible_messages,
            "",
            correction_factors=correction_factors,
        )
        tool_tokens = estimate_tools_tokens(context.tools, correction_factors=correction_factors)
        pressure = self.pressure_policy.evaluate(
            request,
            estimated_tokens=raw_estimate.total,
            tool_output_tokens=_tool_output_tokens(visible_messages),
            history_tokens=history_tokens,
        )
        compact_summary, compact_cursor = self._maybe_compact_context(
            visible_messages,
            pressure=pressure,
            run_id=_request_run_id(request),
        )

        memory_recall = self._recall_memory(context, request, plan_state, run_signals)
        candidates = {
            "system": self._system_items(context.system_prompt, plan_state),
            "task_plan": _plan_items(plan_state, run_signals),
            "working_set": self._working_items(snapshot, delta, stale_items, artifact_refs),
            "memory": _memory_items(memory_recall),
            "conversation": _conversation_items(
                visible_messages,
                compacted_until_message_id=compact_cursor,
                compact_summary=compact_summary,
            ),
        }
        selected, dropped, sections = _select_context(candidates, pressure.level, pressure.effective_budget)
        view = ContextView(
            system=[item.content for item in selected["system"]],
            task_plan=[item.content for item in selected["task_plan"]],
            working_set=[item.content for item in selected["working_set"]],
            memory=[item.content for item in selected["memory"]],
            conversation=[item.content for item in selected["conversation"]],
        )
        selected_messages = _select_messages(
            visible_messages,
            pressure.level,
            self.tool_ledger,
            compacted_until_message_id=compact_cursor,
        )
        system_prompt = _compose_system_prompt(context, view)
        after_estimate = estimate_context(
            selected_messages,
            system_prompt,
            context.tools,
            correction_factors=correction_factors,
        )
        tokens_by_layer = {
            layer: sum(
                estimate_text_tokens(line, correction_factors=correction_factors).total
                for line in getattr(view, layer)
            )
            for layer in _LAYER_ORDER
        }
        tokens_by_layer["runtime"] = 0
        tokens_by_layer["tools"] = tool_tokens.total

        dynamic_text = "\n".join(
            [
                *view.task_plan,
                *view.working_set,
                *view.memory,
                *view.conversation,
            ]
        )
        report = ContextReport(
            context_id=f"ctx_{_hash_text(dynamic_text)}",
            repository_fingerprint=snapshot.fingerprint,
            total_budget_tokens=pressure.effective_budget,
            estimated_tokens_before=raw_estimate.total,
            estimated_tokens_after=after_estimate.total,
            sections=sections,
            selected_items=_selected_summaries(selected),
            stale_items=stale_items,
            dropped_items=dropped,
            repository_delta=delta,
            retrieved_memory_ids=[item.record.id for item in memory_recall.retrieved],
            memory_retrieval_reasons={
                item.record.id: list(item.reasons)
                for item in memory_recall.retrieved
            },
            dropped_memory_ids=list(memory_recall.dropped),
            dropped_memory_reasons=dict(memory_recall.dropped),
            memory_tokens=tokens_by_layer["memory"],
            context_mode=_context_mode(context),
            pressure=pressure,
            context_view=view,
            artifact_refs=artifact_refs,
            tokens_by_layer=tokens_by_layer,
            compact_summary=compact_summary,
            prefix_hash=_hash_text("\n".join(view.system)),
            dynamic_hash=_hash_text(dynamic_text),
            estimation={
                "raw_estimate": raw_estimate.total,
                "estimated_after": after_estimate.total,
                "by_type": dict(raw_estimate.by_type),
            },
        )
        self.store.append_context_ledger(_projection_payload(report, run_id=_request_run_id(request)))
        self.store.update_context_meta({"last_context_id": report.context_id})
        return PreparedAgentContext(
            system_prompt=system_prompt,
            messages=selected_messages,
            tools=list(context.tools),
            report=report,
        )

    def finalize_run(self, result: object) -> None:
        return None

    def _maybe_compact_context(
        self,
        messages: list[Message],
        *,
        pressure: ContextPressure,
        run_id: str | None,
    ) -> tuple[str, str | None]:
        meta = self.store.read_meta() or {}
        context_meta = meta.get("context") if isinstance(meta.get("context"), dict) else {}
        current_cursor = _optional_text(context_meta.get("compacted_until_message_id"))
        current_summary = _optional_text(context_meta.get("last_compact_summary")) or ""
        if pressure.level != "critical":
            return current_summary, current_cursor

        groups = _message_groups(messages)
        if len(groups) <= 6:
            return current_summary, current_cursor
        compacted_groups = groups[:-6]
        cursor = _session_message_id(compacted_groups[-1][-1])
        if cursor is None or cursor == current_cursor:
            return current_summary, current_cursor

        summary = _deterministic_compact_summary(compacted_groups)
        source_ids = [
            message_id
            for group in compacted_groups
            for message in group
            if (message_id := _session_message_id(message)) is not None
        ]
        self.store.append_context_ledger(
            {
                "type": "context_compaction",
                "context_id": f"ctx_compact_{_hash_text(summary + cursor)}",
                "run_id": run_id,
                "created_at": _utc_now_iso(),
                "compacted_until_message_id": cursor,
                "summary": summary,
                "source_message_ids": source_ids,
                "artifact_refs": [],
                "fallback": True,
            }
        )
        self.store.update_context_meta(
            {
                "compacted_until_message_id": cursor,
                "last_compact_summary": summary,
                "last_compacted_at": _utc_now_iso(),
            }
        )
        return summary, cursor

    def _observe_tools(
        self,
        context: AgentContext,
        request: ContextPreparationRequest,
        snapshot: RepositorySnapshot,
    ) -> list[ContextArtifactRef]:
        artifact_refs = []
        run_id = _request_run_id(request)
        for message in context.messages:
            if not isinstance(message, ToolResultMessage):
                continue
            self.state.observe_tool_result(
                message,
                repository_fingerprint=snapshot.fingerprint,
            )
            if _should_archive_tool_result(message):
                artifact_refs.append(
                    self.tool_ledger.record_tool_result(run_id=run_id, message=message).artifact
                )
        artifact_refs.extend(self.tool_ledger.artifact_refs())
        return _dedupe_artifacts(artifact_refs)

    def _recall_memory(
        self,
        context: AgentContext,
        request: ContextPreparationRequest,
        plan_state: Mapping[str, object] | None,
        run_signals: Mapping[str, object] | None,
    ) -> MemoryRecall:
        if self.memory_retriever is None:
            return MemoryRecall()
        query = MemoryQuery(
            latest_user_message=_latest_user_text(context.messages),
            raw_user_request=_plan_objective(plan_state),
            goal=_plan_objective(plan_state),
            current_mode=_plan_string(plan_state, "origin_mode") or _context_mode(context),
            verification_status=_signal_text(run_signals, "verification_status"),
            blocked_reason=_signal_error_text(run_signals),
            active_paths=sorted(self.state.active_files),
            changed_paths=[],
            session_id=self.session_id,
            run_id=_request_run_id(request),
            limit=5,
        )
        return self.memory_retriever.recall(query)

    def _system_items(
        self,
        system_prompt: str,
        plan_state: Mapping[str, object] | None,
    ) -> list[ContextItem]:
        lines = [line.strip() for line in system_prompt.splitlines() if line.strip()]
        return [
            _item(f"system:{index}", "system", line, "system_prompt", 100 - index)
            for index, line in enumerate(lines[:10])
        ]

    def _working_items(
        self,
        snapshot: RepositorySnapshot,
        delta: RepositoryDelta,
        stale_items: list[str],
        artifact_refs: list[ContextArtifactRef],
    ) -> list[ContextItem]:
        lines = [render_repository_snapshot(snapshot, delta)]
        if stale_items:
            lines.append("Stale context: " + ", ".join(stale_items[:10]))
        for active in sorted(
            self.state.active_files.values(),
            key=lambda item: (item.role, item.access_count, item.last_accessed_at),
            reverse=True,
        )[:10]:
            lines.append(
                f"Active file: {active.path} role={active.role} freshness={active.freshness} reason={active.reason}"
            )
        for evidence in self.state.evidence[-8:]:
            lines.append(f"Evidence from {evidence.source}: {evidence.content}")
        for ref in artifact_refs[:8]:
            lines.append(f"Artifact: {ref.path} summary={ref.summary}")
        return [
            _item(f"working:{index}", "working_set", line, "session_context", 80 - index)
            for index, line in enumerate(lines)
            if line
        ]


@dataclass(frozen=True)
class ToolLedgerEntry:
    tool_call_id: str
    run_id: str | None
    tool_name: str
    status: str
    artifact: ContextArtifactRef
    affected_paths: list[str]
    verification: dict[str, object] | None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact"] = asdict(self.artifact)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolLedgerEntry:
        artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {}
        verification = payload.get("verification")
        return cls(
            tool_call_id=str(payload.get("tool_call_id") or ""),
            run_id=payload.get("run_id") if isinstance(payload.get("run_id"), str) else None,
            tool_name=str(payload.get("tool_name") or ""),
            status=str(payload.get("status") or "success"),
            artifact=ContextArtifactRef(
                kind=str(artifact.get("kind") or "tool_output"),
                path=str(artifact.get("path") or ""),
                source_hash=artifact.get("source_hash") if isinstance(artifact.get("source_hash"), str) else None,
                summary=str(artifact.get("summary") or ""),
                original_tokens=_int(artifact.get("original_tokens")),
                visible_tokens=_int(artifact.get("visible_tokens")),
            ),
            affected_paths=[
                str(path)
                for path in payload.get("affected_paths", [])
                if isinstance(path, str)
            ],
            verification=verification if isinstance(verification, dict) else None,
            error_code=payload.get("error_code") if isinstance(payload.get("error_code"), str) else None,
        )


class ToolArtifactLedger:
    """Archive tool outputs and expose compact references to the prompt."""

    def __init__(self, *, store: SessionStore) -> None:
        self.store = store
        self.workspace_dir = store.workspace_dir
        self.session_id = store.session_id
        self.ledger_file = store.layout.context_ledger_file
        self.artifact_dir = store.layout.tool_outputs_dir

    def record_tool_result(
        self,
        *,
        run_id: str | None,
        message: ToolResultMessage,
    ) -> ToolLedgerEntry:
        text = _tool_text(message)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self._entry_for_call(message.tool_call_id)
        if existing is not None and existing.artifact.source_hash == digest:
            return existing

        artifact_path = (
            Path(".codepilot")
            / "sessions"
            / self.session_id
            / "artifacts"
            / "tool_outputs"
            / f"{_safe_artifact_stem(message.tool_call_id)}_{digest[:12]}.txt"
        )
        target = self.workspace_dir / artifact_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")

        summary = _summary_for_tool_result(message, text)
        visible = _projection_text(message, artifact_path.as_posix(), summary)
        entry = ToolLedgerEntry(
            tool_call_id=message.tool_call_id,
            run_id=run_id,
            tool_name=message.tool_name,
            status=message.status,
            artifact=ContextArtifactRef(
                kind="tool_output",
                path=artifact_path.as_posix(),
                source_hash=digest,
                summary=summary,
                original_tokens=estimate_context_tokens([message], ""),
                visible_tokens=estimate_context_tokens(
                    [
                        ToolResultMessage(
                            tool_call_id=message.tool_call_id,
                            tool_name=message.tool_name,
                            content=[TextContent(text=visible)],
                            status=message.status,
                        )
                    ],
                    "",
                ),
            ),
            affected_paths=list(message.affected_paths),
            verification=dict(message.verification) if message.verification else None,
            error_code=message.error_code,
        )
        self._append(entry)
        return entry

    def project_tool_result(
        self,
        message: ToolResultMessage,
        *,
        preserve_full: bool,
    ) -> ToolResultMessage:
        if preserve_full and len(_tool_text(message)) <= _LONG_TOOL_OUTPUT_CHARS:
            return message
        entry = self._entry_for_call(message.tool_call_id)
        if entry is None:
            entry = self.record_tool_result(run_id=None, message=message)
        text = _projection_text(message, entry.artifact.path, entry.artifact.summary)
        return ToolResultMessage(
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            content=[TextContent(text=text)],
            status=message.status,
            is_error=message.is_error,
            approved=message.approved,
            approval_id=message.approval_id,
            error_code=message.error_code,
            exit_code=message.exit_code,
            affected_paths=list(message.affected_paths),
            workspace_changed=message.workspace_changed,
            diff_summary=message.diff_summary,
            verification=dict(message.verification) if message.verification else None,
            details=message.details,
            timestamp=message.timestamp,
            metadata={**message.metadata, "artifact_ref": entry.artifact.path},
        )

    def load_entries(self) -> list[ToolLedgerEntry]:
        if not self.ledger_file.exists():
            return []
        entries: list[ToolLedgerEntry] = []
        for line in self.ledger_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("type") == "tool_artifact":
                entries.append(ToolLedgerEntry.from_dict(payload))
        return entries

    def artifact_refs(self) -> list[ContextArtifactRef]:
        return [entry.artifact for entry in self.load_entries()]

    def _entry_for_call(self, tool_call_id: str) -> ToolLedgerEntry | None:
        for entry in reversed(self.load_entries()):
            if entry.tool_call_id == tool_call_id:
                return entry
        return None

    def _append(self, entry: ToolLedgerEntry) -> None:
        self.store.append_context_ledger({"type": "tool_artifact", **entry.to_dict()})


def calibrate_context_usage(
    *,
    workspace_dir: str | Path,
    provider: str | None,
    model: str | None,
    report: Mapping[str, object] | None,
    actual_input_tokens: int,
) -> None:
    if not report or actual_input_tokens <= 0:
        return
    estimation = report.get("estimation")
    if not isinstance(estimation, Mapping):
        return
    raw_estimate = estimation.get("raw_estimate")
    by_type = estimation.get("by_type")
    if not isinstance(raw_estimate, int) or not isinstance(by_type, dict):
        return
    ContextUsageCalibrator(workspace_dir).update(
        provider=provider,
        model=model,
        raw_estimate=raw_estimate,
        actual_input_tokens=actual_input_tokens,
        breakdown={
            str(key): int(value)
            for key, value in by_type.items()
            if isinstance(value, int) and not isinstance(value, bool)
        },
    )


def build_context_freshness_notice(freshness: Any) -> UserMessage | None:
    if not freshness.requires_steering():
        return None
    payload = freshness.to_event_payload()
    lines = [
        "[Context Freshness]",
        f"status={freshness.status}",
    ]
    if freshness.changed_paths:
        lines.append("changed_files=" + ", ".join(freshness.changed_paths))
    if freshness.missing_paths:
        lines.append("missing_files=" + ", ".join(freshness.missing_paths))
    lines.append("旧工具结果可能已过期；依赖这些文件前请重新读取。")
    return UserMessage(
        content=[TextContent(text="\n".join(lines))],
        metadata={"context_freshness": payload},
    )


def compare_snapshots(
    previous: RepositorySnapshot | None,
    current: RepositorySnapshot,
) -> RepositoryDelta:
    if previous is None:
        return RepositoryDelta()
    old_paths = set(previous.top_level_entries)
    new_paths = set(current.top_level_entries)
    old_status = _status_map(previous.git_status)
    new_status = _status_map(current.git_status)
    modified = sorted(
        path
        for path, status in new_status.items()
        if status != "??"
        and not _is_internal_path(path)
        and (
            old_status.get(path) != status
            or previous.dirty_path_hashes.get(path) != current.dirty_path_hashes.get(path)
        )
    )
    deleted = sorted(
        {
            *[
                path
                for path, status in new_status.items()
                if "D" in status and not _is_internal_path(path)
            ],
            *(path for path in (old_paths - new_paths) if not _is_internal_path(path)),
        }
    )
    return RepositoryDelta(
        added_paths=sorted((new_paths - old_paths) | {p for p, s in new_status.items() if s == "??"}),
        modified_paths=modified,
        deleted_paths=deleted,
        branch_changed=previous.branch != current.branch,
        head_changed=previous.head_sha != current.head_sha,
        instructions_changed=previous.instruction_hashes != current.instruction_hashes,
    )


def render_repository_snapshot(
    snapshot: RepositorySnapshot,
    delta: RepositoryDelta,
) -> str:
    lines = [
        "## Repository Context",
        f"- Repository fingerprint: {snapshot.fingerprint[:12]}",
        f"- Workspace: {snapshot.workspace_root}",
        f"- Project type: {snapshot.project_type or 'unknown'}",
        f"- Manifests: {', '.join(snapshot.manifest_files) or '(none)'}",
        f"- Top-level: {', '.join(snapshot.top_level_entries) or '(empty)'}",
        f"- Test directories: {', '.join(snapshot.test_directories) or '(none)'}",
        f"- Instruction files: {', '.join(snapshot.instruction_files) or '(none)'}",
        f"- Git branch: {snapshot.branch or 'unknown'}",
        f"- HEAD: {snapshot.head_sha or 'unknown'}",
        f"- Working tree changes: {len(snapshot.git_status)}",
    ]
    if delta.changed:
        lines.extend(
            [
                "### Changes since previous model call",
                f"- Added: {', '.join(delta.added_paths) or '(none)'}",
                f"- Modified: {', '.join(delta.modified_paths) or '(none)'}",
                f"- Deleted: {', '.join(delta.deleted_paths) or '(none)'}",
                f"- Branch changed: {delta.branch_changed}",
                f"- HEAD changed: {delta.head_changed}",
                f"- Instructions changed: {delta.instructions_changed}",
            ]
        )
    return "\n".join(lines)


def _select_context(
    candidates: dict[str, list[ContextItem]],
    pressure_level: str,
    total_budget: int,
) -> tuple[dict[str, list[ContextItem]], list[DroppedContextItem], list[ContextSectionReport]]:
    selected: dict[str, list[ContextItem]] = {}
    dropped: list[DroppedContextItem] = []
    sections: list[ContextSectionReport] = []
    for layer in _LAYER_ORDER:
        items = sorted(candidates[layer], key=lambda item: (item.priority, -item.estimated_tokens), reverse=True)
        layer_budget = max(32, int(total_budget * _LAYER_BUDGET_RATIOS[layer]))
        keep_count = _KEEP_COUNTS.get(pressure_level, _KEEP_COUNTS["tight"]).get(layer, 10)
        kept: list[ContextItem] = []
        token_total = 0
        for item in items:
            over_budget = token_total + item.estimated_tokens > layer_budget
            over_count = len(kept) >= keep_count
            if over_budget or over_count:
                dropped.append(
                    DroppedContextItem(
                        item_id=item.id,
                        section=layer,
                        reason="over_budget" if over_budget else "low_relevance",
                        source=item.source,
                    )
                )
                continue
            kept.append(item)
            token_total += item.estimated_tokens
        selected[layer] = kept
        sections.append(
            ContextSectionReport(
                name=layer,
                budget_tokens=layer_budget,
                candidate_items=len(items),
                selected_items=len(kept),
                estimated_tokens_before=sum(item.estimated_tokens for item in items),
                estimated_tokens_after=token_total,
                reduction_policy="priority_budget",
            )
        )
    return selected, dropped, sections


def _plan_items(
    plan_state: Mapping[str, object] | None,
    run_signals: Mapping[str, object] | None,
) -> list[ContextItem]:
    if plan_state is None and run_signals is None:
        return []
    approved = (
        _plan_string(plan_state, "status") == "active"
        and _plan_string(plan_state, "approval_state") == "approved"
    )
    lines = [
        (
            "Approved Execution Contract: execute this canonical plan; do not replace it "
            "with a newly invented plan."
            if approved
            else "Task Plan: canonical runtime state for the current task."
        ),
        f"Objective: {_plan_objective(plan_state)}",
        f"Summary: {_plan_string(plan_state, 'summary') or '(none)'}",
        f"Plan status: {_plan_string(plan_state, 'status') or 'none'}",
        f"Approval: {_plan_string(plan_state, 'approval_state') or 'not_required'}",
        f"Origin mode: {_plan_string(plan_state, 'origin_mode') or 'build'}",
        f"Verification: {_signal_text(run_signals, 'verification_status') or 'unknown'}",
    ]
    last_error = _signal_error_text(run_signals)
    if last_error:
        lines.append(f"Last error: {last_error}")
    items = plan_state.get("items") if isinstance(plan_state, Mapping) else None
    if isinstance(items, list):
        active_items = [
            item
            for item in items[:MAX_PLAN_ITEMS]
            if isinstance(item, Mapping)
            and item.get("status") in {"pending", "in_progress"}
        ]
        completed_items = [
            item
            for item in items[:MAX_PLAN_ITEMS]
            if isinstance(item, Mapping) and item.get("status") == "completed"
        ]
        for item in active_items:
            if isinstance(item, Mapping):
                lines.append(
                    "Step "
                    f"{item.get('id')}: {item.get('step')} [{item.get('status')}] "
                    f"details={item.get('details') or '(none)'} "
                    f"verification={item.get('verification') or '(none)'}"
                )
        if completed_items:
            lines.append(
                "Completed steps: "
                + ", ".join(
                    f"{item.get('id')}={item.get('step')}"
                    for item in completed_items
                )
            )
    return [
        _item(f"task_plan:{index}", "task_plan", line, "plan_state", 75 - index)
        for index, line in enumerate(lines)
        if line.strip()
    ]


def _memory_items(recall: MemoryRecall) -> list[ContextItem]:
    return [
        _item(
            item.record.id,
            "memory",
            f"{render_memory(item.record)} [reasons={', '.join(item.reasons)}]",
            "memory_recall",
            70 - index,
        )
        for index, item in enumerate(recall.retrieved[:5])
    ]


def _without_published_plan_summaries(
    messages: list[Message],
    plan_state: Mapping[str, object] | None,
) -> list[Message]:
    if (
        _plan_string(plan_state, "status") != "active"
        or _plan_string(plan_state, "approval_state") != "approved"
    ):
        return list(messages)
    return [
        message
        for message in messages
        if not (
            getattr(message, "role", None) == "assistant"
            and isinstance(getattr(message, "metadata", None), dict)
            and message.metadata.get("message_kind") == "plan_summary"
        )
    ]


def _conversation_items(
    messages: list[Message],
    *,
    compacted_until_message_id: str | None = None,
    compact_summary: str = "",
) -> list[ContextItem]:
    items: list[ContextItem] = []
    if compact_summary:
        items.append(
            _item(
                "conversation:compact_summary",
                "conversation",
                f"Compacted earlier conversation: {compact_summary}",
                "compact_summary",
                65,
            )
        )
    visible_messages = _messages_after_compact_cursor(
        messages,
        compacted_until_message_id=compacted_until_message_id,
    )
    for index, message in enumerate(visible_messages[-8:]):
        text = _message_preview(message)
        if text:
            items.append(
                _item(
                    f"conversation:{index}",
                    "conversation",
                    f"{getattr(message, 'role', 'message')}: {text}",
                    "recent_messages",
                    40 - index,
                )
            )
    return items


def _select_messages(
    messages: list[Message],
    pressure_level: str,
    ledger: ToolArtifactLedger,
    *,
    compacted_until_message_id: str | None = None,
) -> list[Message]:
    messages = _messages_after_compact_cursor(
        messages,
        compacted_until_message_id=compacted_until_message_id,
    )
    if pressure_level == "normal":
        return list(messages)
    keep_groups = {"tight": 12, "critical": 6}.get(pressure_level, 12)
    groups = _message_groups(messages)
    selected = [message for group in groups[-keep_groups:] for message in group]
    return [
        ledger.project_tool_result(message, preserve_full=pressure_level == "tight")
        if isinstance(message, ToolResultMessage)
        else message
        for message in selected
    ]


def _message_groups(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if _assistant_tool_call_ids(message):
            ids = set(_assistant_tool_call_ids(message))
            group = [message]
            index += 1
            while index < len(messages):
                next_message = messages[index]
                if not isinstance(next_message, ToolResultMessage):
                    break
                if next_message.tool_call_id not in ids:
                    break
                group.append(next_message)
                index += 1
            groups.append(group)
            continue
        if isinstance(message, ToolResultMessage) and groups:
            groups[-1].append(message)
        else:
            groups.append([message])
        index += 1
    return groups


def _compose_system_prompt(context: AgentContext, view: ContextView) -> str:
    runtime_state = (
        context.runtime_state
        if isinstance(context.runtime_state, Mapping)
        else {}
    )
    sections = [
        ("Mode Policy", [_mapping_text(runtime_state, "mode_policy")]),
        ("Runtime State", _runtime_state_lines(runtime_state)),
        ("Task Plan", view.task_plan),
        ("Working Set", view.working_set),
        ("Memory", view.memory),
        ("Conversation", view.conversation),
        ("Available Tools", _tool_prompt_lines(context.tools)),
    ]
    parts = [context.system_prompt.rstrip()]
    for name, lines in sections:
        visible = [line for line in lines if line]
        if visible:
            parts.append(f"## {name}\n" + "\n".join(f"- {line}" for line in visible))
    return "\n\n".join(part for part in parts if part).strip()


def _runtime_state_lines(runtime_state: Mapping[str, object]) -> list[str]:
    labels = (
        ("run_id", "Run"),
        ("mode", "Mode"),
        ("checkpoint_phase", "Checkpoint"),
        ("plan_status", "Plan status"),
        ("plan_approval_state", "Plan approval"),
        ("verification_status", "Verification"),
        ("directive", "Continuation directive"),
    )
    return [
        f"{label}: {value}"
        for key, label in labels
        if (value := _mapping_text(runtime_state, key))
    ]


def _tool_prompt_lines(tools: list[object]) -> list[str]:
    lines: list[str] = []
    for tool in tools:
        if isinstance(tool, Mapping):
            name = _optional_text(tool.get("name"))
            description = _optional_text(tool.get("description"))
        else:
            name = _optional_text(getattr(tool, "name", None))
            description = _optional_text(getattr(tool, "description", None))
        if name:
            lines.append(f"{name}: {description or 'See the runtime tool schema.'}")
    return lines


def _mapping_text(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    return value.strip() if isinstance(value, str) else ""


def _projection_payload(report: ContextReport, *, run_id: str | None) -> dict[str, Any]:
    return {
        "type": "context_projection",
        "context_id": report.context_id,
        "run_id": run_id,
        "created_at": _utc_now_iso(),
        "pressure": asdict(report.pressure) if report.pressure else None,
        "tokens_by_layer": dict(report.tokens_by_layer),
        "selected_items": list(report.selected_items),
        "dropped_items": [asdict(item) for item in report.dropped_items],
        "memory_ids": list(report.retrieved_memory_ids),
        "memory_retrieval_reasons": dict(report.memory_retrieval_reasons),
        "dropped_memory_ids": list(report.dropped_memory_ids),
        "dropped_memory_reasons": dict(report.dropped_memory_reasons),
        "memory_tokens": report.memory_tokens,
        "artifact_refs": [asdict(item) for item in report.artifact_refs],
        "compact_summary": report.compact_summary or None,
        "runner_preflight": report.runner_preflight.to_dict(),
        "prefix_hash": report.prefix_hash,
        "dynamic_hash": report.dynamic_hash,
    }


def _selected_summaries(selected: dict[str, list[ContextItem]]) -> list[dict[str, object]]:
    return [
        {
            "id": item.id,
            "layer": layer,
            "kind": item.kind,
            "path": item.path or "",
            "source": item.source,
            "estimated_tokens": item.estimated_tokens,
            "freshness": item.freshness,
        }
        for layer, items in selected.items()
        for item in items
    ]


def _item(item_id: str, kind: str, content: str, source: str, priority: int) -> ContextItem:
    return ContextItem(
        id=item_id,
        kind=kind,
        content=content,
        source=source,
        trust="observed" if source != "memory_recall" else "derived",
        priority=priority,
        estimated_tokens=estimate_text_tokens(content).total,
        freshness="fresh",
    )


def _plan_state_from_context(
    context: AgentContext,
    plan_state_store: PlanStateStore,
) -> Mapping[str, object] | None:
    if isinstance(context.plan_state, Mapping):
        return context.plan_state
    return plan_state_store.current()


def _run_signals_from_context(context: AgentContext) -> Mapping[str, object] | None:
    return context.run_signals if isinstance(context.run_signals, Mapping) else None


def _plan_objective(plan_state: Mapping[str, object] | None) -> str:
    if plan_state is None:
        return ""
    value = plan_state.get("objective")
    return value if isinstance(value, str) else ""


def _plan_string(plan_state: Mapping[str, object] | None, key: str) -> str | None:
    if plan_state is None:
        return None
    value = plan_state.get(key)
    return value if isinstance(value, str) and value else None


def _context_mode(context: AgentContext) -> str:
    text = _latest_user_text(context.messages).lower()
    if _signal_error_text(context.run_signals):
        return "repair"
    if any(word in text for word in ("why", "how", "explain", "为什么", "解释")):
        return "qa"
    return "act"


def _latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(getattr(block, "text", "") for block in content)
    return ""


def _messages_after_compact_cursor(
    messages: list[Message],
    *,
    compacted_until_message_id: str | None,
) -> list[Message]:
    if not compacted_until_message_id:
        return list(messages)
    for index, message in enumerate(messages):
        if _session_message_id(message) == compacted_until_message_id:
            return list(messages[index + 1 :])
    return list(messages)


def _session_message_id(message: Message) -> str | None:
    metadata = getattr(message, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("session_message_id")
    return value if isinstance(value, str) and value else None


def _deterministic_compact_summary(groups: list[list[Message]]) -> str:
    messages = [message for group in groups for message in group]
    counts: dict[str, int] = {}
    user_requests: list[str] = []
    tool_names: list[str] = []
    for message in messages:
        role = str(getattr(message, "role", "message"))
        counts[role] = counts.get(role, 0) + 1
        if isinstance(message, UserMessage):
            text = _message_preview(message, limit=120)
            if text:
                user_requests.append(text)
        if isinstance(message, ToolResultMessage) and message.tool_name:
            tool_names.append(message.tool_name)
    parts = [
        f"{len(messages)} earlier messages compacted",
        "roles=" + ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())),
    ]
    if user_requests:
        parts.append("recent_user=" + user_requests[-1])
    if tool_names:
        parts.append("tools=" + ", ".join(_dedupe(tool_names)[-6:]))
    return "; ".join(parts)


def _message_preview(message: Message, *, limit: int = 240) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return " ".join(content.split())[:limit]
    if isinstance(content, list):
        text = "".join(getattr(block, "text", "") for block in content if isinstance(block, TextContent))
        return " ".join(text.split())[:limit]
    return ""


def _tool_output_tokens(messages: list[Message]) -> int:
    return sum(
        estimate_context_tokens([message], "")
        for message in messages
        if isinstance(message, ToolResultMessage)
    )


def _request_run_id(request: ContextPreparationRequest) -> str | None:
    signal = request.signal
    if isinstance(signal, Mapping):
        value = signal.get("run_id")
        return value if isinstance(value, str) else None
    return None


def _request_provider_model(request: ContextPreparationRequest) -> tuple[str | None, str | None]:
    signal = request.signal
    if not isinstance(signal, Mapping):
        return None, None
    provider = signal.get("provider")
    model = signal.get("model")
    return (
        provider if isinstance(provider, str) and provider else None,
        model if isinstance(model, str) and model else None,
    )


def _signal_text(signal: object, key: str) -> str | None:
    if not isinstance(signal, Mapping):
        return None
    value = signal.get(key)
    return value if isinstance(value, str) and value else None


def _signal_error_text(signal: object) -> str | None:
    if not isinstance(signal, Mapping):
        return None
    last_error = signal.get("last_error")
    if isinstance(last_error, Mapping):
        code = last_error.get("error_code") or last_error.get("code")
        tool = last_error.get("tool_name")
        parts = [str(part) for part in (tool, code) if isinstance(part, str) and part]
        return ":".join(parts) if parts else None
    return None


def _dedupe_artifacts(items: list[Any]) -> list[ContextArtifactRef]:
    seen: set[str] = set()
    out: list[ContextArtifactRef] = []
    for item in items:
        if not isinstance(item, ContextArtifactRef):
            continue
        key = item.path
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _assistant_tool_call_ids(message: Message) -> list[str]:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    return [block.id for block in content if isinstance(block, ToolCall) and block.id]


def _tool_result_text(message: ToolResultMessage, *, limit: int = 1200) -> str:
    parts = [getattr(block, "text", "") for block in message.content]
    return "".join(part for part in parts if part).strip()[:limit]


def _metadata_paths(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    paths: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in paths:
            paths.append(text)
    return paths


def _is_successful_read_result(message: ToolResultMessage) -> bool:
    status = str(getattr(message, "status", "") or "success")
    return message.tool_name == "read" and status == "success"


def _tool_text(message: ToolResultMessage) -> str:
    return "".join(getattr(block, "text", "") for block in message.content)


def _should_archive_tool_result(message: ToolResultMessage) -> bool:
    if isinstance(message.metadata.get("artifact_ref"), str):
        return True
    return len(_tool_text(message)) > _LONG_TOOL_OUTPUT_CHARS


def _safe_stem(value: str) -> str:
    raw = value or "tool"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:64]


def _safe_artifact_stem(value: str) -> str:
    """Keep archived tool-output paths short enough for nested workspaces."""

    stem = _safe_stem(value).strip("._-")
    return (stem[:8].strip("._-") or "tool")


def _summary_for_tool_result(message: ToolResultMessage, text: str) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) > 280:
        compact = f"{len(text.splitlines())} lines, {len(text)} chars archived"
    paths = ", ".join(message.affected_paths)
    prefix = f"{message.tool_name} status={message.status}"
    if paths:
        prefix += f" paths={paths}"
    return f"{prefix}: {compact}" if compact else prefix


def _projection_text(message: ToolResultMessage, artifact_path: str, summary: str) -> str:
    lines = [
        "[Tool output archived]",
        f"tool={message.tool_name}",
        f"status={message.status}",
        f"artifact={artifact_path}",
        f"summary={summary}",
    ]
    if message.verification:
        lines.append(f"verification={message.verification}")
    return "\n".join(lines)


def _git_lines(root: Path, args: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if args == ["status", "--porcelain"]:
        return [
            line
            for line in lines
            if len(line) < 4 or not _is_internal_path(line[3:].split(" -> ")[-1])
        ]
    return lines


def _status_map(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:].split(" -> ")[-1]
        if not _is_internal_path(path):
            result[path] = status
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_internal_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return False
    return normalized.split("/", 1)[0] in _INTERNAL_DIR_NAMES


def _dirty_path_hashes(root: Path, status_lines: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in _status_map(status_lines):
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue
        if target.is_file():
            hashes[path] = _sha256(target)
        elif target.is_dir():
            hashes[path] = _directory_fingerprint(target)
        elif not target.exists():
            hashes[path] = "<missing>"
    return hashes


def _directory_fingerprint(path: Path, *, limit: int = 100) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and not any(part in _INTERNAL_DIR_NAMES for part in item.relative_to(path).parts)
        ),
        key=lambda item: item.as_posix(),
    )
    for item in files[:limit]:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        try:
            digest.update(_sha256(item).encode("ascii"))
        except FileNotFoundError:
            continue
    digest.update(f"count:{len(files)}".encode("ascii"))
    return digest.hexdigest()


def _ensure_file_role(value: str) -> ContextFileRole:
    if value not in _CONTEXT_FILE_ROLES:
        raise ValueError(f"Unknown active file role: {value}")
    return cast(ContextFileRole, value)


def _ensure_context_evidence_kind(value: str) -> ContextEvidenceKind:
    if value not in _CONTEXT_EVIDENCE_KINDS:
        raise ValueError(f"Unknown context evidence kind: {value}")
    return cast(ContextEvidenceKind, value)


def _ensure_context_freshness(value: str) -> ContextFreshness:
    if value not in _CONTEXT_FRESHNESS_VALUES:
        raise ValueError(f"Unknown context freshness: {value}")
    return cast(ContextFreshness, value)


def _ensure_context_trust(value: str) -> ContextTrust:
    if value not in _CONTEXT_TRUST_VALUES:
        raise ValueError(f"Unknown context trust: {value}")
    return cast(ContextTrust, value)


def _active_file_rank(path: str, active: ActiveFile) -> tuple[int, int, int, float, str]:
    return (
        _ACTIVE_FILE_ROLE_PRIORITY.get(active.role, 0),
        _ACTIVE_FILE_FRESHNESS_PRIORITY.get(active.freshness, 0),
        active.access_count,
        active.last_accessed_at,
        path,
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


__all__ = [
    "ActiveFile",
    "ContextEvidence",
    "ContextEvidenceKind",
    "ContextFileRole",
    "ContextGovernor",
    "ContextPressurePolicy",
    "FileSummary",
    "RepositoryTracker",
    "SessionContextState",
    "ToolArtifactLedger",
    "ToolLedgerEntry",
    "build_context_freshness_notice",
    "calibrate_context_usage",
    "compare_snapshots",
    "render_repository_snapshot",
]
