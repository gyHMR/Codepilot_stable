from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .admission import AdmissionPolicy, contains_sensitive_content, normalize_content
from .contracts import (
    AddMemory,
    ApproveMemory,
    DeleteMemory,
    DisableMemory,
    EditMemory,
    EnableMemory,
    HistoryMemory,
    ListMemory,
    MemoryActor,
    MemoryCommand,
    MemoryCommandResult,
    MemoryProposal,
    MemoryProposalBatch,
    MemoryProposalReceipt,
    MemoryQuery,
    MemoryRecallResult,
    MemoryRecord,
    PurgeMemory,
    RejectMemory,
    ShowMemory,
)
from .recall import MemoryRecallEngine
from .repository import (
    LegacyMemoryMigrator,
    MemoryMigrationReport,
    ProjectMemoryRepository,
    UserMemoryRepository,
)


class MemoryService:
    """Canonical implementation of Memory recall, proposal and management ports."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        user_home: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        auto_migrate: bool = True,
    ) -> None:
        self.user_repository = UserMemoryRepository(user_home)
        self.project_repository = ProjectMemoryRepository(workspace_dir)
        self.admission = AdmissionPolicy()
        self.retriever = MemoryRecallEngine(
            self.user_repository,
            self.project_repository,
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.migrator = LegacyMemoryMigrator(
            self.user_repository,
            self.project_repository,
        )
        if auto_migrate:
            report = self.migrator.dry_run()
            if report.required:
                self.migrator.migrate()

    def recall(self, query: MemoryQuery) -> MemoryRecallResult:
        return self.retriever.recall(query)

    def submit_proposals(
        self,
        batch: MemoryProposalBatch,
    ) -> MemoryProposalReceipt:
        records: list[MemoryRecord] = []
        rejected: list[str] = []
        for proposal in batch.proposals:
            existing = tuple(self._repository(proposal.scope).all_records())
            decision = self.admission.evaluate(
                proposal,
                existing=existing,
                origin=batch.origin,
                verification_passed=batch.verification_passed,
            )
            if decision.disposition == "duplicate":
                duplicate = self._require(decision.existing_id or "")
                records.append(duplicate)
                continue
            if decision.disposition != "accept":
                rejected.append(f"{proposal.key}:{decision.reason}")
                continue
            if decision.source is None or decision.status is None:
                raise RuntimeError("accepted memory is missing server-owned fields")
            record = MemoryRecord(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                scope=decision.proposal.scope,
                type=decision.proposal.type,
                key=decision.proposal.key,
                content=decision.proposal.content,
                source=decision.source,
                status=decision.status,
                updated_at=self._now(),
            )
            self._repository(record.scope).save(record)
            self._ensure_invariants(record.scope, record.key)
            records.append(record)
        return MemoryProposalReceipt(records=tuple(records), rejected=tuple(rejected))

    def execute(
        self,
        command: MemoryCommand,
        actor: MemoryActor,
    ) -> MemoryCommandResult:
        del actor
        if isinstance(command, AddMemory):
            receipt = self.submit_proposals(
                MemoryProposalBatch(
                    session_id="management",
                    run_id=f"command_{uuid.uuid4().hex[:12]}",
                    origin="user_explicit",
                    verification_passed=True,
                    proposals=(
                        MemoryProposal(
                            scope=command.scope,
                            type=command.type,
                            key=command.key,
                            content=command.content,
                        ),
                    ),
                )
            )
            if receipt.rejected:
                reason = receipt.rejected[0].split(":", 1)[-1]
                raise ValueError(f"memory conflict or rejection: {reason}")
            return MemoryCommandResult(receipt.records, "memory added")
        if isinstance(command, ListMemory):
            records = self._all_records()
            if command.scope is not None:
                records = [record for record in records if record.scope == command.scope]
            if command.status is not None:
                records = [record for record in records if record.status == command.status]
            return MemoryCommandResult(tuple(records), "memory listed")
        if isinstance(command, ShowMemory):
            return MemoryCommandResult((self._require(command.memory_id),), "memory shown")
        if isinstance(command, ApproveMemory):
            record = self._require_status(command.memory_id, {"candidate"})
            repository = self._repository(record.scope)
            for current in repository.records_for_key(record.key):
                if current.id != record.id and current.status in {"active", "disabled"}:
                    repository.save(
                        replace(current, status="superseded", updated_at=self._now())
                    )
            approved = replace(
                record,
                source="user_explicit",
                status="active",
                updated_at=self._now(),
            )
            repository.save(approved)
            self._ensure_invariants(record.scope, record.key)
            return MemoryCommandResult((approved,), "memory approved")
        if isinstance(command, RejectMemory):
            record = self._require_status(command.memory_id, {"candidate"})
            rejected = replace(record, status="deleted", updated_at=self._now())
            self._repository(record.scope).save(rejected)
            return MemoryCommandResult((rejected,), "memory rejected")
        if isinstance(command, EditMemory):
            return MemoryCommandResult((self._edit(command),), "memory edited")
        if isinstance(command, DisableMemory):
            return MemoryCommandResult(
                (self._change_status(command.memory_id, "active", "disabled"),),
                "memory disabled",
            )
        if isinstance(command, EnableMemory):
            record = self._require_status(command.memory_id, {"disabled"})
            active = next(
                (
                    item
                    for item in self._repository(record.scope).records_for_key(record.key)
                    if item.id != record.id and item.status == "active"
                ),
                None,
            )
            if active is not None:
                raise ValueError(f"memory conflict with active record: {active.id}")
            enabled = replace(record, status="active", updated_at=self._now())
            self._repository(record.scope).save(enabled)
            self._ensure_invariants(record.scope, record.key)
            return MemoryCommandResult((enabled,), "memory enabled")
        if isinstance(command, DeleteMemory):
            record = self._require_status(
                command.memory_id,
                {"candidate", "active", "disabled", "superseded"},
            )
            deleted = replace(record, status="deleted", updated_at=self._now())
            self._repository(record.scope).save(deleted)
            return MemoryCommandResult((deleted,), "memory deleted")
        if isinstance(command, HistoryMemory):
            records = self._repository(command.scope).records_for_key(command.key)
            return MemoryCommandResult(tuple(records), "memory history")
        if isinstance(command, PurgeMemory):
            record = self._require(command.memory_id)
            self._repository(record.scope).purge(record.id)
            return MemoryCommandResult(message="memory purged")
        raise TypeError(f"unsupported memory command: {type(command).__name__}")

    def admit_user_prompt(
        self,
        text: str,
        *,
        session_id: str,
        run_id: str,
    ) -> MemoryProposalReceipt:
        parsed = _explicit_prompt_proposal(text)
        if parsed is None:
            return MemoryProposalReceipt()
        return self.submit_proposals(
            MemoryProposalBatch(
                session_id=session_id,
                run_id=run_id,
                origin="user_explicit",
                verification_passed=True,
                proposals=(parsed,),
            )
        )

    def dry_run_legacy_migration(self) -> MemoryMigrationReport:
        return self.migrator.dry_run()

    def migrate_legacy(self) -> MemoryMigrationReport:
        return self.migrator.migrate()

    def _edit(self, command: EditMemory) -> MemoryRecord:
        old = self._require_status(
            command.memory_id,
            {"candidate", "active", "disabled"},
        )
        content = normalize_content(command.content)
        if contains_sensitive_content(content):
            raise ValueError("memory contains sensitive content")
        if content == old.content:
            return old
        repository = self._repository(old.scope)
        repository.save(replace(old, status="superseded", updated_at=self._now()))
        new_status = "candidate" if old.status == "candidate" else "active"
        edited = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            scope=old.scope,
            type=old.type,
            key=old.key,
            content=content,
            source="user_explicit",
            status=new_status,
            updated_at=self._now(),
        )
        repository.save(edited)
        self._ensure_invariants(old.scope, old.key)
        return edited

    def _change_status(
        self,
        memory_id: str,
        expected: str,
        target: str,
    ) -> MemoryRecord:
        record = self._require_status(memory_id, {expected})
        changed = replace(record, status=target, updated_at=self._now())
        self._repository(record.scope).save(changed)
        return changed

    def _require_status(
        self,
        memory_id: str,
        statuses: set[str],
    ) -> MemoryRecord:
        record = self._require(memory_id)
        if record.status not in statuses:
            expected = ", ".join(sorted(statuses))
            raise ValueError(
                f"memory {memory_id} must be in one of [{expected}], got {record.status}"
            )
        return record

    def _require(self, memory_id: str) -> MemoryRecord:
        for repository in (self.project_repository, self.user_repository):
            record = repository.get(memory_id)
            if record is not None:
                return record
        raise ValueError(f"memory not found: {memory_id}")

    def _repository(
        self,
        scope: str,
    ) -> UserMemoryRepository | ProjectMemoryRepository:
        if scope == "user":
            return self.user_repository
        if scope == "project":
            return self.project_repository
        raise ValueError(f"unknown memory scope: {scope}")

    def _all_records(self) -> list[MemoryRecord]:
        return sorted(
            [
                *self.user_repository.all_records(),
                *self.project_repository.all_records(),
            ],
            key=lambda record: (record.updated_at, record.id),
        )

    def _ensure_invariants(self, scope: str, key: str) -> None:
        current = [
            record
            for record in self._repository(scope).records_for_key(key)
            if record.status in {"active", "disabled"}
        ]
        candidates = [
            record
            for record in self._repository(scope).records_for_key(key)
            if record.status == "candidate"
        ]
        if len(current) > 1:
            raise RuntimeError(f"multiple current memories for {scope}:{key}")
        if len(candidates) > 1:
            raise RuntimeError(f"multiple candidate memories for {scope}:{key}")

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


def render_memory(record: MemoryRecord) -> str:
    return f"[{record.scope}/{record.type}] {record.key}: {record.content}"


def _explicit_prompt_proposal(text: str) -> MemoryProposal | None:
    content = str(text or "").strip()
    marker = re.search(
        r"(?:请记住|记住|remember)\s*[:：]?\s*(.+)$",
        content,
        flags=re.IGNORECASE,
    )
    if marker is None:
        return None
    memory_content = normalize_content(marker.group(1))
    if not memory_content:
        return None
    digest = hashlib.sha256(memory_content.encode("utf-8")).hexdigest()[:12]
    return MemoryProposal(
        scope="project",
        type="project",
        key=f"project.note.{digest}",
        content=memory_content,
    )


__all__ = ["MemoryService", "render_memory"]
