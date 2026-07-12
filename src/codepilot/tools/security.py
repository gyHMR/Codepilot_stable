from __future__ import annotations

"""Canonical tool policy, resource and effect value objects."""

import fnmatch
import hashlib
import json
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Literal, Mapping, TypeAlias, TypeVar, cast


ToolMode: TypeAlias = Literal["plan", "execute", "unrestricted"]
RiskLevel: TypeAlias = Literal["low", "medium", "high", "critical"]
ApprovalPolicy: TypeAlias = Literal["never", "on_risk", "always"]
ToolEffectKind: TypeAlias = Literal[
    "filesystem_read",
    "filesystem_write",
    "filesystem_delete",
    "process_spawn",
    "network_access",
    "credential_access",
    "external_state_read",
    "external_state_write",
    "session_state_read",
    "session_state_write",
]
EffectStatus: TypeAlias = Literal["started", "completed", "partial", "unknown"]
EffectCertainty: TypeAlias = Literal["observed", "reported", "inferred"]
ContentTrust: TypeAlias = Literal["trusted", "untrusted"]
PermissionEffect: TypeAlias = Literal["allow", "deny", "ask"]
RuleSource: TypeAlias = Literal["configuration", "project", "runtime"]
ApprovalScope: TypeAlias = Literal["once", "session", "project"]
ApprovalDecisionValue: TypeAlias = Literal["approve", "deny"]

_TOOL_MODES = frozenset({"plan", "execute", "unrestricted"})
_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
_APPROVAL_POLICIES = frozenset({"never", "on_risk", "always"})
_EFFECT_KINDS = frozenset(
    {
        "filesystem_read",
        "filesystem_write",
        "filesystem_delete",
        "process_spawn",
        "network_access",
        "credential_access",
        "external_state_read",
        "external_state_write",
        "session_state_read",
        "session_state_write",
    }
)
_EFFECT_STATUSES = frozenset({"started", "completed", "partial", "unknown"})
_EFFECT_CERTAINTIES = frozenset({"observed", "reported", "inferred"})
_CONTENT_TRUST = frozenset({"trusted", "untrusted"})
_PERMISSION_EFFECTS = frozenset({"allow", "deny", "ask"})
_RULE_SOURCES = frozenset({"configuration", "project", "runtime"})
_APPROVAL_SCOPES = frozenset({"once", "session", "project"})
_APPROVAL_DECISIONS = frozenset({"approve", "deny"})
_RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_FORBIDDEN_MODEL_AUTH_KEYS = frozenset(
    {
        "allow_dangerous",
        "bypass_approval",
        "ignore_workspace_boundary",
        "trusted",
        "sudo",
        "force_without_approval",
    }
)


@dataclass(frozen=True)
class TimeoutPolicy:
    default_execution_ms: int
    max_execution_ms: int
    idle_timeout_ms: int | None = None
    cleanup_grace_ms: int = 5_000

    def __post_init__(self) -> None:
        _require_positive_int(self.default_execution_ms, "default_execution_ms")
        _require_positive_int(self.max_execution_ms, "max_execution_ms")
        if self.max_execution_ms < self.default_execution_ms:
            raise ValueError("max_execution_ms cannot be less than default_execution_ms")
        if self.idle_timeout_ms is not None:
            _require_positive_int(self.idle_timeout_ms, "idle_timeout_ms")
        _require_non_negative_int(self.cleanup_grace_ms, "cleanup_grace_ms")


@dataclass(frozen=True)
class ConcurrencyPolicy:
    mode: Literal["parallel", "serial"]
    group: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"parallel", "serial"}:
            raise ValueError(f"Unknown concurrency mode: {self.mode}")
        group = _optional_text(self.group)
        if self.mode == "serial" and group is None:
            raise ValueError("serial concurrency requires a group")
        object.__setattr__(self, "group", group)


@dataclass(frozen=True)
class OutputLimits:
    max_data_bytes: int = 256_000
    max_content_bytes: int = 128_000
    max_artifacts: int = 16
    max_artifact_bytes: int = 16_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_data_bytes",
            "max_content_bytes",
            "max_artifacts",
            "max_artifact_bytes",
        ):
            _require_positive_int(getattr(self, name), name)


@dataclass(frozen=True)
class OutputTrustPolicy:
    default_content_trust: ContentTrust = "trusted"
    allow_structurally_validated: bool = False

    def __post_init__(self) -> None:
        trust = _clean_text(self.default_content_trust)
        if trust not in _CONTENT_TRUST:
            raise ValueError(f"Unknown content trust: {self.default_content_trust}")
        if not isinstance(self.allow_structurally_validated, bool):
            raise TypeError("allow_structurally_validated must be bool")
        object.__setattr__(self, "default_content_trust", cast(ContentTrust, trust))


@dataclass(frozen=True)
class ToolPolicy:
    allowed_modes: frozenset[ToolMode]
    declared_effects: frozenset[ToolEffectKind]
    required_permissions: frozenset[str]
    base_risk: RiskLevel
    approval: ApprovalPolicy
    timeout: TimeoutPolicy
    concurrency: ConcurrencyPolicy
    output_limits: OutputLimits
    output_trust: OutputTrustPolicy

    def __post_init__(self) -> None:
        modes = frozenset(_clean_text(value) for value in self.allowed_modes)
        if not modes or not modes <= _TOOL_MODES:
            raise ValueError(f"Invalid allowed tool modes: {sorted(modes)}")
        effects = frozenset(_clean_text(value) for value in self.declared_effects)
        if not effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid declared effects: {sorted(effects)}")
        permissions = frozenset(_require_text(value, "required permission") for value in self.required_permissions)
        risk = _clean_text(self.base_risk)
        if risk not in _RISK_LEVELS:
            raise ValueError(f"Unknown tool risk level: {self.base_risk}")
        approval = _clean_text(self.approval)
        if approval not in _APPROVAL_POLICIES:
            raise ValueError(f"Unknown approval policy: {self.approval}")
        for name, expected in (
            ("timeout", TimeoutPolicy),
            ("concurrency", ConcurrencyPolicy),
            ("output_limits", OutputLimits),
            ("output_trust", OutputTrustPolicy),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        object.__setattr__(self, "allowed_modes", cast(frozenset[ToolMode], modes))
        object.__setattr__(self, "declared_effects", cast(frozenset[ToolEffectKind], effects))
        object.__setattr__(self, "required_permissions", permissions)
        object.__setattr__(self, "base_risk", cast(RiskLevel, risk))
        object.__setattr__(self, "approval", cast(ApprovalPolicy, approval))


@dataclass(frozen=True)
class ToolResource:
    uri: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _require_text(self.uri, "resource uri"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "resource metadata"))


@dataclass(frozen=True)
class ToolEffect:
    kind: ToolEffectKind
    resource: ToolResource
    operation: str
    status: EffectStatus
    certainty: EffectCertainty

    def __post_init__(self) -> None:
        kind = _clean_text(self.kind)
        if kind not in _EFFECT_KINDS:
            raise ValueError(f"Unknown tool effect: {self.kind}")
        if not isinstance(self.resource, ToolResource):
            raise TypeError("effect resource must be ToolResource")
        status = _clean_text(self.status)
        if status not in _EFFECT_STATUSES:
            raise ValueError(f"Unknown effect status: {self.status}")
        certainty = _clean_text(self.certainty)
        if certainty not in _EFFECT_CERTAINTIES:
            raise ValueError(f"Unknown effect certainty: {self.certainty}")
        object.__setattr__(self, "kind", cast(ToolEffectKind, kind))
        object.__setattr__(self, "operation", _require_text(self.operation, "effect operation"))
        object.__setattr__(self, "status", cast(EffectStatus, status))
        object.__setattr__(self, "certainty", cast(EffectCertainty, certainty))


@dataclass(frozen=True)
class ToolAccessRequest:
    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    effects: frozenset[ToolEffectKind]
    risk: RiskLevel
    reason: str
    safe_preview: Mapping[str, object] = field(default_factory=dict)
    approval_scopes: frozenset[ApprovalScope] = frozenset(
        {"once", "session", "project"}
    )

    def __post_init__(self) -> None:
        actions = tuple(_require_text(value, "access action") for value in self.actions)
        if not actions:
            raise ValueError("access actions cannot be empty")
        resources = tuple(self.resources)
        if any(not isinstance(value, ToolResource) for value in resources):
            raise TypeError("access resources must be ToolResource values")
        effects = frozenset(_clean_text(value) for value in self.effects)
        if not effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid access effects: {sorted(effects)}")
        risk = _clean_text(self.risk)
        if risk not in _RISK_LEVELS:
            raise ValueError(f"Unknown access risk: {self.risk}")
        scopes = frozenset(_clean_text(value) for value in self.approval_scopes)
        if not scopes or not scopes <= _APPROVAL_SCOPES:
            raise ValueError(f"Invalid access approval scopes: {sorted(scopes)}")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "effects", cast(frozenset[ToolEffectKind], effects))
        object.__setattr__(self, "risk", cast(RiskLevel, risk))
        object.__setattr__(self, "reason", _require_text(self.reason, "access reason"))
        object.__setattr__(self, "safe_preview", _freeze_mapping(self.safe_preview, "safe preview"))
        object.__setattr__(
            self,
            "approval_scopes",
            cast(frozenset[ApprovalScope], scopes),
        )


TInput = TypeVar("TInput")


@dataclass(frozen=True)
class ToolAccessResolution(Generic[TInput]):
    input: TInput
    access: ToolAccessRequest

    def __post_init__(self) -> None:
        if not isinstance(self.access, ToolAccessRequest):
            raise TypeError("access must be ToolAccessRequest")
        object.__setattr__(self, "input", deepcopy(self.input))


@dataclass(frozen=True)
class PermissionRule:
    action_pattern: str
    resource_pattern: str
    effect: PermissionEffect
    modes: frozenset[ToolMode] = frozenset()
    source: RuleSource = "configuration"
    priority: int = 0

    def __post_init__(self) -> None:
        effect = _clean_text(self.effect)
        source = _clean_text(self.source)
        modes = frozenset(_clean_text(value) for value in self.modes)
        if effect not in _PERMISSION_EFFECTS:
            raise ValueError(f"Unknown permission effect: {self.effect}")
        if source not in _RULE_SOURCES:
            raise ValueError(f"Unknown permission rule source: {self.source}")
        if not modes <= _TOOL_MODES:
            raise ValueError(f"Invalid permission rule modes: {sorted(modes)}")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("permission rule priority must be int")
        object.__setattr__(self, "action_pattern", _require_text(self.action_pattern, "action pattern"))
        object.__setattr__(
            self,
            "resource_pattern",
            _require_text(self.resource_pattern, "resource pattern"),
        )
        object.__setattr__(self, "effect", cast(PermissionEffect, effect))
        object.__setattr__(self, "modes", cast(frozenset[ToolMode], modes))
        object.__setattr__(self, "source", cast(RuleSource, source))


@dataclass(frozen=True)
class PermissionDecision:
    effect: PermissionEffect
    reason: str
    matched_rule: PermissionRule | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        effect = _clean_text(self.effect)
        if effect not in _PERMISSION_EFFECTS:
            raise ValueError(f"Unknown permission decision: {self.effect}")
        object.__setattr__(self, "effect", cast(PermissionEffect, effect))
        object.__setattr__(self, "reason", _require_text(self.reason, "permission reason"))
        object.__setattr__(self, "details", _freeze_mapping(self.details, "permission details"))

    @property
    def allowed(self) -> bool:
        return self.effect == "allow"

    @property
    def denied(self) -> bool:
        return self.effect == "deny"

    @property
    def requires_approval(self) -> bool:
        return self.effect == "ask"


@dataclass(frozen=True)
class PermissionEngine:
    rules: tuple[PermissionRule, ...] = ()
    denied_effects: frozenset[ToolEffectKind] = frozenset()
    approval_effects: frozenset[ToolEffectKind] = frozenset()

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if any(not isinstance(rule, PermissionRule) for rule in rules):
            raise TypeError("permission rules must be PermissionRule values")
        denied_effects = frozenset(self.denied_effects)
        approval_effects = frozenset(self.approval_effects)
        if not denied_effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid denied effects: {sorted(denied_effects)}")
        if not approval_effects <= _EFFECT_KINDS:
            raise ValueError(f"Invalid approval effects: {sorted(approval_effects)}")
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "denied_effects", denied_effects)
        object.__setattr__(self, "approval_effects", approval_effects)

    def decide(self, request, policy: ToolPolicy, access: ToolAccessRequest) -> PermissionDecision:
        if _contains_forbidden_auth(request.arguments):
            return PermissionDecision("deny", "model_authorization_forbidden")
        if request.mode not in policy.allowed_modes:
            return PermissionDecision("deny", "tool_mode_denied")
        if not access.effects <= policy.declared_effects:
            return PermissionDecision("deny", "tool_effect_policy_violation")
        if access.safe_preview.get("shell_class") == "high_risk":
            return PermissionDecision("deny", "shell_high_risk_forbidden")
        if access.effects & self.denied_effects:
            return PermissionDecision("deny", "runtime_effect_denied")
        if access.effects & self.approval_effects:
            return PermissionDecision("ask", "runtime_effect_requires_approval")

        matched = self._matched_rule(request.mode, access)
        if matched is not None:
            return PermissionDecision(matched.effect, f"permission_rule_{matched.effect}", matched)
        if policy.approval == "always":
            return PermissionDecision("ask", "tool_policy_requires_approval")
        risk = max(_RISK_RANK[policy.base_risk], _RISK_RANK[access.risk])
        if policy.approval == "on_risk" and risk >= _RISK_RANK["medium"]:
            return PermissionDecision("ask", "tool_risk_requires_approval")
        return PermissionDecision("allow", "tool_policy_allow")

    def _matched_rule(self, mode: ToolMode, access: ToolAccessRequest) -> PermissionRule | None:
        resource_uris = tuple(resource.uri for resource in access.resources) or ("",)
        effect_rank = {"allow": 0, "ask": 1, "deny": 2}
        def rank(rule: PermissionRule) -> tuple[int, int, int]:
            return (
                rule.priority,
                _pattern_specificity(rule.action_pattern) + _pattern_specificity(rule.resource_pattern),
                effect_rank[rule.effect],
            )
        selected = []
        for action in access.actions:
            for resource_uri in resource_uris:
                matches = [
                    rule
                    for rule in self.rules
                    if (not rule.modes or mode in rule.modes)
                    and fnmatch.fnmatchcase(action, rule.action_pattern)
                    and fnmatch.fnmatchcase(resource_uri, rule.resource_pattern)
                ]
                if not matches:
                    return None
                selected.append(max(matches, key=rank))
        for effect in ("deny", "ask", "allow"):
            candidates = [rule for rule in selected if rule.effect == effect]
            if candidates:
                return max(candidates, key=rank)
        return None


@dataclass(frozen=True)
class ApprovalChallenge:
    approval_id: str
    request_fingerprint: str
    run_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    registration_id: str
    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    effects: frozenset[ToolEffectKind]
    risk: RiskLevel
    reason: str
    safe_preview: Mapping[str, object]
    allowed_scopes: frozenset[ApprovalScope] = frozenset({"once"})
    expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        scopes = frozenset(_clean_text(value) for value in self.allowed_scopes)
        if not scopes or not scopes <= _APPROVAL_SCOPES:
            raise ValueError(f"Invalid approval scopes: {sorted(scopes)}")
        for name in (
            "approval_id",
            "request_fingerprint",
            "run_id",
            "session_id",
            "tool_call_id",
            "tool_name",
            "registration_id",
            "reason",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        object.__setattr__(self, "actions", tuple(_require_text(value, "approval action") for value in self.actions))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "effects", frozenset(self.effects))
        object.__setattr__(self, "allowed_scopes", cast(frozenset[ApprovalScope], scopes))
        object.__setattr__(self, "safe_preview", _freeze_mapping(self.safe_preview, "safe preview"))
        if self.expires_at_ms is not None:
            _require_non_negative_int(self.expires_at_ms, "expires_at_ms")


@dataclass(frozen=True)
class ApprovalResponse:
    approval_id: str
    request_fingerprint: str
    decision: ApprovalDecisionValue
    scope: ApprovalScope = "once"
    reason: str = ""

    def __post_init__(self) -> None:
        decision = _clean_text(self.decision)
        scope = _clean_text(self.scope)
        if decision not in _APPROVAL_DECISIONS:
            raise ValueError(f"Unknown approval decision: {self.decision}")
        if scope not in _APPROVAL_SCOPES:
            raise ValueError(f"Unknown approval scope: {self.scope}")
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(
            self,
            "request_fingerprint",
            _require_text(self.request_fingerprint, "request_fingerprint"),
        )
        object.__setattr__(self, "decision", cast(ApprovalDecisionValue, decision))
        object.__setattr__(self, "scope", cast(ApprovalScope, scope))
        object.__setattr__(self, "reason", _clean_text(self.reason))


@dataclass(frozen=True)
class ApprovalGrant:
    grant_id: str
    approval_id: str
    request_fingerprint: str
    scope: ApprovalScope
    actions: tuple[str, ...]
    resources: tuple[ToolResource, ...]
    issued_at_ms: int
    expires_at_ms: int | None

    def __post_init__(self) -> None:
        scope = _clean_text(self.scope)
        if scope not in _APPROVAL_SCOPES:
            raise ValueError(f"Unknown approval scope: {self.scope}")
        object.__setattr__(self, "grant_id", _require_text(self.grant_id, "grant_id"))
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(
            self,
            "request_fingerprint",
            _require_text(self.request_fingerprint, "request_fingerprint"),
        )
        object.__setattr__(self, "scope", cast(ApprovalScope, scope))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "resources", tuple(self.resources))
        _require_non_negative_int(self.issued_at_ms, "issued_at_ms")
        if self.expires_at_ms is not None:
            _require_non_negative_int(self.expires_at_ms, "expires_at_ms")

    def expired(self, now_ms: int | None = None) -> bool:
        now = int(time.time() * 1000) if now_ms is None else now_ms
        return self.expires_at_ms is not None and now >= self.expires_at_ms


def approval_fingerprint(request, access: ToolAccessRequest) -> str:
    payload = {
        "run_id": request.run_id,
        "session_id": request.session_id,
        "tool_call_id": request.tool_call_id,
        "tool_name": request.tool_name,
        "registration_id": request.registration_id,
        "arguments": _json_value(request.arguments),
        "actions": list(access.actions),
        "resources": [resource.uri for resource in access.resources],
        "effects": sorted(access.effects),
        "approval_scopes": sorted(access.approval_scopes),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_approval_challenge(
    request,
    access: ToolAccessRequest,
    *,
    reason: str,
    ttl_ms: int = 300_000,
) -> ApprovalChallenge:
    now_ms = int(time.time() * 1000)
    return ApprovalChallenge(
        approval_id="approval_" + uuid.uuid4().hex[:20],
        request_fingerprint=approval_fingerprint(request, access),
        run_id=request.run_id,
        session_id=request.session_id,
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        registration_id=request.registration_id,
        actions=access.actions,
        resources=access.resources,
        effects=access.effects,
        risk=access.risk,
        reason=reason,
        safe_preview=access.safe_preview,
        allowed_scopes=access.approval_scopes,
        expires_at_ms=now_ms + ttl_ms,
    )


def issue_approval_grant(
    challenge: ApprovalChallenge,
    response: ApprovalResponse,
) -> ApprovalGrant:
    now_ms = int(time.time() * 1000)
    return ApprovalGrant(
        grant_id="grant_" + uuid.uuid4().hex[:20],
        approval_id=challenge.approval_id,
        request_fingerprint=challenge.request_fingerprint,
        scope=response.scope,
        actions=challenge.actions,
        resources=challenge.resources,
        issued_at_ms=now_ms,
        expires_at_ms=challenge.expires_at_ms,
    )


def approval_challenge_data(challenge: ApprovalChallenge) -> dict[str, object]:
    return {
        "approval_id": challenge.approval_id,
        "request_fingerprint": challenge.request_fingerprint,
        "run_id": challenge.run_id,
        "session_id": challenge.session_id,
        "tool_call_id": challenge.tool_call_id,
        "tool_name": challenge.tool_name,
        "registration_id": challenge.registration_id,
        "actions": list(challenge.actions),
        "resources": [resource.uri for resource in challenge.resources],
        "effects": sorted(challenge.effects),
        "risk": challenge.risk,
        "reason": challenge.reason,
        "safe_preview": dict(challenge.safe_preview),
        "allowed_scopes": sorted(challenge.allowed_scopes),
        "expires_at_ms": challenge.expires_at_ms,
    }


def _contains_forbidden_auth(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(_FORBIDDEN_MODEL_AUTH_KEYS.intersection(str(key) for key in value)) or any(
            _contains_forbidden_auth(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_auth(item) for item in value)
    return False


def _pattern_specificity(pattern: str) -> int:
    return sum(1 for char in pattern if char not in "*?[]")


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return cast(Mapping[str, object], _freeze_value(value))


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_text(value: object, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object) -> str | None:
    return _clean_text(value) or None


__all__ = [
    "ApprovalChallenge",
    "ApprovalGrant",
    "ApprovalPolicy",
    "ApprovalResponse",
    "ApprovalScope",
    "ConcurrencyPolicy",
    "ContentTrust",
    "EffectCertainty",
    "EffectStatus",
    "OutputLimits",
    "OutputTrustPolicy",
    "PermissionDecision",
    "PermissionEffect",
    "PermissionEngine",
    "PermissionRule",
    "RiskLevel",
    "TimeoutPolicy",
    "ToolAccessRequest",
    "ToolAccessResolution",
    "ToolEffect",
    "ToolEffectKind",
    "ToolMode",
    "ToolPolicy",
    "ToolResource",
    "approval_challenge_data",
    "approval_fingerprint",
    "build_approval_challenge",
    "issue_approval_grant",
]
