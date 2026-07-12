from __future__ import annotations

"""Static Skill Package discovery, validation, and resource boundaries."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, Mapping, cast

import yaml


SkillSource = Literal["workspace", "external"]
SkillTrust = Literal["workspace", "explicitly_configured"]

_ENTRYPOINT = "SKILL.md"
_RESOURCE_DIRS = frozenset({"references", "scripts", "assets"})
_MANIFEST_FIELDS = frozenset(
    {
        "name",
        "version",
        "description",
        "command",
        "allowed_modes",
        "required_tools",
        "required_mcp",
    }
)
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_CAPABILITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MAX_ENTRYPOINT_BYTES = 256_000
_MAX_RESOURCE_BYTES = 256_000
_MAX_RESOURCE_FILES = 256


class SkillPackageError(ValueError):
    """Raised when a Skill Package violates its static contract."""


@dataclass(frozen=True)
class SkillManifest:
    name: str
    version: str
    description: str
    command: str | None = None
    allowed_modes: frozenset[str] = frozenset({"plan", "execute"})
    required_tools: tuple[str, ...] = ()
    required_mcp: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillPackage:
    manifest: SkillManifest
    root: Path
    entrypoint: Path
    source: SkillSource
    trust: SkillTrust
    digest: str

    def read_instructions(self) -> str:
        _meta, body = _read_entrypoint(self.entrypoint)
        return body

    def list_resources(self) -> tuple[str, ...]:
        resources: list[str] = []
        for directory in sorted(_RESOURCE_DIRS):
            base = self.root / directory
            if not base.exists():
                continue
            if not base.is_dir() or base.is_symlink():
                raise SkillPackageError(
                    f"skill resource directory must be a real directory: {base}"
                )
            for candidate in sorted(base.rglob("*")):
                if candidate.is_symlink():
                    raise SkillPackageError(
                        f"skill resources cannot contain symbolic links: {candidate}"
                    )
                if candidate.is_file():
                    resources.append(candidate.relative_to(self.root).as_posix())
                    if len(resources) > _MAX_RESOURCE_FILES:
                        raise SkillPackageError(
                            f"skill {self.manifest.name} contains more than "
                            f"{_MAX_RESOURCE_FILES} resource files"
                        )
        return tuple(resources)

    def resolve_resource(self, raw_path: str) -> Path:
        normalized = str(raw_path or "").strip().replace("\\", "/")
        relative = PurePosixPath(normalized)
        if (
            not normalized
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] not in _RESOURCE_DIRS
        ):
            raise SkillPackageError(
                "skill resource path must be relative to references/, scripts/, or assets/"
            )
        candidate = self.root.joinpath(*relative.parts)
        resolved_root = self.root.resolve(strict=True)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (FileNotFoundError, ValueError) as exc:
            raise SkillPackageError(f"skill resource not found: {normalized}") from exc
        if not resolved.is_file() or candidate.is_symlink():
            raise SkillPackageError(f"skill resource is not a regular file: {normalized}")
        return resolved

    def read_resource(self, raw_path: str) -> str:
        path = self.resolve_resource(raw_path)
        return _read_utf8_file(path, max_bytes=_MAX_RESOURCE_BYTES, label="skill resource")


@dataclass
class SkillCatalog:
    packages: list[SkillPackage] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    loaded_paths: list[str] = field(default_factory=list)
    _by_name: dict[str, SkillPackage] = field(default_factory=dict, init=False, repr=False)
    _by_command: dict[str, SkillPackage] = field(default_factory=dict, init=False, repr=False)

    def add(self, package: SkillPackage) -> None:
        name = package.manifest.name
        command = package.manifest.command
        if name in self._by_name:
            raise SkillPackageError(f"duplicate skill name: {name}")
        if command and command in self._by_command:
            raise SkillPackageError(f"duplicate skill command: /{command}")
        self.packages.append(package)
        self._by_name[name] = package
        if command:
            self._by_command[command] = package
        self.loaded_paths.append(str(package.root))

    def find(self, value: str) -> SkillPackage | None:
        key = str(value or "").strip().lstrip("/").lower()
        return self._by_name.get(key) or self._by_command.get(key)


def load_skill_catalog(
    workspace_dir: str | Path,
    configured_paths: list[str] | None = None,
) -> SkillCatalog:
    """Discover package directories and load only their validated manifests."""

    workspace = Path(workspace_dir).resolve()
    default_root = workspace / ".codepilot" / "skills"
    catalog = SkillCatalog()
    candidates: list[tuple[Path, SkillSource, SkillTrust]] = []
    seen: set[str] = set()

    def add_candidate(path: Path, source: SkillSource, trust: SkillTrust) -> None:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            catalog.errors.append(f"skill path does not exist: {path}")
            return
        key = str(resolved).casefold()
        if key in seen:
            return
        seen.add(key)
        candidates.append((resolved, source, trust))

    if default_root.exists():
        if not default_root.is_dir():
            catalog.errors.append(f"workspace skill root is not a directory: {default_root}")
        else:
            for path in sorted(default_root.iterdir()):
                if path.is_dir() and (path / _ENTRYPOINT).exists():
                    add_candidate(path, "workspace", "workspace")

    for raw in configured_paths or []:
        target = Path(raw)
        if not target.is_absolute():
            target = workspace / target
        if target.is_file():
            catalog.errors.append(
                f"flat skill files are not supported; configure a package directory: {target}"
            )
            continue
        if not target.exists():
            catalog.errors.append(f"skill path does not exist: {target}")
            continue
        if not target.is_dir():
            catalog.errors.append(f"skill path is not a directory: {target}")
            continue
        if (target / _ENTRYPOINT).is_file():
            add_candidate(target, "external", "explicitly_configured")
            continue
        for path in sorted(target.iterdir()):
            if path.is_dir() and (path / _ENTRYPOINT).exists():
                add_candidate(path, "external", "explicitly_configured")

    for root, source, trust in candidates:
        try:
            catalog.add(_load_package(root, source=source, trust=trust))
        except Exception as exc:
            catalog.errors.append(f"{root}: {exc}")
    return catalog


def _load_package(root: Path, *, source: SkillSource, trust: SkillTrust) -> SkillPackage:
    if root.is_symlink():
        raise SkillPackageError(f"skill package root cannot be a symbolic link: {root}")
    entrypoint = root / _ENTRYPOINT
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise SkillPackageError(f"skill package must contain a regular {_ENTRYPOINT}")
    meta, body = _read_entrypoint(entrypoint)
    manifest = _parse_manifest(meta)
    if root.name != manifest.name:
        raise SkillPackageError(
            f"skill directory name '{root.name}' must match manifest name '{manifest.name}'"
        )
    digest = hashlib.sha256(
        (manifest.name + "\0" + manifest.version + "\0" + body).encode("utf-8")
    ).hexdigest()
    return SkillPackage(
        manifest=manifest,
        root=root,
        entrypoint=entrypoint,
        source=source,
        trust=trust,
        digest="sha256:" + digest,
    )


def _read_entrypoint(path: Path) -> tuple[Mapping[str, object], str]:
    text = _read_utf8_file(path, max_bytes=_MAX_ENTRYPOINT_BYTES, label="SKILL.md")
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise SkillPackageError("SKILL.md must begin with YAML frontmatter")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as exc:
        raise SkillPackageError("SKILL.md frontmatter is not closed") from exc
    try:
        raw_meta = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        raise SkillPackageError(f"invalid SKILL.md YAML frontmatter: {exc}") from exc
    if not isinstance(raw_meta, dict):
        raise SkillPackageError("SKILL.md frontmatter must be a YAML object")
    if any(not isinstance(key, str) for key in raw_meta):
        raise SkillPackageError("SKILL.md frontmatter keys must be strings")
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise SkillPackageError("SKILL.md instruction body cannot be empty")
    return cast(Mapping[str, object], raw_meta), body


def _parse_manifest(raw: Mapping[str, object]) -> SkillManifest:
    unknown = sorted(set(raw) - _MANIFEST_FIELDS)
    if unknown:
        raise SkillPackageError(f"unknown skill manifest fields: {', '.join(unknown)}")
    name = _manifest_identifier(raw.get("name"), "name", _NAME_RE)
    version = _manifest_identifier(raw.get("version"), "version", _VERSION_RE)
    description = _required_string(raw.get("description"), "description")
    if len(description) > 500 or "\n" in description:
        raise SkillPackageError("skill description must be one line and at most 500 characters")
    command_value = raw.get("command")
    command = (
        _manifest_identifier(command_value, "command", _NAME_RE)
        if command_value is not None
        else None
    )
    allowed_modes = frozenset(_string_list(raw.get("allowed_modes"), "allowed_modes", default=("plan", "execute")))
    if not allowed_modes or not allowed_modes <= {"plan", "execute"}:
        raise SkillPackageError("allowed_modes must contain only plan and/or execute")
    required_tools = _capability_list(raw.get("required_tools"), "required_tools")
    required_mcp = _capability_list(raw.get("required_mcp"), "required_mcp")
    return SkillManifest(
        name=name,
        version=version,
        description=description,
        command=command,
        allowed_modes=allowed_modes,
        required_tools=required_tools,
        required_mcp=required_mcp,
    )


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SkillPackageError(f"skill manifest {field_name} must be a non-empty string")
    return value.strip()


def _manifest_identifier(value: object, field_name: str, pattern: re.Pattern[str]) -> str:
    text = _required_string(value, field_name)
    if not pattern.fullmatch(text):
        raise SkillPackageError(f"invalid skill manifest {field_name}: {text}")
    return text


def _string_list(
    value: object,
    field_name: str,
    *,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SkillPackageError(f"skill manifest {field_name} must be a list of strings")
    cleaned = tuple(item.strip() for item in value)
    if any(not item for item in cleaned) or len(set(cleaned)) != len(cleaned):
        raise SkillPackageError(f"skill manifest {field_name} contains empty or duplicate values")
    return cleaned


def _capability_list(value: object, field_name: str) -> tuple[str, ...]:
    values = _string_list(value, field_name)
    if any(not _CAPABILITY_RE.fullmatch(item) for item in values):
        raise SkillPackageError(f"skill manifest {field_name} contains an invalid capability name")
    return values


def _read_utf8_file(path: Path, *, max_bytes: int, label: str) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        raise SkillPackageError(f"{label} exceeds the {max_bytes}-byte limit: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillPackageError(f"{label} must be UTF-8 text: {path}") from exc


__all__ = [
    "SkillCatalog",
    "SkillManifest",
    "SkillPackage",
    "SkillPackageError",
    "SkillSource",
    "SkillTrust",
    "load_skill_catalog",
]
