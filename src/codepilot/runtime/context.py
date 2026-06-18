from __future__ import annotations

"""Runtime context facts used by prompt rendering."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from codepilot.extensions.types import LoadedExtensions
from codepilot.sessions.memory import load_global_memory

from .config import ResolvedRuntimeConfig

_MANIFEST_PROJECT_TYPES = {
    "pyproject.toml": "Python",
    "package.json": "JavaScript/TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
}
_TEST_DIR_NAMES = {"test", "tests", "spec", "specs"}
_TOP_LEVEL_LIMIT = 30


@dataclass(frozen=True)
class RepositoryBootstrap:
    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    git_branch: str | None
    git_dirty: bool | None


@dataclass(frozen=True)
class RuntimeContext:
    repository_context: str
    prompt_guidelines: list[str]
    append_sections: list[str]
    tool_snippets: dict[str, str]
    memory_text: str


def build_repository_bootstrap(workspace: Path) -> RepositoryBootstrap:
    root = workspace.resolve()
    entries = _top_level_entries(root)
    manifest_files = [name for name in _MANIFEST_PROJECT_TYPES if (root / name).is_file()]
    project_type = _project_type(manifest_files)
    test_directories = [
        entry
        for entry in entries
        if entry.endswith("/") and entry[:-1] in _TEST_DIR_NAMES
    ]
    git_branch, git_dirty = _git_status(root)
    return RepositoryBootstrap(
        workspace_root=str(root).replace("\\", "/"),
        project_type=project_type,
        manifest_files=manifest_files,
        top_level_entries=entries,
        test_directories=test_directories,
        git_branch=git_branch,
        git_dirty=git_dirty,
    )


def render_repository_context(bootstrap: RepositoryBootstrap) -> str:
    project_type = bootstrap.project_type or "unknown"
    manifests = ", ".join(bootstrap.manifest_files) if bootstrap.manifest_files else "(none)"
    top_level = ", ".join(bootstrap.top_level_entries) if bootstrap.top_level_entries else "(empty)"
    tests = ", ".join(bootstrap.test_directories) if bootstrap.test_directories else "(none)"
    branch = bootstrap.git_branch or "unknown"
    dirty = "unknown" if bootstrap.git_dirty is None else ("modified" if bootstrap.git_dirty else "clean")
    return "\n".join(
        [
            "## Repository Context",
            f"- Workspace: {bootstrap.workspace_root}",
            f"- Project type: {project_type}",
            f"- Manifests: {manifests}",
            f"- Top-level: {top_level}",
            f"- Test directories: {tests}",
            f"- Git branch: {branch}",
            f"- Working tree: {dirty}",
        ]
    )


def build_runtime_context(
    workspace: Path,
    config: ResolvedRuntimeConfig,
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> RuntimeContext:
    prompt_guidelines = [
        *(config.prompt_guidelines or []),
        *loaded_extensions.prompt_guidelines,
        *loaded_skills.prompt_guidelines,
    ]
    prompt_guidelines.extend([f"[skill-diagnostic] {d}" for d in loaded_skills.diagnostics])

    append_sections = []
    if config.append_system_prompt:
        append_sections.append(config.append_system_prompt)
    append_sections.extend(loaded_extensions.append_prompts)
    append_sections.extend(loaded_skills.append_prompts)
    if config.prompt_debug_sources:
        append_sections.append("\n".join(_build_prompt_debug_lines(loaded_extensions, loaded_skills)))

    return RuntimeContext(
        repository_context=render_repository_context(build_repository_bootstrap(workspace)),
        prompt_guidelines=prompt_guidelines,
        append_sections=append_sections,
        tool_snippets=config.tool_snippets or {},
        memory_text=load_global_memory(workspace),
    )


def _top_level_entries(root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    items = sorted(root.iterdir(), key=lambda item: item.name.lower())
    entries = []
    for item in items[:_TOP_LEVEL_LIMIT]:
        entries.append(f"{item.name}/" if item.is_dir() else item.name)
    return entries


def _project_type(manifest_files: list[str]) -> str | None:
    for manifest in _MANIFEST_PROJECT_TYPES:
        if manifest in manifest_files:
            return _MANIFEST_PROJECT_TYPES[manifest]
    return None


def _git_status(root: Path) -> tuple[str | None, bool | None]:
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None, None
    if branch.returncode != 0 or status.returncode != 0:
        return None, None
    branch_text = branch.stdout.strip() or None
    return branch_text, bool(status.stdout.strip())


def _build_prompt_debug_lines(
    loaded_extensions: LoadedExtensions,
    loaded_skills: LoadedExtensions,
) -> list[str]:
    debug_lines: list[str] = ["## Prompt Sources", "### extensions"]
    debug_lines.extend([f"- {p}" for p in loaded_extensions.loaded_paths] or ["- (none)"])
    debug_lines.append("### skills")
    debug_lines.extend([f"- {p}" for p in loaded_skills.loaded_paths] or ["- (none)"])
    if loaded_extensions.errors or loaded_skills.errors:
        debug_lines.append("### errors")
        debug_lines.extend([f"- {e}" for e in [*loaded_extensions.errors, *loaded_skills.errors]])
    if loaded_skills.diagnostics:
        debug_lines.append("### diagnostics")
        debug_lines.extend([f"- {d}" for d in loaded_skills.diagnostics])
    return debug_lines
