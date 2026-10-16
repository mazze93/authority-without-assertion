"""Protocol freeze: executable preregistration constraints.

A run is allowed only when protocol/freeze.yaml is frozen and complete,
research.yaml agrees with it, the requested backend is covered, the local
runtime (when applicable) matches the frozen facts, and the checkout is the
exact clean tracked source state named by the frozen (annotated) protocol tag,
with no untracked files among the protocol and source paths.

Every run record carries a stamp() of the facts the guard admitted, and
stamp_problems() lets scoring refuse records that lack one or mix protocols.

Pure except for load_freeze(), collect_local_facts(), and collect_source_facts().
"""
from __future__ import annotations

import importlib.metadata
import os
import pathlib
import platform
import subprocess
from typing import Any

import yaml

LOCAL_REQUIRED = (
    "repo_id", "revision", "artifact_path", "mlx_version", "mlx_lm_version",
    "python_version", "macos_version", "temperature_requested",
    "temperature_behavior", "licence_reviewed_at", "redistributable", "attested_by",
)
API_REQUIRED = (
    "model_requested", "terms_reviewed_at", "redistributable",
    "temperature_requested", "temperature_behavior", "attested_by",
)
ENVIRONMENT_FACTS = ("revision", "mlx_version", "mlx_lm_version", "python_version", "macos_version")
TEMPERATURE_BEHAVIORS = ("accepted", "rejected", "ignored")
# Paths whose contents define what a trial is. Untracked files here would be
# loaded (a scenario glob, an importable module) without being frozen.
PROTOCOL_PATHS = ("protocol", "src", "research.yaml")
# Generated files never loaded as protocol: Python bytecode caches, the
# `pip install -e` metadata directory, and macOS Finder metadata.
GENERATED_DIR_MARKERS = ("__pycache__", ".egg-info")
GENERATED_FILE_NAMES = (".DS_Store",)


def load_freeze(root: pathlib.Path) -> dict[str, Any]:
    return yaml.safe_load((root / "protocol" / "freeze.yaml").read_text())["protocol_freeze"]


def _pkg(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def hf_snapshot(repo_id: str) -> tuple[str | None, str | None]:
    """(revision, path) of the locally cached snapshot; never downloads."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None, None
    try:
        path = snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:
        return None, None
    return pathlib.PurePath(path).name, str(path)


def collect_local_facts(repo_id: str) -> dict[str, Any]:
    """Facts from the environment that will actually execute the local pilot."""
    revision, path = hf_snapshot(repo_id)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "artifact_path": path,
        "mlx_version": _pkg("mlx"),
        "mlx_lm_version": _pkg("mlx-lm"),
        "huggingface_hub_version": _pkg("huggingface-hub"),
        "python_version": platform.python_version(),
        "macos_version": platform.mac_ver()[0] or None,
    }


def _git(root: pathlib.Path, *args: str, stdin: str | None = None) -> str | None:
    try:
        p = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return p.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _is_generated(path: str) -> bool:
    parts = path.split("/")
    return path.endswith(".pyc") or parts[-1] in GENERATED_FILE_NAMES or any(
        part.endswith(marker) for part in parts[:-1] for marker in GENERATED_DIR_MARKERS
    )


def _untracked_protocol_files(root: pathlib.Path) -> list[str] | None:
    """Untracked files, ignored ones included, under PROTOCOL_PATHS."""
    out = _git(root, "ls-files", "--others", "-z", "--", *PROTOCOL_PATHS)
    if out is None:
        return None
    return sorted(p for p in out.split("\0") if p and not _is_generated(p))


def _worktree_mismatch(root: pathlib.Path) -> list[str] | None:
    """Tracked paths whose working-tree content differs from HEAD.

    Hashes the files themselves instead of asking `git status`, which a
    skip-worktree or assume-unchanged flag silences.
    """
    listing = _git(root, "ls-tree", "-r", "-z", "HEAD")
    if listing is None:
        return None
    regular: list[tuple[str, str]] = []
    mismatched: list[str] = []
    for entry in filter(None, listing.split("\0")):
        meta, path = entry.split("\t", 1)
        mode, _kind, blob = meta.split()
        full = root / path
        if mode == "160000":                      # submodule: not file content
            continue
        if mode == "120000":
            if not full.is_symlink():
                mismatched.append(path)
                continue
            got = _git(root, "hash-object", "--stdin", stdin=os.readlink(full))
            if got != blob:
                mismatched.append(path)
            continue
        if not full.is_file() or full.is_symlink():
            mismatched.append(path)
            continue
        regular.append((path, blob))
    if regular:
        hashes = _git(root, "hash-object", "--stdin-paths",
                      stdin="\n".join(p for p, _ in regular) + "\n")
        if hashes is None:
            return None
        got = hashes.splitlines()
        if len(got) != len(regular):
            return None
        mismatched += [p for (p, blob), h in zip(regular, got) if h != blob]
    return sorted(mismatched)


def collect_source_facts(root: pathlib.Path, tag: str | None = None) -> dict[str, Any]:
    """Identity of the checked-out protocol source.

    Untracked run/preflight artifacts are deliberately ignored: they are data,
    not protocol mutations. Untracked files under PROTOCOL_PATHS, and any
    tracked file whose content differs from HEAD, block execution.
    """
    head = _git(root, "rev-parse", "HEAD")
    tag_text = _git(root, "tag", "--points-at", "HEAD")
    diff = _git(root, "status", "--porcelain", "--untracked-files=no")
    ref = f"refs/tags/{tag}" if tag else None
    return {
        "head": head,
        "tags_at_head": tag_text.splitlines() if tag_text is not None else None,
        "tracked_dirty": None if diff is None else bool(diff),
        "tag_type": _git(root, "cat-file", "-t", ref) if ref else None,
        "tag_object": _git(root, "rev-parse", ref) if ref else None,
        "untracked_protocol_files": _untracked_protocol_files(root),
        "worktree_mismatch": _worktree_mismatch(root),
    }


def requested_temperature(backend: dict[str, Any]) -> float | None:
    """The temperature a run sends: research.yaml's value, or None (omitted).

    The guard and the runner both read it here, so they cannot disagree about
    a default.
    """
    return backend.get("temperature")


def _temperature_problems(name: str, entry: dict[str, Any], backend: dict[str, Any]) -> list[str]:
    behavior = entry.get("temperature_behavior")
    requested = entry.get("temperature_requested")
    sent = requested_temperature(backend)
    if behavior not in TEMPERATURE_BEHAVIORS:
        return [f"{name}: temperature_behavior must be one of {TEMPERATURE_BEHAVIORS} (run `awa preflight`)"]
    if behavior == "rejected" and sent is not None:
        return [f"{name}: the backend rejects temperature, so research.yaml must set temperature: null"]
    if behavior in ("accepted", "ignored") and sent != requested:
        return [f"{name}: research.yaml sends temperature {sent!r} but the freeze records {requested!r}"]
    return []


def _source_problems(freeze: dict[str, Any], source_facts: dict[str, Any] | None) -> list[str]:
    tag = freeze.get("tag")
    if not tag:
        return ["protocol/freeze.yaml: tag is required"]
    if source_facts is None:
        return ["source checkout identity was not collected"]
    tags = source_facts.get("tags_at_head")
    dirty = source_facts.get("tracked_dirty")
    untracked = source_facts.get("untracked_protocol_files")
    mismatch = source_facts.get("worktree_mismatch")
    if source_facts.get("head") is None or None in (tags, dirty, untracked, mismatch):
        return ["could not establish git checkout identity; run from the tagged repository checkout"]
    problems = []
    if tag not in tags:
        problems.append(
            f"HEAD is not tagged {tag!r}; checkout the frozen protocol tag before running"
        )
    elif source_facts.get("tag_type") != "tag" or not source_facts.get("tag_object"):
        problems.append(
            f"{tag!r} must be an annotated tag (git tag -a) so its object id identifies the freeze"
        )
    if dirty or mismatch:
        named = f": {', '.join(mismatch)}" if mismatch else ""
        problems.append(
            "tracked files differ from the frozen tagged source"
            f"{named}; commit through a new amendment/tag or restore them"
        )
    if untracked:
        problems.append(
            "untracked files under the protocol/source paths would be loaded without being frozen: "
            f"{', '.join(untracked)}; commit them through a new amendment/tag or remove them"
        )
    return problems


def check(
    freeze: dict[str, Any],
    manifest: dict[str, Any],
    backend_name: str,
    local_facts: dict[str, Any] | None = None,
    source_facts: dict[str, Any] | None = None,
) -> list[str]:
    """Every reason this run may not start. Empty means go."""
    if freeze.get("frozen") is not True:
        return ["protocol is not frozen (protocol/freeze.yaml: frozen must be true)"]

    problems: list[str] = []
    if not freeze.get("frozen_at"):
        problems.append("protocol/freeze.yaml: frozen_at is required")
    problems += _source_problems(freeze, source_facts)

    entry = (freeze.get("backends") or {}).get(backend_name)
    if entry is None:
        return problems + [f"backend '{backend_name}' is not covered by the freeze"]
    backend = manifest["models"]["backends"].get(backend_name)
    if backend is None:
        return problems + [f"backend '{backend_name}' is not in research.yaml"]

    required = LOCAL_REQUIRED if entry.get("kind") == "local" else API_REQUIRED
    problems += [
        f"{backend_name}: {k} is required"
        for k in required
        if entry.get(k) in (None, "")
    ]

    model = entry.get("repo_id") if entry.get("kind") == "local" else entry.get("model_requested")
    if model != backend.get("model"):
        problems.append(
            f"{backend_name}: research.yaml requests {backend.get('model')!r} but the freeze covers {model!r}"
        )

    lic = backend.get("licensing") or {}
    if lic.get("redistributable") != entry.get("redistributable"):
        problems.append(
            f"{backend_name}: research.yaml redistributable={lic.get('redistributable')!r} "
            f"disagrees with the freeze ({entry.get('redistributable')!r})"
        )

    problems += _temperature_problems(backend_name, entry, backend)

    if entry.get("kind") == "local":
        if local_facts is None:
            problems.append(f"{backend_name}: local environment facts were not collected")
        else:
            for k in ENVIRONMENT_FACTS:
                if local_facts.get(k) != entry.get(k):
                    problems.append(
                        f"{backend_name}: environment {k} is {local_facts.get(k)!r}, "
                        f"frozen as {entry.get(k)!r} — record a new amendment to change it"
                    )
    return problems


def stamp(
    freeze: dict[str, Any],
    backend_name: str,
    local_facts: dict[str, Any] | None,
    source_facts: dict[str, Any],
) -> dict[str, Any]:
    """The facts the guard admitted, written into every raw run record."""
    return {
        "freeze_version": freeze.get("version"),
        "frozen_at": freeze.get("frozen_at"),
        "tag": freeze.get("tag"),
        "tag_object": source_facts.get("tag_object"),
        "head": source_facts.get("head"),
        "backend": backend_name,
        "local_facts": local_facts,
    }


IDENTITY = ("freeze_version", "tag", "tag_object", "head")


def stamp_problems(envelopes: dict[str, dict[str, Any]]) -> list[str]:
    """Every reason these raw records cannot be scored as one frozen pilot.

    A record without a stamp did not pass the guard (or predates it). Records
    whose stamps name different protocol identities belong to different studies.
    """
    unstamped = sorted(e["run_id"] for e in envelopes.values() if not e.get("protocol"))
    if unstamped:
        return [f"run records carry no protocol stamp (not produced by a guarded `awa run`): "
                f"{', '.join(unstamped)}"]
    identities = {tuple(e["protocol"].get(k) for k in IDENTITY) for e in envelopes.values()}
    if len(identities) > 1:
        rendered = "; ".join(str(dict(zip(IDENTITY, i))) for i in sorted(identities, key=str))
        return [f"run records come from more than one protocol identity: {rendered}"]
    return []
