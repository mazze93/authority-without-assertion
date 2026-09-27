"""Protocol freeze: the executable form of the preregistration amendment.

A run is allowed only when protocol/freeze.yaml says the protocol is frozen,
every required value is filled, research.yaml agrees with it, it covers the
backend being run, and — for a local backend — the environment the runner is
in matches what was frozen. Prose may describe the freeze; this module is what
enforces it.

Pure except for `load_freeze` and `collect_local_facts`.
"""
from __future__ import annotations

import importlib.metadata
import pathlib
import platform
from typing import Any

import yaml

LOCAL_REQUIRED = ("repo_id", "revision", "artifact_path", "mlx_version", "mlx_lm_version",
                  "python_version", "macos_version", "temperature_requested",
                  "temperature_behavior", "licence_reviewed_at", "redistributable", "attested_by")
API_REQUIRED = ("model_requested", "terms_reviewed_at", "redistributable",
                "temperature_requested", "temperature_behavior", "attested_by")
# Facts the runner re-collects at run time and must match exactly.
ENVIRONMENT_FACTS = ("revision", "mlx_version", "mlx_lm_version", "python_version", "macos_version")
TEMPERATURE_BEHAVIORS = ("accepted", "rejected", "ignored")


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
    except Exception:            # not cached, or offline lookup failed: report absence, not a guess
        return None, None
    return pathlib.PurePath(path).name, str(path)


def collect_local_facts(repo_id: str) -> dict[str, Any]:
    """What this environment actually is. Run on the machine that runs the pilot."""
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


def _temperature_problems(name: str, entry: dict[str, Any], backend: dict[str, Any]) -> list[str]:
    behavior, requested, sent = entry.get("temperature_behavior"), entry.get("temperature_requested"), backend.get("temperature")
    if behavior not in TEMPERATURE_BEHAVIORS:
        return [f"{name}: temperature_behavior must be one of {TEMPERATURE_BEHAVIORS} (run `awa preflight`)"]
    if behavior == "rejected" and sent is not None:
        return [f"{name}: the backend rejects temperature, so research.yaml must set temperature: null"]
    if behavior in ("accepted", "ignored") and sent != requested:
        return [f"{name}: research.yaml sends temperature {sent!r} but the freeze records {requested!r}"]
    return []


def check(freeze: dict[str, Any], manifest: dict[str, Any], backend_name: str,
          local_facts: dict[str, Any] | None = None) -> list[str]:
    """Every reason this run may not start. Empty means go."""
    if freeze.get("frozen") is not True:
        return ["protocol is not frozen (protocol/freeze.yaml: frozen must be true)"]
    problems: list[str] = []
    if not freeze.get("frozen_at"):
        problems.append("protocol/freeze.yaml: frozen_at is required")
    entry = (freeze.get("backends") or {}).get(backend_name)
    if entry is None:
        return problems + [f"backend '{backend_name}' is not covered by the freeze"]
    backend = manifest["models"]["backends"].get(backend_name)
    if backend is None:
        return problems + [f"backend '{backend_name}' is not in research.yaml"]

    required = LOCAL_REQUIRED if entry.get("kind") == "local" else API_REQUIRED
    problems += [f"{backend_name}: {k} is required" for k in required if entry.get(k) in (None, "")]

    model = entry.get("repo_id") if entry.get("kind") == "local" else entry.get("model_requested")
    if model != backend.get("model"):
        problems.append(f"{backend_name}: research.yaml requests {backend.get('model')!r} but the freeze covers {model!r}")
    lic = backend.get("licensing") or {}
    if lic.get("redistributable") != entry.get("redistributable"):
        problems.append(f"{backend_name}: research.yaml redistributable={lic.get('redistributable')!r} "
                        f"disagrees with the freeze ({entry.get('redistributable')!r})")
    problems += _temperature_problems(backend_name, entry, backend)

    if entry.get("kind") == "local":
        if local_facts is None:
            problems.append(f"{backend_name}: local environment facts were not collected")
        else:
            for k in ENVIRONMENT_FACTS:
                if local_facts.get(k) != entry.get(k):
                    problems.append(f"{backend_name}: environment {k} is {local_facts.get(k)!r}, "
                                    f"frozen as {entry.get(k)!r} — record a new amendment to change it")
    return problems
