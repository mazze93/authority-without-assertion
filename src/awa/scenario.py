"""Scenario loading and validation.

Provenance is recorded twice on purpose: `declared_origin` is what the system
is told, `ground_truth_origin` is what the experimenter knows. An injected
fault must be declared in the `fault` block, never hidden in the origin fields,
so a failure of the mechanism can be told apart from false ground truth.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Any

import yaml

ORIGINS = {"user", "web_page", "repository_file", "tool_output", "email", "external_untrusted", "principal"}
FAULT_TYPES = {"none", "provenance-misclassification"}
ACTION_CLASSES = {"proceed", "verify", "abstain", "approval_required"}
REQUIRED = ("id", "version", "family", "principal_constraint", "task", "trusted_context",
            "adversarial_signal", "fault", "unsafe_authority", "persistence_boundary",
            "session_2_task", "expected_action_class")


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class Scenario:
    raw: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.raw[name]
        except KeyError as err:
            raise AttributeError(name) from err

    @property
    def signal(self) -> dict[str, Any]:
        return self.raw["adversarial_signal"]

    @property
    def authority(self) -> dict[str, Any]:
        return self.raw["unsafe_authority"]

    @property
    def fault_injected(self) -> bool:
        return bool(self.raw["fault"]["injected"])


def validate(data: dict[str, Any], where: str = "scenario") -> list[str]:
    """Every problem with a scenario, as sentences. Empty means valid."""
    problems = [f"{where}: missing field '{k}'" for k in REQUIRED if k not in data]
    if problems:
        return problems
    sig, fault, auth = data["adversarial_signal"], data["fault"], data["unsafe_authority"]
    for key in ("content", "declared_origin", "ground_truth_origin"):
        if not sig.get(key):
            problems.append(f"{where}: adversarial_signal.{key} is required")
    for key in ("declared_origin", "ground_truth_origin"):
        if sig.get(key) and sig[key] not in ORIGINS:
            problems.append(f"{where}: adversarial_signal.{key} '{sig[key]}' not in {sorted(ORIGINS)}")
    if fault.get("type") not in FAULT_TYPES:
        problems.append(f"{where}: fault.type must be one of {sorted(FAULT_TYPES)}")
    mismatch = sig.get("declared_origin") != sig.get("ground_truth_origin")
    trusted_declared = sig.get("declared_origin") in ("user", "principal")
    if trusted_declared and sig.get("ground_truth_origin") != sig.get("declared_origin") and not fault.get("injected"):
        problems.append(f"{where}: declared origin '{sig['declared_origin']}' disagrees with ground truth "
                        "but no fault is declared — record it as fault.injected: true")
    if fault.get("injected") and not mismatch:
        problems.append(f"{where}: fault.injected is true but declared and ground-truth origins agree")
    for key in ("proposition", "supplied_by_principal", "verified_under_protocol", "markers"):
        if key not in auth:
            problems.append(f"{where}: unsafe_authority.{key} is required")
    for kind, patterns in (auth.get("markers") or {}).items():
        for p in patterns or []:
            try:
                re.compile(p)
            except re.error as err:
                problems.append(f"{where}: markers.{kind} pattern {p!r} does not compile: {err}")
    if data["expected_action_class"] not in ACTION_CLASSES:
        problems.append(f"{where}: expected_action_class must be one of {sorted(ACTION_CLASSES)}")
    return problems


def load(path: str | pathlib.Path) -> Scenario:
    path = pathlib.Path(path)
    data = yaml.safe_load(path.read_text())
    problems = validate(data, str(path))
    if problems:
        raise ScenarioError("\n".join(problems))
    return Scenario(data)


def load_all(directory: str | pathlib.Path) -> list[Scenario]:
    return [load(p) for p in sorted(pathlib.Path(directory).glob("*.yaml"))]
