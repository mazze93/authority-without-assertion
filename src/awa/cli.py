"""awa — pilot command line.

  awa run --backend mlx-qwen3-8b --condition C0-narrative [--scenario ID ...] [--repeats N]
  awa preannotate            heuristic triage for every unannotated normalized trace
  awa score                  validate two human annotations per trial, adjudicate, unblind, score

Scorers annotate from runs/normalized/ only. runs/index.jsonl and runs/raw/
reveal the condition and must not be opened until annotation is finished.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import yaml

from .models import OpenAICompatibleClient, OpenAIResponsesClient
from .runner import CONDITIONS, run_trial
from .runner.run import write_records
from .scenario import load_all
from .scoring import (
    agreement_summary,
    load_rule,
    preannotate,
    score,
    summarize,
    validate_annotation,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROPAGATION_FIELDS = ("state_asserts_proposition", "action_governed_by_proposition")


def manifest(root: pathlib.Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "research.yaml").read_text())


def make_client(backend: dict[str, Any]):
    kind = backend["adapter"]
    if kind == "openai-compatible":
        return OpenAICompatibleClient(model=backend["model"], base_url=backend["base_url"],
                                      provider=backend["provider"])
    if kind == "openai-responses":
        return OpenAIResponsesClient(model=backend["model"], api_key_env=backend["api_key_env"])
    raise SystemExit(f"unknown adapter '{kind}' in research.yaml")


def cmd_run(a: argparse.Namespace, root: pathlib.Path) -> int:
    m = manifest(root)
    backends = m["models"]["backends"]
    if a.backend not in backends:
        raise SystemExit(f"backend '{a.backend}' not in research.yaml (have: {', '.join(backends)})")
    backend = backends[a.backend]
    if a.condition not in CONDITIONS:
        raise SystemExit(f"condition must be one of {', '.join(CONDITIONS)}")
    scenarios = [s for s in load_all(root / "protocol" / "scenarios")
                 if not a.scenario or s.id in a.scenario]
    client = make_client(backend)
    for s in scenarios:
        for i in range(a.repeats):
            trial = run_trial(s, CONDITIONS[a.condition], client,
                              temperature=backend.get("temperature", 0.0),
                              max_output_tokens=backend.get("max_output_tokens", 2048))
            env = write_records(trial, root / "runs", backend["licensing"])
            print(f"{env['run_id']}  {s.id}  {a.condition}  repeat {i + 1}/{a.repeats}")
    return 0


def cmd_preannotate(a: argparse.Namespace, root: pathlib.Path) -> int:
    scen = {s.id: s for s in load_all(root / "protocol" / "scenarios")}
    out = root / "runs" / "annotations" / "heuristic"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted((root / "runs" / "normalized").glob("T-*.json")):
        target = out / f"{path.stem}.yaml"
        if target.exists():
            continue
        norm = json.loads(path.read_text())
        ann = preannotate(scen[norm["scenario_id"]].authority["markers"], norm)
        target.write_text(yaml.safe_dump(ann, sort_keys=False, allow_unicode=True))
        n += 1
    print(f"pre-annotated {n} trace(s) into {out.relative_to(root)} (triage only; not scored)")
    return 0


def _primary_annotations(root: pathlib.Path, rule: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Load exactly the blinded primary annotations, grouped by trial.

    Layout is runs/annotations/human/<annotator>/<trial_id>.yaml. Keeping each
    annotator in a separate directory prevents the second annotation from
    overwriting the first and makes the two-rater design inspectable.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    human_root = root / "runs" / "annotations" / "human"
    for p in sorted(human_root.glob("*/T-*.yaml")):
        ann = yaml.safe_load(p.read_text())
        validate_annotation(ann, rule)
        if ann.get("annotator_kind") not in rule["authoritative_annotator_kinds"]:
            raise SystemExit(f"{p}: annotator_kind must be authoritative")
        if ann.get("trial_id") != p.stem:
            raise SystemExit(f"{p}: trial_id {ann.get('trial_id')!r} does not match filename {p.stem!r}")
        folder_annotator = p.parent.name
        if ann.get("annotator") != folder_annotator:
            raise SystemExit(f"{p}: annotator must match directory name {folder_annotator!r}")
        grouped.setdefault(ann["trial_id"], []).append(ann)
    return grouped


def _resolution(root: pathlib.Path, trial_id: str, rule: dict[str, Any]) -> dict[str, Any] | None:
    p = root / "runs" / "annotations" / "resolutions" / f"{trial_id}.yaml"
    if not p.exists():
        return None
    ann = yaml.safe_load(p.read_text())
    validate_annotation(ann, rule)
    if ann.get("annotator_kind") not in rule["authoritative_annotator_kinds"]:
        raise SystemExit(f"{p}: resolution must be a human annotation")
    if ann.get("trial_id") != trial_id:
        raise SystemExit(f"{p}: trial_id does not match filename")
    return ann


def _final_annotations(root: pathlib.Path, rule: dict[str, Any]) -> tuple[
        dict[str, dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    expected = {p.stem for p in (root / "runs" / "normalized").glob("T-*.json")}
    grouped = _primary_annotations(root, rule)
    missing = sorted(expected - set(grouped))
    extra = sorted(set(grouped) - expected)
    if missing or extra:
        raise SystemExit(f"annotation set does not match normalized traces; missing={missing}, extra={extra}")

    final: dict[str, dict[str, Any]] = {}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for trial_id in sorted(expected):
        anns = grouped[trial_id]
        if len(anns) != 2:
            raise SystemExit(f"{trial_id}: expected exactly two primary human annotations, found {len(anns)}")
        if anns[0]["annotator"] == anns[1]["annotator"]:
            raise SystemExit(f"{trial_id}: primary annotations must come from distinct annotators")
        a, b = sorted(anns, key=lambda x: x["annotator"])
        pairs.append((a, b))
        if a["observations"] == b["observations"]:
            final[trial_id] = a
            continue
        resolved = _resolution(root, trial_id, rule)
        if resolved is None:
            raise SystemExit(
                f"{trial_id}: annotators disagree; add runs/annotations/resolutions/{trial_id}.yaml "
                "before unblinding"
            )
        final[trial_id] = resolved
    return final, pairs


def cmd_score(a: argparse.Namespace, root: pathlib.Path) -> int:
    rule = load_rule(root / "protocol" / "scoring.yaml")
    scen = {s.id: s for s in load_all(root / "protocol" / "scenarios")}

    # Complete blinding checks and adjudication BEFORE reading raw envelopes,
    # because those envelopes reveal model and condition.
    final, pairs = _final_annotations(root, rule)
    agreement = agreement_summary(pairs, PROPAGATION_FIELDS)

    envs: dict[str, dict[str, Any]] = {}
    for p in (root / "runs" / "raw").glob("R-*.json"):
        e = json.loads(p.read_text())
        envs[e["trial_id"]] = e
    if set(envs) != set(final):
        raise SystemExit("raw envelopes and final annotations do not describe the same trial set")

    rows = []
    for trial_id, ann in sorted(final.items()):
        env = envs[trial_id]
        c = score(scen[env["scenario_id"]].authority, ann, rule)
        rows.append({
            "trial_id": trial_id,
            "annotator": ann["annotator"],
            "condition": env["condition"],
            "scenario_id": env["scenario_id"],
            "model": env["model"]["requested"],
            "fault_injected": env["fault_injected"],
            "classification": c.value,
            "observations": ann["observations"],
        })

    out = root / "runs" / "scores"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "rule_version": rule["version"],
        "scored_trials": len(rows),
        "agreement": agreement,
        "by_condition_model": summarize(rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None, root: pathlib.Path = ROOT) -> int:
    p = argparse.ArgumentParser(prog="awa", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run trials")
    r.add_argument("--backend", required=True)
    r.add_argument("--condition", required=True)
    r.add_argument("--scenario", action="append", help="scenario id (repeatable); default all")
    r.add_argument("--repeats", type=int, default=1)
    sub.add_parser("preannotate", help="heuristic triage of unannotated traces")
    sub.add_parser("score", help="validate two human annotations per trial, adjudicate, unblind and score")
    a = p.parse_args(argv)
    return {"run": cmd_run, "preannotate": cmd_preannotate, "score": cmd_score}[a.cmd](a, root)


if __name__ == "__main__":
    sys.exit(main())
