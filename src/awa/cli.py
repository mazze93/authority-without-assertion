"""awa — pilot command line.

  awa run --backend mlx-qwen3-8b --condition C0-narrative [--scenario ID ...] [--repeats N]
  awa preannotate            heuristic triage for every unannotated normalized trace
  awa score                  score HUMAN annotations, unblind, write runs/scores/summary.json

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
from .scoring import load_rule, preannotate, score, summarize

ROOT = pathlib.Path(__file__).resolve().parents[2]


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


def cmd_score(a: argparse.Namespace, root: pathlib.Path) -> int:
    rule = load_rule(root / "protocol" / "scoring.yaml")
    scen = {s.id: s for s in load_all(root / "protocol" / "scenarios")}
    envs = {}
    for p in (root / "runs" / "raw").glob("R-*.json"):
        e = json.loads(p.read_text())
        envs[e["trial_id"]] = e
    rows, skipped = [], []
    for p in sorted((root / "runs" / "annotations" / "human").glob("T-*.yaml")):
        ann = yaml.safe_load(p.read_text())
        if ann.get("annotator_kind") not in rule["authoritative_annotator_kinds"]:
            skipped.append(p.name)
            continue
        env = envs[ann["trial_id"]]
        c = score(scen[env["scenario_id"]].authority, ann, rule)
        rows.append({"trial_id": ann["trial_id"], "annotator": ann["annotator"],
                     "condition": env["condition"], "scenario_id": env["scenario_id"],
                     "model": env["model"]["requested"], "fault_injected": env["fault_injected"],
                     "classification": c.value, "observations": ann["observations"]})
    out = root / "runs" / "scores"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"rule_version": rule["version"], "scored": len(rows), "skipped_non_human": skipped,
               "by_condition": summarize(rows)}
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
    sub.add_parser("score", help="score human annotations and unblind")
    a = p.parse_args(argv)
    return {"run": cmd_run, "preannotate": cmd_preannotate, "score": cmd_score}[a.cmd](a, root)


if __name__ == "__main__":
    sys.exit(main())
