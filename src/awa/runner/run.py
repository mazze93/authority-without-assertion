"""One trial: session 1 (T0–T3), persistence (T4), session boundary, session 2
(T5–T7). Writes an immutable raw record plus a condition-blind normalized trace.

Raw records are never overwritten; corrections are new records.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import secrets
from datetime import datetime, timezone
from typing import Any

from ..models.base import ModelClient, ModelRequest, ModelResponse
from ..scenario import Scenario
from .conditions import AGENT_ROLE, Condition

ORIGIN_LABEL = {
    "web_page": "Retrieved web page",
    "repository_file": "Repository file",
    "tool_output": "Tool output",
    "email": "Email",
    "external_untrusted": "External content",
}


def signal_message(s: Scenario) -> dict[str, str]:
    """Deliver the signal as the system would see it: under its DECLARED origin."""
    sig = s.signal
    if sig["declared_origin"] in ("user", "principal"):
        return {"role": "user", "content": sig["content"]}
    label = ORIGIN_LABEL.get(sig["declared_origin"], "Content")
    return {"role": "user", "content": f"[{label} — source: {sig.get('source', 'unknown')}]\n{sig['content']}"}


def session_1(s: Scenario) -> list[dict[str, str]]:
    context = "\n".join(f"- {c}" for c in s.trusted_context)
    return [
        {"role": "system", "content": f"{AGENT_ROLE}\n\nPrincipal's standing constraint: {s.principal_constraint}"},
        {"role": "user", "content": f"{s.task}\n\nProject context:\n{context}"},
        signal_message(s),
    ]


def session_2(s: Scenario, state: str) -> list[dict[str, str]]:
    """The new instance gets ONLY the reconstructed state — not the constraint."""
    msgs = [{"role": "system", "content": f"{AGENT_ROLE}\n\nState carried over from the previous session:\n{state}"}]
    if s.raw.get("correction"):
        msgs.append({"role": "user", "content": s.correction})
    msgs.append({"role": "user", "content": s.session_2_task})
    return msgs


def _event(stage: str, request: ModelRequest, response: ModelResponse) -> dict[str, Any]:
    return {"stage": stage, "messages": list(request.messages), "response": response.to_dict()}


def run_trial(s: Scenario, condition: Condition, client: ModelClient,
              temperature: float | None = 0.0, max_output_tokens: int = 2048) -> dict[str, Any]:
    """Execute the three model calls; return the full in-memory trial."""
    def ask(messages: list[dict[str, str]]) -> tuple[ModelRequest, ModelResponse]:
        req = ModelRequest(messages=tuple(messages), temperature=temperature,
                           max_output_tokens=max_output_tokens)
        return req, client.generate(req)

    s1 = session_1(s)
    r1, a1 = ask(s1)                                                     # T3
    s1_full = s1 + [{"role": "assistant", "content": a1.text}]
    r2, summary = ask(s1_full + [{"role": "user", "content": condition.summary_instruction}])  # T4
    r3, a2 = ask(session_2(s, summary.text))                             # T5–T7
    return {
        "scenario": s,
        "condition": condition,
        "events": [_event("s1_action", r1, a1), _event("persist_summary", r2, summary),
                   _event("s2_action", r3, a2)],
        "texts": {"s1_action": a1.text, "reconstructed_state": summary.text, "s2_action": a2.text},
        "model": a1,
    }


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def new_run_id() -> str:
    return "R-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)


def trial_id(run_id: str) -> str:
    """Opaque id for blind scoring; the mapping lives only in the raw record."""
    return "T-" + hashlib.sha256(run_id.encode()).hexdigest()[:12]


def normalize(trial: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Condition-blind view: no condition, model, or harness fields — only what a
    scorer needs. (A treatment's own output, such as source labels written by
    the model, is the thing being measured and stays in.)"""
    s: Scenario = trial["scenario"]
    return {"trial_id": trial_id(run_id), "scenario_id": s.id, "scenario_version": s.version,
            **trial["texts"]}


def _write_new(path: pathlib.Path, data: bytes) -> None:
    with open(path, "xb") as fh:                   # "x": fail rather than overwrite
        fh.write(data)


def write_records(trial: dict[str, Any], out_dir: str | pathlib.Path,
                  licensing: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    """Persist raw envelope + events + normalized trace; append to the index."""
    out = pathlib.Path(out_dir)
    raw_dir, norm_dir = out / "raw", out / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id()
    s: Scenario = trial["scenario"]
    m: ModelResponse = trial["model"]
    events = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in trial["events"]).encode()
    norm = normalize(trial, run_id)
    norm_bytes = (json.dumps(norm, ensure_ascii=False, indent=2) + "\n").encode()
    envelope = {
        "run_id": run_id,
        "trial_id": norm["trial_id"],
        "scenario_id": s.id, "scenario_version": s.version,
        "fault_injected": s.fault_injected,
        "condition": trial["condition"].id,
        "model": {"provider": m.provider, "requested": m.model_requested,
                  "resolved_per_call": [e["response"]["model_resolved"] for e in trial["events"]],
                  "parameters": m.request_parameters,
                  "parameters_hash": _sha(json.dumps(m.request_parameters, sort_keys=True).encode())},
        "events": {"path": f"{run_id}.events.jsonl", "sha256": _sha(events)},
        "normalized": {"path": f"../normalized/{norm['trial_id']}.json", "sha256": _sha(norm_bytes)},
        "licensing": licensing,
        "scores": None,          # scores live in annotations, never in the raw record
    }
    _write_new(raw_dir / f"{run_id}.events.jsonl", events)
    _write_new(norm_dir / f"{norm['trial_id']}.json", norm_bytes)
    _write_new(raw_dir / f"{run_id}.json", (json.dumps(envelope, indent=2) + "\n").encode())
    with open(out / "index.jsonl", "a") as fh:
        fh.write(json.dumps({"run_id": run_id, "trial_id": norm["trial_id"], "scenario_id": s.id,
                             "condition": trial["condition"].id, "provider": m.provider,
                             "model": m.model_requested}) + "\n")
    return envelope
