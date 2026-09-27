"""Capability preflight: one non-study call that establishes how a backend
treats `temperature`.

It uses no scenario, is not a trial, and writes only to preflight/, which is
outside runs/ and never enters any FAPR calculation. A successful call
establishes `accepted`, not that sampling changed; `ignored` is recorded only
when the runtime says so explicitly, which this probe cannot detect, so it is
left for a human to set by hand with a note.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from .models.base import ModelRequest, sha256_hex, utc_now

PROBE = ModelRequest(messages=({"role": "user", "content": "Reply only with: probe"},),
                     temperature=0.0, max_output_tokens=16)


def classify(status: int, body: bytes) -> str:
    """accepted | rejected | error. A 4xx that names temperature is a rejection;
    any other failure is an error and establishes nothing."""
    if 200 <= status < 300:
        return "accepted"
    if 400 <= status < 500 and b"temperature" in body.lower():
        return "rejected"
    return "error"


def probe(client: Any, transport: Any) -> dict[str, Any]:
    """Send the probe through the backend's own payload builder and endpoint."""
    body = client.payload(PROBE)
    url, headers = client.endpoint()
    started = utc_now()
    status, raw = transport(url, headers, json.dumps(body).encode("utf-8"))
    return {
        "kind": "capability-preflight",
        "not_a_trial": True,
        "provider": client.provider,
        "model_requested": client.model,
        "url": url,
        "request": {k: v for k, v in body.items() if k not in ("messages", "input")},
        "status": status,
        "temperature_behavior": classify(status, raw),
        "raw_response_sha256": sha256_hex(raw),
        "raw_response": raw.decode("utf-8", errors="replace"),
        "started_at": started,
        "completed_at": utc_now(),
    }


def write(record: dict[str, Any], root: pathlib.Path, backend_name: str) -> pathlib.Path:
    out = root / "preflight"
    out.mkdir(exist_ok=True)
    path = out / f"{backend_name}-temperature-0.json"
    with open(path, "x") as fh:                       # write-once, like the run records
        fh.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path
