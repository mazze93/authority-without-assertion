"""The model contract.

A ModelRequest is provider-neutral. A ModelResponse keeps every piece of
provider metadata needed to reconstruct the run: what was requested, what the
provider says it actually served, the exact parameters sent, usage, timing and
a hash of the raw response body. Normalization must never drop these.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

# (url, headers, body) -> (status, raw response bytes). Injected so adapters are
# testable against recorded fixtures without network access.
Transport = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[dict[str, str], ...]          # [{"role": ..., "content": ...}]
    temperature: float | None = 0.0               # None: omit (some models reject it)
    seed: int | None = None
    max_output_tokens: int = 2048
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: str
    model_requested: str
    model_resolved: str | None
    request_parameters: dict[str, Any]
    usage: dict[str, Any]
    raw_response_sha256: str
    started_at: str
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelClient(Protocol):
    provider: str
    model: str

    def generate(self, request: ModelRequest) -> ModelResponse: ...


class ModelCallError(RuntimeError):
    """A provider returned a non-success status or an unparseable body."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def http_post_json(url: str, headers: dict[str, str], body: bytes,
                   timeout: float = 300.0) -> tuple[int, bytes]:
    """Default transport. HTTPS for remote hosts is the caller's configuration."""
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:          # keep the body: it explains the failure
        return err.code, err.read()


def post(transport: Transport, url: str, headers: dict[str, str],
         payload: dict[str, Any]) -> tuple[bytes, dict[str, Any], str, str]:
    """Send payload; return (raw bytes, parsed json, started_at, completed_at)."""
    started = utc_now()
    status, raw = transport(url, headers, json.dumps(payload).encode("utf-8"))
    completed = utc_now()
    if status >= 400:
        raise ModelCallError(f"{url} returned HTTP {status}: {raw[:500]!r}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ModelCallError(f"{url} returned a body that is not JSON: {err}") from err
    return raw, parsed, started, completed
