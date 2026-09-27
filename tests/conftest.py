"""Shared fixtures. No test touches the network: adapters get a recorded-response transport."""
from __future__ import annotations

import json
import pathlib

import pytest

from awa.models.base import ModelRequest, ModelResponse

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class RecordingTransport:
    """Returns a canned response and remembers what was sent."""
    def __init__(self, body: bytes, status: int = 200):
        self.body, self.status, self.calls = body, status, []

    def __call__(self, url, headers, body):
        self.calls.append({"url": url, "headers": headers, "json": json.loads(body)})
        return self.status, self.body


class ScriptedClient:
    """A ModelClient that replies from a script, in order: s1 action, summary, s2 action."""
    provider, model = "scripted", "scripted-v1"

    def __init__(self, replies: list[str]):
        self.replies, self.requests = list(replies), []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(text=self.replies.pop(0), provider=self.provider,
                             model_requested=self.model, model_resolved="scripted-v1-resolved",
                             request_parameters={"temperature": request.temperature},
                             usage={}, raw_response_sha256="sha256:0",
                             started_at="t0", completed_at="t1")


@pytest.fixture
def recording():
    return RecordingTransport


@pytest.fixture
def scripted():
    return ScriptedClient
