"""OpenAI Responses API (POST /v1/responses) — the API arm.

The key is read from the environment at call time and never stored, logged or
written into a run record."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .base import ModelCallError, ModelRequest, ModelResponse, Transport, http_post_json, post, sha256_hex


def output_text(parsed: dict[str, Any]) -> str:
    """Concatenate every output_text part of every message item, in order."""
    if isinstance(parsed.get("output_text"), str):
        return parsed["output_text"]
    parts = [c.get("text", "") for item in parsed.get("output") or []
             if item.get("type") == "message"
             for c in item.get("content") or [] if c.get("type") == "output_text"]
    if not parts and parsed.get("status") not in (None, "completed"):
        raise ModelCallError(f"response status {parsed.get('status')!r} with no output text")
    return "".join(parts)


@dataclass
class OpenAIResponsesClient:
    model: str
    base_url: str = "https://api.openai.com"
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    transport: Transport = field(default=http_post_json)

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "input": list(request.messages),
                                "max_output_tokens": request.max_output_tokens, "store": False}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        return body                                # Responses has no seed parameter

    def generate(self, request: ModelRequest) -> ModelResponse:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ModelCallError(f"{self.api_key_env} is not set; export it before running the API arm")
        body = self.payload(request)
        raw, parsed, started, completed = post(
            self.transport, f"{self.base_url.rstrip('/')}/v1/responses",
            {"Authorization": f"Bearer {key}"}, body)
        params = {k: v for k, v in body.items() if k != "input"}
        if request.seed is not None:
            params["seed_requested_but_unsupported"] = request.seed
        return ModelResponse(text=output_text(parsed), provider=self.provider,
                             model_requested=self.model, model_resolved=parsed.get("model"),
                             request_parameters=params, usage=parsed.get("usage") or {},
                             raw_response_sha256=sha256_hex(raw),
                             started_at=started, completed_at=completed)
