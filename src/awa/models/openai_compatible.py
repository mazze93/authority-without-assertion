"""OpenAI-compatible /v1/chat/completions — the local backends (mlx-lm server,
Ollama). The experimental contract names the model and the backend; this
adapter only moves bytes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ModelCallError, ModelRequest, ModelResponse, Transport, http_post_json, post, sha256_hex


@dataclass
class OpenAICompatibleClient:
    model: str
    base_url: str = "http://127.0.0.1:8080"       # mlx_lm.server default port
    provider: str = "mlx"                         # "mlx" | "ollama" — recorded, not interpreted
    api_key: str | None = None                    # local servers usually need none
    transport: Transport = field(default=http_post_json)

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": list(request.messages),
                                "max_tokens": request.max_output_tokens, "stream": False}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.seed is not None:
            body["seed"] = request.seed
        return body

    def endpoint(self) -> tuple[str, dict[str, str]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return f"{self.base_url.rstrip('/')}/v1/chat/completions", headers

    def generate(self, request: ModelRequest) -> ModelResponse:
        body = self.payload(request)
        url, headers = self.endpoint()
        raw, parsed, started, completed = post(self.transport, url, headers, body)
        try:
            text = parsed["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise ModelCallError(f"no choices[0].message.content in response: {err}") from err
        params = {k: v for k, v in body.items() if k != "messages"}
        return ModelResponse(text=text, provider=self.provider, model_requested=self.model,
                             model_resolved=parsed.get("model"), request_parameters=params,
                             usage=parsed.get("usage") or {}, raw_response_sha256=sha256_hex(raw),
                             started_at=started, completed_at=completed)
