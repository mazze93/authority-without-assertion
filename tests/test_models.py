import pytest

from awa.models import OpenAICompatibleClient, OpenAIResponsesClient, ModelRequest
from awa.models.base import ModelCallError

from conftest import fixture_bytes

REQ = ModelRequest(messages=({"role": "user", "content": "hi"},), temperature=0.0, max_output_tokens=64)


def test_responses_client_joins_output_text_and_keeps_provider_metadata(recording, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    t = recording(fixture_bytes("openai_responses.json"))
    r = OpenAIResponsesClient(model="gpt-5.6-terra", transport=t).generate(REQ)
    assert r.text == "I will not deploy 4.2.0 until you approve it."
    assert (r.provider, r.model_requested, r.model_resolved) == ("openai", "gpt-5.6-terra",
                                                                "gpt-5.6-terra-2026-09-01")
    assert r.usage["total_tokens"] == 134 and r.raw_response_sha256.startswith("sha256:")
    sent = t.calls[0]
    assert sent["url"].endswith("/v1/responses") and sent["json"]["input"][0]["content"] == "hi"
    assert sent["json"]["store"] is False and "input" not in r.request_parameters


def test_responses_client_never_records_the_key(recording, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    r = OpenAIResponsesClient(model="m", transport=recording(fixture_bytes("openai_responses.json"))).generate(REQ)
    assert "sk-secret-value" not in repr(r.to_dict())


def test_responses_client_requires_key(recording, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelCallError, match="OPENAI_API_KEY"):
        OpenAIResponsesClient(model="m", transport=recording(b"{}")).generate(REQ)


def test_seed_is_recorded_as_unsupported_on_responses(recording, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    req = ModelRequest(messages=REQ.messages, seed=7)
    r = OpenAIResponsesClient(model="m", transport=recording(fixture_bytes("openai_responses.json"))).generate(req)
    assert r.request_parameters["seed_requested_but_unsupported"] == 7


def test_chat_client_parses_and_omits_null_temperature(recording):
    t = recording(fixture_bytes("chat_completions.json"))
    req = ModelRequest(messages=REQ.messages, temperature=None)
    r = OpenAICompatibleClient(model="mlx-community/Qwen3-8B-4bit", transport=t).generate(req)
    assert r.text.startswith("Upgrading") and r.provider == "mlx"
    assert "temperature" not in t.calls[0]["json"] and t.calls[0]["url"].endswith("/v1/chat/completions")


def test_http_errors_surface_with_body(recording):
    t = recording(b'{"error": "model not loaded"}', status=503)
    with pytest.raises(ModelCallError, match="503"):
        OpenAICompatibleClient(model="m", transport=t).generate(REQ)
