"""The freeze guard: a run starts only from a frozen, complete, consistent protocol."""
import copy
import json

import pytest

from awa import freeze, preflight
from awa.cli import main
from awa.models import OpenAICompatibleClient, OpenAIResponsesClient

from conftest import ROOT

MANIFEST = freeze.yaml.safe_load((ROOT / "research.yaml").read_text())
FACTS = {"repo_id": "mlx-community/Qwen3-8B-4bit", "revision": "abc123", "artifact_path": "/cache/abc123",
         "mlx_version": "0.30.0", "mlx_lm_version": "0.28.0", "huggingface_hub_version": "0.35.0",
         "python_version": "3.12.4", "macos_version": "26.1"}


def frozen():
    """The repository's freeze, completed as an author would complete it."""
    fz = copy.deepcopy(freeze.load_freeze(ROOT))
    fz["frozen"], fz["frozen_at"] = True, "2026-09-27"
    fz["backends"]["mlx-qwen3-8b"].update(
        {k: FACTS[k] for k in ("revision", "artifact_path", "mlx_version", "mlx_lm_version",
                               "python_version", "macos_version")},
        temperature_behavior="accepted", licence_reviewed_at="2026-09-27", redistributable=True,
        attested_by="author")
    fz["backends"]["openai-gpt-5.6-terra"]["temperature_behavior"] = "accepted"
    return fz


def manifest(redistributable_local=True):
    m = copy.deepcopy(MANIFEST)
    m["models"]["backends"]["mlx-qwen3-8b"]["licensing"]["redistributable"] = redistributable_local
    return m


def test_the_repository_freeze_is_not_yet_frozen():
    assert freeze.check(freeze.load_freeze(ROOT), MANIFEST, "mlx-qwen3-8b", FACTS) == [
        "protocol is not frozen (protocol/freeze.yaml: frozen must be true)"]


def test_a_complete_consistent_freeze_passes_for_both_backends():
    fz = frozen()
    assert freeze.check(fz, manifest(), "mlx-qwen3-8b", FACTS) == []
    assert freeze.check(fz, manifest(), "openai-gpt-5.6-terra") == []


def test_every_required_value_must_be_present():
    fz = frozen()
    fz["backends"]["mlx-qwen3-8b"]["revision"] = None
    assert any("revision is required" in p for p in freeze.check(fz, manifest(), "mlx-qwen3-8b", FACTS))


def test_uncovered_backend_is_refused():
    assert freeze.check(frozen(), manifest(), "ollama-qwen3-8b") == [
        "backend 'ollama-qwen3-8b' is not covered by the freeze"]


def test_research_yaml_must_agree_with_the_freeze():
    m = manifest()
    m["models"]["backends"]["openai-gpt-5.6-terra"]["model"] = "gpt-5.6-sol"
    assert any("requests 'gpt-5.6-sol'" in p for p in freeze.check(frozen(), m, "openai-gpt-5.6-terra"))
    assert any("redistributable" in p for p in
               freeze.check(frozen(), manifest(redistributable_local=None), "mlx-qwen3-8b", FACTS))


@pytest.mark.parametrize("fact", ["revision", "mlx_lm_version", "python_version", "macos_version", "mlx_version"])
def test_environment_drift_is_refused(fact):
    drifted = {**FACTS, fact: "something-else"}
    problems = freeze.check(frozen(), manifest(), "mlx-qwen3-8b", drifted)
    assert any(f"environment {fact}" in p and "new amendment" in p for p in problems)


def test_temperature_must_match_the_recorded_behavior():
    fz = frozen()
    fz["backends"]["openai-gpt-5.6-terra"]["temperature_behavior"] = "rejected"
    assert any("temperature: null" in p for p in freeze.check(fz, manifest(), "openai-gpt-5.6-terra"))
    fz["backends"]["openai-gpt-5.6-terra"]["temperature_behavior"] = None
    assert any("awa preflight" in p for p in freeze.check(fz, manifest(), "openai-gpt-5.6-terra"))


def test_run_refuses_before_any_model_call(tmp_path):
    import shutil
    for part in ("protocol", "research.yaml"):
        src = ROOT / part
        (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp_path / part)
    with pytest.raises(SystemExit, match="protocol is not frozen"):
        main(["run", "--backend", "openai-gpt-5.6-terra", "--condition", "C0-narrative"], root=tmp_path)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("status,body,expected", [
    (200, b'{"choices": []}', "accepted"),
    (400, b'{"error": "Unsupported parameter: temperature"}', "rejected"),
    (400, b'{"error": "bad model"}', "error"),
    (503, b'{"error": "temperature service down"}', "error"),
])
def test_preflight_classification(status, body, expected):
    assert preflight.classify(status, body) == expected


def test_preflight_record_is_write_once_and_keyless(tmp_path, recording, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    client = OpenAIResponsesClient(model="gpt-5.6-terra")
    t = recording(b'{"status": "completed", "output": []}')
    rec = preflight.probe(client, t)
    assert rec["temperature_behavior"] == "accepted" and rec["not_a_trial"] is True
    assert rec["request"]["temperature"] == 0.0 and "sk-secret-value" not in json.dumps(rec)
    path = preflight.write(rec, tmp_path, "openai-gpt-5.6-terra")
    assert path.parent.name == "preflight"
    with pytest.raises(FileExistsError):
        preflight.write(rec, tmp_path, "openai-gpt-5.6-terra")


def test_preflight_uses_the_backends_own_endpoint(recording):
    t = recording(b'{"choices": [{"message": {"content": "probe"}}]}')
    rec = preflight.probe(OpenAICompatibleClient(model="m", base_url="http://127.0.0.1:8080"), t)
    assert t.calls[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert rec["temperature_behavior"] == "accepted"
