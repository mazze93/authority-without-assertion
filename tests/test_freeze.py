"""The freeze guard: a run starts only from a frozen, complete, consistent protocol."""
import copy
import json

import pytest

from awa import freeze, preflight
from awa.cli import main
from awa.models import OpenAICompatibleClient, OpenAIResponsesClient

from conftest import ROOT

MANIFEST = freeze.yaml.safe_load((ROOT / "research.yaml").read_text())
FACTS = {
    "repo_id": "mlx-community/Qwen3-8B-4bit",
    "revision": "abc123",
    "artifact_path": "/cache/abc123",
    "mlx_version": "0.30.0",
    "mlx_lm_version": "0.28.0",
    "huggingface_hub_version": "0.35.0",
    "python_version": "3.12.4",
    "macos_version": "26.1",
}
SOURCE = {
    "head": "deadbeef",
    "tags_at_head": ["pilot-protocol-v0.1.0"],
    "tracked_dirty": False,
    "tag_type": "tag",
    "tag_object": "cafef00d",
    "untracked_protocol_files": [],
    "worktree_mismatch": [],
}


def frozen():
    """The repository's freeze, completed as an author would complete it."""
    fz = copy.deepcopy(freeze.load_freeze(ROOT))
    fz["frozen"], fz["frozen_at"] = True, "2026-09-27"
    fz["backends"]["mlx-qwen3-8b"].update(
        {k: FACTS[k] for k in (
            "revision", "artifact_path", "mlx_version", "mlx_lm_version",
            "python_version", "macos_version",
        )},
        temperature_behavior="accepted",
        licence_reviewed_at="2026-09-27",
        redistributable=True,
        attested_by="author",
    )
    fz["backends"]["openai-gpt-5.6-terra"]["temperature_behavior"] = "accepted"
    return fz


def manifest(redistributable_local=True):
    m = copy.deepcopy(MANIFEST)
    m["models"]["backends"]["mlx-qwen3-8b"]["licensing"]["redistributable"] = redistributable_local
    return m


def check(fz, m, backend, facts=None, source=SOURCE):
    return freeze.check(fz, m, backend, facts, source)


def test_the_repository_freeze_is_not_yet_frozen():
    assert freeze.check(freeze.load_freeze(ROOT), MANIFEST, "mlx-qwen3-8b", FACTS) == [
        "protocol is not frozen (protocol/freeze.yaml: frozen must be true)"
    ]


def test_a_complete_consistent_freeze_passes_for_both_backends():
    fz = frozen()
    assert check(fz, manifest(), "mlx-qwen3-8b", FACTS) == []
    assert check(fz, manifest(), "openai-gpt-5.6-terra") == []


def test_every_required_value_must_be_present():
    fz = frozen()
    fz["backends"]["mlx-qwen3-8b"]["revision"] = None
    assert any("revision is required" in p for p in check(fz, manifest(), "mlx-qwen3-8b", FACTS))


def test_uncovered_backend_is_refused():
    assert check(frozen(), manifest(), "ollama-qwen3-8b") == [
        "backend 'ollama-qwen3-8b' is not covered by the freeze"
    ]


def test_research_yaml_must_agree_with_the_freeze():
    m = manifest()
    m["models"]["backends"]["openai-gpt-5.6-terra"]["model"] = "gpt-5.6-sol"
    assert any("requests 'gpt-5.6-sol'" in p for p in check(frozen(), m, "openai-gpt-5.6-terra"))
    assert any(
        "redistributable" in p
        for p in check(frozen(), manifest(redistributable_local=None), "mlx-qwen3-8b", FACTS)
    )


@pytest.mark.parametrize("fact", ["revision", "mlx_lm_version", "python_version", "macos_version", "mlx_version"])
def test_environment_drift_is_refused(fact):
    drifted = {**FACTS, fact: "something-else"}
    problems = check(frozen(), manifest(), "mlx-qwen3-8b", drifted)
    assert any(f"environment {fact}" in p and "new amendment" in p for p in problems)


def test_frozen_tag_must_point_at_head():
    source = {**SOURCE, "tags_at_head": []}
    assert any(
        "HEAD is not tagged" in p
        for p in check(frozen(), manifest(), "openai-gpt-5.6-terra", source=source)
    )


def test_tracked_source_drift_is_refused_but_untracked_data_is_outside_this_check():
    source = {**SOURCE, "tracked_dirty": True}
    assert any(
        "tracked files differ" in p
        for p in check(frozen(), manifest(), "openai-gpt-5.6-terra", source=source)
    )


def test_untracked_protocol_files_are_refused():
    source = {**SOURCE, "untracked_protocol_files": ["protocol/scenarios/draft.yaml"]}
    assert any(
        "untracked files" in p and "draft.yaml" in p
        for p in check(frozen(), manifest(), "openai-gpt-5.6-terra", source=source)
    )


def test_content_drift_hidden_from_status_is_refused():
    source = {**SOURCE, "tracked_dirty": False, "worktree_mismatch": ["research.yaml"]}
    assert any(
        "tracked files differ" in p and "research.yaml" in p
        for p in check(frozen(), manifest(), "openai-gpt-5.6-terra", source=source)
    )


def test_a_lightweight_tag_is_refused():
    source = {**SOURCE, "tag_type": "commit"}
    assert any(
        "annotated tag" in p
        for p in check(frozen(), manifest(), "openai-gpt-5.6-terra", source=source)
    )


@pytest.mark.parametrize("key", ["untracked_protocol_files", "worktree_mismatch"])
def test_uncollected_source_facts_fail_closed(key):
    source = {**SOURCE, key: None}
    assert any(
        "could not establish git checkout identity" in p
        for p in check(frozen(), manifest(), "openai-gpt-5.6-terra", source=source)
    )


def test_missing_git_identity_fails_closed():
    source = {"head": None, "tags_at_head": None, "tracked_dirty": None}
    assert any(
        "could not establish git checkout identity" in p
        for p in check(frozen(), manifest(), "openai-gpt-5.6-terra", source=source)
    )


def test_temperature_must_match_the_recorded_behavior():
    fz = frozen()
    fz["backends"]["openai-gpt-5.6-terra"]["temperature_behavior"] = "rejected"
    assert any("temperature: null" in p for p in check(fz, manifest(), "openai-gpt-5.6-terra"))
    fz["backends"]["openai-gpt-5.6-terra"]["temperature_behavior"] = None
    assert any("awa preflight" in p for p in check(fz, manifest(), "openai-gpt-5.6-terra"))


def test_guard_and_runner_read_the_same_temperature():
    m = manifest()
    del m["models"]["backends"]["openai-gpt-5.6-terra"]["temperature"]
    fz = frozen()
    fz["backends"]["openai-gpt-5.6-terra"]["temperature_behavior"] = "rejected"
    # With the key absent the run omits temperature, so a 'rejected' freeze passes ...
    assert freeze.requested_temperature(m["models"]["backends"]["openai-gpt-5.6-terra"]) is None
    assert check(fz, m, "openai-gpt-5.6-terra") == []
    # ... and an 'accepted' freeze of 0.0 does not, because nothing would be sent.
    fz["backends"]["openai-gpt-5.6-terra"]["temperature_behavior"] = "accepted"
    assert any("sends temperature None" in p for p in check(fz, m, "openai-gpt-5.6-terra"))


def _env(run_id, **protocol):
    stamp = None if protocol.get("absent") else {
        "freeze_version": "0.1.0", "tag": "pilot-protocol-v0.1.0",
        "tag_object": "cafef00d", "head": "deadbeef", **protocol,
    }
    return {"run_id": run_id, "protocol": stamp}


def test_scoring_refuses_unstamped_or_mixed_records():
    assert freeze.stamp_problems({"T-1": _env("R-1"), "T-2": _env("R-2")}) == []
    unstamped = freeze.stamp_problems({"T-1": _env("R-1"), "T-2": _env("R-2", absent=True)})
    assert len(unstamped) == 1 and "R-2" in unstamped[0] and "no protocol stamp" in unstamped[0]
    mixed = freeze.stamp_problems({"T-1": _env("R-1"), "T-2": _env("R-2", head="0ther")})
    assert len(mixed) == 1 and "more than one protocol identity" in mixed[0]


def test_stamp_records_what_the_guard_admitted():
    s = freeze.stamp(frozen(), "mlx-qwen3-8b", FACTS, SOURCE)
    assert s["tag_object"] == "cafef00d" and s["head"] == "deadbeef"
    assert s["frozen_at"] == "2026-09-27" and s["local_facts"] == FACTS


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


def test_preflight_record_is_keyless_and_append_only(tmp_path, recording, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    client = OpenAIResponsesClient(model="gpt-5.6-terra")
    t = recording(b'{"status": "completed", "output": []}')
    rec = preflight.probe(client, t)
    assert rec["temperature_behavior"] == "accepted" and rec["not_a_trial"] is True
    assert rec["request"]["temperature"] == 0.0 and "sk-secret-value" not in json.dumps(rec)

    first = preflight.write(rec, tmp_path, "openai-gpt-5.6-terra")
    assert first.parent.name == "preflight"
    with pytest.raises(FileExistsError):
        preflight.write(rec, tmp_path, "openai-gpt-5.6-terra")

    retry = {**rec, "started_at": "2026-09-27T01:23:45.999+00:00"}
    second = preflight.write(retry, tmp_path, "openai-gpt-5.6-terra")
    assert second != first
    assert first.exists() and second.exists()


def test_preflight_error_does_not_need_to_be_deleted_before_retry(tmp_path):
    error = {
        "raw_response_sha256": "sha256:" + "a" * 64,
        "started_at": "2026-09-27T01:00:00.000+00:00",
    }
    accepted = {
        "raw_response_sha256": "sha256:" + "b" * 64,
        "started_at": "2026-09-27T01:01:00.000+00:00",
    }
    p1 = preflight.write(error, tmp_path, "backend")
    p2 = preflight.write(accepted, tmp_path, "backend")
    assert p1 != p2


def test_preflight_uses_the_backends_own_endpoint(recording):
    t = recording(b'{"choices": [{"message": {"content": "probe"}}]}')
    rec = preflight.probe(
        OpenAICompatibleClient(model="m", base_url="http://127.0.0.1:8080"), t
    )
    assert t.calls[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert rec["temperature_behavior"] == "accepted"
