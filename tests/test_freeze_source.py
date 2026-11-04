"""The guard against a real git checkout.

test_freeze.py exercises check() with injected facts. These tests build a
frozen, tagged repository and drive `awa run` end to end, so they also cover
what collect_source_facts() can and cannot see. Each case is one way a run
could start from a protocol other than the frozen one. No network: the model
client is scripted.
"""
import json
import os
import shutil
import subprocess

import pytest
import yaml

import awa.cli as cli
from awa.cli import main

from conftest import ROOT, ScriptedClient

TAG = "pilot-protocol-v0.1.0"
BACKEND = "openai-gpt-5.6-terra"
RUN = ["run", "--backend", BACKEND, "--condition", "C0-narrative", "--scenario", "dependency-update"]
GIT_ENV = {
    "GIT_AUTHOR_NAME": "probe", "GIT_AUTHOR_EMAIL": "probe@example.invalid",
    "GIT_COMMITTER_NAME": "probe", "GIT_COMMITTER_EMAIL": "probe@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
}


# ── fixtures ────────────────────────────────────────────────────────────────

def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                          capture_output=True, env={**os.environ, **GIT_ENV}).stdout.strip()


def _freeze_api_backend(repo):
    path = repo / "protocol" / "freeze.yaml"
    doc = yaml.safe_load(path.read_text())
    fz = doc["protocol_freeze"]
    fz["frozen"], fz["frozen_at"] = True, "2026-10-16"
    fz["backends"][BACKEND]["temperature_behavior"] = "accepted"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


@pytest.fixture
def clients(monkeypatch):
    """Every scripted client `awa run` creates; the guard must refuse before any."""
    made = []
    def make_client(backend):
        made.append(ScriptedClient(["s1", "state", "s2"] * 10))
        return made[-1]
    monkeypatch.setattr(cli, "make_client", make_client)
    return made


@pytest.fixture
def repo(tmp_path, clients):
    """A frozen checkout at an annotated protocol tag."""
    r = tmp_path / "repo"
    r.mkdir()
    for part in ("protocol", "research.yaml", "src"):
        src = ROOT / part
        if src.is_dir():
            shutil.copytree(src, r / part, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        else:
            shutil.copy(src, r / part)
    _freeze_api_backend(r)
    git(r, "init", "-q")
    git(r, "add", ".")
    git(r, "commit", "-qm", "frozen protocol")
    git(r, "tag", "-a", TAG, "-m", "freeze")
    return r


def calls(clients):
    return sum(len(c.requests) for c in clients)


# ── the clean case ──────────────────────────────────────────────────────────

def test_a_clean_frozen_checkout_runs_and_stamps_every_record(repo, clients):
    assert main(RUN, root=repo) == 0
    assert calls(clients) == 3
    [raw] = [p for p in (repo / "runs" / "raw").glob("R-*.json")]
    stamp = json.loads(raw.read_text())["protocol"]
    assert stamp["tag"] == TAG and stamp["head"] == git(repo, "rev-parse", "HEAD")
    assert stamp["tag_object"] == git(repo, "rev-parse", f"refs/tags/{TAG}")
    assert stamp["tag_object"] != stamp["head"]          # annotated: its own object
    norm = next((repo / "runs" / "normalized").glob("T-*.json")).read_text()
    assert TAG not in norm and stamp["head"] not in norm  # blinding untouched


def test_run_and_preflight_artifacts_do_not_block_the_next_run(repo, clients):
    assert main(RUN, root=repo) == 0
    (repo / "preflight").mkdir()
    (repo / "preflight" / "probe.json").write_text("{}")
    assert main(RUN, root=repo) == 0
    assert calls(clients) == 6


# ── each way a different protocol could reach the model ─────────────────────

def refused(repo, clients, match):
    with pytest.raises(SystemExit, match=match):
        main(RUN, root=repo)
    assert calls(clients) == 0
    assert not (repo / "runs" / "raw").exists()


def test_an_untracked_scenario_is_refused(repo, clients):
    src = repo / "protocol" / "scenarios" / "dependency-update.yaml"
    (src.parent / "zz-unregistered.yaml").write_text(src.read_text())
    refused(repo, clients, r"untracked files .*zz-unregistered\.yaml")


def test_an_ignored_file_under_source_paths_is_still_refused(repo, clients):
    (repo / ".gitignore").write_text("*.yaml.local\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-qm", "ignore")
    git(repo, "tag", "-fa", TAG, "-m", "re-freeze")
    (repo / "protocol" / "scenarios" / "draft.yaml.local").write_text("x: 1\n")
    refused(repo, clients, r"untracked files .*draft\.yaml\.local")


def test_bytecode_caches_and_finder_files_are_not_protocol(repo):
    cache = repo / "src" / "awa" / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "cli.cpython-311.pyc").write_bytes(b"\0")
    (repo / "src" / "awa.egg-info").mkdir()
    (repo / "src" / "awa.egg-info" / "PKG-INFO").write_text("x")
    (repo / "protocol" / "scenarios" / ".DS_Store").write_bytes(b"\0")
    assert main(RUN, root=repo) == 0


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_an_edit_hidden_from_git_status_is_refused(repo, clients, flag):
    git(repo, "update-index", flag, "research.yaml")
    text = (repo / "research.yaml").read_text()
    (repo / "research.yaml").write_text(text.replace("max_output_tokens: 2048", "max_output_tokens: 9"))
    assert git(repo, "status", "--porcelain", "--untracked-files=no") == ""   # status is blind
    refused(repo, clients, r"tracked files differ .*research\.yaml")


def test_a_deleted_tracked_file_is_refused(repo, clients):
    git(repo, "update-index", "--skip-worktree", "protocol/scoring.yaml")
    (repo / "protocol" / "scoring.yaml").unlink()
    refused(repo, clients, r"tracked files differ .*scoring\.yaml")


def test_a_lightweight_tag_is_refused(repo, clients):
    git(repo, "tag", "-d", TAG)
    git(repo, "tag", TAG)
    refused(repo, clients, "annotated tag")


def test_a_moved_tag_runs_but_its_records_name_a_different_identity(repo):
    """Prevention ends here: a local tag can be moved. Detection does not."""
    assert main(RUN, root=repo) == 0
    first = json.loads(next((repo / "runs" / "raw").glob("R-*.json")).read_text())["protocol"]
    (repo / "research.yaml").write_text(
        (repo / "research.yaml").read_text().replace("max_output_tokens: 2048", "max_output_tokens: 9"))
    git(repo, "commit", "-qam", "drift")
    git(repo, "tag", "-fa", TAG, "-m", "moved")
    assert main(RUN, root=repo) == 0
    stamps = {json.loads(p.read_text())["protocol"]["tag_object"]
              for p in (repo / "runs" / "raw").glob("R-*.json") if not p.name.endswith(".events.jsonl")}
    assert first["tag_object"] in stamps and len(stamps) == 2
