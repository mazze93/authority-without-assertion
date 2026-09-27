import copy

import pytest

from awa.scenario import ScenarioError, load, load_all, validate

from conftest import ROOT

SCEN = ROOT / "protocol" / "scenarios"


def test_all_pilot_scenarios_are_valid():
    ids = {s.id for s in load_all(SCEN)}
    assert ids == {"dependency-update", "forged-approval", "summary-laundering"}


def test_the_negative_control_declares_its_fault_openly():
    s = load(SCEN / "forged-approval.yaml")
    assert s.fault_injected and s.fault["observable_to_system"] is False
    assert s.signal["declared_origin"] == "user" and s.signal["ground_truth_origin"] == "external_untrusted"


def test_a_hidden_provenance_fault_is_rejected():
    data = copy.deepcopy(load(SCEN / "forged-approval.yaml").raw)
    data["fault"] = {"type": "none", "injected": False, "observable_to_system": True}
    assert any("no fault is declared" in p for p in validate(data))


def test_a_declared_fault_must_actually_mismatch():
    data = copy.deepcopy(load(SCEN / "dependency-update.yaml").raw)
    data["fault"]["injected"] = True
    data["adversarial_signal"]["declared_origin"] = "external_untrusted"
    assert any("origins agree" in p for p in validate(data))


def test_bad_marker_regex_is_reported(tmp_path):
    data = copy.deepcopy(load(SCEN / "dependency-update.yaml").raw)
    data["unsafe_authority"]["markers"]["asserted"] = ["(unclosed"]
    import yaml
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ScenarioError, match="does not compile"):
        load(p)
