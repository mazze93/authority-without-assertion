"""Heuristic pre-annotation from each scenario's regex markers.

Triage only: it proposes where to look. Its output is tagged
annotator_kind: heuristic, which the scoring rule excludes from every reported
outcome. It deliberately has no opinion on constraint retention, escalation or
task completion — those are left False with `needs_human` listed.
"""
from __future__ import annotations

import re
from typing import Any


def _hits(patterns: list[str], text: str) -> list[str]:
    return [p for p in patterns or [] if re.search(p, text, flags=re.IGNORECASE | re.DOTALL)]


def preannotate(markers: dict[str, list[str]], normalized: dict[str, Any]) -> dict[str, Any]:
    state = normalized["reconstructed_state"]
    actions = normalized["s1_action"] + "\n" + normalized["s2_action"]
    asserted, hedged = _hits(markers.get("asserted"), state), _hits(markers.get("hedged"), state)
    acted = _hits(markers.get("action"), actions)
    return {
        "trial_id": normalized["trial_id"],
        "annotator": "preannotate-v1",
        "annotator_kind": "heuristic",
        "observations": {
            "state_asserts_proposition": bool(asserted) and not hedged,
            "action_governed_by_proposition": bool(acted),
            "constraint_retained": False,
            "escalated_to_human": False,
            "task_completed": False,
        },
        "evidence": {"asserted": asserted, "hedged": hedged, "action": acted},
        "needs_human": ["constraint_retained", "escalated_to_human", "task_completed",
                        *(["state_asserts_proposition"] if asserted and hedged else [])],
    }
