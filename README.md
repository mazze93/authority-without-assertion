# Authority Without Assertion

*Evidence-gated state for agentic systems under contextual pressure* — Mazze LeCzzare Frazer

**One proposition:** generated information should not acquire operational
authority merely because it was generated, persisted, or repeated.

This repository measures one failure: **false-authority propagation**. An
unsupported claim crosses a session boundary and then governs what the next
agent does. The study compares narrative persistence with evidence-gated
state. See `preregistration.md`.

## Status

**Pilot stage (PR1).** The harness, three scenarios, the scoring rule and the
two baseline conditions are built. No trials have been run. C1 and C2 need
runtime adapters that are deliberately not built yet. The pilot first checks
that the failure happens at a measurable rate.

## Layout

| Path | What | Licence |
|---|---|---|
| `research.yaml` | study manifest: question, conditions, model backends, who may author what | CC BY 4.0 |
| `preregistration.md` | design, scoring, decision rules; amended, never edited | CC BY 4.0 |
| `protocol/scenarios/` | hand-authored scenarios | CC BY 4.0 |
| `protocol/scoring.yaml` | scoring rule v1 (the code reads it) | CC BY 4.0 |
| `graph/` | claims, checks and append-only ledger (Bearing format) | CC BY 4.0 |
| `src/awa/`, `tests/` | harness, adapters, scorer | Apache-2.0 |
| `runs/raw/`, `runs/normalized/` | model outputs | per artifact |
| `runs/annotations/`, `runs/scores/` | annotations and scores | CC BY 4.0 |

The licences are mapped path by path in `REUSE.toml`.

## Running the pilot

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest                       # 28 tests, no network

# local: start an OpenAI-compatible server (e.g. mlx_lm.server) on :8080, then
.venv/bin/python -m awa run --backend mlx-qwen3-8b --condition C0-narrative --repeats 5
# API: export OPENAI_API_KEY in your shell first (never commit it)
.venv/bin/python -m awa run --backend openai-gpt-5.6-terra --condition C0p-prompted-provenance --repeats 5

.venv/bin/python -m awa preannotate    # triage only
# annotators write runs/annotations/human/<trial_id>.yaml from runs/normalized/ only,
# without opening runs/raw/ or runs/index.jsonl (they reveal the condition)
.venv/bin/python -m awa score          # scores human annotations, then unblinds
```

Before the first real run, complete the amendment the preregistration asks for.

## Checking the claim graph

The study's claims use the Bearing ledger from
[intentional-fragility](https://github.com/mazze93/intentional-fragility):

```bash
PYTHONPATH=<intentional-fragility>/engine/src python -m bearing --root . validate
PYTHONPATH=<intentional-fragility>/engine/src python -m bearing --root . fold
```

Every claim currently reads *gone (0)*. That is the correct starting state:
its checks have not been run.
