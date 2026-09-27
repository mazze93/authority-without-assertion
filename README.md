# Authority Without Assertion

*Evidence-gated state for agentic systems under contextual pressure* — Mazze LeCzzare Frazer

**One proposition:** generated information should not acquire operational
authority merely because it was generated, persisted, or repeated.

This repository measures one failure: **false-authority propagation**. An
unsupported claim crosses a session boundary and then governs what the next
agent does. The study compares narrative persistence with evidence-gated
state. See `preregistration.md`.

## Status

**Pilot stage (PR1).** The harness, three scenarios, scoring rule and two
baseline conditions are built. No trials have been run. C1 and C2 need runtime
adapters that are deliberately not built yet. The pilot first checks that the
failure occurs at a measurable rate.

Model identifiers in `research.yaml` are **configuration values**, not claims
that a provider or local runtime has been verified. Every run records both the
requested model and the model identifier reported by the provider.

## Layout

| Path | What | Licence |
|---|---|---|
| `research.yaml` | study manifest: question, conditions, model backends, authority | CC BY 4.0 |
| `preregistration.md` | design, scoring, decision rules; amended after first data, never rewritten | CC BY 4.0 |
| `protocol/scenarios/` | hand-authored scenarios | CC BY 4.0 |
| `protocol/scoring.yaml` | scoring rule v1 | CC BY 4.0 |
| `graph/` | claims, checks and append-only Bearing ledger | CC BY 4.0 |
| `src/awa/`, `tests/` | harness, adapters, scorer | Apache-2.0 |
| `runs/raw/`, `runs/normalized/` | model outputs | per artifact |
| `runs/annotations/`, `runs/scores/` | human annotations and derived scores | CC BY 4.0 |

The licences are mapped path by path in `REUSE.toml`.

## Running the pilot

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest

# local: run everything (server and harness) from ONE environment, so the facts
# the guard re-checks at run time are the facts of the machine doing the work.
# Start an OpenAI-compatible server (for example mlx_lm.server) on :8080
.venv/bin/python -m awa run --backend mlx-qwen3-8b --condition C0-narrative --repeats 5

# API: export OPENAI_API_KEY in your shell first; never commit it
.venv/bin/python -m awa run --backend openai-gpt-5.6-terra --condition C0p-prompted-provenance --repeats 5

.venv/bin/python -m awa preannotate    # heuristic triage only
```

### Freezing the protocol (before the first trial)

`awa run` refuses to start until `protocol/freeze.yaml` passes. The freeze must
be marked frozen and complete, it must agree with `research.yaml`, it must cover
the backend being run, and for the local backend it must match the environment
the harness is running in. **HEAD must also carry the frozen protocol tag and
all tracked files must match that tagged source state.** Untracked run/preflight
artifacts are data and do not invalidate the checkout. There is no override
flag: changing protocol or tracked source means a new amendment and tag.

```bash
.venv/bin/python -m awa freeze-facts --backend mlx-qwen3-8b   # repo id, snapshot revision, versions
.venv/bin/python -m awa preflight --backend mlx-qwen3-8b      # one non-study temperature probe
.venv/bin/python -m awa preflight --backend openai-gpt-5.6-terra
```

1. Copy those facts and the preflight results into `protocol/freeze.yaml`.
2. Record the same values as a dated amendment in `preregistration.md`.
3. Set `frozen: true` and `frozen_at`, merge, and tag `pilot-protocol-v0.1.0`.
4. Only then run the first trial.

Preflight records are immutable, append-only artifacts in `preflight/`. A retry
creates a new timestamp/hash-addressed record rather than deleting an earlier
error. They are never trials, sit outside `runs/`, and are excluded from every
FAPR calculation.

## Blind human annotation

Annotators may inspect only `runs/normalized/`. Do not open `runs/raw/` or
`runs/index.jsonl` before annotation is complete; those files reveal condition
and model.

Each trial requires **two independent human annotations**. Condition/model metadata is withheld, but the treatment can leave recognizable traces in the model's own text; this is metadata masking, not guaranteed perfect blinding:

```text
runs/annotations/
├── human/
│   ├── annotator-a/
│   │   └── T-….yaml
│   └── annotator-b/
│       └── T-….yaml
└── resolutions/
    └── T-….yaml        # only when any observation disagrees
```

The `annotator` value inside each primary annotation must match its directory
name. If the two primaries disagree on any observation, discuss the trace while
remaining blind and write a separate resolution record; never edit either
primary.

Then:

```bash
.venv/bin/python -m awa score
```

`awa score` fails closed unless every normalized trial has exactly two
distinct primary annotations and every disagreement has a resolution. Only
after those checks pass does it read the raw envelopes and unblind. The output
reports pre-adjudication raw agreement and Cohen's κ, plus FAPR stratified by
**condition × provider-reported model × fault bucket** when the reported model is stable across the trial. Requested identifiers remain in the scored record.

## Checking the claim graph

The study's claims use the Bearing ledger from
[intentional-fragility](https://github.com/mazze93/intentional-fragility):

```bash
PYTHONPATH=<intentional-fragility>/engine/src python -m bearing --root . validate
PYTHONPATH=<intentional-fragility>/engine/src python -m bearing --root . fold
```

Every claim currently reads *gone (0)*. That is the correct starting state:
its checks have not been run.
