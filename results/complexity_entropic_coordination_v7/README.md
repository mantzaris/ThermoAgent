# ThermoAgent V7: complexity-dependent generalized-entropic coordination

## Research question

When does distributed generalized-entropic consensus become useful for
communication-efficient monitoring and risk-controlled coordination among
independent autonomous agents in complex, coupled, partially observed
networks?

V7 is a new namespace built from frozen V6 commit
`8013300c23553928a0269e6be27f5baaedee7e53`. V1–V6 artifacts were neither
regenerated nor modified. V6 remains the low-complexity historical boundary.

## What changed from V6

V7 has separate humanitarian multi-commodity and defensive abstract
utility-restoration state machines. Small, medium, and large configurations
use 12/28/52 persistent agents, 8/16/30 operational nodes, and 30/60/100
steps. Agents control multiple assets; resource contention, delayed movement,
topology-dependent service, telemetry corruption, partitions, commitments,
cascades, and multi-step causal chains change future feasibility.

Every agent has its own observation history, belief, memory, utility, assets,
commitments, inbox, outbox, role-specific tools, and action process. Peer
information arrives only through logged messages or fully costed distributed
sketches. The environment validates typed actions but never substitutes an
oracle decision.

## Information measures and control

The Level-2 controller receives normalized Shannon and Tsallis entropy for
`q={0.5,1,1.5,2,3}`, Gini–Simpson impurity, pooled uncertainty,
Jensen–Shannon/Jensen–Tsallis disagreement, graph-weighted disagreement,
consensus residual, and temporal slopes. Gini–Simpson is not the economic
Gini. These are information-theoretic/statistical-mechanics analogies, not
literal physical thermodynamics. Entropy never changes action effects.

Operational proposals, information gathering, communication, and delegation
are separate typed fields. Operational messages and thermodynamic sketches
are accounted independently in messages, bytes, drops, and latency.

## Protocol and stages

- Feasibility gates: all A/B/C gates passed; this did not require a favorable
  entropy effect.
- Frozen protocol: `v7-protocol-candidate-1.0`;
  checksum `760e9d019140dc0a1edf16af76f0d0a393e09d3680a3ece2499e84a8b4d0fff5`.
- Episode counts: `{"formal_communication": 0, "formal_dynamic": 0, "formal_reference": 0, "holdout": 0, "smoke_and_pilots": 43, "validation": 0}`.
- Validation unlocked: `False`.
- Holdout unlocked: `False`.

The formal independent unit is the matched environment panel. Candidate
actions within a panel are never treated as independent replicates. Models use
nested grouped folds, matched 60% action coverage, 10,000 panel bootstraps, and
prespecified same-capacity feature blocks.

## Main findings

Only retained feasibility pilots exist. Their coupling-by-fragmentation coefficient was -0.0588; this is not a formal estimate.

Formal communication H3 status: `not formally tested`.
The final prospective disposition is: **feasibility gates pass; formal protocol may now be frozen**.

Negative and neutral effects, failed feasibility iterations, environment
repairs, missing-PyTorch test evidence, and the unavailable RunPod endpoint are
retained. No result is real-human evidence. The Qwen model, if its gated stage
ran, is one LLM implementation; deterministic pilots are engineering controls.

## Integrity and reproducibility

- Tests: 378 passed under system Python; a retained isolated-venv attempt had 12 missing-PyTorch failures.
- Replay: `45` episodes, `0` mismatches.
- Maximum reconstructed conservation residual: `1.3002932064409833e-12`.
- Primary model: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, NF4/BF16 (only if gated Qwen ran).
- Simulated operator only; no human participants.

## Reproduction order

```bash
./scripts/run-v7-tests.sh
./scripts/run-v7-complexity-audit.sh
./scripts/run-v7-environment-smoke.sh
./scripts/run-v7-pilots.sh
./scripts/run-v7-pilot-iteration2.sh
./scripts/run-v7-pilot-iteration3.sh
./scripts/replay-v7-results.sh
./scripts/evaluate-v7-gates.sh
./scripts/freeze-v7-protocol.sh       # only when gates pass and source is clean
./scripts/run-v7-development.sh       # frozen, resumable formal development
./scripts/train-v7-multiseed.sh       # refuses unless formal gates unlock it
./scripts/run-v7-real-qwen.sh         # refuses unless formal gates unlock it
./scripts/run-v7-validation.sh        # refuses while locked
./scripts/run-v7-holdout.sh           # refuses while locked
./scripts/generate-v7-figures.sh
./scripts/validate-v7-pdfs.sh
./scripts/replay-v7-results.sh
./scripts/index-v7-artifacts.sh
./scripts/build-v7-report.sh
```

## Directory guide

- `protocol/`, `manifests/`: frozen design and seed provenance when unlocked.
- `pilots*/`, `raw/`: retained feasibility summaries and compressed ledgers.
- `development/`, `statistics/`, `tables/`: formal evidence if run.
- `training/`, `qwen/`: gated learned-agent evidence only.
- `figures/pdf/`, `figures/source_data/`: vector figures and exact source tables.
- `reproducibility/`: environment, replay, checksum, and PDF QA.
- `negative_results/`: failed iterations and prospective no-go disposition.

## Limitations and publication readiness

All operators are simulated, domain models are abstract, the pilot utility
action pool remains harm-heavy, graph fidelity is deliberately lower than a
real disaster or utility system, and external validity is untested. A positive
AIJ claim requires validation and a single locked holdout; if either is locked,
this package is an engineering/development study rather than confirmatory
journal evidence.
