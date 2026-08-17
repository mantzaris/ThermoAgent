# Entropy-triggered belief monitoring V8

## Scientific disposition

V8 is a **development-stage no-go study**. It asked whether locally deployable
generalized-information scheduling could reduce exact belief-sketch traffic
while preserving distributed estimation and frozen-agent performance. V7
commit `e46b6738231883e92b9b525ab1c3c190e38391e7` and all earlier namespaces
remain unchanged.

The final pilot repaired two genuine hysteresis defects. The repaired trigger
became information-driven, but it transmitted during nominal pre-disruption
operation at 71–86%, far above the prospectively fixed 10% limit. Neither
declared candidate passed in both applications. Formal development, five-seed
training, validation, and holdout were therefore not run. No confirmatory V8
claim is supported.

## Trigger and actual wire protocol

The candidate score combined 0.45 Jensen–Shannon drift, 0.25 maximum normalized
Tsallis-spectrum drift over q={0.5,1,1.5,2,3}, 0.15 confidence drift, and
0.15 bounded message age. Pilot iteration 3 evaluated `tau_on` 0.11 and 0.115,
with `tau_off=0.04`, two-step cooldown, a 30-step maximum-silence deadline,
partition-recovery refresh, two-hop forwarding, and deterministic uint8 simplex
encoding. The serializer is an actual `TBV8` big-endian binary frame with IDs,
step, confidence, encoding, hop count, payload, and CRC32; byte counts use
`len(serialized_frame)`, not a formula.

The strongest non-entropic comparator was not frozen because the trigger gate
failed. KPI-change 0.12 is shown only as a development diagnostic.

## Development evidence

- humanitarian (n=6 panels): exact sketch-wire byte reduction 0.2396 [0.1497, 0.3343]; distributed-state error change -0.0003 [-0.0007, 0.0000]; pre-disruption transmission rate for tau=0.115 82.7%.
- utility restoration (n=6 panels): exact sketch-wire byte reduction 0.2775 [0.2042, 0.3525]; distributed-state error change 0.0003 [0.0000, 0.0006]; pre-disruption transmission rate for tau=0.115 71.2%.

The tau=0.115 point estimates reduced exact sketch bytes by 24.0% in
humanitarian and 27.8% in utility restoration, but the confidence intervals are
based on only six independent panels per application and the scheduler violated
the nominal-traffic gate. Fully combined byte reductions were 22.5% and 22.9%.
These are pilot diagnostics, not H1 evidence. H2 and H3 were not formally tested.

## Autonomous agents and stage boundary

The retained pilots used the independent persistent V7 agents and deterministic
decentralized rule policy; delivered sketches updated recipient beliefs and
could alter consequential actions. A one-seed, six-episode sequential-IPPO
engineering pilot exercised 1,320 transitions and all four delegation actions.
The prospective five-seed formal training was locked, so V8 does not claim
multi-seed learned-agent replication or frozen-policy noninferiority. Qwen and
human participants were not used.

## Integrity and compute

- Development protocol: `v8-development-3.0-no-go`;
  hash `32b519cd7c282fbc8eecc60757c511bc74322ddf26919a9af08752d74969684f`.
- Completed stage counts: `{"development": 624, "hysteresis_repair_pilot": 60, "hysteresis_repair_pilot_v2": 60, "hysteresis_repair_pilot_v3": 60, "pilots": 196, "pilots_v2": 196, "pilots_v3": 88, "routing_repair_pilot": 8}`.
- Retained invalidated/partial stage accounting:
  `{"development_final_hysteresis_suppression_invalidated": {"complete_episodes": 288, "partial_run_directories": 0}, "development_final_pre_hysteresis_invalidated": {"complete_episodes": 108, "partial_run_directories": 2}}`.
- Sequential RL engineering pilot: one seed, six episodes,
  `1320` temporally linked transitions;
  formal five-seed training was not run.
- Tests: 422 full-suite tests (0 failures, 0 errors, 0 skipped); 43 focused V8 tests (0 failures, 0 errors, 0 skipped).
- Replay: 1688 ledgers,
  0 mismatches; maximum conservation residual
  1.7337242752546445e-12.
- GPU hours, LLM calls, prompt/generated tokens, and cloud cost: zero. Execution
  used local CPU NumPy and the existing RunPod endpoint was unreachable.

## Reproduction order

```bash
./scripts/run-v8-tests.sh
./scripts/run-v8-hysteresis-repair-pilot.sh
./scripts/run-v8-hysteresis-repair-pilot-v2.sh
./scripts/run-v8-hysteresis-repair-pilot-v3.sh
./scripts/analyze-v8-no-go.sh
./scripts/replay-v8-results.sh
./scripts/generate-v8-figures.sh
./scripts/validate-v8-pdfs.sh
./scripts/record-v8-manual-pdf-qa.sh
./scripts/close-v8-development-no-go.sh
./scripts/compact-v8-artifacts.sh
./scripts/build-v8-report.sh
./scripts/index-v8-artifacts.sh
```

The earlier pilots and invalidated development attempts are retained with their
own manifests. `training/NOT_RUN.md`, `validation/NOT_RUN.md`, and
`holdout/NOT_RUN.md` record the prospective stop. See `CLAIMS_MATRIX.md` for
prohibited extensions and `INDEX.csv` for every artifact.

## Limitations and prohibited claims

The final mechanism pilot has only six panels per application, used rule-policy
outcomes, and never reached validation. It cannot establish communication-
efficient monitoring, entropy-specific superiority, downstream learned-policy
retention, real-world utility or humanitarian performance, Qwen behavior, or
human effectiveness. Information entropy is not literal thermodynamics.
