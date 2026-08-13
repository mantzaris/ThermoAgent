# DOET implementation record

Status: local implementation and deterministic preflight complete; real-LLM
validation pending RunPod availability.

Planned implementations are `DOET-rule` (transparent stateful trigger and fixed
mode rules) and `DOET-RL` (the trigger gates expanded options while an independent
decentralized actor chooses among eligible options). Each agent retains its own
private observation, memory, utility, inbox, commitments, planning context, and
authority. The simulator validates actions but does not issue domain decisions.

All entropy sketches, alerts, operational messages, bytes, prompted/generated
tokens, LLM calls, communication-active epochs, latency, and estimated cost will
be included in accounting.

## Implemented components

- `thermoagent/doet.py`: independent per-agent CUSUM/simple-hysteresis state
  machines with direction fixed by development/validation, dwell, cooldown,
  separate on/off/crisis thresholds, confidence attenuation from local
  consensus disagreement, and bounded neighbor-alert evidence.
- Three modes: quiet (local planning and sparse sketching), targeted (bilateral
  information/negotiation), and crisis (coalition and accelerated planning).
- DOET-rule and DOET-RL. In DOET-RL the trigger masks expanded communication
  options; the actor still consumes exactly 24 private/local features and each
  LLM planner retains its own context and authority.
- Explicit `entropy_alert` messages use the ordinary lossy communication
  channel, consume the sender's budget, and are included in message/byte
  accounting. They carry only a coarse anomaly level and recommended mode, not
  an exact entropy value or true disruption label.
- Sparse privacy-preserving gossip retains a prior distributed estimate between
  exchanges and refreshes the agent's own sketch locally. Pairwise matchings
  bound per-round traffic. Every directed sketch is counted.
- Strong `fixed_always_on`, periodic, random budget-matched, private-local-KPI
  CUSUM, global-entropy oracle, and disruption-label oracle controls.
- Periodic and random budget controls preserve local planning every eight
  periods while intensive communication is inactive. Their validation-derived
  rates target DOET's total counted messages, including entropy-sketch traffic,
  through the fixed control's measured messages per active decision; achieved
  budget mismatch is reported rather than silently assumed away.
- The private-KPI comparator now carries its KPI-specific nominal normalizers
  into the holdout. Validation scales its purely local CUSUM residual by the
  clipped DOET/KPI message ratio; this provides a prospective approximate
  traffic match, with achieved mismatch retained as an explicit result.
- Public-route/local-coalition action affordances. The planner sees public
  initial routes and its own known coalition state, never another agent's
  inventory/cost. Closed routes can still fail at execution, preserving genuine
  replanning. This repair addresses the v1 tie mechanism symmetrically across
  every v2 method.
- Multiple-checkpoint experiment matrices with balanced round-robin RL-seed
  assignment and unambiguous run IDs.
- Restartable three-variant multi-seed training. Each of no-entropy, ThermoAgent
  v1-style, and DOET-RL receives five independent initializations and an
  identical 192-episode budget; final checkpoints are selected by fixed budget,
  never outcome. Although this staged PPO is CPU-bound, its elapsed time on the
  reserved Pod is charged to the 35-hour resource budget and reported as such.
- Fail-closed holdout generation and protocol verification before every locked
  sweep; the generator checks checkpoint hashes, validation status, exact
  balance, new seed separation, episode count, a fixed 20,000-replicate
  stratified Monte Carlo precision analysis, and the 35-GPU-hour cap.
- Updated replay includes protocol messages in causal ledger order. All eight
  v2 mock preflight episodes replay exactly.
- Filtered-deployment provenance with exact source checksum and originating
  branch/commit, plus v2-only control synchronization and retrieval so remote
  work cannot replace frozen-v1 result paths.
- The experiment-level `llm_seed` now initializes both PyTorch and CUDA in the
  real Transformers planner. Decoding remains frozen and deterministic, but the
  manifest seed is therefore the seed actually applied at model load.
- Restartable, outcome-sealed v2 job controls; a health command exposes only
  process/completion counts during holdout.
- Episode-paired non-inferiority and communication analysis with 10,000
  hierarchical bootstrap replicates, explicit training-seed resampling,
  preregistered Holm tests, multi-cost Pareto hypervolume, partition/consensus
  mechanisms, message-type accounting, CSV/LaTeX tables, and evidence-bound
  README/paper-summary generation. Failed locked episodes remain public, are
  never imputed, and fail the confirmatory hypothesis classification while
  still allowing explicitly incomplete matched-pair summaries.
- Fail-closed reproduction commands for design, freeze, holdout, replay,
  analysis, vector figures, PDF validation, reporting, and indexing.

## Verification

The current complete suite is 128/128 passing. New tests cover trigger
validation, per-agent state isolation, no global trigger input, dwell/cooldown,
bounded alert propagation, mode cadence, route-information privacy, counted
sketches and alerts, strong fixed communication, DOET-RL actor inputs, unseen
topology connectivity, balanced five-seed assignment, paired/hierarchical
analysis, multi-cost frontier behavior, filtered provenance, and deterministic
replay. A dedicated control test verifies that a zero-rate random gate still
performs quiet local planning without activating communication.

The eight-episode mock preflight completed with zero failures and maximum
absolute material residual below `1.14e-13`; all eight ledgers replayed exactly.
It is an engineering check only and supplies no research claim.

## Filtered remote execution sequence

The source checksum includes `thermoagent/`, `configs/`, `scripts/`, `tests/`,
`pyproject.toml`, and `requirements-runpod.txt`; it intentionally excludes all
result data and credentials. Run these first on the local clean branch:

```bash
./scripts/runpod-smoke-test.sh
./scripts/runpod-sync.sh
./scripts/runpod-sync-v2-controls.sh bootstrap
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/setup-doet-runpod.sh'
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/start-doet-job.sh doet-profile ./scripts/run-doet-profile.sh'
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/doet-job-status.sh'
./scripts/runpod-fetch-v2-results.sh
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/start-doet-job.sh doet-validation ./scripts/run-doet-validation.sh'
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/doet-job-status.sh'
```

The eight-episode real-Qwen profile (two applications, two regimes, two
methods) must be used to append measured calls, tokens, disk, wall time, and
the projected validation/holdout cost to `notes/14_entropy_trigger_protocol.md`
before validation begins. After validation exits successfully, train every seed
without filtering:

`setup-doet-runpod.sh` is deliberately separate from the frozen-v1 setup
script: it writes the dependency snapshot and test log only under the v2
namespace and cannot overwrite `results/reproducibility/environment.json` or
`results/logs/setup/` from the original study.

```bash
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/start-doet-job.sh doet-training ./scripts/train-doet-multiseed.sh'
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/doet-job-status.sh'
./scripts/runpod-fetch-v2-results.sh
./scripts/design-doet-holdout.sh
```

At that boundary, inspect only validation/training evidence, complete
`notes/14_entropy_trigger_protocol.md`, commit the generated config, selected
trigger, all 15 checkpoints, design, statistics code, and protocol text, and
verify a clean tree. Then recapture and deploy that exact committed snapshot:

```bash
git status --short
./scripts/runpod-sync.sh
./scripts/runpod-sync-v2-controls.sh resume
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/freeze-doet-holdout.sh'
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/start-doet-job.sh doet-holdout ./scripts/run-doet-holdout.sh'
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/doet-job-status.sh'
```

The freeze command refuses dirty originating provenance, a branch other than
`entropy-triggered-communication`, a changed source checksum, fewer than 15
checkpoints, failed tests, or a missing design artifact. During the last job,
use only the health command; do not fetch or open partial outcomes. Once it
finishes, fetch and rebuild without rerunning episodes:

```bash
./scripts/runpod-fetch-v2-results.sh
./scripts/rebuild-doet-results.sh
```
