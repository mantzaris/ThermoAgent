# ThermoAgent experimental results

Status: complete frozen evaluation. All 1,096 post-freeze episodes finished,
all 1,096 event ledgers replayed exactly, and all ten paper-facing PDFs passed
mechanical and manual visual QA. No final seed failed or was selectively rerun.

The result is scientifically mixed and mostly negative. Distributed operational
entropy is a useful disruption monitor, but the calibrated free-energy gap is
not. Entropy-conditioned coordination shows modest in-distribution improvements
over its matched no-entropy actor, but those effects do not survive the
prespecified familywise correction and disappear on locked holdout. Strong
fixed, scripted, and centralized alternatives generally match or beat the
autonomous system. In the tested environments, the added autonomy and
communication cost are not justified.

## 1. Research question

Can independent, tool-using autonomous agents use distributed
statistical-mechanics signals to recognize collective instability, negotiate
temporary coalitions, and recover logistics performance under private
information, mixed objectives, disrupted communication, and compound shocks?

ThermoAgent is a shared architecture for two abstract logistics applications.
A frozen language model proposes concise structured domain actions, while a
small PPO metapolicy selects how each agent coordinates. Each actor sees only
local/private features, delivered messages, commitments, and a locally gossiped
entropy/free-energy estimate. The simulator alone validates typed tools and
enforces quantitative state transitions. Exact global state is evaluator-only,
except in the explicitly named oracle ablation.

## 2. What makes the agents independent

Each persistent organization has its own:

- identity, role, obligations, utility weights, risk tolerance, and constraints;
- private observation vault, belief state, working memory, and episodic memory;
- inbox, outbox, trust estimates, commitment ledger, RNG/policy state, and
  separate LLM context;
- planning/replanning loop and role-scoped validated tool set;
- authority to accept, reject, counter, join, refuse, or withdraw.

Private state is not stored in a shared prompt or domain-decision process.
Information crosses agents only through an explicit logged message or public
coarse sketch. A shared model instance is used only for inference efficiency.
The simulator can reject an invalid action, but it never repairs or invents a
domain decision.

The 101-test frozen suite directly covers private-state access denial, explicit
message-only transfer, removal effects on negotiation, divergent utilities,
refusal/revision, decentralized actor features, central-coordinator privacy,
role permissions, conservation, deterministic replay, gossip, structured-output
recovery, atomic publication, and PDF validation. Strict real-model Stage 1
also demonstrates an offer, rejection, counteroffer, failed route action,
successful replan, delivered coalition invitation, and a separate invitee
agent's validated join. Failed predecessors are retained.

## 3. Hardware, model, and serving configuration

- Remote execution path: `/workspace/ThermoAgent` on the existing RunPod Pod.
- GPU: NVIDIA RTX 4090, 24,564 MiB; driver 570.195.03; CUDA driver/runtime
  support 12.8.
- Host: 32 CPU threads, approximately 124 GiB RAM.
- Python/PyTorch: Python 3.12.3; PyTorch 2.8.0+cu128.
- Primary planner: `Qwen/Qwen2.5-7B-Instruct`.
- Immutable model revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Serving: Transformers 4.55.4, Accelerate 1.10.1, bitsandbytes 0.47.0,
  in-process batched inference.
- Precision: bitsandbytes NF4, double quantization, bfloat16 compute.
- Decoding: deterministic (`do_sample=false`, temperature 0, top-p 1), maximum
  2,560 input and 160 generated tokens in comparative runs.
- Comparative LLM seed: 0; RL seed: 3001; prompt revision:
  `planner-json-v6`.
- Corrected CUDA smoke: two prompts, 192 generated tokens in 4.54 s
  (42.3 tokens/s), 7.07 GiB peak allocated VRAM, and 100% JSON/static-schema/
  dynamic-tool validity.

The isolated environment is `/workspace/ThermoAgent/.venv`. Hugging Face and
project caches are under `/workspace/.cache`, outside Git. Base-model weights,
virtual environments, credentials, and caches are not in this result tree.
Exact dependency and hardware captures are in
[`reproducibility/`](reproducibility/).

## 4. Applications, scenarios, and methods

The commercial environment represents suppliers, manufacturers, carriers,
warehouses, and retailers. The humanitarian environment represents abstract
agencies/NGOs, transport, depots, clinics, and communities. Both track conserved
material, production and handling capacities, lead times, shipments, stochastic
demand, backlog/unmet need, costs, commitments, and explicit communication.
The humanitarian environment is not a behavioral model of real organizations.

Scenarios vary shared/moderate/private information; aligned/moderately mixed/
strongly mixed objectives; reliable/intermittent/partitioned communication;
nominal, moderate, correlated, and compound disruptions; and two main system
sizes. Disruptions include capacity impairment, route/warehouse/depot failure,
demand or need surges, communication partition, coordinating-agent loss, and
compound combinations. Holdout uses unseen nine-agent topologies and new
correlated/compound combinations.

### Baselines

1. `centralized_lookahead`: full-information deterministic replanning every
   period; an unattainable upper bound.
2. `centralized_llm`: one legal coordinator using only disclosed reports and
   public route eligibility; absent reports fail closed.
3. `scripted_independent`: persistent private organizations with fixed
   planning/negotiation rules.
4. `autonomous_no_comm`: private LLM agents with no inter-agent messages.
5. `autonomous_fixed_comm`: private LLM agents with fixed periodic communication.
6. `learned_no_entropy`: the matched PPO coordination policy with monitor
   features zeroed.
7. `thermoagent`: distributed entropy/free-energy features plus learned
   coordination.

Ablations add entropy to the LLM without an RL gate, no episodic memory,
activity-matched random gating, exact-global oracle entropy, and a causal
shuffled/delayed entropy signal.

### Completed matrix

| Stage | Applications and sizes | Design | Seeds | Completed |
|---|---|---|---:|---:|
| Engineering | both | deterministic invariants + CUDA/model smoke | fixtures | 101 tests pass |
| Agentic smoke | 2-agent negotiation, 5 commercial, 6 humanitarian | real Qwen capability gates | fixtures | strict v6 pass |
| Eligible pilot | both, 8 agents | 7 methods; nominal + compound partition | 3 paired | 84/84 |
| Main | commercial 8/11; humanitarian 8/10 | 7 methods; 9 targeted cells | 8 paired | 944/944 |
| Ablations | commercial 11; humanitarian 10 | 9 methods; compound partition | 4 paired | 72/72 |
| Locked holdout | both, unseen 9-agent topology | 5 methods; 2 unseen shocks | 4 paired | 80/80 |

Earlier diagnostic pilots and failed smoke attempts remain in the tree. Fifty-
eight completed pre-freeze diagnostic episodes are prospectively marked
`analysis_valid=false`; exclusion rules are in
[`excluded_runs.json`](reproducibility/excluded_runs.json). No post-freeze row
is excluded.

## 5. Outcomes and statistical methods

The primary experimental unit is one complete multi-agent episode.

- Commercial primary outcome: service-loss area under the curve, lower better.
- Humanitarian primary outcome: cumulative unmet weighted need, lower better.

Identical demand/disruption seeds are paired across methods. Main estimates
report paired mean differences, seed-cluster hierarchical bootstrap 95%
intervals, paired win rate, probability of superiority, standardized paired
effect, paired sign-flip tests, and Holm correction over five preregistered
comparisons within each application. Scenario and ablation intervals are
secondary. Failure-aware tables retain asymmetric failures as ranked losses;
all final rows happened to complete. Messages, agents, and timesteps are not
pseudo-replicated as independent outcome units.

In the tables below, improvement is comparator loss minus ThermoAgent loss, so
positive values favor ThermoAgent.

## 6. Main quantitative findings

### Entropy-conditioned actor versus matched no-entropy actor

| Application | ThermoAgent | No entropy | Improvement | 95% CI | Win rate | `d_z` | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Commercial | 16.568 | 16.755 | +0.187 (1.12%) | [0.049, 0.310] | 0.486 | 0.932 | 0.0856 |
| Humanitarian | 5815.83 | 5922.34 | +106.51 (1.80%) | [57.90, 156.58] | 0.514 | 1.386 | 0.0584 |

Both means and hierarchical intervals favor ThermoAgent, but neither comparison
passes the prespecified Holm-adjusted 5% threshold. The result is suggestive,
not confirmatory. Many exact ties and some losses explain why win rates are near
one half despite positive magnitude-weighted differences.

### Stronger alternatives

| Application | Comparator | Improvement | 95% CI | Holm p | Result |
|---|---|---:|---:|---:|---|
| Commercial | fixed communication | -0.722 | [-0.968, -0.478] | 0.0584 | fixed lower loss |
| Commercial | scripted independent | -0.681 | [-1.148, -0.264] | 0.0584 | scripted lower loss |
| Commercial | legal central LLM | -0.310 | [-0.532, -0.088] | 0.0856 | central lower loss |
| Commercial | lookahead upper bound | -15.370 | [-15.770, -14.966] | 0.0584 | lookahead far better |
| Humanitarian | fixed communication | -3.47 | [-50.25, 46.54] | 0.9066 | effectively tied |
| Humanitarian | scripted independent | -143.12 | [-294.31, 14.52] | 0.2568 | uncertain/scripted-favoring |
| Humanitarian | legal central LLM | -182.67 | [-248.16, -117.18] | 0.0584 | central lower loss |
| Humanitarian | lookahead upper bound | -4958.26 | [-5193.08, -4741.31] | 0.0584 | lookahead far better |

The adjusted p-values are coarse with only eight environment-seed clusters and
five comparisons. The intervals, win rates, individual points, and holdout are
therefore reported alongside them rather than reducing the result to a binary
significance label.

### Are autonomous agents more necessary under privacy and misalignment?

No. The fixed-cell response surface is negative everywhere. Relative to one
deployable fixed/legal-central/scripted benchmark chosen once per factor cell,
ThermoAgent's normalized advantage is `-3%` to `-9%` commercially and `-2%` to
`-14%` in humanitarian logistics. More private information or more objective
misalignment did not create an autonomous advantage. This descriptive surface
uses no per-seed oracle selection.

### Locked holdout

On eight paired unseen cells per application, ThermoAgent and learned/no-
entropy tie exactly; ThermoAgent and scripted agents also tie exactly. Fixed
communication is slightly better commercially (-0.167, CI [-0.500, 0]) and ties
in humanitarian. Lookahead is much better in both. The in-distribution entropy
effect therefore does not transfer to the locked topology/shock holdout.

### Ablations

Four seeds per application provide limited power. No ablation survives Holm
correction. ThermoAgent versus no entropy is -0.076 commercial (CI [-0.579,
0.535]) and an exact humanitarian tie. Random gating, shuffled entropy,
entropy-to-LLM, memory removal, and the exact oracle are tied or imprecise. The
ablation set does not isolate a reliable causal value for entropy/free energy,
memory, or learned communication gating.

## 7. Monitoring findings

The broad “entropy/free-energy signal” hypothesis splits sharply:

| Signal | Average precision | ROC AUC | Precision | Recall | Nominal false alarms |
|---|---:|---:|---:|---:|---:|
| Operational entropy | 0.934 | 0.863 | 0.970 | 0.539 | 0.030 |
| Operational energy | 0.885 | 0.800 | 0.689 | 0.925 | 0.755 |
| Free-energy gap | 0.577 | 0.393 | 0.668 | 0.917 | 0.826 |
| Distributed free-energy mean | 0.676 | 0.481 | 0.667 | 0.917 | 0.830 |
| Interaction entropy | 0.551 | 0.411 | 0.095 | 0.010 | 0.181 |

Operational entropy is a useful onset detector at a matched nominal threshold.
Operational energy is predictive but poorly calibrated at that threshold. The
free-energy gap is not a useful high-direction alarm: it falls under disruption
by 0.0108 commercially and 0.0046 in humanitarian logistics, and sensitivity
variants remain weak. These constructs are statistical-mechanics-inspired
operational summaries, not literal thermodynamic quantities.

Where detected, the median alarm is generally at shock onset and before visible
service collapse. The claim is early recognition after onset—not prediction of
an exogenous event before it occurs.

Distributed gossip behaves coherently:

| Application | Communication | Entropy MAE | Free-energy MAE |
|---|---|---:|---:|
| Commercial | reliable | 0.00524 | 0.00327 |
| Commercial | intermittent | 0.01833 | 0.01145 |
| Commercial | partition | 0.03976 | 0.02485 |
| Humanitarian | reliable | 0.00072 | 0.00045 |
| Humanitarian | intermittent | 0.02708 | 0.01692 |
| Humanitarian | partition | 0.06280 | 0.03924 |

Consensus RMSE and estimator error have Spearman association about 0.966 on
reliable links and 0.985--0.996 under impaired links. Commercial source
localization is strong (top-1 0.946--1.0; top-3 1.0). Humanitarian top-1 is mixed
(0--0.089 in several large main cells, 1.0 in small/holdout cells), while top-3
is always 1.0.

## 8. Agentic behavior and communication cost

ThermoAgent visibly coordinates more, but that is not the same as useful
coordination.

| App | Method | Structured valid | Tool valid | Revisions | Proposals | Formed | Useful precision | Breaches | Failed actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Commercial | ThermoAgent | 0.928 | 0.534 | 16.13 | 28.07 | 8.54 | 0.059 | 0.93 | 63.49 |
| Commercial | no entropy | 0.922 | 0.589 | 14.89 | 19.57 | 6.69 | 0.032 | 1.96 | 53.40 |
| Humanitarian | ThermoAgent | 0.975 | 0.579 | 13.82 | 27.33 | 9.83 | 0.028 | 0.79 | 55.54 |
| Humanitarian | no entropy | 0.956 | 0.651 | 11.00 | 18.85 | 7.33 | 0.013 | 2.47 | 40.82 |

Across post-freeze ThermoAgent rows, commercial agents sent 15,449 ordinary
messages and 303,482 monitor sketches; humanitarian agents sent 14,393 ordinary
messages and 287,722 sketches. ThermoAgent forms more coalitions and breaches
fewer commitments, but has lower valid-tool rates, more failed actions, and
useful-coalition precision below 6%. The Pareto result is unfavorable.

## 9. Hypothesis disposition

| Hypothesis | Disposition | Evidence |
|---|---|---|
| Independent-agent boundary is enforced | Supported as engineering invariant | privacy/authority tests and strict Stage 1 |
| Both simulators conserve resources and replay | Supported | 1,096/1,096 final ledgers, max residual <4.55e-13 |
| Distributed gossip tracks global monitoring when connected | Supported | low reliable-link error; predictable partition degradation |
| Operational entropy recognizes collective instability | Supported for onset detection | AP 0.934, ROC AUC 0.863, 3.0% false alarms |
| Calibrated free-energy gap is a useful high-direction signal | Unsupported | ROC AUC 0.393, high false alarms, reversed mean direction |
| Entropic RL improves matched non-entropic RL | Mixed | suggestive main means; no Holm confirmation; exact holdout ties |
| Autonomous-agent value rises with privacy/misalignment | Unsupported | every necessity-map cell negative |
| Autonomous agents beat strong centralized/scripted alternatives | Unsupported | fixed/scripted/legal central generally match or win |
| Coalitions improve recovery enough to justify cost | Unsupported | frequent coalitions, low useful precision, unfavorable Pareto |
| Benefit transfers across applications/topologies | Mixed/unsupported | humanitarian main suggestion, no locked-holdout effect |

## 10. Figures

All files below are vector PDFs. Their 180-DPI previews are in
[`figures/previews/`](figures/previews/). Mechanical and manual QA is recorded
in [`pdf_qa/report.json`](reproducibility/pdf_qa/report.json).

1. [`system_architecture.pdf`](figures/pdf/system_architecture.pdf) — independent
   agent boundary, local RL option selection, frozen separate-context planner,
   typed simulator/tools, explicit communication, and distributed monitor.
2. [`entropy_dynamics.pdf`](figures/pdf/entropy_dynamics.pdf) — commercial
   compound-shock trajectories for entropy, energy, free-energy gap, and
   fulfillment, aligned to disruption and median detection.
3. [`main_performance.pdf`](figures/pdf/main_performance.pdf) — all main methods
   and scenarios with uncertainty and individual paired-seed points.
4. [`communication_performance_pareto.pdf`](figures/pdf/communication_performance_pareto.pdf)
   — primary loss versus ordinary plus monitor messages; shows ThermoAgent's
   communication burden.
5. [`agentic_necessity_map.pdf`](figures/pdf/agentic_necessity_map.pdf) — fixed-
   benchmark response surface over privacy and objective misalignment; every
   observed cell is negative.
6. [`recovery_curves.pdf`](figures/pdf/recovery_curves.pdf) — compound-shock
   service-loss trajectories across methods and applications.
7. [`ablation_effects.pdf`](figures/pdf/ablation_effects.pdf) — nine
   parameter-matched monitoring/coordination variants with four seed points.
8. [`network_snapshots_commercial.pdf`](figures/pdf/network_snapshots_commercial.pdf)
   — one replayed ThermoAgent case at nominal, onset, negotiation, coalition,
   and recovery stages; physical, communication, negotiation, commitment, and
   coalition encodings are distinct.
9. [`network_snapshots_humanitarian.pdf`](figures/pdf/network_snapshots_humanitarian.pdf)
   — corresponding representative humanitarian sequence. Network figures are
   descriptive case studies, not additional replicates.
10. [`agentic_metrics.pdf`](figures/pdf/agentic_metrics.pdf) — episode points and
    intervals for validity, agreement, revisions, coalition quality, breaches,
    and contradictions.

The frozen generator's first final render had three layout defects. The
post-freeze, presentation-only wrapper documented in
[`postfreeze_figure_polish.json`](reproducibility/postfreeze_figure_polish.json)
corrected labels/limits without modifying frozen data, statistics, or protocol
code.

## 11. Tables and statistics

- [`method_summary.csv`](tables/method_summary.csv) — application/scenario/
  method means and bootstrap intervals for primary outcome, fulfillment,
  fairness, communication, tokens, and wall time.
- [`primary_paired_comparisons.csv`](statistics/primary_paired_comparisons.csv)
  — preregistered main paired effects, intervals, win rates, standardized
  effects, sign-flip tests, and Holm adjustment.
- [`scenario_paired_comparisons.csv`](statistics/scenario_paired_comparisons.csv)
  and [`main_all_method_paired_comparisons.csv`](statistics/main_all_method_paired_comparisons.csv)
  — secondary cell-level and all-method effects.
- [`main_failure_aware_comparisons.csv`](statistics/main_failure_aware_comparisons.csv)
  and [`holdout_failure_aware_comparisons.csv`](statistics/holdout_failure_aware_comparisons.csv)
  — planned-pair accounting with failures ranked as losses.
- [`holdout_paired_comparisons.csv`](statistics/holdout_paired_comparisons.csv)
  and [`ablation_paired_comparisons.csv`](statistics/ablation_paired_comparisons.csv)
  — locked robustness and four-seed ablation effects.
- [`monitoring_summary.csv`](statistics/monitoring_summary.csv),
  [`detection_episode_summary.csv`](statistics/detection_episode_summary.csv),
  [`monitoring_predictive_value.csv`](statistics/monitoring_predictive_value.csv),
  and [`free_energy_calibration.csv`](statistics/free_energy_calibration.csv) —
  detector discrimination, timing, future-loss association, and calibration.
- [`distributed_convergence.csv`](statistics/distributed_convergence.csv) and
  [`estimator_comparison.csv`](statistics/estimator_comparison.csv) — consensus
  error and exact/delayed/noisy/absent estimator comparisons.
- [`energy_weight_sensitivity.csv`](statistics/energy_weight_sensitivity.csv),
  [`interaction_entropy_regimes.csv`](statistics/interaction_entropy_regimes.csv),
  [`source_localization_summary.csv`](statistics/source_localization_summary.csv)
  — fixed-weight robustness, joint interaction regimes, and localization.
- [`completion_by_method.csv`](statistics/completion_by_method.csv),
  [`failed_episodes.csv`](statistics/failed_episodes.csv), and
  [`excluded_episodes.csv`](statistics/excluded_episodes.csv) — transparent
  completion, failure, and prospective exclusion accounting.
- [`compute_accounting.json`](statistics/compute_accounting.json) — totals over
  all retained processed episode records.

Analysis-ready records are
[`episodes.csv`](processed/episodes.csv),
[`time_series.csv`](processed/time_series.csv), and
[`agent_metrics.csv`](processed/agent_metrics.csv). The first is the valid unit
for system-performance inference; timestep and agentic rows answer their own
monitoring/behavior questions only.

## 12. Directory and provenance map

- `manifests/`: one full manifest per episode plus
  [`main_sweep.json`](manifests/main_sweep.json),
  [`ablations_sweep.json`](manifests/ablations_sweep.json), and
  [`holdout_sweep.json`](manifests/holdout_sweep.json).
- `smoke/`: CUDA/model and real-agent capability outputs, including failures.
- `pilot/`, `main/`, `ablations/`, `holdout/`: stage completion tables/history.
- `raw/`: complete episode JSON plus gzip-compressed event ledgers. Ledgers
  include observations, memory retrieval, LLM request/response, tools/results,
  messages/offers/counters, commitments, coalitions, disruptions, and
  transitions.
- `processed/`: analysis-ready cross-stage tables.
- `statistics/` and `tables/`: inferential and descriptive outputs above.
- `figures/pdf/` and `figures/previews/`: final vector figures and previews.
- `networks/`: graph artifacts and reserved snapshot/animation locations.
- `logs/setup/`: environment setup and hardware checks.
- `logs/inference/`: model smoke and planner calls where separately captured.
- `logs/training/`: staged PPO episode/update logs, including failed starts.
- `logs/evaluation/`: episode summaries; `logs/jobs/` has detached sweep logs.
- `logs/analysis/`: combined replay/analysis operational logs.
- `checkpoints/`: only ~30 KiB coordination actors; archived superseded actors
  remain for provenance. No base weights.
- `reproducibility/`: environment lock, hardware/source checksum, nominal
  calibration, exclusions, budget, replay reports, immutable protocol, figure
  polish, and PDF QA.
- [`INDEX.csv`](INDEX.csv): 5,626-artifact catalog with type, stage, application,
  method/scenario/seed where applicable, timestamp, generating command, and
  SHA-256 checksum.

The ongoing decision/failure/reproduction record is in [`../notes/`](../notes/),
especially [`09_main_findings.md`](../notes/09_main_findings.md),
[`10_failures_negative_results_and_limitations.md`](../notes/10_failures_negative_results_and_limitations.md),
and [`12_paper_claims_and_evidence.md`](../notes/12_paper_claims_and_evidence.md).

## 13. Exact reproduction commands

Run inside `/workspace/ThermoAgent`:

```bash
# Environment, provenance, and tests
./scripts/setup-runpod.sh
./scripts/capture-reproducibility.sh
./scripts/run-tests.sh -q

# Nominal-only calibration, monitor selection, and staged PPO
./scripts/run-calibration.sh
./scripts/train-policies.sh

# Real-model and pilot qualification
./scripts/run-model-smoke.sh
./scripts/run-agentic-smoke.sh
./scripts/run-pilot.sh

# Freeze once, then execute the final matrices
./scripts/freeze-protocol.sh
./scripts/run-main.sh
./scripts/run-ablations.sh
./scripts/run-holdout.sh

# Immutable replay and frozen analysis
./scripts/replay-results.sh
./scripts/analyze-results.sh

# Base vector figures, presentation-only final polish, and mechanical QA
./scripts/generate-figures.sh
./results/reproducibility/tools/polish-figures.sh
```

After personally inspecting all ten files under `figures/previews/`, record the
manual step and refresh the index:

```bash
.venv/bin/python -m thermoagent mark-visual-qa \
  --results results \
  --reviewer "<reviewer>" \
  --note "All ten final previews inspected at original resolution"
.venv/bin/python -m thermoagent index --results results
```

To rebuild derived artifacts from retained raw ledgers without rerunning an LLM
episode:

```bash
./results/reproducibility/tools/rebuild-final-results.sh
```

That wrapper intentionally leaves manual visual approval pending. Every sweep
uses deterministic run IDs, resumes already-complete rows, atomically publishes
episodes, and retains failures/timeouts. The immutable protocol is
[`protocol_freeze.json`](reproducibility/protocol_freeze.json); all 36 frozen
hashes must verify before interpreting a rerun.

## 14. Compute, tokens, and approximate cost

| Stage | Episodes | Including-load hours | LLM calls | Prompt tokens | Generated tokens |
|---|---:|---:|---:|---:|---:|
| Main | 944 | 15.405 | 57,588 | 97,812,326 | 3,706,858 |
| Ablations | 72 | 2.146 | 7,793 | 12,584,017 | 511,402 |
| Holdout | 80 | 1.040 | 4,152 | 6,073,489 | 249,888 |
| Post-freeze total | 1,096 | 18.592 | 69,533 | 116,469,832 | 4,468,148 |

All retained processed stages, including diagnostics, total 1,334 episode
records, 20.174 summed episode-hours, 76,939 LLM calls, 126,742,334 prompt
tokens, and 4,952,073 generated tokens. The latter episode-hour number excludes
some one-time loads and idle Pod time.

RunPod's 2026 RTX 4090 guide lists `$0.34/hour` Community and `$0.69/hour`
Secure rates; the Pod pricing documentation says compute is billed by the
second and the deployment console is authoritative:

- https://www.runpod.io/articles/guides/nvidia-rtx-4090
- https://docs.runpod.io/pods/pricing

At those two reference rates, the exact post-freeze active job time is about
`$6.32`--`$12.83`. Applying them to all retained summed episode time gives an
approximate `$6.86`--`$13.92` before unallocated model-load, idle, and storage
charges. This is an estimate, not a claim about the user's invoice.

## 15. Negative results, failures, and limitations

Negative evidence is retained rather than tuned away:

- eligible pilot controls were unfavorable, and the final protocol was not
  changed to reverse them;
- the main entropy effect misses multiplicity correction and disappears on
  holdout;
- every autonomy-necessity factor cell is negative;
- free energy has the wrong/unstable alarm direction and high false alarms;
- frequent coalitions have very low measured utility precision;
- Qwen sometimes emits semantically wrong tools despite valid JSON;
- retained Stage 1 and pre-freeze runs document self-target, coalition-member,
  route, privacy, RNG, actor-feature, and partition-timing failures.

Important limitations are one 7B model, deterministic decoding, one final RL
seed, staged/mock-planner PPO rather than online LLM PPO, only eight main and
four holdout/ablation environment seeds, sparse populations for 27 macrostates,
large mandatory sketch traffic, abstract dynamics, action-saturated holdout
cells, and mixed humanitarian source localization. The numerical lookahead is
an upper bound with impossible information. Nothing here validates real
humanitarian behavior or production deployment safety.

The final analysis produced constant-input warnings in monitoring cells where a
signal did not vary; those correlations remain missing. A duplicate read-only
analyzer was accidentally started after a yielded SSH call; both deterministic
passes finished and the index was rebuilt once. No raw or frozen artifact was
mutated. Full details are in the failure notes.

## 16. Remaining experiments

High-value extensions are a prospectively redesigned free-energy reference,
multiple RL and stochastic LLM seeds, a second open-weight model, online
LLM-coupled policy refinement, larger locked topologies, stronger action-
sensitivity checks, richer central optimization, and external domain
calibration. A second model was not added post hoc: the retained work was near
the 24-hour planning envelope, and a fair model comparison needs new prompt and
throughput qualification. These are future experiments, not missing rows in
the frozen matrix.

## 17. Evidence readiness

- **Engineering demonstration:** adequate. Independence, validation,
  conservation, real-model capabilities, replay, monitoring, and restartable
  execution are directly demonstrated.
- **Workshop paper:** adequate if framed narrowly as a rigorous architecture,
  boundary study, and negative/mixed result. It is not adequate for a headline
  that ThermoAgent improves logistics.
- **Full AIJ submission:** not adequate. It needs multi-model/training-seed
  replication, broader locked environments, stronger causal/action-sensitive
  tests, and external validation.

No background experiment or analysis process should remain. Once the final
local fetch, protocol verification, and repository hygiene checks pass, it is
safe to stop the Pod. The final user handoff records that check explicitly.
