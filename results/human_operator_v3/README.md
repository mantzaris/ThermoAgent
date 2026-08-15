# ThermoHITL v3: complete development-stage no-go result

Status: **complete, prospectively stopped before validation, RL training, and
confirmatory holdout**. All v3 evidence uses simulated operators. No human
participants were recruited and no claim about human usability, trust, fatigue,
workload, ethics, or operational safety is supported.

V3 is additive. It does not modify or reinterpret the frozen v1 study or the
completed negative v2 namespace. The v2 finding that motivated this work remains
unchanged: its entropy communication trigger never activated and communication
was not shown to be necessary. V3 corrected actionability and coordination
mechanics in development, then stopped because thermodynamic features failed the
prospective same-information incremental-value gate across both applications.

## Research question and outcome

Can privacy-preserving distributed operational entropy and energy triage scarce
simulated-operator attention among genuinely independent logistics agents and
improve outcomes per operator minute relative to equally informed conventional
KPI policies?

The proposed positive answer was **not established**. Five of six development
gates passed. Gate 5 failed because the thermodynamic view added value in the
abstract humanitarian application but not in commercial logistics. Under the
predeclared all-gates rule, validation, five-seed learned-policy training, and a
new locked holdout were never run. The repository represents missing stages as
explicit `NOT_RUN` records, never as zero-valued or imputed observations.

What is established at development level is narrower:

- v3 agents remain independent and operator views enforce a hashed information
  boundary;
- corrected typed actions can progress through transit to demand and alter loss;
- fixed communication is materially useful in the revised v3 mechanics;
- bounded authorized simulated-operator interventions can change commitments,
  accepted actions, material arrival, and outcomes;
- a transparent trigger activates promptly without nominal/pre-disruption false
  activation on the final development panels; and
- thermodynamic features have application-dependent, not general, incremental
  decision value.

These are engineering and exploratory development results, not confirmatory
evidence for ThermoHITL.

## Mechanism and independence boundary

Each persistent organization has a separate identity, system context, private
observation vault, utility, memory, commitment ledger, inbox/outbox, planner
state, RNG state, tool permissions, thermodynamic estimate, escalation state,
and action authority. Agents may accept, reject, counter, withdraw, request
help, or continue without an operator response. A shared model instance batches
inference only; it does not share contexts or make centralized domain decisions.

Execution follows the audited chain:

`distributed state -> independent request -> bounded attention queue ->`
`authorized hashed view -> typed operator intervention -> autonomous response ->`
`accepted action/commitment -> transit -> demand arrival -> primary loss`.

Advisory directives retain agent refusal. Mandatory emergency overrides are
typed, scoped, time limited, and followed by a logged return of authority. The
simulator validates but never invents an ordinary agent decision.

## Applications and operator authority

Commercial logistics uses private suppliers, manufacturers, carriers,
warehouses, and retailers. The operator may authorize emergency routes or
resources, temporary coarse information sharing, priority changes, bounded
constraint relaxation, or conflict resolution.

Humanitarian logistics uses abstract NGOs/agencies, transport providers, depots,
clinics, and communities. The simulated incident commander may authorize
abstract emergency access/resources, priority changes, temporary hubs or
sharing, and bounded conflict resolution. This is not a model of real
humanitarian organizations or ethical decision making.

Operator profiles vary finite slots, response latency, service time, accuracy,
fatigue sensitivity/recovery, and risk aversion. The primary profile is
`high_accuracy_bounded`; `oracle` is an explicitly unattainable upper bound.
Profile values are in [`operator_profiles.csv`](tables/operator_profiles.csv).

## Thermodynamic variables

Operational energy is the fixed development score

`E = .24 backlog + .22 unmet + .16 congestion + .14 lateness`
`    + .12 commitment risk + .12 route/safety risk`.

The weights came from service-first application utilities, not observed
treatment effects. Entropy is reported through flow-allocation entropy, local
belief entropy, and bounded Jensen–Shannon disagreement. Agents exchange coarse
link-local sketches and maintain distributed entropy/energy estimates,
two-sided nominal residuals, slope, and consensus confidence. Exact evaluator
state is analysis-only. The Helmholtz-like diagnostic `F = E - T*S` uses bounded
disruption volatility as temperature and is not a literal physical quantity or
the primary trigger.

The selected development trigger used `tau_on=1.5`, `tau_off=0.6`,
`actionable_tau_on=1.1`, minimum dwell 2, and cooldown 3. It was never promoted
to validation.

## Functional dashboard and view experiment

The dashboard under `thermoagent/dashboard/` is part of the decision mechanism,
not a post-hoc illustration. It renders exactly the schema-validated payload
consumed by the simulated operator and supports deterministic GPU-free replay,
live mock simulation, play/pause/step/rewind, jump-to-alert, matched branch
inspection, and SVG/JSON export.

Implemented views are local KPI, entropy only, energy only, combined
thermodynamics, thermodynamics plus disagreement, and a separately labeled
evaluator-global oracle. Payloads exclude other agents' raw private state, RNG,
future disruptions, and counterfactual outcomes; each payload is canonically
hashed and logged. See the [dashboard guide](../../thermoagent/dashboard/README.md),
[data dictionary](../../thermoagent/dashboard/DATA_DICTIONARY.md), and
[information boundary](../../thermoagent/dashboard/INFORMATION_BOUNDARIES.md).

## Hardware, model, and software

- Existing RunPod path: `/workspace/ThermoAgent`; no second Pod was created.
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB; driver 570.195.03; CUDA 12.8.
- Host: 32 CPU threads, approximately 124 GiB RAM.
- Python 3.12.3; PyTorch 2.8.0+cu128.
- Planner: `Qwen/Qwen2.5-7B-Instruct` at immutable revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Transformers 4.55.4, Accelerate 1.10.1, bitsandbytes 0.47.0.
- NF4 double quantization with BF16 computation; deterministic generation.
- Final planner prompt revision:
  `planner-json-v9-human-affordance-repair-turn`.
- Existing isolated environment and model cache were reused; no compatible
  package was reinstalled.

The non-secret environment capture is
[`runpod_environment.json`](reproducibility/runpod_environment.json).

## Experiment inventory

| Evidence | Episodes or units | Status |
|---|---:|---|
| Antecedent v2 actionability diagnosis | 952 v2 ledgers | derived diagnostic; v2 unchanged |
| Deterministic v3 development | 809 episodes | complete |
| Retained real-Qwen v8 qualification | 4 episodes | failed structured-validity gate |
| Versioned real-Qwen v9 retry | 4 episodes | qualification passed |
| Exact v3 replay | 817/817 ledgers | zero mismatches |
| Validation | 0 | prospectively not run |
| Independent RL training seeds | 0 | prospectively not run |
| Locked holdout | 0 | prospectively not run; no outcomes opened |

The primary unit for outcome analysis is a complete episode. Development paired
effects use common scenario seeds and a fixed-seed 10,000-replicate bootstrap.
No messages, agents, or timesteps are treated as independent outcome units.

## Prospective gate results

| Gate | Outcome | Key evidence |
|---|---|---|
| 1 Engineering | **pass** | 183 tests; 817 exact replays; zero privacy/nonfinite/mismatch failures; max conservation residual `6.82e-13` |
| 2 Agent actionability | **pass** | Qwen v9 first pass 97.89%, after one repair 100%, accepted-to-next/demand 84.21%; deterministic causal mechanics passed |
| 3 Coordination necessity | **pass** | fixed communication reduced aggregate loss 48.45% commercial and 52.54% humanitarian; all three important regimes improved |
| 4 Human causal usefulness | **pass** | bounded KPI-triggered operator reduced loss 1.59% commercial and 2.80% humanitarian; complete chains in both applications |
| 5 Thermodynamic information value | **fail** | commercial failed both ranking and 5% utility criteria; humanitarian passed |
| 6 Trigger feasibility | **pass** | 100% timely disrupted-episode activation; 0% missed, pre-disruption, and nominal false activation; 97 beneficial complete probes |

The authoritative machine-readable decision is
[`gate_status.json`](development/gate_status.json).

### Actionability diagnosis and correction

The frozen-v2 diagnosis found 1,983 material proposals, 408 accepted (20.57%),
and 396 accepted actions arriving (97.06%). The main historical bottleneck was
proposal validation/acceptance, not transit conditional on acceptance.

In the retained v3 Qwen v8 attempt, 52/305 responses were invalid: 40 used a
target outside the agent's local affordance, 11 invented a tool name, and one
missed required fields. First-pass and post-repair validity were both 82.95%,
although all 15 accepted material actions reached demand. Diagnosis showed that
the repair instruction had been appended inside Qwen's assistant turn. The
versioned v9 prompt encoded correction as a new user turn and foregrounded
exact local tools/targets. On the prespecified four-episode retry it achieved
278/284 first-pass valid (97.89%), 284/284 after one repair, 19 accepted material
actions, and 16/19 reaching demand (84.21%). The failed attempt and the guarded
run-ID collision are retained.

### Development outcome effects

Lower loss is better. These intervals are exploratory development evidence.

| Comparison | Application | Relative loss change | Bootstrap 95% CI | Paired episodes |
|---|---|---:|---:|---:|
| fixed communication vs no communication | commercial | -48.45% | [-54.86%, -42.31%] | 15 |
| fixed communication vs no communication | humanitarian | -52.54% | [-60.72%, -44.90%] | 15 |
| KPI-triggered bounded operator vs autonomy | commercial | -1.59% | [-2.17%, -1.02%] | 20 |
| KPI-triggered bounded operator vs autonomy | humanitarian | -2.80% | [-4.37%, -1.40%] | 20 |

ThermoHITL-versus-KPI superiority, non-inferiority to always-on review, and a
confirmatory loss/effort Pareto frontier were not tested because Gate 5 failed.
In preliminary development only, ThermoHITL-rule averaged 27.3 versus 31.8
operator minutes commercially and 48.3 versus 51.9 humanitarian minutes for the
KPI trigger, but those unmatched exploratory values are not an operator-effort
claim.

### Same-information incremental value

The monitoring model was evaluated on held-out development environment seeds
12405 and 12406 with the same private-local KPI boundary and attention budget.

| Application | Δ average precision | Δ ROC AUC | Relative budgeted-utility gain | Gate |
|---|---:|---:|---:|---|
| Commercial | -0.0147 | -0.0066 | +2.86% | fail |
| Humanitarian | +0.0649 | +0.0249 | +211.84% | pass through utility/AP |

The commercial result fails the preregistered `>=0.05` AP/AUC or `>=5%`
utility criterion. Therefore cross-application thermodynamic information value
is unsupported even though the humanitarian result is strong on development.

### Trigger and counterfactual mechanism

Across 30 disrupted final-candidate episodes, the trigger activated before
sustained collapse in 30/30, missed 0/30, and had zero pre-disruption false
activation. Across 10 nominal episodes, false activation was 0/10. Disrupted
episodes averaged 7.33 requests, 3.97 completed interventions, and 38.2
simulated-operator minutes; this is nontrivial, not always-on use.

For ThermoHITL-rule development probes, the complete causal chain occurred 98
times commercially and 80 times humanitarianly. Five humanitarian probes were
harmful, and many alerts changed no primary outcome; these negative cases are
retained. KPI-triggered probes produced 36 commercial and 32 humanitarian
complete chains. Per-intervention effects are mechanistic secondary evidence;
episode-level intention-to-treat comparisons remain primary.

## Hypotheses

| Hypothesis | Outcome |
|---|---|
| H1 ThermoHITL beats same-information KPI trigger | not tested |
| H2 non-inferior to always-on review with less effort | not tested |
| H3 better loss/effort Pareto frontier | not tested |
| H4 timely activation with bounded false alarms | supported in development only |
| H5 complete positive causal intervention chain | supported in development only, with retained null/harmful cases |
| H6 thermodynamics adds value beyond KPIs | unsupported cross-application |
| H7 distributed robustness under partitions | not tested beyond abbreviated development diagnostics |
| H8 cross-application primary conclusion | unsupported/not tested |

## Compute and failures

- Manifest-accounted episode GPU time: 0.12277 single-GPU hours.
- Conservative model-initialization/guarded-launch estimate: 0.06667 hours.
- Total additional v3 estimate: **0.18943 single-GPU hours**.
- Approximate GPU cost at $0.34/hour: **$0.0644**.
- Real-Qwen calls: 589; prompt tokens: 1,084,097; generated tokens: 32,250.
- Validation/training/holdout GPU use: zero.

The retry logger initially lacked a nested directory; the complete first run was
retained. A later wrongly scoped stage environment variable caused the run-ID
collision guard to stop a launch before any episode. Neither attempt overwrote
data or became a selective scientific rerun. See `logs/diagnostics/`.

## Figures

All 19 evidence-bearing paper-facing files are vector PDFs. Poppler opened each PDF, detected
fonts, rendered a preview, and the original-resolution previews passed visual
inspection. The QA record is [`report.json`](reproducibility/pdf_qa/report.json).

1. [thermohitl_architecture.pdf](figures/pdf/thermohitl_architecture.pdf) — independent agents, distributed monitor, attention queue, bounded operator, and returned autonomy.
2. [operator_dashboard_overview.pdf](figures/pdf/operator_dashboard_overview.pdf) — data-populated vector replay of the functional information-limited dashboard; the underlying commercial and humanitarian SVG exports and replay metadata are in [`dashboard/populated_replays/`](dashboard/populated_replays/).
3. [energy_entropy_phase_plane.pdf](figures/pdf/energy_entropy_phase_plane.pdf) — nominal-standardized development trajectories, the prospective projected trigger boundary, and actual disruption/intervention points.
4. [network_operator_sequence.pdf](figures/pdf/network_operator_sequence.pdf) — quiet, disruption, request, intervention, and response network sequence.
5. [trigger_and_intervention_dynamics.pdf](figures/pdf/trigger_and_intervention_dynamics.pdf) — trigger features, queue, workload, autonomy, interventions, and loss.
6. [operator_view_incremental_value.pdf](figures/pdf/operator_view_incremental_value.pdf) — absolute utility, paired and relative differences, cluster-bootstrap intervals, panel counts, and regime results, explicitly limited to development evidence.
7. [loss_operator_effort_pareto.pdf](figures/pdf/loss_operator_effort_pareto.pdf) — exploratory development loss/effort points; not a confirmatory frontier.
8. [primary_effect_forest.pdf](figures/pdf/primary_effect_forest.pdf) — paired gate-3/gate-4 effects with bootstrap intervals.
9. [causal_intervention_effects.pdf](figures/pdf/causal_intervention_effects.pdf) — paired per-intervention effect distributions.
10. [intervention_funnel.pdf](figures/pdf/intervention_funnel.pdf) — separate autonomous-action progression and counterfactual-probe causal stages, with distinct denominators.
11. [operator_workload_performance.pdf](figures/pdf/operator_workload_performance.pdf) — normalized loss versus operator minutes and queue demand.
12. [attention_allocation_heatmap.pdf](figures/pdf/attention_allocation_heatmap.pdf) — normalized incident attention priorities over time.
13. [monitoring_incremental_value.pdf](figures/pdf/monitoring_incremental_value.pdf) — ranking and causal utility beyond local KPIs.
14. [trigger_timing_and_false_alarms.pdf](figures/pdf/trigger_timing_and_false_alarms.pdf) — timing, misses, and false activations.
15. [partition_robustness.pdf](figures/pdf/partition_robustness.pdf) — honest commercial-only aborted v1 partition diagnostic.
16. [commercial_case_study.pdf](figures/pdf/commercial_case_study.pdf) — one complete development commercial ledger case.
17. [humanitarian_case_study.pdf](figures/pdf/humanitarian_case_study.pdf) — one complete abstract humanitarian ledger case.
18. [thermodynamic_ablation.pdf](figures/pdf/thermodynamic_ablation.pdf) — entropy, energy, free-energy, disagreement, and combined development diagnostics.
19. [actionability_diagnostics.pdf](figures/pdf/actionability_diagnostics.pdf) — deterministic mechanics, retained Qwen v8 failure, and qualified v9 retry.

The explicit `PROSPECTIVELY NOT RUN` training-seed panel is retained as a
non-result in [`reproducibility/not_run_figures/`](reproducibility/not_run_figures/);
it is deliberately excluded from the publication figure set.

## Tables and statistics

`tables/` contains the experimental design, six gates, actionability funnel,
operator profiles/views, trigger parameters, communication accounting,
development paired effects, counterfactual chains, same-information monitoring,
compute, failures, and hypothesis outcomes. Required but ineligible analyses
(`main_paired_comparisons`, `noninferiority_analysis`,
`communication_reductions`, `pareto_operating_points`, `ablation_results`, RL
seeds, and holdout) contain explicit one-row `not_run` records.

`statistics/` contains the 10,000-replicate bootstrap outputs, actionability and
counterfactual summaries, and the analysis manifest. `raw/` contains one
immutable episode record and compressed event ledger per run. `processed/`
contains analysis-ready development rows. `manifests/` binds source, seeds,
model/prompt, tokens/calls, wall time, checksums, and completion status.

## Directory map

- `protocol/`: development protocol plus future IRB-dependent human-study templates.
- `diagnostics/`, `development/`, `monitoring/`: antecedent and gate evidence.
- `operator_models/`, `dashboard/`: simulated-operator and interface records.
- `validation/`, `training/`, `checkpoints/`, `holdout_locked/`: explicit fail-closed stop records.
- `counterfactuals/`, `raw/`, `processed/`, `statistics/`, `tables/`: evidence and analysis.
- `figures/pdf/`, `figures/previews/`: vector figures and rendered previews.
- `logs/`, `manifests/`, `reproducibility/`: execution, replay, environment, checksums, and QA.
- [`INDEX.csv`](INDEX.csv): size and SHA-256 catalog for every v3 artifact.

## Reproduction

From a fresh clone, use the compatible environment described in
`requirements-runpod.txt`. Replay, analysis, figures, PDF QA, and dashboard
replay do not require a GPU.

```bash
./scripts/run-human-operator-tests.sh
./scripts/run-human-operator-diagnostics.sh
./scripts/run-human-operator-development.sh
./scripts/run-human-operator-monitoring.sh
./scripts/replay-human-operator-results.sh
./scripts/analyze-human-operator-results.sh
./scripts/generate-human-operator-figures.sh
./scripts/validate-human-operator-pdfs.sh
./scripts/run-human-operator-dashboard.sh --episode \
  results/human_operator_v3/raw/development_trigger_candidate_n10_v4/<run-id>/episode.json
```

The complete fail-closed rebuild is:

```bash
./scripts/rebuild-human-operator-results.sh
```

The following commands intentionally fail while Gate 5 remains failed:

```bash
./scripts/train-human-operator-multiseed.sh
./scripts/run-human-operator-validation.sh
./scripts/design-human-operator-holdout.sh
./scripts/freeze-human-operator-holdout.sh
./scripts/run-human-operator-holdout.sh
```

The exact remote qualification workflow and filtered synchronization commands
are in `notes/33_thermohitl_reproduction_and_compute.md`.

## Limitations and readiness

The environment is synthetic; operators are simulated; mechanism development
used deterministic planners extensively; the real-Qwen qualification has only
four episodes per attempt; no learned v3 policy or independent RL seed exists;
no validation/holdout was opened; the partition study was not completed; and
commercial thermodynamic incremental value failed. Development mechanics were
purposefully made coordination- and intervention-actionable, so their large
coordination effect is a gate result, not an external-validity estimate.

The evidence is adequate for a reproducible engineering demonstration and a
useful negative/boundary report. It is **not sufficient for the intended
positive AIJ submission**. A future versioned study would need to improve
commercial same-information decision value without tuning on a new holdout,
re-pass all gates, train at least five independent seeds, and complete fresh
validation and sealed holdout evaluation. A real-human claim additionally
requires separate institutional approval and participant evidence.
