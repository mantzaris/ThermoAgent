# Distributed Operational Entropy Triggering (DOET)

Status: locked v2 analysis `complete`; 0 failed holdout episodes. Journal-readiness classification: **insufficient for the intended AIJ submission**.

This directory is a separate result namespace for the second ThermoAgent study. The frozen v1 data and interpretation remain unchanged in the parent results tree.

## Research question and method

Can a privacy-preserving distributed operational-entropy estimate tell genuinely independent logistics agents when to leave quiet local operation, start targeted bilateral coordination, or activate bounded crisis-coalition coordination—while preserving always-on fixed communication performance with less communication and inference?

DOET maintains a separate state machine inside every agent. Its private state records that agent's gossiped entropy estimate and recent change, nominal center and variance, local surprisal, consensus confidence, communication availability, and time since its own last intensive coordination event. The selected trigger statistic uses only the locally calibrated entropy residual, surprisal, consensus confidence, and explicitly delivered bounded neighbor alerts; the retained auxiliary fields support audit and candidate ablations. It never receives evaluator-global entropy or the true disruption label. Low-cost entropy sketches are explicitly counted. Recipients independently decide whether to escalate. The trigger changes communication eligibility; it does not accept contracts or make domain decisions for an agent.

Each organization retains its own identity, private observations/memory/utility, planning context, inbox/outbox, trust state, commitment ledger, policy recurrent state, and accept/reject/counter/withdraw authority. The quantitative simulator validates typed tools and conservation but cannot invent a domain action.

## Frozen original-study evidence

V1 remains a mixed/negative result: 101 tests passed; 1,096 post-freeze ledgers replayed exactly; operational entropy monitored disruption well (AP 0.934, ROC AUC 0.863), the free-energy gap did not; small in-distribution ThermoAgent improvements did not survive correction; the 80-episode holdout tied the matched no-entropy policy exactly; and fixed communication won every necessity-map cell.

The v2 tie diagnosis found that the raw IEEE-754 primary outcomes—not rounded displays—were identical in all 16 ThermoAgent/no-entropy pairs even though option choices diverged on 58.2% of common commercial epochs and 46.2% of humanitarian epochs. Fifty-four of 57 learned-policy material actions failed; the three successful ThermoAgent shipments reached only intermediate nodes and none reached demand. The policies therefore communicated differently without changing demand-reaching material flow. Entropy features varied, were not masked, and changed only 2.69%/3.49% of actor actions when zeroed. V1 had one RL seed (3001).

Monitoring controls show that full/global ordinary KPIs already classify disruption perfectly in these synthetic trajectories, so entropy has no incremental global predictive value. Under the execution-realistic restriction to one agent's private local KPIs, adding distributed entropy improved development AP by about 0.097 in both applications and AUC by about 0.17. The seen v1 holdout retained calibration gains but not a robust ranking gain. The defensible contribution is therefore a compressed distributed trigger under restricted information—not a claim that entropy universally beats centralized KPIs.

## Hardware, model, and design

- GPU: existing RunPod RTX 4090, 24 GB; CUDA 12.8; no additional Pod.
- Planner: `Qwen/Qwen2.5-7B-Instruct` at immutable revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Precision: bitsandbytes NF4, bfloat16 compute, double quantization.
- Prompt revision: `planner-json-v7-route-affordances`; deterministic decoding; max 2560 input and 160 generated tokens.
- Selected trigger: `hysteresis_low` (`hysteresis`, `low` residual, `rho=0.6`, `kappa=0.0`, `tau_on=1.2`, `tau_off=0.4`, crisis threshold `2.8`, dwell `2`, cooldown `2`, propagation `neighbor`).
- Learned replications: 5 independent RL seeds per learned method; 15/15 fixed-budget trainings completed; no outcome-selected checkpoints.
- New holdout: 144 base matched panels and 696 method episodes on unseen `tri_region_bridge_v2`; four compute-priority methods use all panels, while secondary comparators use the same preregistered non-nominal subset. There are 16 primary-method seeds per application in each isolated, partition, correlated, and compound-OOD regime plus 8 nominal seeds per application.
- Primary benchmark: `fixed_always_on`; 2% relative non-inferiority margin.
- Primary unit: one complete multi-agent episode. Analysis uses paired panels, 10,000 hierarchical bootstrap replicates, explicit RL-seed variation, one-sided non-inferiority bounds, and Holm correction for H1/H2 across applications.

## Locked-holdout primary findings

| Application | Loss degradation vs fixed | One-sided 95% upper | Non-inferior | Message reduction (95% CI) |
|---|---:|---:|:---:|---:|
| Commercial | 1.00% | 1.56% | yes | 72.4% [71.0%, 73.7%] |
| Humanitarian | 0.38% | 0.58% | yes | 74.2% [73.0%, 75.4%] |

Positive reductions mean DOET used less than fixed communication. Every message total includes operational packets, alerts, and entropy-sketch gossip.

| Application | Structured bytes | Prompt tokens | Generated tokens | LLM calls | Inference latency | Wall-clock time |
|---|---:|---:|---:|---:|---:|---:|
| Commercial | 68.6% | 37.8% | 41.5% | 36.8% | 38.8% | 38.8% |
| Humanitarian | 70.6% | 41.9% | 42.5% | 38.5% | 39.9% | 39.9% |

## Critical mechanistic result

- `doet_rule`: 0/144 episodes activated; 0 total activations; mean quiet-mode fraction 1.000; maximum observed trigger residual 0.618 versus `tau_on=1.200`.
- `doet_rl`: 0/144 episodes activated; 0 total activations; mean quiet-mode fraction 1.000; maximum observed trigger residual 0.618 versus `tau_on=1.200`.

The selected entropy trigger never activated in the locked holdout. This repeats the preregistered validation warning: all four entropy candidates also had zero activations there, but the frozen selector lacked a minimum-activation eligibility gate and chose among them on non-inferiority and communication. The formal H1/H2 endpoint results therefore show that the frozen sparse quiet-mode schedule was close to fixed communication while using less communication; they do **not** show that operational entropy successfully triggered timely coordination. This is the central negative finding and prevents the intended causal event-trigger contribution.

### Exploratory signal and oracle controls

- All 60 local DOET-variant episodes and all 12 exact-global-entropy oracle episodes had zero activations.
- The private-KPI control activated in 12/12 episodes and changed mean loss by -0.023% while using 72.1% more messages than selected DOET.
- The putative disruption-label oracle activated in 12/12 episodes and changed mean loss by -0.300% while using 40.0% more messages than selected DOET. Ledger timing shows both active controls first fired at period 0, eight periods before disruption; these are false activations, not timely alarms. The binary-label oracle inherited the selected low-direction transform, so label 0 was treated as anomalously low; it is retained as an invalid exploratory oracle implementation, not an upper bound.

Did DOET satisfy the complete preregistered Pareto criterion? **no**. Normalized hypervolume increased in all frozen comparator/cost cells (**yes**), but H3 also required loss-message nondominance. No communication dominated DOET-rule on loss and messages in both applications (**yes**), so a hypervolume-only positive claim is not permitted.

## Preregistered hypotheses

- `H1` — **supported**. Frozen success criterion: DOET-rule non-inferior to fixed in both applications after Holm correction
- `H2` — **supported**. Frozen success criterion: message reduction CI excludes zero and mean reduction >=20% in both applications after Holm correction
- `H3` — **unsupported**. Frozen success criterion: DOET-rule is loss-message nondominated and strictly increases the frozen normalized frontier hypervolume for messages, prompt tokens, calls, and latency in both applications
- `H4` — **unsupported**. Frozen success criterion: >=75% first post-disruption activation before sustained severe collapse (service loss >=0.90 for three consecutive periods), severe collapse observed in every non-nominal episode, <=10% pre-disruption false activation, and <=10% nominal episode false activation
- `H5` — **unsupported**. Frozen success criterion: non-inferior in partition and compound-partition regimes in both applications, with positive consensus-RMSE/degradation slope and Pearson r >=0.20 in each application
- `H6` — **supported**. Frozen success criterion: H1 and H2 supported in both applications


Supported: H1, H2, H6. Unsupported: H3, H4, H5.

Negative, mixed, and failed findings are retained in `tables/hypothesis_outcomes.csv`, `tables/failed_runs.csv`, the validation candidate table, and the mechanistic outputs. No claim of literal thermodynamics, realistic humanitarian behavior, or autonomous-agent necessity is made unless the corresponding evidence supports it.

## Compute and communication accounting

- Holdout episodes: 696 (0 failed).
- Summed episode wall time: 14.875 hours.
- LLM calls: 56,653; prompt tokens: 100,908,718; generated tokens: 3,745,964.
- All counted messages: 78,759; structured bytes: 23,475,368.
- Approximate GPU cost at $0.34/hour: $5.06. One-time model load and non-GPU local diagnostics are reported separately in manifests.
- Total additional model-smoke/profile/validation/training/holdout/authorized-ablation Pod time including model loads: 22.062 single-GPU hours; approximate cost $7.50. CPU-bound staged PPO time on the reserved Pod is included. This is the value audited against the 35-hour cap.

## Artifacts

- `protocol/`: selected trigger, power/precision analysis, matched design, and immutable holdout freeze.
- `diagnostics/`: exact v1 tie mechanism and action/communication divergence.
- `monitoring/`: entropy-versus-KPI detectors, incremental value, lead time, and localization.
- `training/` and `checkpoints/`: all RL seeds, curves, fixed-budget selection, and small policy checkpoints.
- `validation/`: all trigger candidates, selected operating point, and budget-matched control rates.
- `holdout_locked/` and `raw/holdout_locked/`: episode summaries and event-sourced ledgers.
- `processed/`, `statistics/`, and `tables/`: paired analysis, bootstrap output, Pareto frontiers, mechanisms, failures, and CSV/LaTeX tables.
- `figures/pdf/` and `figures/previews/`: vector paper figures and rendered previews; `reproducibility/pdf_qa/` contains mechanical and visual QA.
- `logs/` and `manifests/`: restartable run status, exact model/config/seeds/tokens/checksums, and failure records.

### Figure guide

- [`doet_architecture.pdf`](figures/pdf/doet_architecture.pdf): decentralized sketches, local trigger state, three communication modes, and retained agent authority.
- [`original_holdout_tie_diagnostics.pdf`](figures/pdf/original_holdout_tie_diagnostics.pdf): v1 action and communication divergence despite exact service-outcome ties.
- [`monitoring_baseline_comparison.pdf`](figures/pdf/monitoring_baseline_comparison.pdf): entropy detectors against thresholds, rolling statistics, change detectors, and multivariate KPI models.
- [`entropy_incremental_value.pdf`](figures/pdf/entropy_incremental_value.pdf): incremental entropy value under global and private-local observability.
- [`trigger_dynamics.pdf`](figures/pdf/trigger_dynamics.pdf): aligned entropy, trigger statistic, active agents, messages, and service loss; it explicitly displays the absent activation.
- [`performance_communication_pareto.pdf`](figures/pdf/performance_communication_pareto.pdf): common-panel loss-message frontier, including no communication and strong controls.
- [`noninferiority_forest.pdf`](figures/pdf/noninferiority_forest.pdf): paired DOET-rule degradation intervals against the frozen 2% margin by application and regime.
- [`communication_reduction.pdf`](figures/pdf/communication_reduction.pdf): fully counted reductions in messages, bytes, tokens, calls, latency, and wall time.
- [`multiple_seed_learning_curves.pdf`](figures/pdf/multiple_seed_learning_curves.pdf): all five independent seeds for each learned method.
- [`training_seed_variability.pdf`](figures/pdf/training_seed_variability.pdf): checkpoint-level outcome and communication variability in locked evaluation.
- [`holdout_primary_results.pdf`](figures/pdf/holdout_primary_results.pdf): common matched-panel primary outcomes with seed points.
- [`partition_robustness.pdf`](figures/pdf/partition_robustness.pdf): consensus error, loss degradation, and the zero-activation result under partitions.
- [`trigger_ablation_effects.pdf`](figures/pdf/trigger_ablation_effects.pdf): validation candidates plus prospectively specified exploratory signal/oracle controls.
- [`commercial_event_case_study.pdf`](figures/pdf/commercial_event_case_study.pdf) and [`humanitarian_event_case_study.pdf`](figures/pdf/humanitarian_event_case_study.pdf): disruption-aligned episode sequences with the absent trigger visibly annotated.
- [`network_snapshots_entropy_trigger.pdf`](figures/pdf/network_snapshots_entropy_trigger.pdf): deterministic physical/communication network snapshots showing that quiet mode persisted rather than implying an unobserved escalation.

### Table guide

- `experimental_design.csv`: episode counts, matched seeds, systems, regimes, and methods; the `.tex` companion is publication-ready.
- `rl_training_seed_results.csv`: evaluation variability for all independent learned checkpoints; `training/seed_manifest.csv` records selection and hashes.
- `trigger_parameters.csv`, `communication_budgets.csv`, and `achieved_budget_match.csv`: frozen trigger and communication-control settings plus realized matching error.
- `monitoring_comparison.csv`: detector performance; the richer source tables under `monitoring/` retain prevalence, timing, localization, and incremental-value analyses.
- `main_paired_comparisons.csv`, `noninferiority_analysis.csv`, and `communication_reductions.csv`: paired effects, hierarchical intervals, formal margins, and fully counted savings.
- `pareto_operating_points.csv`: common-panel loss/cost points and dominance flags.
- `holdout_results.csv`: application/scenario/method summaries; episode-level rows remain under `holdout_locked/` and `processed/`.
- `mechanistic_summary.csv` and `rl_option_selection.csv`: trigger/collapse/mode behavior and learned option distributions.
- `trigger_ablation_results.csv` and `extended_ablation_results.csv`: validation candidates and post-holdout exploratory signal/oracle controls.
- `compute_token_accounting.*` and `total_compute_accounting.*`: holdout-only and complete additional-resource accounting.
- `failed_runs.csv`: all locked failures (empty because all 696 completed); training attempts and any retained setup failures remain in their stage logs.
- `hypothesis_outcomes.csv`: frozen H1--H6 decisions; the `.tex` companion is publication-ready.

## Reproduction commands

```bash
./scripts/run-entropy-trigger-diagnostics.sh
./scripts/run-monitoring-validation-v2.sh
./scripts/run-doet-calibration.sh
./scripts/run-doet-profile.sh
./scripts/run-doet-validation.sh
./scripts/train-doet-multiseed.sh
./scripts/design-doet-holdout.sh
./scripts/freeze-doet-holdout.sh
./scripts/run-doet-holdout.sh
./scripts/run-doet-ablations.sh  # exploratory, after the locked holdout
./scripts/rebuild-doet-results.sh
```

The locked episodes were executed from commit `09ac91b72dd7fb5151fc6af2c28da9855653b2dc` with source checksum `655cb19264b51a33b47273c28c990f07eb85a0f9caa54da2b8ab4d96509e06c9`; the authoritative values are also stored in `reproducibility/execution_source.json` and every run manifest. For filtered RunPod deployment, use `./scripts/runpod-sync.sh`, then `./scripts/runpod-sync-v2-controls.sh bootstrap`. Fetch only this study with `./scripts/runpod-fetch-v2-results.sh`; the command never overwrites the frozen v1 namespace. Exact sequencing and restart instructions are in `notes/14_entropy_trigger_protocol.md` and `notes/15_entropy_trigger_implementation.md`.

## Limitations and readiness

These are abstract logistics simulators, one 7B model family, deterministic decoding, a small discrete coordination policy, synthetic disruption processes, and a single 4090 execution environment. Full/global KPI detectors can dominate entropy when centralized observability is available. Balanced learned-checkpoint evaluation exposes five training seeds but does not cross every checkpoint with every panel. Communication cost uses measured messages/bytes/tokens/calls/latency and a transparent hourly-rate estimate, not a deployment-specific network tariff.

Current classification: **insufficient for the intended AIJ submission**. The completed platform and boundary result are suitable as an engineering demonstration, but the intended entropy-triggered positive contribution is not presently sufficient for an AIJ submission. See `PAPER_SUMMARY.md` and `notes/19_entropy_trigger_paper_claims.md` for the exact allowed claims and remaining work.
