# ThermoHITL V5 results

## Status and research question

V5 is a completed **development-stage no-go study**. It asks whether
privacy-preserving distributed belief entropy and inter-agent disagreement
improve allocation of a scarce simulated operator's attention beyond ordinary
local KPIs when genuinely independent autonomous agents hold fragmented
information.

The answer from V5 development is **no**. Coordination and bounded operator
intervention are causally useful in the designed environments, but the frozen
KPI-plus-entropy/disagreement triage policy did not beat KPI-only triage in
either primary application. Several required gates failed, so validation and
the sealed holdout were never run. There were no human participants.

V5 builds from the immutable V4 snapshot `8ccd27df248940fc0cbb55c43a30949de3370533`
through the non-semantic V4 maintenance commit
`d39eb2eefefa54259a2bafc6dcd6e9b0dbde2ffe`. Formal V5 development used branch
`thermodynamic-human-oversight-v5`, execution commit
`9e38e78033c34c543f33d0ee41613e56d68a73f8`, source checksum
`b24b01b1e4d20d9e96e842f6edadbffdc9956a7c901be8d15ca52238744cc026`, and
protocol checksum
`687aee0ebde467c3d5b5919906345a570abb9454b3fdbbd33c9b2a965d575770`.
V1-V4 artifacts and conclusions remain unchanged.

## Applications and agents

Each competitive panel contains four simultaneous incidents and 12
independent agents. Commercial logistics is a prespecified KPI-sufficient
boundary application; humanitarian logistics and abstract cyber-physical
utility restoration are the two primary applications. Utility cyber events
are offline defensive simulator state transitions—there is no real-system
access, network probing, exploit code, credential use, or operational claim.

Every agent has a separate identity, private observation, belief, memory,
utility, inbox, commitment ledger, typed action authority, and decision loop.
Messages and distributed sketches are explicit ledger events. Partitions block
delivery. The simulator validates actions and conserves resources but does not
silently replace agent decisions with an oracle. Deterministic independent
planners are engineering controls; real Qwen and decentralized RL evidence are
reported separately.

## State and oversight mechanism

For agent (i), local belief entropy is

`H_i(t) = -sum_s p_i(s,t) log p_i(s,t)`.

V5 also records mean and dispersion of local entropy, bounded Jensen–Shannon
disagreement, a gossip-derived entropy estimate, slope, consensus residual,
and consensus confidence. Operational energy is a normalized aggregate of
visible service stress, backlog, delay, resource scarcity, safety risk, and
commitment strain. Energy and `F = E - T_eff H` are secondary diagnostics, not
literal physical thermodynamics.

The simulated operator sees only a hashed authorized payload, can select at
most two incidents, incurs intervention time/cost, and may abstain under low
consensus. Candidate actions include imperfect delayed verification, peer
evidence, emergency resources, rerouting/reconfiguration, repair capacity,
isolation/quarantine, commitment revision, defer, and no action. Wrong choices
can be neutral or harmful. Matched counterfactual branches restore an identical
stored stochastic tape.

## Design and evidence inventory

- Four retained pilot iterations: 672 episodes; the first three document
  unsuccessful development mechanics and the fourth passed engineering
  evaluability.
- One invalidated concurrent-launch namespace: 840 ledgers retained; no row is
  inferentially eligible.
- Valid formal development: 840 independent panels and 30,240 candidate
  interventions, covering 20 seeds, three applications, seven regimes, and two
  information conditions.
- Sketch ablation: 768 episodes across none, periodic, event-triggered, and
  always-on exchange.
- Real Qwen qualification: 36 episodes and 108 agent decision epochs.
- Decentralized RL: two methods × five independent seeds × 30,000 planned
  decision epochs; 302,080 sampled training and 21,600 evaluation decisions.
- Exact replay: 3,156 event ledgers, zero mismatches, maximum conservation
  residual 0.0.
- Validation: 0 episodes, prospectively locked.
- Sealed holdout: 0 episodes, never opened.

The primary independent unit is an environment panel, not a candidate action,
agent, message, or time step. The analysis uses grouped nested cross-fitting,
paired panel contrasts, 10,000-replicate cluster bootstraps, and a
full-refit, within-stratum permutation falsification analysis. The first two
permutation outputs used an inconsistent fit boundary; they are preserved as
superseded, and the correction is documented in
[`protocol/development_analysis_correction_addendum.md`](protocol/development_analysis_correction_addendum.md).

## Main development findings

### Coordination and operator action are consequential

Fixed decentralized communication reduced loss relative to no communication:

| Application | Absolute reduction | Relative reduction | 90% cluster CI | Panels changed |
|---|---:|---:|---:|---:|
| Commercial | 0.2732 | 7.72% | [0.2141, 0.3331] | 63.3% |
| Humanitarian | 0.2261 | 6.48% | [0.1706, 0.2848] | 54.2% |
| Utility restoration | 0.4503 | 12.17% | [0.3896, 0.5117] | 69.2% |

The bounded simulated-operator oracle reduced loss by 13.24%, 12.92%, and
12.57%, respectively, with positive 95% intervals and about 14.2-14.4 operator
minutes per panel. Complete alert-to-service chains occurred in 78.3%-82.9% of
panels. These establish actionability, not human effectiveness.

### Primary incremental-value endpoint fails

At the matched two-incident budget, thermodynamic triage minus KPI-only utility
was:

| Application | KPI utility | Thermodynamic utility | Paired gain | 95% cluster CI |
|---|---:|---:|---:|---:|
| Commercial boundary | 0.0426 | 0.0275 | -0.0151 | [-0.0489, 0.0174] |
| Humanitarian | 0.0074 | -0.0042 | -0.0116 | [-0.0279, 0.0043] |
| Utility restoration | 0.0260 | 0.0161 | -0.0100 | [-0.0277, 0.0068] |

The thermodynamic policy selected harmful interventions more often in both
primary applications. Single entropy/disagreement variables were not shortcut
labels (AUC about 0.51-0.54; common support 0.83-0.85), but the combined block
still provided no incremental decision utility.

The private-minus-public interactions were +0.0136 [−0.0047, 0.0321] in
humanitarian logistics and +0.0055 [−0.0157, 0.0269] in utility restoration.
They do not support the fragmented-observability mechanism.

### Safety, triggering, and communication

Low-consensus abstention reduced harmful interventions from 35 to 19 in
humanitarian panels and from 29 to 9 in utility panels, improving utility in
both. This supports a narrow safety boundary independent of the failed primary
triage claim.

Trigger feasibility failed: nominal false activation was 30.0%-32.5%, and
utility restoration had 74.2% timely activation and 25.8% misses. Event-driven
sketching reduced private-condition bytes by 61.6%-63.5% versus always-on
sketching, but thermodynamic monitoring still fails cost-adjusted usefulness
because its causal gain is negative. Formal development counted 19,545 sketch
messages (3,521,543 bytes) in addition to 10,080 operational messages
(1,218,510 bytes).

## Real-Qwen and RL evidence

The real model was `Qwen/Qwen2.5-7B-Instruct`, immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`, Transformers 4.55.4,
bitsandbytes NF4, BF16 computation, and PyTorch 2.8.0+cu128 on one RTX 4090.
All 108 outputs were valid on the first attempt; material acceptance was
83.3%-86.1%. Commercial and humanitarian service-reaching rates (22.2% and
38.9%) missed the 45% gate, and utility agents had zero private-evidence action
divergence. Gate 2 therefore failed.

All ten decentralized PPO-style runs completed without seed removal, but both
methods collapsed to `no_action`: mean reward 0, action diversity one, and
entropy-policy gain 0 for every seed. That reproducible degeneracy fails Gate
10 and is not presented as successful learning.

## Prospective gates

| Gate | Result | Consequence |
|---|---|---|
| 1. Engineering integrity | Pass | 238 tests; exact replay and conservation |
| 2. Autonomous-agent validity | **Fail** | Qwen thresholds not met |
| 3. Coordination necessity | Pass | Both primary applications |
| 4. Human causal usefulness | Pass | Simulated bounded operator only |
| 5. Thermodynamic incremental value | **Fail** | Negative primary estimates |
| 6. Trigger/triage feasibility | **Fail** | False activations and misses |
| 7. Mechanism specificity | **Fail** | Interaction intervals cross zero |
| 8. Safety/abstention | Pass | Harm reduction in both primary applications |
| 9. Communication-cost feasibility | **Fail** | No positive causal value after sketch cost |
| 10. Multi-seed stability | **Fail** | Stable single-action collapse |

Because all ten gates were required, validation remained locked and the sealed
holdout was not run.

## Compute accounting

Real-Qwen qualification made 108 calls, used 104,836 prompt and 6,116 generated
tokens, and took 109.34 seconds including model load: 0.0304 single-GPU hours,
approximately USD 0.010 at the project's illustrative USD 0.34/hour. RL used
CPU (898.7 summed seconds). The V5 execution remained far below the 50-hour and
USD 40 caps. No background job is required to inspect or replay results.

## Figures

All paper figures are true vector PDFs under [`figures/pdf/`](figures/pdf/),
with 240-DPI PNGs under [`figures/png/`](figures/png/) and mechanical/visual QA
under [`reproducibility/pdf_qa/`](reproducibility/pdf_qa/).

1. [`v5_architecture.pdf`](figures/pdf/v5_architecture.pdf): independence,
   sketches, budgeted oversight, causal branch, and the prospective stop.
2. [`three_application_overview.pdf`](figures/pdf/three_application_overview.pdf):
   application roles, outcomes, and boundary status.
3. [`utility_restoration_multilayer_network.pdf`](figures/pdf/utility_restoration_multilayer_network.pdf):
   abstract service, logistics, and communication layers.
4. [`distributed_entropy_communication_network.pdf`](figures/pdf/distributed_entropy_communication_network.pdf):
   node entropy, disagreement, and confidence over ad-hoc links.
5. [`energy_entropy_disagreement_phase_plane.pdf`](figures/pdf/energy_entropy_disagreement_phase_plane.pdf):
   actual development coordinates and cross-fitted selections.
6. [`populated_operator_dashboard.pdf`](figures/pdf/populated_operator_dashboard.pdf):
   functional utility replay export; three application exports are in
   `dashboard_exports/`.
7. [`operator_dashboard_kpi_vs_entropy.pdf`](figures/pdf/operator_dashboard_kpi_vs_entropy.pdf):
   matched operator payloads.
8. [`causal_alert_to_outcome_funnels.pdf`](figures/pdf/causal_alert_to_outcome_funnels.pdf):
   separate autonomous-action and intervention-probe populations.
9. [`feature_block_incremental_value.pdf`](figures/pdf/feature_block_incremental_value.pdf):
   causal utility for seven prespecified feature blocks.
10. [`paired_cluster_effect_forest.pdf`](figures/pdf/paired_cluster_effect_forest.pdf):
    primary negative effects and intervals.
11. [`fragmented_vs_public_interaction.pdf`](figures/pdf/fragmented_vs_public_interaction.pdf):
    unsupported mechanism interaction.
12. [`communication_cost_service_loss_pareto.pdf`](figures/pdf/communication_cost_service_loss_pareto.pdf):
    loss versus bytes including sketches.
13. [`operator_effort_service_loss_pareto.pdf`](figures/pdf/operator_effort_service_loss_pareto.pdf):
    loss versus simulated operator minutes.
14. [`intervention_harm_benefit_distribution.pdf`](figures/pdf/intervention_harm_benefit_distribution.pdf):
    beneficial, neutral, and harmful counterfactual effects.
15. [`low_consensus_abstention.pdf`](figures/pdf/low_consensus_abstention.pdf):
    forced selection versus safe abstention.
16. [`multiseed_rl_learning_curves.pdf`](figures/pdf/multiseed_rl_learning_curves.pdf):
    every RL seed, including collapse.
17. [`multiseed_policy_evaluation.pdf`](figures/pdf/multiseed_policy_evaluation.pdf):
    seed-level evaluation rewards.
18. [`calibration_reliability.pdf`](figures/pdf/calibration_reliability.pdf):
    cross-fitted action-value reliability.
19. [`competitive_panel_selection_comparison.pdf`](figures/pdf/competitive_panel_selection_comparison.pdf):
    win/tie/loss fractions across 120 disrupted panels per application.
20. [`regime_specific_heterogeneity.pdf`](figures/pdf/regime_specific_heterogeneity.pdf):
    no hidden regime rescue.
21. [`trigger_timing_and_false_alarms.pdf`](figures/pdf/trigger_timing_and_false_alarms.pdf):
    timely, missed, and nominal false activation.

## Tables and provenance

[`tables/`](tables/) contains the experimental design, gate outcomes,
actionability, primary paired effects, coordination, operator causal effects,
fragmentation interaction, abstention, trigger burden, sketch accounting,
counterfactual interventions, RL seeds, hypotheses, and compute accounting.
[`INDEX.csv`](INDEX.csv) inventories every artifact with size and SHA-256.
Raw ledgers are under `raw/`; manifests, failed-run registry, exclusion ledger,
protocol deviations, exact replay, source checksums, and environment metadata
are under `manifests/`, `negative_results/`, and `reproducibility/`.

## Reproduction

```bash
./scripts/run-human-operator-v5-tests.sh
./scripts/run-human-operator-v5-development.sh
./scripts/run-human-operator-v5-sketch-ablation.sh
./scripts/analyze-human-operator-v5-results.sh
./scripts/run-human-operator-v5-real-qwen.sh       # RTX 4090 / cached model
./scripts/run-human-operator-v5-training.sh
./scripts/replay-human-operator-v5-results.sh
./scripts/evaluate-human-operator-v5-gates.sh
./scripts/generate-human-operator-v5-figures.sh
./scripts/validate-human-operator-v5-pdfs.sh
./scripts/build-human-operator-v5-report.sh
./scripts/verify-human-operator-v5-artifacts.sh
```

Replay and dashboard operation require no GPU. Launch a populated replay with
`./scripts/run-human-operator-v5-dashboard.sh --episode <episode.json>`.
Validation and holdout commands exit nonzero with an explicit gate-lock message.

## Limitations and readiness

The primary causal evidence uses simulated environments, deterministic
independent-agent controls, and simulated operators. Real Qwen evidence is a
small qualification set, not formal outcome evaluation. The learned policies
collapsed. No real operator, utility, humanitarian organization, cyber system,
or field deployment was studied. The environments remain abstractions, and
the refit-permutation correction is post-development secondary analysis.

The repository supports an engineering demonstration and a transparent
development/boundary workshop report. It does **not** support a confirmatory
journal manuscript or an Artificial Intelligence submission centered on the
proposed positive claim. A future study needs a redesigned learnable action
policy, lower false activation, successful autonomous qualification, positive
prospective validation, and a truly sealed holdout.
