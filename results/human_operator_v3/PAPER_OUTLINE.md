# AIJ-oriented paper outline (24–27 pages plus supplement)

Working title: **ThermoHITL: Mechanism-Gated Thermodynamic Triage for
Human-on-the-Loop Oversight of Decentralized Logistics Agents**

Current disposition: outline only. V3 stopped prospectively before validation
and holdout because the cross-application thermodynamic-information gate failed.
The outline marks the evidence that exists and the confirmatory evidence that
would be required; it is not a disguised full manuscript.

## Main paper

### 1. Introduction (2 pages)

- Scarce supervisory attention in decentralized, privately informed logistics.
- Why monitoring accuracy is insufficient without an observable causal path.
- The v2 negative result: inactive entropy trigger and absent communication need.
- Research question and mechanism-gated philosophy.
- Contributions that can currently be claimed: architecture, audited information
  boundary, actionable v3 simulator, counterfactual machinery, and a negative
  cross-domain information-value boundary. Do not claim confirmatory control value.

Evidence: `development_gates.csv`, `actionability_diagnostics.csv`,
`thermohitl_architecture.pdf`, v2 result README.

### 2. Related work (2–2.5 pages)

- Adjustable autonomy and mixed-initiative systems.
- Human-on/in/over-the-loop distinctions.
- Event-triggered and attention-constrained control.
- Multi-agent LLM autonomy and private-state separation.
- Operational anomaly detection and statistical-mechanics-inspired summaries.
- Counterfactual simulation and causal-mechanism audits.

No empirical novelty claim belongs in this section.

### 3. Problem formulation (2 pages)

- Agents, private observations, utilities, memories, commitments, and typed tools.
- Dynamic physical and communication graphs.
- Primary losses: commercial service-loss AUC and humanitarian weighted unmet need.
- Finite operator slots, queue, workload, service time, latency, accuracy, fatigue.
- Attention-allocation objective and loss per operator minute.
- Execution-time information sigma-algebras for each view condition.

Evidence: operator-view schemas and tests; `human_trial_schema.json` is future-only.

### 4. ThermoHITL architecture (2.5 pages)

- Independent decision loops and absence of a hidden central domain planner.
- Local operational energy and component weights.
- Flow entropy, belief entropy, Jensen–Shannon disagreement, nominal residuals.
- Link-respecting bounded gossip and consensus confidence.
- Diagnostic free energy and nonphysical interpretation.
- Independent escalation, capacity-constrained queue, bounded intervention, and
  returned authority.
- Dashboard as an experimental information channel, not post-hoc decoration.

Figures: `thermohitl_architecture.pdf`, `operator_dashboard_overview.pdf`,
`energy_entropy_phase_plane.pdf`.

### 5. Applications and v3 action mechanics (2 pages)

- Commercial roles, authority constraints, disruption regimes, emergency tools.
- Abstract humanitarian roles, priority conflicts, access/resource tools.
- Typed validation, one bounded repair, lead times, transit, arrival, and demand.
- Conservation and exogenous emergency-resource accounting.
- Development-only changes relative to frozen v1/v2.

Evidence: `actionability_diagnostics.csv`, action/tool events, conservation tests,
`actionability_diagnostics.pdf`.

### 6. Simulated operator and dashboard (2 pages)

- Explicit simulated-operator label and real-human boundary.
- Six bounded profiles and two privileged oracle references.
- Queueing, workload recovery, fatigue/accuracy degradation, intervention cost.
- KPI, entropy, energy, combined, disagreement, and oracle view schemas.
- Payload validation, hashing, provenance, replay, and privacy enforcement.

Figures: `operator_dashboard_overview.pdf`, `operator_workload_performance.pdf`.
Evidence: operator action/view hashes and dashboard schema tests.

### 7. Prospective mechanism gates and experimental protocol (2.5 pages)

- Six gates and thresholds fixed before expensive stages.
- Development/validation/holdout separation and fail-closed controls.
- Qwen qualification, repair limit, and retained infrastructure retry.
- Intended five-seed learned-policy design, prospective holdout, outcome seal.
- Why validation/training/holdout were not run after the no-go decision.

Table: `development_gates.csv`; records under `validation/`, `training/`, and
`holdout_locked/` explicitly say `NOT_RUN` rather than containing imputed values.

### 8. Statistical and causal methodology (2 pages)

- Complete episode as unit; paired common scenario seeds.
- Fixed-seed 10,000-replicate bootstrap for development summaries.
- Prospective hierarchical bootstrap and Holm family if a future study unlocks.
- Same-information held-out development monitoring comparison.
- Paired branching with simulator, RNG, agent, queue, and view-state restoration.
- Intention-to-treat primacy and per-intervention mediation as secondary evidence.

Evidence: `analysis_manifest.json`, counterfactual ledgers, replay report.

### 9. Development results (3.5–4 pages)

#### 9.1 Engineering and replay

- Test/replay totals, zero mismatches, conservation residual, privacy checks.

#### 9.2 Actionability

- Mock and real-Qwen structured validity.
- Accepted-to-transit/next/demand funnel.
- Retained failed Qwen attempt and versioned development retry.

#### 9.3 Coordination and bounded-human usefulness

- Fixed communication versus no communication by application/regime.
- KPI-triggered bounded operator versus autonomy.
- Complete alert-to-outcome counterfactual chains.

#### 9.4 Thermodynamic information value

- Same-information AP, ROC AUC, Brier score, and budgeted utility.
- Commercial failure and humanitarian success; cross-application Gate 5 failure.

#### 9.5 Trigger feasibility

- Timing, false activation, activation counts, and causal probes.

Figures: `primary_effect_forest.pdf`, `intervention_funnel.pdf`,
`monitoring_incremental_value.pdf`, `trigger_timing_and_false_alarms.pdf`,
`causal_intervention_effects.pdf`.

### 10. Mechanistic case studies (1.5 pages)

- Commercial event chain and intervention effect.
- Humanitarian event chain and intervention effect.
- Quiet, disruption, request, queue, intervention, autonomous response, arrival.
- Explicitly distinguish episode trajectory from paired intervention branch.

Figures: `commercial_case_study.pdf`, `humanitarian_case_study.pdf`,
`network_operator_sequence.pdf`, `trigger_and_intervention_dynamics.pdf`.

### 11. Negative result and boundary analysis (1.5 pages)

- Why trigger feasibility did not imply incremental information value.
- Commercial thermodynamic features slightly worsened held-out ranking.
- Humanitarian utility gain did not establish cross-domain generality.
- Real-Qwen validity limitations and the difference between material progression
  conditional on acceptance and end-to-end proposal success.
- No trained-policy, Pareto, robustness, or confirmatory claims.

Figures: `thermodynamic_ablation.pdf`, `partition_robustness.pdf`, explicit
`training_seed_curves.pdf` NOT-RUN panel.

### 12. Discussion (1.5–2 pages)

- What prospective gates prevented: another inactive or monitoring-only holdout.
- When distributed summaries may still be useful: high disagreement and abstract
  humanitarian allocation, subject to prospective replication.
- General lesson: actionability, necessity, human authority, and incremental
  information must be established before optimizing sparse oversight.
- Threats to construct, internal, external, and ecological validity.

### 13. Real-human study boundary and ethics (0.75 page)

- Simulated operators do not establish usability, trust, workload, ethics, or safety.
- Future IRB/ethics review, consent, privacy, accessibility, fatigue, and training.
- Prepared protocol/schema/randomization hooks only.

### 14. Conclusion (0.5 page)

- Mechanism-gated platform succeeded technically; proposed cross-domain positive
  contribution did not clear development.
- A future versioned study must improve same-information commercial value and
  real-planner validity before any locked evaluation.

## Main tables

1. Architecture, applications, operator authority, and information views.
2. Prospective gates, thresholds, outcomes, and evidence paths.
3. Actionability and full material-progression funnel.
4. Same-information incremental monitoring/decision value by application.
5. Development paired effects and counterfactual causal chains.
6. Compute, tokens, calls, failures, and stages not run.

## Optional supplement (15–25 pages)

- S1: complete typed tool and intervention schemas.
- S2: agent-independence and operator-view threat model.
- S3: thermodynamic definitions, nominal calibration, weight sensitivities.
- S4: gossip algorithm and partition diagnostics.
- S5: operator profiles, queue dynamics, fatigue equations.
- S6: all development matrices and seeds.
- S7: Qwen prompt revisions and retained repair failure analysis.
- S8: counterfactual snapshot/RNG restoration proof-by-test and replay schema.
- S9: every seed-level development point and failed attempt.
- S10: figure/PDF QA, environment lock, source checksums, reproduction commands.
- S11: proposed future human-study protocol and power-analysis template.

## Claims-to-evidence constraints

| Candidate claim | Evidence | Current status |
|---|---|---|
| Agents remain independent under oversight | privacy/authority tests; view hashes; architecture figure | confirmed engineering |
| V3 actions can reach demand and change loss | actionability table; funnel; counterfactual events | confirmed development |
| Coordination is necessary in v3 mechanics | paired coordination table/forest | confirmed development only |
| Bounded intervention can change outcomes | paired counterfactual table/case studies | confirmed development only |
| Thermodynamics adds value beyond same KPIs in commercial | monitoring table | unsupported |
| Thermodynamics adds value beyond same KPIs in humanitarian | monitoring table | supported development only |
| Cross-application information value | Gate 5 | unsupported |
| ThermoHITL beats KPI trigger | no validation/holdout artifact | untested |
| ThermoHITL is non-inferior to always-on review | no validation/holdout artifact | untested |
| ThermoHITL improves a loss/effort Pareto frontier | no eligible main experiment | untested |
| Robustness under partitions | abbreviated development diagnostic only | untested confirmatorily |
| Actual human usability/workload benefit | no participants | untested |

