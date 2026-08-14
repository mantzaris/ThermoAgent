# DOET paper claims and evidence map

Status: final for v2. `Confirmed` below means the exact preregistered endpoint
passed on the new locked holdout; it does not upgrade a failed mechanism into a
causal claim.

| Candidate claim | Supporting table/statistic | Supporting figure | Status and allowed wording |
|---|---|---|---|
| The v1 exact ties came from identical policies. | `diagnostics/action_divergence.csv`, `diagnostics/feature_usage.csv` | `original_holdout_tie_diagnostics.pdf` | **Unsupported.** Policies diverged on 58.2% commercial and 46.2% humanitarian common epochs. Ties arose because divergent learned actions did not reach demand: 54/57 material calls failed and the other 3 stopped at intermediate nodes. |
| Operational entropy universally adds predictive information beyond ordinary KPIs. | `monitoring/incremental_value.csv`, `tables/monitoring_comparison.csv` | `monitoring_baseline_comparison.pdf`, `entropy_incremental_value.pdf` | **Unsupported.** Full evaluator KPI logistic models already reach AP/AUC 1.0; entropy adds zero rank value. |
| Distributed entropy adds value when ordinary inputs are restricted to private local KPIs. | `monitoring/incremental_value.csv` | `entropy_incremental_value.pdf` | **Suggestive development result.** AP rises by about 0.097--0.098 and AUC by 0.169--0.171 on original-main diagnostics. The seen v1 holdout does not preserve a robust rank gain. |
| DOET-rule is non-inferior to fixed communication. | `tables/noninferiority_analysis.csv`, `statistics/hierarchical_bootstrap.json` | `noninferiority_forest.pdf` | **Confirmed endpoint H1.** Aggregate relative degradation is 0.997% commercial (one-sided upper 1.565%) and 0.382% humanitarian (upper 0.584%), below the frozen 2% margin after Holm correction. Commercial isolated and partition regime-level bounds exceed 2%, so do not claim uniform regime non-inferiority. |
| DOET-rule reduces fully counted communication by at least 20%. | `tables/communication_reductions.csv`, `statistics/main_paired_comparisons.csv` | `communication_reduction.pdf` | **Confirmed endpoint H2.** Messages fall 72.35% and 74.22%, including entropy sketches; bytes, tokens, calls, latency, and wall time also fall with intervals excluding zero. |
| Operational entropy triggered those savings. | `tables/mechanistic_summary.csv`, `processed/mechanistic_events.csv` | `trigger_dynamics.pdf`, both event case studies | **Unsupported.** DOET-rule and DOET-RL each activated in 0/144 episodes; maximum statistic 0.618 versus `tau_on=1.2`. Savings are attributable only to the frozen sparse quiet schedule. |
| DOET improves the performance--communication Pareto frontier. | `statistics/pareto_points.csv`, `statistics/pareto_frontier_hypervolume.csv` | `performance_communication_pareto.pdf` | **Unsupported H3 / mixed diagnostic.** Frozen hypervolume increases, but no communication dominates DOET-rule on both loss and messages in both applications, violating the full preregistered criterion. |
| DOET activates after disruption and before severe service collapse. | `tables/mechanistic_summary.csv` | `trigger_dynamics.pdf`, `commercial_event_case_study.pdf`, `humanitarian_event_case_study.pdf` | **Unsupported H4.** There were no activations. The figures must retain the explicit zero-activation annotation. |
| Distributed triggering is robust to delayed/noisy/partitioned communication. | `statistics/partition_consensus_relationship.csv`, `tables/mechanistic_summary.csv` | `partition_robustness.pdf` | **Unsupported H5.** Partition episodes had no activations, and consensus-error/degradation slopes and correlations were negative rather than predictably positive. |
| The formal non-inferiority and communication endpoints replicate across applications. | `tables/hypothesis_outcomes.csv` | `holdout_primary_results.pdf`, `communication_reduction.pdf` | **Confirmed endpoint H6.** H1/H2 pass in commercial and humanitarian applications. Allowed wording must immediately state that the entropy mechanism did not activate. |
| DOET-RL improves fixed communication. | DOET-RL rows in `statistics/main_paired_comparisons.csv`, `statistics/training_seed_variability.csv` | `training_seed_variability.pdf` | **Suggestive/descriptive.** Mean loss is 0.961% lower commercial and 1.382% lower humanitarian with 86.50%/84.85% fewer messages across five training seeds. Because entropy never activated, attribute this to the learned sparse coordination policy, not DOET triggering. |
| Exact global entropy would have activated when distributed entropy did not. | `tables/extended_ablation_results.csv`, `tables/extended_ablation_mechanisms.csv` | `trigger_ablation_effects.pdf` | **Unsupported exploratory result.** All 12 exact-global-entropy oracle episodes had zero activations. |
| The KPI and disruption-label controls provide valid timely upper bounds. | `tables/extended_ablation_mechanisms.csv` | `trigger_ablation_effects.pdf` | **Unsupported / implementation failure retained.** Both fired at period 0, eight periods before disruption. The label oracle inherited the low-direction transform and is invalid as an oracle upper bound. |
| Independent autonomous agents are necessary in these environments. | `statistics/pareto_points.csv` plus frozen v1 necessity map | `performance_communication_pareto.pdf` plus frozen v1 `agentic_necessity_map.pdf` | **Unsupported.** No communication dominates DOET-rule in the v2 common panel, and the frozen v1 necessity map favored fixed communication in every cell. |
| The constructs are literal physical entropy, energy, or temperature. | Method definitions only | none | **Not claimed.** These are statistical-mechanics-inspired operational constructs. |
| DOET v2 is ready as the intended positive AIJ contribution. | Entire result set and H1--H6 table | all paper PDFs | **Unsupported.** Engineering demonstration: yes. Intended AIJ mechanism: insufficient because activation, timely response, partition robustness, and autonomous necessity were not established. |

## Hypothesis disposition

- H1: **confirmed** as an aggregate non-inferiority endpoint.
- H2: **confirmed** as fully counted communication superiority.
- H3: **unsupported** because DOET-rule is dominated by no communication.
- H4: **unsupported** because there were zero trigger activations.
- H5: **unsupported** because there were zero partition activations and no
  predicted consensus-error relationship.
- H6: **confirmed only as the conjunction of H1/H2 in both applications**; the
  causal cross-application entropy claim is unsupported.

## Paper decision

Do not use the provisional positive contribution statement as written. A
scientifically honest short paper or technical report can present the platform,
preregistered negative mechanistic result, monitoring observability boundary,
and replay/tie diagnosis. The intended positive AIJ submission requires a new
study with a trigger that passes a prospectively enforced minimum-activation and
timing gate on validation, plus a fresh locked holdout. Thresholds must not be
retuned on v2 holdout values.

The paper-facing synthesis is
`results/entropy_triggered_v2/PAPER_SUMMARY.md`; the machine-readable decisions
are `results/entropy_triggered_v2/tables/hypothesis_outcomes.csv`.
