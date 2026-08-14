# DOET development, validation, and comparative results

Status: complete. Development and validation are exploratory/model-selection
stages; the 696-episode holdout is the only new confirmatory evaluation.

## Validation decision

The real-Qwen validation matrix contained 144 episodes, separate from
development and from the new holdout. Four prospectively specified trigger
candidates were compared by the frozen lexicographic selector. It selected
`hysteresis_low` with low-direction residual, `tau_on=1.2`, `tau_off=0.4`,
crisis threshold `2.8`, minimum dwell `2`, cooldown `2`, and neighbor
propagation. The retained CUSUM state parameters are `rho=0.6` and `kappa=0.0`.
Application- and role-conditioned nominal normalizers came only from nominal
development/calibration data.

The selector exposed an important warning before holdout: every entropy
candidate activated zero times. Selected `hysteresis_low` increased commercial
loss by about 0.792%, tied humanitarian loss, and reduced counted messages by
about 70.7%. Because the frozen selector did not require a nonzero, timely
activation rate, it selected a quiet schedule. This was recorded before the
holdout and no threshold was changed afterward.

## Multiple-seed training

Learned non-entropic coordination, ThermoAgent v1, and DOET-RL each used five
independent PPO initializations: 7301--7305. All 15 fixed-budget runs completed;
none was removed or retrained. Checkpoints were chosen by the same fixed rule,
not by holdout performance. DOET-RL also had zero trigger activations during
training, so its active-decision trajectory count and PPO update count are
lower. This is a retained design limitation, not a failed seed.

## Locked comparison with always-on fixed communication

The primary preregistered method was DOET-rule and the primary benchmark was
`fixed_always_on`. The primary unit was a complete paired episode. All values
below use 64 matched non-nominal panels per application and 10,000 hierarchical
bootstrap replicates.

| Application | Mean loss difference | Relative degradation | One-sided 95% upper | Message reduction (95% CI) |
|---|---:|---:|---:|---:|
| Commercial | +0.1246 service-loss AUC | +0.997% | +1.565% | 72.35% [70.98%, 73.67%] |
| Humanitarian | +15.012 weighted unmet need | +0.382% | +0.584% | 74.22% [73.05%, 75.43%] |

Both application-level upper confidence bounds are below the frozen 2%
non-inferiority margin, and both message-reduction intervals exclude zero and
exceed the 20% practical target after Holm correction. H1 and H2 therefore pass
as formal endpoint tests.

The communication savings include entropy sketches and all operational,
negotiation, coalition, and commitment traffic. Relative to fixed, DOET-rule
also reduced:

- structured bytes by 68.58% commercial and 70.57% humanitarian;
- prompt tokens by 37.81% and 41.93%;
- generated tokens by 41.49% and 42.48%;
- LLM calls by 36.77% and 38.46%;
- measured LLM latency by 38.84% and 39.87%;
- episode wall time by 38.82% and 39.86%.

These formal results cannot be attributed to entropy triggering. DOET-rule and
DOET-RL each activated in 0/144 holdout episodes, remained in quiet mode for
100% of periods, and sent no entropy alerts. The largest realized trigger
statistic was 0.510 commercial and 0.618 humanitarian, below `tau_on=1.2`.
Operational entropy itself varied and reached standardized residuals above 3;
the selected low-direction transform and stateful threshold did not convert
those changes into activation.

## Regime boundaries

The aggregate commercial non-inferiority endpoint passes, but the regime-level
one-sided upper bound exceeds 2% for isolated disruption (4.098%) and
communication partition (2.626%). Commercial correlated and compound-OOD pass.
All four humanitarian non-nominal regimes pass. Communication reductions remain
large in every regime.

Partition robustness is unsupported mechanistically. Consensus error does not
predictably increase degradation: commercial slope -0.743 with Pearson
`r=-0.195` (`p=0.284`), humanitarian slope -0.121 with `r=-0.103`
(`p=0.574`). Neither partition regime produced a trigger activation.

## Pareto and stronger controls

On the prospectively common method/panel subset, no communication has both the
lowest loss and zero messages in each application:

| Application | No communication | DOET-RL | Fixed | DOET-rule |
|---|---:|---:|---:|---:|
| Commercial loss / messages | 12.993 / 0 | 13.290 / 29.58 | 13.292 / 222.67 | 13.582 / 58.25 |
| Humanitarian loss / messages | 4061.29 / 0 | 4199.11 / 31.92 | 4267.02 / 224.92 | 4293.76 / 55.58 |

Thus DOET-rule is dominated and H3 fails even though the frozen normalized
hypervolume calculation improves when DOET is added to comparator-only
frontiers. The hypervolume result cannot be reported alone.

DOET-RL descriptively improves fixed while using much less communication:
commercial relative loss -0.961% with 86.50% fewer messages; humanitarian
-1.382% with 84.85% fewer messages. It was evaluated across all five training
seeds. Because its entropy gate never activated, this is evidence about the
learned sparse/quiet coordination policy, not evidence that entropy caused the
improvement.

## Exploratory post-holdout controls

The 96-row ablation design was checksum-frozen after the holdout and before its
first episode. It used three new seeds, two applications, and correlated plus
compound-partition regimes. It is exploratory, not confirmatory.

- All 60 local DOET-variant episodes and all 12 exact-global-entropy trigger
  oracle episodes had zero activations and matched the selected DOET loss within
  each application/scenario cell.
- Removing distributed gossip reduced average messages from 68.75 to 51.5 with
  no loss change, further showing that sketch cost bought no activation here.
- The private-local-KPI CUSUM control activated in 12/12 episodes and used
  72.1% more messages than selected DOET, while changing mean within-cell loss
  by only -0.023%.
- The putative disruption-label oracle activated in 12/12 episodes and used
  40.0% more messages, with mean within-cell loss change -0.300%.
  Event timing revealed that both active controls first fired at period 0,
  eight periods before disruption. The binary label inherited the selected
  low-direction transform, so healthy label 0 was treated as anomalously low.
  It is an invalid oracle implementation and is retained as a failure, not an
  upper bound.

All 96 control episodes completed. A replay-order defect initially reported 12
metric mismatches only for the alerting controls: replay applied protocol alerts
after, rather than before, the same-period public metric snapshot. The replay
engine and a regression test were corrected; no episode was rerun. All 936 v2
ledgers now replay exactly.

## Evidence classification

Formally supported: H1, H2, and endpoint-defined H6. Unsupported: H3, H4, H5.
The pass on H6 means only that H1/H2 pass in both applications; it does not show
a cross-application entropy-trigger mechanism. The intended AIJ-positive DOET
claim is not supported because activation, timely response, partition mechanism,
and autonomous-agent necessity were not demonstrated.

Authoritative artifacts:

- `results/entropy_triggered_v2/statistics/main_paired_comparisons.csv`
- `results/entropy_triggered_v2/statistics/hierarchical_bootstrap.json`
- `results/entropy_triggered_v2/statistics/pareto_points.csv`
- `results/entropy_triggered_v2/tables/mechanistic_summary.csv`
- `results/entropy_triggered_v2/tables/extended_ablation_results.csv`
- `results/entropy_triggered_v2/tables/extended_ablation_mechanisms.csv`
- `results/entropy_triggered_v2/tables/hypothesis_outcomes.csv`
