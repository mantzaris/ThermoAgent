# Data dictionary

## Publication data

`results/JSTAT/source_data/` contains exactly one CSV for each numbered
publication figure. Each file is produced from retained aggregate tables and is
the numerical source for the same-numbered PDF in `paper/JSTAT/figures/`.

`results/JSTAT/figure_catalog.csv` maps each PDF to its source table, estimand,
scientific purpose, supported claim, limitation, and content hashes.

## Evidence-stage tables

- `results/JSTAT/stages/discovery/`: exploratory local-response and collective
  effects used to motivate later prospective work.
- `results/JSTAT/stages/replication/`: prospective Qwen replication tables and
  the microscopic-response table used to fit the kinetic surrogate.
- `results/JSTAT/stages/corrected_quench/`: quench trajectories, nominal
  geometry, recovery summaries, representation comparisons, delayed
  sensitivities, and the historical correction record.
- `results/JSTAT/stages/cross_model/`: final two-model panel summaries,
  hypothesis effects, sensitivity analyses, collective-observable tables,
  surrogate comparisons, and compact reproducibility records.

## Core fields and units

| Field | Meaning | Unit or range |
|---|---|---|
| `belief_magnetization` | mean categorical belief spin | [-1, 1] |
| `action_magnetization` | mean categorical action spin | [-1, 1] |
| `belief_action_overlap` | mean within-agent belief-action product | [-1, 1] |
| `reference_energy_per_agent` | symmetric-layer effective compatibility coordinate | effective units/agent |
| `configuration_entropy` | Shannon entropy of prospectively defined collective states | nats |
| `entropy_rate` | bounded-history transition unpredictability | nats/update |
| `total_correlation` | marginal entropies minus joint entropy | nats |
| `susceptibility` | finite-system magnetization fluctuation | dimensionless descriptive scale |
| `macrostate_distance` | leave-one-cluster-out standardized distance from nominal geometry | metric-specific distance |
| `adjusted_pathwise_irreversibility_nats_per_update` | raw block reversal divergence minus shuffled floor | nats/attempted update |
| `connected_correlation` | time-window connected belief correlation | [-1, 1] |
| `integrated_autocorrelation_time_updates` | truncated sum of normalized magnetization autocorrelation | attempted updates |
| `binder_cumulant` | finite-size order-parameter shape statistic | dimensionless |

Negative bias-adjusted information quantities are retained rather than clipped.
Confidence intervals and tests are summarized over complete graph/environment
trajectory clusters; rows inside a trajectory are not independent replicates.
