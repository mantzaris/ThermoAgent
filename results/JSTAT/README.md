# JSTAT evidence package

This directory contains the compact publication-facing evidence for the final
ThermoAgent study. It preserves the distinct discovery, replication,
corrected-quench, and cross-model stages without distributing raw prompts,
completions, model weights, or complete trajectory records.

## Final experiment

- Two pinned model families: Qwen and Granite.
- Six graph/environment clusters per model.
- Four matched conditions per cluster.
- 48 trajectories, 45 sweeps per trajectory, 16 attempted updates per sweep.
- 34,560 formal decisions.
- Complete graph/environment trajectory cluster as the inferential unit.

The frozen protocol is `configs/statmech_llm/cross_model/protocol.yaml`; its
SHA-256 is
`863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9`.

## Layout

- `source_data/`: one immutable numerical source CSV per publication figure.
- `figure_catalog.csv`: figure-to-source provenance and scientific purpose.
- `stages/`: compact aggregate and reproducibility records for each evidential
  role.
- `reproducibility/`: publication hash inventory and semantic-consolidation
  manifest.
- `provenance/figure_inventory.csv`: SHA-256 classification of all 153 retained
  stage figure artifacts.
- `provenance/script_inventory.csv`: dependency-closure decision for every
  script present before consolidation.

The primary machine-readable dispositions are in
`stages/cross_model/statistics/primary_results.json`, with cluster values and
test details in `stages/cross_model/tables/hypothesis_effects.csv`.

## Interpretation

Effective reference energy is a model-relative compatibility observable.
Adjusted path-reversal divergence is a coarse-grained temporal-asymmetry
statistic whose arm-level values may be negative after shuffle-floor
subtraction. Neither is a claim of literal thermodynamic energy or exact
entropy production. The study is finite (`N=16`) and supports no
thermodynamic-limit transition or universality claim.

See `docs/data_dictionary.md` and `docs/validation.md` for fields, units,
correction history, and validation rules.
