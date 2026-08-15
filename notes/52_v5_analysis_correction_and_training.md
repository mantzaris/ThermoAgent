# V5 analysis correction and multi-seed training

## Refit-permutation correction

The first two 199-replicate development permutation jobs exposed an analysis
boundary inconsistency: the helper refit only non-nominal panels, whereas the
primary pipeline fit all development regimes and then evaluated its primary
effect on disrupted panels. No validation or holdout data were involved.

The inconsistent outputs are retained under
`results/human_operator_v5/superseded/` and are ineligible. The corrected
procedure keeps nominal panels in every grouped fit, excludes them only from
the evaluation contrast, preserves the frozen strata, regularization grid,
seeds, and 199 replicate count, and refits the complete pipeline after every
block permutation. This repair cannot change the already negative primary
point estimates or unlock Gate 5.

The corrected 199-replicate jobs completed for both applications. Humanitarian
true/permuted mean gains were `-0.0116410` and `+0.0025069`; utility-restoration
values were `-0.0099697` and `+0.0010033`. The corrected files are the
authoritative secondary falsification outputs. They preserve the primary
negative interpretation.

## Training evaluation addendum

The development protocol already froze two decentralized PPO-style methods,
five seeds (52001-52005), 30,000 decision epochs, and a final-checkpoint rule.
The separate development-only evaluation seeds 52101-52110 were recorded in
`configs/human_operator_v5_training.yaml` before training. They were not chosen
from training or validation outcomes. The canonical formal-development YAML
was restored byte-for-byte to checksum
`687aee0ebde467c3d5b5919906345a570abb9454b3fdbbd33c9b2a965d575770`.

All ten training runs completed, with no seed removed. Both KPI-only and
entropy/disagreement policies converged to a degenerate single-action policy
with zero mean evaluation reward, zero between-seed variance, and action
diversity one. The entropy policy's mean gain was exactly zero. This fails the
prospective learning-stability gate: reproducible collapse is not useful
stability. The result is development evidence and cannot substitute for a
validation or holdout comparison.
