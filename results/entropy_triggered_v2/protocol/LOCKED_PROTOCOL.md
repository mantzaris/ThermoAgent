# DOET locked-holdout protocol

This file is part of the pre-outcome freeze set. The machine-readable
`holdout_freeze.json`, created only after real validation and all fifteen
training checkpoints exist, records the exact source, configuration,
calibration, trigger-selection, checkpoint, design, and analysis checksums.

The confirmatory primary method is `doet_rule`; `doet_rl` is the learned
secondary method and cannot replace it after outcomes are observed. The benchmark is
`fixed_always_on`. The primary outcomes are service-loss area under the curve
for commercial logistics and cumulative unmet weighted need for humanitarian
logistics. The relative non-inferiority margin is 2%. Communication includes
all operational packets, entropy sketches, alert messages, structured bytes,
prompt/generated tokens, LLM calls, and inference latency. H1 and H2 are
evaluated in both applications with the preregistered paired hierarchical
analysis and Holm correction. H2's frozen primary unit is fully counted
messages and requires a positive confidence bound plus at least 20% mean
reduction; other costs remain required outputs. H3 requires message
nondominance and strict normalized hypervolume gains for messages, prompt
tokens, calls, and latency in both applications. H4 requires at least 75% of
non-nominal episodes to first activate at or after disruption and before
sustained severe service collapse, at most 10% of non-nominal episodes to
activate before the disruption, and at most 10% of nominal episodes to activate
falsely. Sustained severe collapse is confirmed at the third consecutive
post-disruption period with normalized service loss at least 0.90; activation
at that third period is not counted as leading collapse, and an episode without
confirmed collapse receives no before-collapse success credit. The development-
only evaluability audit and its input checksums are frozen in
`protocol/h4_evaluability_audit.json`. H5 requires non-inferiority in both
partition families and both applications, a positive consensus-error/
degradation slope, and Pearson `r >= 0.20` in each application. Failed runs and
all five RL seeds are retained; any failed locked episode prevents a supported
confirmatory classification, though complete matched pairs remain descriptive.

The new holdout uses environment seeds 8101–8116 for four non-nominal regimes,
8201–8208 for nominal false-activation panels, LLM seed 9101, a 16-period
horizon, and the unseen `tri_region_bridge_v2` topology. Fixed always-on,
learned non-entropic, DOET-rule, and DOET-RL run on all 144 panels. Five
secondary methods run on the same preregistered common non-nominal subset; all
Pareto comparisons restrict every method to these identical panels. The
preferred subset uses seeds 8101, 8106, and 8111. If measured validation plus
five-seed training makes that design exceed 35 single-GPU hours, the generator
uses the first fitting runtime-only fallback in this frozen order:
`[8101, 8111]`, then `[8106]`. Priority-method panels and RL-seed replication
are never reduced by this rule. This compute-capped allocation was selected
from measured throughput before validation outcomes. It is not run until the
freeze record exists and verifies.
Thresholds, budget-matched controls, and operating points come only from the
separate validation set.

Freeze requires all 144 validation episodes to complete without failure and
all 144 event ledgers to replay exactly. It also requires all fifteen fixed-
budget training runs and rejects any DOET-RL checkpoint whose recorded nominal-
normalizer checksum differs from the validation-selected calibration. A resumed
episode is accepted only when both its episode and compressed event ledger match
the manifest checksums; otherwise it is retained as a failure without rerun.

Before freeze, a 20,000-replicate fixed-seed stratified Monte Carlo precision
analysis and measured throughput must show that validation, all fifteen
training runs, the measured real-Qwen profile and model smoke, a 0.1-hour
unmeasured setup reserve, and the projected runtime-selected holdout fit within
35 additional single-GPU hours. The projection includes a 15% buffer. If even
the 616-episode minimum preregistered design does not fit, the generator fails
closed before freeze. CPU-bound training time on the reserved Pod is included.
