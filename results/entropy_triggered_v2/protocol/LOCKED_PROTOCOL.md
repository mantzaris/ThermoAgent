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
non-nominal episodes to activate before service collapse and at most 10% of
nominal episodes to activate falsely. H5 requires non-inferiority in both
partition families and both applications, a positive consensus-error/
degradation slope, and Pearson `r >= 0.20` in each application. Failed runs and
all five RL seeds are retained; any failed locked episode prevents a supported
confirmatory classification, though complete matched pairs remain descriptive.

The new holdout uses environment seeds 8101–8116 for four non-nominal regimes,
8201–8208 for nominal false-activation panels, LLM seed 9101, a 16-period
horizon, and the unseen `tri_region_bridge_v2` topology. Fixed always-on,
learned non-entropic, DOET-rule, and DOET-RL run on all 144 panels. Five
secondary methods run on the preregistered common non-nominal subset using
seeds 8101, 8106, and 8111 in every application/regime cell; all Pareto
comparisons restrict every method to these identical panels. This
compute-capped allocation was selected from measured throughput before
validation outcomes. It is not run until the freeze record exists and verifies.
Thresholds, budget-matched controls, and operating points come only from the
separate validation set.

Before freeze, a 20,000-replicate fixed-seed stratified Monte Carlo precision
analysis and measured throughput must show that validation, all fifteen
training runs, the measured real-Qwen profile and model smoke, a 0.1-hour
unmeasured setup reserve, and the projected 696-episode holdout fit within 35
additional single-GPU hours. CPU-bound training time on the reserved Pod is
included.
