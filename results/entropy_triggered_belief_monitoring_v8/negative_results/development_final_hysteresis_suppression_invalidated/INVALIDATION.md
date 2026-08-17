# Invalidated final-development attempt: hysteresis suppression

This 288-arm development-only batch completed without infrastructure failure,
but it is ineligible for V8 method selection. It was aggregated before any V8
protocol freeze and before validation or holdout execution.

The post-batch mechanism audit found that the nominal generalized-information
arm transmitted 12,236 frames: 4,224 initial references, 7,992 maximum-silence
refreshes, 19 partition-recovery frames, and only one information-score event.
The off latch used the complete score, including monotonically increasing
message age. In normal trajectories the latch therefore almost never released.
The apparent communication result described an age-based refresh scheduler,
not the intended generalized-information event trigger.

All raw outputs and aggregate results are retained under the adjacent analysis
directory and `raw/development_final_hysteresis_suppression_invalidated/`.
They must not be used as formal V8 evidence. The repair excludes age from the
off-latch release statistic while retaining age in the activation score. Fresh
pilot and development seeds are required after the repair.
