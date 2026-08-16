# V6 frozen development findings

This note was written only after the complete frozen development and dynamic
analysis batch closed. It reports development evidence, not validation or
holdout evidence. The operator is simulated; no human participants were
studied.

## Prospective primary comparison

The development-selected strongest non-entropic comparator was
`kpi_confidence`. At matched dynamic autonomous-action coverage, the combined
generalized-entropic controller reduced harmful-action rate in both primary
applications, but neither effect reached the prospectively frozen practical
minimum of 0.03:

- Humanitarian, private fragmented information: reduction `0.02258`, paired
  95% cluster-bootstrap interval `[0.01307, 0.03296]`, causal-utility change
  `+0.09874`, relative service-loss change `-0.01747`.
- Utility restoration, private fragmented information: reduction `0.01224`,
  interval `[0.00256, 0.02237]`, causal-utility change `+0.05111`, relative
  service-loss change `-0.00865`.
- Commercial boundary application: reduction `0.01731`, interval
  `[0.00888, 0.02588]`, causal-utility change `+0.06562`, relative
  service-loss change `-0.00900`.

These positive point estimates and intervals are not enough to pass Gate 5:
the practical threshold was fixed before formal development and was not
lowered after seeing the results.

## Fragmentation mechanism

The private-fragmented minus public-shared incremental harm reduction was:

- Humanitarian: `0.01908`, interval `[0.00833, 0.02990]`.
- Utility restoration: `0.00916`, interval `[-0.00147, 0.02001]`.
- Commercial boundary: `0.00706`, interval `[-0.00225, 0.01655]`.

The humanitarian interaction is directionally clear but narrowly below the
frozen `0.02` practical threshold. Utility restoration does not replicate the
interaction. Gate 7 therefore fails.

## Entropy family

The prespecified development rule selected `tsallis_q_0_5` from
`q in {0.5, 1, 1.5, 2, 3}`. This selection used development evidence only and
is not a validation- or holdout-selected q. The full-refit, severity/regime/
split-stratified permutation analysis comparing the generalized spectrum with
the Shannon/Jensen-Shannon family found:

- Humanitarian: observed reduction `0.00617`, one-sided Monte Carlo
  `p=0.004975`, Holm-adjusted `p=0.009950`.
- Utility restoration: observed reduction `0.00166`, `p=0.079602`,
  Holm-adjusted `p=0.079602`.

Thus the generalized family beyond Shannon/JS is supported only in the
humanitarian development application; it does not replicate across the two
primary applications.

## Distributed communication

All thermodynamic-sketch traffic is counted. Across the 720 communication
episodes, event-triggered exchange averaged `96.08` sketch messages and
`9,608.33` sketch bytes per episode, compared with `245.97` messages and
`24,596.67` bytes for always-on exchange. Event triggering reduced total
messages by approximately `51.7%--52.2%` and total bytes by
`48.4%--48.8%` across applications. Its distributed-estimation MAE was
`0.08741`, compared with `0.04832` always-on, `0.13866` periodic, and
`0.18824` without sketches. Matched safety differences between event-triggered
and always-on exchange had intervals spanning zero in every application.

## Timing and operator burden

In private fragmented development panels, the combined controller activated
by step 4 in `91.1%` of disrupted humanitarian panels and `95.6%` of utility
panels. Pre-disruption false activation was `0.56%` and `1.67%`, respectively;
nominal false activation was zero. Mean escalation burden was `3.61` and
`3.63` cases per disrupted panel, above the frozen maximum `3.5`; this is an
additional Gate 6 failure even though service and utility results were
favorable.

## Learnability audit qualification

The supervised action-value ceiling produced positive utility over no action
and five or six distinct selected actions per application. The primary risk
cross-fitting correctly isolated environment seed, topology family, and
scenario family in every fold. A post-outcome audit found, however, that the
separate pooled learnability diagnostic reused numeric environment seeds
across applications: all five `environment_seed_disjoint` flags are false,
while topology- and scenario-family flags are true. Its numerical ceiling is
retained but is methodologically compromised and cannot be used to rescue or
unlock the study. The deviation is recorded in
`results/generalized_entropic_consensus_v6/reproducibility/protocol_deviations.csv`.

## Disposition before agent qualification closes

Gate 5 and Gate 7 already fail on the complete frozen development evidence.
Consequently validation and holdout cannot run regardless of the later PPO
and Qwen qualification results. Those prespecified stages are still completed
because they measure agentic validity and training stability, not because they
can override the no-go.
