# V6 prospective protocol draft

This note records decisions after engineering pilots but before formal V6
development, validation, or holdout outcomes. The machine-readable protocol is
`configs/generalized_entropic_consensus_v6.yaml`. Its status remains `draft`
until the implementation, tests, exact source commit, and checksum manifest are
frozen.

## Claim hierarchy

The primary claim concerns selective safety: at 50% autonomous-action coverage,
a combined generalized-entropic controller must reduce harmful physical actions
by at least three percentage points relative to the strongest frozen
non-entropic uncertainty baseline, with a paired panel-bootstrap 95% lower bound
above zero, in both humanitarian and utility restoration. Commercial logistics
is a prespecified boundary application.

The strongest non-entropic comparator is selected once on development using
mean risk-coverage area, breaking ties with Brier score. The pilot currently
suggests predictive uncertainty, but the formal rule—not that observed label—is
frozen. Tests first compare the generalized family. Individual q values are
interpreted only after a family-level success and Holm adjustment.

## Feature design

The primary block combines ordinary KPI and action-policy uncertainty with a
small interpretable thermodynamic set: Shannon local uncertainty, pooled-minus-
average uncertainty, Jensen-Shannon disagreement, a Tsallis spectrum contrast,
a Jensen-Tsallis spectrum contrast, graph-weighted disagreement, consensus
residual, and temporal disagreement change. Exact q=2 and Gini-Simpson values
are not both entered into the same primary regression because they are
mathematically equivalent after normalization. All q-family measures remain
available as prespecified ablations.

The strongest non-entropic family includes KPI logistic risk, KPI plus
predictive-action uncertainty, and a grouped one-sided split-conformal upper
risk bound. The conformal calibration family is disjoint from both the outer
test family and the model-fitting families. Transparent fixed, random,
margin, maximum-probability, ensemble-proxy, individual entropy, graph, and
oracle rankings are secondary risk-coverage references. The oracle is never
deployable.

Regularization is selected only inside nested grouped folds. The independent
unit is the environment panel. Groups isolate environment seed, topology,
scenario family, and disruption family. Candidate decisions from one panel may
not cross splits.

## Gate rationale

Three percentage points is the smallest practical harm-rate reduction: at 50%
coverage and up to twenty post-disruption eligible physical decisions per panel, it corresponds to preventing
roughly one harmful autonomous action per 1.7 panels, rather than a negligible
ranking improvement. The 2% service-loss margin prevents a controller from
appearing safe solely by withholding useful action. The private-minus-public
interaction must exceed two percentage points because the scientific mechanism
is fragmented observability, not generic model capacity.

Event-triggered sketches must reduce both messages and bytes by 40% relative to
always-on sketches and keep distributed-estimation MAE below 0.12. All sketch
traffic, latency, model calls, and tokens are counted. Five independently
initialized sequential PPO seeds are mandatory for every primary learned
method; no failed seed may be dropped.

## Power and planned compute

The machine-readable plan is
`results/generalized_entropic_consensus_v6/protocol/power_and_compute_plan.json`.
A conservative paired-SD floor of 0.10 gives approximate power 0.992 for 210
development panels, 0.908 for 120 validation panels, and 0.950 for 144 holdout
panels per primary application/information condition at the frozen 0.03
practical effect. These are normal-approximation planning values, not outcome
tests.

Five methods by five sequential PPO seeds, model loading/profile, 150 Qwen
episodes, analysis, and a 15% reserve project to 11.50 single-GPU hours and
USD 3.91 at USD 0.34/hour. Qwen contains 2,700 primary agent decisions before
any single allowed repair, with a projected 2,730 calls, 2.73 million prompt
tokens, and 253,000 generated tokens. Projected storage is 4.3 GiB. This is
below the 50-hour and USD 40 caps. A measured remote profile will be recorded
before full Qwen execution.

No validation or holdout seed will be opened while this protocol is draft.

## Timing repair and outcome-blind escalation calibration

Before protocol freeze, review identified that the initial V6 implementation
began decisions at the disruption epoch and unconditionally escalated the
highest-risk unserved proposal. This repeated a V5 timing limitation. The
repaired design adds an explicit nominal incident mode and a step-0
pre-disruption decision. No future mode enters pre-disruption beliefs. Rare
fixed-rate ambiguous nominal evidence makes false alerts observable rather
than structurally impossible.

Pilot `pilot_v11_timing_final` used untouched seeds 60981–60985 and included
nominal, telemetry-integrity, partition, and compound regimes. Thresholds
0.50–0.80 were compared using only activation, timing, and burden—not harm,
utility, or service outcomes. The frozen rule selects the highest threshold
that passes the timing criteria for both candidate primary controllers. It
selected 0.80. Predictive uncertainty activated 93.3% of disrupted primary
panels, 76.7% by step 4, with 2.60 mean escalations, 0% pre-disruption false
activation, and 0% nominal-panel activation. Combined generalized entropy
activated 100%, 83.3% by step 4, with 2.70 mean escalations, 0%
pre-disruption false activation, and 5% nominal-panel activation. These are
pilot calibration results, not evidence for selective-safety efficacy.
