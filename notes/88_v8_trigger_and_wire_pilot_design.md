# V8 trigger and wire pilot design

This design was recorded before running V8 pilot panels.

## Candidate family

The pilot compares always-on, none, periodic, matched random, local KPI change,
predictive-uncertainty change, L1 belief drift, age of information, the frozen
V7 Shannon-change rule, Jensen-Shannon drift, Tsallis-spectrum drift, and the
full generalized-information score. Thresholds are a small prespecified grid;
there is no dense search. Every non-none scheduler sends an initial reference.
Every deployable event scheduler has a finite maximum-silence deadline.

The full score uses weights `(JS, spectrum, confidence, age) =
(0.45, 0.25, 0.15, 0.15)`. Candidate `tau_on` values are 0.08 and 0.14;
`tau_off=0.04`, cooldown two simulation steps, and maximum silence 30 steps.
The entropy spectrum is fixed at q in `{0.5, 1, 1.5, 2, 3}`.

## Encoding rule

FP32, FP16, and bounded uint8-simplex frames are exercised with the same
always-on schedule. FP16 will be selected if its mean L1 round-trip error is at
most 0.001 and it is smaller than FP32. uint8 is eligible only if its mean L1
error is at most 0.005 and it does not worsen pilot estimation by more than
0.002. If both pass, prefer the smaller encoding; otherwise prefer FP16, then
FP32. Encoding is frozen before formal development and held constant across
primary schedulers.

## Trigger selection rule

A generalized-information candidate is eligible only if, in each application,
its paired mean actual-wire byte reduction from always-on is at least 25%, its
increase in normalized time-integrated belief-estimation error is at most 0.02,
its pointwise p95 error is at most 0.08, and its detection-delay degradation is
at most one application decision interval. Rank eligible candidates by:

1. smallest worst-application estimation-error increase;
2. smallest worst-application detection-delay increase;
3. greatest mean byte reduction;
4. deterministic configuration digest.

The strongest non-entropic comparator is chosen at the closest actual byte
budget (mean absolute log-byte ratio first), then by lower disagreement-error,
lower detection delay, and lower service loss. Candidate non-entropic methods
are periodic, matched random, KPI change, predictive-uncertainty change, L1
belief drift, and age of information. This choice does not depend on whether
the generalized trigger beats it.

## Pilot scope and progression

Four retained panels per application cover small, medium, and large systems,
multiple topology families, low/high fragmentation, and medium/high disruption.
One-hop forwarding is used in the pilot. Pilot outcomes may choose the frozen
configuration but may not be reported as validation or holdout evidence.
