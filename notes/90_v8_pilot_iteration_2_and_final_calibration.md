# V8 pilot iteration 2 and final calibration plan

Pilot iteration 2 completed 196/196 arms with zero execution failures and a
scheduler-invariant evaluator target. It confirmed that always-on exchange
improves the primary *distributed-state* estimate once belief-vector error and
distributed-disagreement error are both represented, but neither tested
generalized threshold met every prespecified constraint. `tau_on=0.08` was too
close to always-on (only about 6--9% sketch-byte reduction), while `tau_on=0.14`
reduced sketch bytes by about 31--36% but produced a worst-panel belief-MAE p95
increase of about 0.019 in utility restoration.

Before pilot iteration 3 outcomes, the primary estimation estimand is defined
as the equally weighted mean of two normalized errors:

1. MAE between the locally reconstructed pooled belief and the evaluator pool
   of current independent private evidence;
2. absolute error in Jensen-Shannon disagreement.

The components remain reported separately. Equal weighting prevents a method
from appearing accurate merely because one local belief happens to be near the
pool while its disagreement estimate is absent. The mean noninferiority margin
remains 0.02. The belief-MAE p95 increase over always-on remains capped at 0.01.
No fixed absolute p95 ceiling is used because the always-on floor itself varies
with graph size, scope overlap, and latency; the absolute p95 is still reported.

The valid matched encoding pilot selected the actual uint8-simplex frame:
approximately 0.0017 mean L1 quantization error and fewer wire bytes than FP16.
All iteration-3 schedulers therefore use that same encoding.

Iteration 3 is the last trigger-calibration pilot. Its generalized thresholds
are the prespecified interpolation set `{0.10, 0.11, 0.12, 0.125, 0.13}`.
It also reruns the matched-budget KPI, predictive-uncertainty, random, and
periodic competitors on the same new panels. If no generalized threshold passes
the unchanged reduction, mean-error, p95-increase, and delay criteria, V8 stops
before formal development rather than adding more thresholds.
