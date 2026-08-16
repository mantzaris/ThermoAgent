# V7 prospective protocol candidate

This note records the candidate formal design while the final retained
feasibility pilot is still running. It is not a frozen protocol and does not
unlock formal development. The machine-readable source is
`configs/v7_protocol_candidate.yaml`.

## Decisions and alternatives

The formal risk estimand will compare a regularized logistic non-entropic
controller with a same-capacity entropic controller. A larger boosting model
was considered, but it would confound feature value with model capacity. Fixed
transparent rules remain diagnostic references. Model regularization will be
selected only inside grouped training folds.

The independent unit is a complete matched environment panel. All decisions
from one environment seed, graph instance, scenario-factor combination, and
information condition remain in the same fold. Candidate actions are not
replicates.

The candidate development design uses 50 reference panels per application:
27 medium-size coupling-by-fragmentation response-surface panels, enough extra
high/high medium panels to reach 12, four high/high small and four high/high
large panels, and six public-shared high/high panels. This concentrates power
on the primary interaction and high-complexity estimands instead of expanding
an uncontrolled factorial.

The practical high-complexity harm-reduction target is 0.04 absolute. That is
roughly one prevented harmful action per 25 executed actions and is large
enough to matter under a 60% autonomous-action coverage budget. The candidate
complexity-interaction target is 0.02 absolute from low/low to high/high.
Neither value was selected from a positive V7 effect: the first two retained
pilots produced a negative interaction and did not meet action-pool validity.

Event-triggered sketch communication must reduce both total messages and total
bytes by at least 20% relative to always-on exchange. The paired upper 95%
bound on harm degradation must be at most 0.02 and distributed-estimation MAE
must not exceed 0.08. Operational, negotiation, evidence, sketch, escalation,
dropped-message, and byte accounting are all included.

Realized autonomous coverage is also frozen as a validity condition. The
mean absolute paired coverage difference between the two primary controllers
must not exceed 0.02. Formal progression additionally requires zero episode
failures, zero replay mismatches, zero privacy failures, and independently
reconstructed conservation residual no greater than `1e-9`; favorable output
cannot compensate for an engineering failure.

## Progression rule

Iteration 3 is the last planned outcome-informed environment repair. If the
prospective feasibility gates fail, the protocol candidate will remain
unfrozen, formal development will not run, and V7 will be reported as an
engineering and feasibility boundary result. If all three feasibility gates
pass, this candidate will be reviewed for internal consistency, hashed,
committed as the execution source, and only then used to generate untouched
development, validation, and holdout manifests.

RL and real-Qwen work are not allowed merely because the simulator runs. They
are unlocked only after the formal development reference panels establish a
measurable primary mechanism opportunity. The existing RunPod endpoint was
unreachable with `connection refused` during the V7 feasibility phase; no new
Pod was created.
