# Operational thermodynamics in v3

These are statistical-mechanics-inspired operational summaries, not physical
energy, entropy, free energy, or temperature.

Operational energy uses fixed service-priority weights:

`E = .24 backlog + .22 unmet + .16 congestion + .14 lateness`
`    + .12 commitment risk + .12 route/safety risk`.

Entropy components are normalized material-flow allocation entropy and belief
entropy over local disruption states. Agent disagreement is bounded
Jensen–Shannon divergence across received sketches. Agents compute raw values,
two-sided deviations from nominal development calibration, slope, optional
acceleration, and consensus confidence. Gossip respects time-varying links and
partitions. Evaluator-global estimates are retained only for error analysis.

The free-energy diagnostic is `F = E - T*S`; `T` is bounded disruption
volatility/decision stochasticity. Because v1 free-energy gap was weak, v3 never
uses raw free energy as the only primary signal.

The final development trigger candidate used `tau_on=1.5`, `tau_off=0.6`,
`actionable_tau_on=1.1`, dwell 2, cooldown 3. It activated timely in every final
disrupted development episode with zero nominal or pre-disruption false
activation. This timing result did not rescue the failed incremental-value gate.

