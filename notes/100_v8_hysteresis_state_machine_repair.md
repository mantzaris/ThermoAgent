# V8 hysteresis state-machine repair

Recorded before executing fresh seeds 8812401--8812506. Iteration 2 retained the
5% gate and evaluated lower on thresholds, but both thresholds produced
identical traffic: 1.68% information-score transmissions in humanitarian and
5.16% in utility restoration. This invariance identified a state-machine
defect, not a threshold-selection result.

The previous latch suppressed every new score above `tau_on` until innovation
first fell below `tau_off`. That is inappropriate when the score is defined
relative to the last transmitted belief: a new high excursion represents new
information and must be eligible after cooldown. The corrected Schmitt-style
event semantics are:

- below `tau_off`: release/re-arm;
- between `tau_off` and `tau_on`: suppress while latched;
- at or above `tau_on`: transmit, including a fresh high continuation;
- always apply cooldown, maximum silence, and partition-recovery overrides.

No threshold or gate changes are made. Pilot iteration 3 reuses the declared
0.11 and 0.115 candidates on untouched seeds. If neither passes the unchanged
mechanism and nominal-traffic criteria, V8 stops before formal development.
