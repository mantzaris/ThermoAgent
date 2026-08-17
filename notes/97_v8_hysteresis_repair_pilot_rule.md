# V8 hysteresis-repair pilot rule

Recorded before execution of fresh seeds 8812001--8812106. The repair pilot is
mechanism qualification, not validation. It compares the two already declared
on thresholds, always/no-exchange anchors, and the previously selected KPI
candidate under the corrected off-latch statistic.

A generalized candidate is mechanism-eligible only if, in each application:

- at least one `generalized_information_on` transmission occurs;
- information-score transmissions comprise at least 5% of non-initial,
  non-partition-recovery transmissions;
- the candidate neither remains inactive nor becomes always-on;
- every panel integrates at least one delivered peer belief;
- equal-Shannon mode-switch and sender-privacy unit tests pass;
- actual uint8 wire serialization tests pass.

The 5% criterion is deliberately modest: maximum-silence frames remain a
required stale-state safety mechanism, but an information trigger must show
that its information score actually schedules a nontrivial portion of traffic.
If both thresholds qualify, retain the earlier stable selection order: lower
worst-application estimation error, then higher byte reduction, then stable
name. A replacement 48-panel development batch will use entirely fresh seeds
after this pilot. The invalidated batch is not pooled into that analysis.
