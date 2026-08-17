# V8 pre-freeze final selection rule

Recorded while the corrected 288-arm `development_final` batch was still
running and before its aggregate results were opened.

The candidate set remains exactly the two previously declared generalized
thresholds (`tau_on` 0.125 and 0.13), always-on/no-exchange anchors, and the
two previously declared KPI budget interpolation points (0.10 and 0.12).
There is no further threshold search.

An eligible generalized trigger must pass, in both applications, all of the
following panel-bootstrap or all-panel conditions:

- lower 95% bound for sketch-message reduction at least 0.25;
- lower 95% bound for exact sketch-wire-byte reduction at least 0.25;
- upper 95% bound for mean primary distributed-state-error increase at most
  0.02;
- upper 95% bound for pointwise-p95 primary-error increase at most 0.01;
- upper 95% bound for detection-delay degradation at most five simulator
  steps (one decision epoch);
- every panel integrates at least one delivered peer belief after two-hop
  forwarding;
- mean pre-disruption non-initial transmission rate at most 0.10.

The last quantity treats the first reference frame as protocol initialization,
not a false transmission. It uses only trigger evaluations strictly before the
domain-defined disruption onset. This feasibility diagnostic was added before
the completed batch was aggregated because the earlier selection code had not
made the nominal-traffic requirement explicit.

If both generalized candidates are eligible, choose the one with the lowest
worst-application mean primary error, then higher mean byte reduction, then a
stable name tie-break. Choose the strongest non-entropic comparator by the
already written lexicographic rule: worst-application log-byte-budget distance,
then primary error, disagreement error, delay, service loss, and stable name.

H1 and H3 are progression criteria after the five-seed frozen autonomous
policy is evaluated. H2 entropy-specific matched-byte superiority remains a
separate extension and cannot block an otherwise valid communication-
efficient monitoring study.
