# V8 formal development iteration 1

The complete 624-arm batch used 24 independent panels per application and
finished with zero execution failures. This remains development evidence.

The generalized trigger at `tau_on=0.125` reduced sketch messages and actual
wire bytes by 0.281 (95% panel-bootstrap interval 0.249--0.314) in
humanitarian logistics and 0.333 (0.298--0.371) in utility restoration. Mean
primary distributed-state error increased by 0.00066 (0.00018--0.00120) and
0.00132 (0.00080--0.00192), respectively. Detection delay did not degrade.

The strongest matched-budget non-entropic scheduler remained KPI-change at
0.12. Generalized information improved primary error over KPI-change in
humanitarian logistics by 0.00146 (0.00065--0.00243), but was worse in utility
restoration by 0.00092 (0.00030--0.00160 in the opposite direction). The
cross-application entropy-specific H2 extension therefore failed in this
development iteration and is not being recast as positive.

The progression feasibility gate failed because four large utility grid
panels recorded zero recipient belief integrations despite 6,500--7,371
delivered sketch frames each. Inspection showed an architectural reachability
boundary: the one-hop V8 network caches and forwards a sketch, but a recipient
integrates it only when the target asset lies within its authority scope. On
these grid instances, agents sharing an asset were more than one forwarding
hop apart. This is not evidence that communication had no content; it is a
routing/TTL failure that prevents the required causal path into the action
policy.

Before any validation seed or outcome exists, one repair pilot is authorized:
increase the explicit forwarding limit from one to two on the four failed
panels, leaving the serializer, thresholds, beliefs, environment, policies,
and evaluator unchanged. The repair counts every forwarded frame and byte. If
two hops do not produce a recipient update in every panel, V8 stops rather
than weakening the gate. If it succeeds, one final retained 24-panel-per-
application development iteration will compare the already calibrated 0.125
and 0.13 generalized thresholds under two-hop forwarding. The selection rule
is fixed before that iteration: require all H1 noninferiority constraints and
all-panel belief integration, then minimize worst-application primary error;
use byte reduction and stable candidate name only as tie breaks.

Before any final-iteration comparative output was inspected, KPI-change at
0.10 was added alongside 0.12. Pilot traffic at 0.06 and 0.12 bracketed the
generalized trigger, and 0.10 is a single prespecified interpolation intended
to make the non-entropic byte budget closer. Comparator selection still uses
the frozen lexicographic budget-distance rule; this is not a search over KPI
outcome performance.
