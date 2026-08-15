# V4 dashboard and information boundaries

The v4 dashboard is part of the tested mechanism, not a post-hoc decorative
figure. `V4DashboardReplay` reads event-sourced topology, explicit queue and
operator events, public material/service events, and schema-validated
`operator_view_v4` payloads. It does not use evaluator time-series state to fill
the deployable view.

## Authorized data

- public agent identity, role, location, topology, communication links, and
  temporarily authorized emergency edges;
- feature fields allowed by the assigned view condition;
- bounded alert reason, predicted benefit/uncertainty, priority, and time to
  collapse;
- queue, one-slot budget, workload, and simulated operator minutes;
- typed operator acknowledgements and public material/service progress; and
- contributor/provenance metadata and the SHA-256 hash of the exact payload.

## Forbidden normal-view data

- raw private agent observations, memory, utility, inventory, cost, or
  telemetry;
- evaluator-global thermodynamic state;
- true disruption or true telemetry-corruption labels;
- future events or loss;
- RNG state;
- counterfactual outcomes; and
- oracle state.

Utility restoration may show a coarse visible telemetry-confidence state; it
cannot show whether the simulator marked telemetry as truly corrupted. Tests
inject forbidden fields and require rejection, verify payload hashes, and
verify deterministic replay.

Three actual populated replay exports—commercial, humanitarian, and utility—are
under `results/human_operator_v4/dashboard_exports/` as SVG and PDF. The
publication overview is copied from the functional utility replay export, not
redrawn by hand. The future human-study protocol, schema, randomization
template, and power template are technical preparation only; no participant was
studied.
