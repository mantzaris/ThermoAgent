# Environment design

## Shared quantitative kernel

Both applications use `thermoagent.environment.LogisticsEnvironment`. Each
organization controls one abstract node in the current implementation; the
identity API allows one organization to own more than one node in a later
topology. State includes inventory, production/handling capacity, stochastic
demand, backlog, impairment, delay, costs, in-transit shipments, commitments,
communication links, and temporary coalitions.

Production is an explicitly counted exogenous material inflow. Demand service
is an explicitly counted sink. A shipment removes inventory at dispatch and
adds it at arrival. Every step verifies:

`initial inventory + cumulative production = current inventory + in-transit + cumulative delivered`.

The commercial default has 11 agents (2 suppliers, 2 manufacturers, 2
carriers, 2 warehouses, 3 retailers). The humanitarian default has 10 agents
(2 NGOs, 1 agency, 2 transport providers, 2 depots, 2 clinics, 1 community
coordinator). Smaller pilot topologies retain at least one demand node.

## Private variables

Every agent receives only its own exact inventory, capacity/impairment, demand,
backlog, delay, service shortfall, commitment strain, private marginal cost,
and noisy local forecast. Public identity and topology are visible. Other
operational information crosses boundaries through messages or binned public
summaries. The evaluator alone can call `full_state_for_evaluator()`.

## Disruptions

- nominal: no exogenous impairment;
- moderate: one upstream node loses 45% capacity and one physical route closes;
- correlated: multiple upstream/logistics nodes lose 80% capacity, regional
  demand grows, a quarter of physical routes close, and lead times increase;
- compound: correlated capacity/demand shock plus a warehouse/depot outage,
  roughly one-third route closure, a two-period network lead penalty, and the
  configured communication degradation;
- reliable, intermittent, and partitioned communication graphs independently
  determine packet delivery.

The richer physical-route/lead-time implementation was added while the initial
pilot was already running. Therefore that pilot is explicitly `pilot v1` for
planner/monitor/throughput diagnosis. A separate stress pilot must validate the
revised compound dynamics before protocol freeze; v1 outcomes will not be mixed
with v2 as if they came from one data-generating process.

The locked holdout uses `holdout_nine_agent`: two sparse regional communication
paths joined by one bridge and a reduced source--destination graph. This is a
real connectivity change, not only a topology label. Every demand node remains
reachable from at least two sources where the population permits it.

These are abstract research environments, not behavioral models of real firms
or humanitarian actors.

## Evaluator metrics and replay

In addition to primary service loss, the evaluator records fulfillment,
backlog, recovery, cost, fairness, welfare, minimum/mean local utility, material
and transport efficiency, on-time delivery, messages/bytes/delivery rate,
offers/agreements/individual rationality, breaches, memory accuracy, trust
calibration, disclosure, and coalition usefulness. Metrics absent an
opportunity retain an explicit denominator/count so they are not interpreted as
population estimates.

Every episode records simulator tool calls and results in a compressed event
ledger. `scripts/replay-results.sh` instantiates the frozen scenario and replays
the quantitative transitions from those calls without invoking an LLM. Main,
ablation, and holdout replays must match period metrics and tool-result codes.
