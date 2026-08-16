# V7 humanitarian multi-commodity environment

## Domain state

The simulator contains depots, hubs, shelters, clinics, five commodity classes
(food, water, medical supplies, fuel, and shelter material), shared vehicles,
fuel, emergency reserves, route capacity, route travel time, local demand,
backlog, priorities, commitments, and shipments in transit.

Persistent agents represent NGOs, transport organizations, local authorities,
assessment teams, clinics, shelters, depots, and hubs. Every agent controls or
observes multiple locations. Scopes overlap, but peer observations remain
private unless a message is delivered.

## Domain-specific transitions

- demand evolves at each step and is served from destination inventory;
- dispatch removes stock from a source and reserves a vehicle and fuel;
- cargo arrives only after the graph-derived route travel time;
- correlated aftershocks disable route edges and communication edges;
- lost cargo is recorded explicitly;
- water shortage raises later medical need, producing a cross-commodity
  cascade;
- priority changes can help the selected site while reducing other sites'
  effective priority, allowing bounded harm;
- a cancelled healthy dispatch is harmful, while cancelling a risky dispatch
  can help.

## Accounting

Commodity conservation is reconstructed as initial stock minus inventory at
all locations, in-transit stock, consumed stock, and losses. Vehicle and fuel
balances are reconstructed separately. A deliberate fault-injection test adds
unexplained stock and must make the audit fail.

## Outcome vector

The simulator retains weighted unmet critical need, service-loss AUC, simulated
critical-shortage exposure, time to first critical delivery, delivery volume,
waste, allocation inequality, commitment failure, harmful/beneficial/neutral
actions, causal utility, service-reaching actions, and communication cost. The
simulated shortage metric is not a validated mortality model.
