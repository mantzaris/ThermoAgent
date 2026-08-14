# Dashboard information boundaries

The dashboard is part of the tested decision mechanism. It renders the same
serialized payload supplied to the simulated operator; it does not query the
environment for a richer display-only state.

Normal execution permits only:

- explicitly shared public infrastructure and communication availability;
- requesting-agent identity, role, request reason, and coarse requested action;
- locally available KPI fields included by the assigned view condition;
- bounded privacy-preserving entropy/energy sketches included by that view;
- queue, workload, fatigue, latency, and prior typed operator actions; and
- public acknowledgements and material-stage outcomes.

Normal execution forbids raw observations from another agent, exact private
utility/cost/inventory/capacity, memory contents, RNG state, future disruption,
evaluator-global entropy or energy, and counterfactual outcomes. The
`evaluator_global_oracle` condition is separately labeled and never used as a
feasible primary policy.

Every operator payload is schema validated, canonically serialized, SHA-256
hashed, and logged before attention allocation or intervention. Replay checks
the stored hash. Tests inject forbidden fields and require rejection.

The simulated operator may alter only a typed authority, information, priority,
constraint, or feasibility record. Advisory directives preserve agent refusal;
temporary emergency overrides are mandatory only for their recorded scope and
duration, after which authority returns to the agents.

