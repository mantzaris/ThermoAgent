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

For v4 utility restoration, the normal view may show a coarse visible telemetry
confidence state and conflicting-summary disagreement. It may not show the true
cyber-compromise label, true incident mode, hidden resource requirement, raw
private telemetry, or future cascade. The dashboard builds v4 frames only from
topology, explicit messages/actions/material results, and validated
`operator_view_v4` events; evaluator episode metrics are analysis-only.

The simulated operator may alter only a typed authority, information, priority,
constraint, or feasibility record. Advisory directives preserve agent refusal;
temporary emergency overrides are mandatory only for their recorded scope and
duration, after which authority returns to the agents.

## V6 evaluator-only replay

V6 normal frames are reconstructed exclusively from hashed `operator_view`
events whose audience is `simulated_operator`. They can show the proposed
action, action-value margin, local/generalized uncertainty, pooled uncertainty,
disagreement, consensus residual, sketch contributors, missing agents, and the
bounded queue state. They cannot show the incident's true mode, correct action,
future stochastic tape, or matched causal effect.

The optional evaluator-analysis panel is a separate endpoint and is never
loaded by the ordinary dashboard. It reads only
`v6_counterfactual_branch` events addressed `private_to="evaluator"`, carries a
permanent privileged-analysis warning, and exposes matched loss with/without an
action solely for retrospective causal-chain inspection. No agent, delegation
policy, or simulated-operator policy can call this view during execution.
