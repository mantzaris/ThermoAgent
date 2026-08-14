# Agent independence and operator authority

Each organization owns a distinct private vault, identity/role prompt, utility,
memory, commitments, inbox/outbox, planner state, tool permissions, RNG state,
distributed monitor state, and escalation controller. Shared batching does not
share hidden context. Information crosses boundaries only through event-sourced
messages, public infrastructure, coarse sketches, and signed directives.

The simulator may validate, reject, and apply a typed action. It never repairs
an invalid domain decision, chooses an offer response, or accepts a coalition on
an agent's behalf. Agents retain accept/reject/counter/withdraw authority.

Normal operator views cannot contain raw private costs, exact private inventory
or capacity, memory, agent RNG, future disruptions, evaluator-global entropy or
energy, or counterfactual results. Every view is validated and hashed before
allocation. The evaluator oracle uses a separate privileged schema.

Operator tools include bounded emergency route/resource authorization, temporary
coarse sharing, priority adjustment, constraint relaxation, conflict resolution,
temporary emergency override, and return of control. Tests cover advisory
refusal, mandatory scope, duration, conservation, privacy injection, hash
mutation, and autonomy transitions.

