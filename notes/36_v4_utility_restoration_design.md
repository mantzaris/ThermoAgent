# V4 utility-restoration simulator design

## Scope and safety boundary

The third application is a defensive, abstract simulator of electric-utility
service restoration logistics after stochastic cyber-physical disruption. It
does not connect to, scan, probe, model protocols for, or issue commands to any
real utility or operational-technology asset. Cyber events are categorical
simulator state changes: missing, delayed, conflicting, or low-confidence
telemetry; command unavailability; abstract resource-database inconsistency;
and correlated physical failure.

## Layered network

The service layer contains distribution zones, a substation, a microgrid, and
critical hospital, water, shelter, and communications loads. The communication
layer is a separately mutable graph carrying bounded belief sketches and
messages. The restoration-logistics layer contains field crews, a repair depot,
spares, fuel, mobile generation, travel routes, and verification queues. A
service edge, communication link, and logistics route are never treated as the
same edge.

## Independent organizations

Persistent agents represent a distribution zone, substation, microgrid,
crew dispatch, parts depot, mobile-generation/fuel logistics, hospital, water,
communications, and incident coordination. Each has a separate private-state
vault, utility, context, memory, inbox, commitment ledger, and typed authority.
No agent owns another agent's vault. The incident coordinator receives only
explicit messages and public/coarse sketches; it is not an evaluator-global
planner.

## Matched observability mechanism

Development panels deliberately match local visible service severity while
varying belief coherence. In a coherent physical fault, agents receive
consistent evidence and can restore service autonomously. In a fragmented or
potentially corrupted telemetry case, the same local service deficit is paired
with incompatible private beliefs. A bounded operator can authorize
verification, temporary information sharing, an emergency logistics edge, or
a priority change. These interventions modify information or feasible action
space; they do not directly set the outcome.

This mechanism is prospectively falsifiable. Thermodynamic information should
add little once all information is public, and shuffled or KPI-stratum-permuted
entropy/disagreement should not reproduce the benefit. Commercial logistics is
the pre-specified boundary application because its local KPI view includes the
actionable route/contract flags that are absent from fragmented humanitarian
and utility views.

## Conserved quantities and feasibility

The utility simulator accounts for spare parts, crews, crew time, fuel, mobile
generators, repair capacity, service capacity, commitments, and resource
transit. It rejects negative inventories, duplicate crew/generator assignment,
impossible concurrent use, and service creation without repair, local
generation, or an authorized emergency resource. Every accepted action and
stage transition is event sourced.
