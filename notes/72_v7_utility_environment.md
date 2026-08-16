# V7 defensive cyber-physical utility-restoration environment

## Scope and safety boundary

The utility application is an offline abstract simulation. "Cyber" disruptions
are corrupted or stale telemetry, disabled communication edges, inconsistent
state, and defensive isolation. The code contains no target, protocol,
credential, exploit, malware, or real operational procedure.

## Layers and resources

The environment represents a topology-dependent physical service graph and a
separate ad-hoc agent communication graph. Nodes are sources, distribution
assets, or critical loads. Persistent organizations include zone operators,
crew dispatch, cyber defense, communications, resource allocation, and
critical-load representatives.

Shared resources include crews, spares, mobile generation, and fuel. Repair
actions require crew travel and a spare; restoration completes after a delay.
Reconfiguration changes actual service-edge feasibility. Defensive isolation
can prevent propagation but removes current service and is harmful when based
on wrong evidence. Mobile generation consumes a conserved generator and fuel.

## Cascades and outcomes

Service is propagated by graph reachability from source nodes through available
and non-isolated components. Correlated physical failures and telemetry
integrity loss start after the nominal period. Failed neighbors produce a
coupling-dependent probability of later component failure. Outcomes include
critical unserved load, service-loss AUC, restoration time, cascade count and
depth, unsafe switching/isolation, crew utilization and travel, duplicate work
orders, causal utility, service-reaching actions, belief-confidence recovery,
and fully counted communication.
