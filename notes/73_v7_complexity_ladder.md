# V7 prospective complexity ladder

This note records the engineering factors before pilot outcomes are analyzed.

The three simulator levels are 12/28/52 persistent agents, 8/16/30 operational
nodes, and 30/60/100 time steps. Complexity is not implemented by cloning
independent incidents. Each level changes the number of agents sharing assets,
the graph, resource contention, action delay exposure, communication paths,
and number of correlated disruptions.

Coupling, information fragmentation, and network disruption each have fixed
parameter encodings of 0.20, 0.55, and 0.85 for low, medium, and high. These
encodings were chosen as simulator design levels, not from comparative outcome
inspection. The pilot includes factorial off-diagonal combinations (high
coupling/low fragmentation and low coupling/high fragmentation) so an
interaction is identifiable rather than inferred from one diagonal.

Humanitarian topology families are random geometric, small-world, and modular.
Utility families are grid, scale-free, and modular. Structural diagnostics and
graph hashes are stored. A test rejects graph-isomorphic topology families.

The formal "high complexity" region, effect-size threshold, panel count, and
held-out topology plan will be frozen only after retained-pilot variance and
runtime profiling. No validation or holdout inputs have been executed.
