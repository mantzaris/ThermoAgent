# V7 environment architecture

## Decision timing

This design record was written during Stage B, after the V6 audit and before
the retained V7 feasibility-pilot analysis was opened.

## Shared infrastructure boundary

V7 shares only:

- append-only event sourcing and deterministic ledger IDs;
- persistent-agent privacy vaults, inboxes, outboxes, and commitments;
- structural graph generation and diagnostics;
- explicit message delivery and per-edge communication accounting;
- generalized-information calculations;
- Level-2 risk-controller and replay interfaces.

`HumanitarianV7Environment` and `UtilityRestorationV7Environment` implement
separate domain initialization, observations, disruption mechanics, action
validation, delayed completions, resource accounting, service transitions,
and outcome vectors. Neither application calls a common service-deficit
transition function.

## Complexity ladder used for engineering

| level | persistent agents | operational nodes | horizon | decision interval |
|---|---:|---:|---:|---:|
| small | 12 | 8 | 30 | 3 |
| medium | 28 | 16 | 60 | 4 |
| large | 52 | 30 | 100 | 5 |

These values meet the requested target ranges while remaining feasible on one
RTX 4090 and on local CPU for deterministic diagnostics. Formal sample sizes
and exact complexity regions remain unfrozen until retained pilot profiling is
complete.

## Coupling mechanisms

Coupling is causal rather than a label. Humanitarian agents contend for shared
vehicles, fuel, reserves, routes, and destination priorities. Water shortages
increase later medical demand. Utility agents contend for field crews, spares,
mobile generation, fuel, service edges, and communication restoration.
Physical failures can disconnect downstream loads and can induce dependent
failures. In both applications, partitions disable actual message delivery.

Each completed action writes a causal-chain identifier and stages. Current
engineering chains include scheduling, resource/authority change, arrival or
restoration, and service-outcome change.

## Information boundary

Agent policies receive only private observations, their persistent beliefs and
memory, commitments, and explicitly delivered messages. Generalized entropy
uses messages actually delivered through available graph edges. Evaluator-only
counterfactual branches and conservation audits are marked `private_to`
`evaluator`; they are not deployable inputs.
