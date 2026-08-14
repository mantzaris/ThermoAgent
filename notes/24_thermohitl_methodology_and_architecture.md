# ThermoHITL methodology and architecture

## Motivation from v2

V2 saved communication but its entropy trigger activated zero times; no
communication dominated DOET-rule, and many proposed actions never became
accepted demand-reaching flow. V3 therefore required an observed causal chain
and six gates before any expensive evaluation. It did not retune or replace v2.

## Design alternatives and decisions

- **Operator implementation:** chose bounded simulated operators for scalable
  experiments; actual participants were excluded absent IRB/ethics authority.
- **Dashboard stack:** chose a dependency-light HTML/SVG server over Streamlit
  so replay works in a fresh clone without GPU or new packages.
- **Supervisory authority:** chose typed, scoped changes to feasibility,
  information, priority, or constraints. A prose message cannot mutate state.
- **Counterfactual evaluation:** chose paired cloning from an eligible state,
  restoring exogenous RNG, agent state, queue, workload, commitments, and view.
- **Learned allocator:** planned compact PPO/contextual-bandit variants, but
  prospective Gate 5 failed before training; no model was fit.
- **Scientific stop:** required all gates, rather than accepting humanitarian
  success alone. This prevented post-hoc domain selection.

## Shared architecture

Agents continue autonomous local planning in quiet mode, exchange bounded
coarse sketches over the available communication graph, independently escalate,
and continue if the operator cannot respond. The allocator ranks only authorized
view payloads. Completed operator actions enter the same typed tool/event path as
other simulator mutations. Advisory actions may be refused; mandatory override
scope and expiry are logged; authority is explicitly returned.

Primary outcomes remain commercial service-loss AUC and humanitarian cumulative
weighted unmet need. One complete episode is the outcome unit.
