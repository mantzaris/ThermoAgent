# ThermoHITL actionability diagnostics

Status: complete; Gate 2 passed after a retained failed qualification and one
versioned development retry.

The v2 antecedent established that many proposed material actions failed
validation or did not reach demand. V3 will measure the full funnel for every
material proposal:

1. first-pass structured output;
2. static and dynamic tool validation;
3. optional single repair;
4. material action accepted;
5. shipment entered transit;
6. arrival at the next intended node;
7. arrival at a demand node;
8. short-horizon loss difference;
9. episode-level paired loss difference.

Each stage will be event sourced with a proposal identifier. Development may
adjust v3-only horizon, lead time, prompt affordances, typed schema, routing, or
repair prompt until gate 2 passes. The frozen v1/v2 simulator is not modified.

## Results

The v2 antecedent contained 1,983 material proposals, of which 408 were
accepted (20.57%) and 396 arrived (97.06% conditional on acceptance). This
localized the main old failure to validation/acceptance.

The deterministic final-candidate stage made 5,045 structured proposals at
100% validity. It accepted 1,191 material actions; 96.73% progressed to the
next stage and demand. Paired probes changed the primary outcome 53 times
commercially and 45 times humanitarianly.

The retained Qwen v8 run made 305 proposals at 82.95% first/final validity.
Forty failures selected a target outside the agent-local affordance, eleven
invented a tool, and one omitted fields. All repairs failed because the repair
instruction was encoded after the Qwen assistant-generation marker. All 15
accepted material actions nevertheless reached demand.

Prompt v9 encoded the repair as a new user turn and foregrounded exact local
tool/target lists. The four-episode versioned retry made 284 proposals: 97.89%
were valid first pass and 100% after at most one repair. Nineteen material
actions were accepted and 16 (84.21%) reached both the next stage and demand.
Commercial and humanitarian qualifications both passed. No failed row was
deleted or overwritten.
